import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


def normalize_pc(x):
    """Centre and unit-scale a point cloud. x: (B, N, C), returns same shape."""
    xyz = x[..., :3]
    centre = xyz.mean(dim=1, keepdim=True)
    xyz = xyz - centre
    scale = xyz.norm(dim=-1).max(dim=1, keepdim=True)[0].unsqueeze(-1).clamp(min=1e-6)
    xyz = xyz / scale
    if x.shape[-1] > 3:
        out = x.clone()
        out[..., :3] = xyz
        return out
    return xyz


class MAB(nn.Module):
    """Multi-head Attention Block. x attends to y."""
    def __init__(self, dim_q, dim_kv, dim_out, num_heads, ln=True, slot_att=False, dropout=0.0):
        super().__init__()
        assert dim_out % num_heads == 0
        self.num_heads = num_heads
        self.dim_split = dim_out // num_heads
        self.slot_att = slot_att
        self.fc_q = nn.Linear(dim_q, dim_out, bias=False)
        self.fc_k = nn.Linear(dim_kv, dim_out, bias=False)
        self.fc_v = nn.Linear(dim_kv, dim_out, bias=False)
        self.fc_o = nn.Linear(dim_out, dim_q, bias=False)
        self.ffn1 = nn.Linear(dim_q, dim_q)
        self.ffn2 = nn.Linear(dim_q, dim_q)
        if ln:
            self.ln1 = nn.LayerNorm(dim_q)
            self.ln2 = nn.LayerNorm(dim_q)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        # x: (B, N, dim_q), y: (B, M, dim_kv)
        B = x.shape[0]
        q = self.fc_q(x)  # (B, N, D)
        k = self.fc_k(y)  # (B, M, D)
        v = self.fc_v(y)  # (B, M, D)

        # split heads: (H*B, N, Dh)
        q_ = torch.cat(q.split(self.dim_split, -1), 0)
        k_ = torch.cat(k.split(self.dim_split, -1), 0)
        v_ = torch.cat(v.split(self.dim_split, -1), 0)

        scale = math.sqrt(self.dim_split)
        attn = q_.bmm(k_.transpose(1, 2)) / scale  # (H*B, N, M)

        if self.slot_att:
            # softmax over queries (slot attention)
            attn = attn.softmax(dim=1)
            att = attn.bmm(v_)  # (H*B, N, Dh)
            att = att / (attn.sum(dim=-1, keepdim=True) + 1e-5)
        else:
            attn = attn.softmax(dim=2)
            att = attn.bmm(v_)  # (H*B, N, Dh)

        att = torch.cat(att.split(B, 0), -1)  # (B, N, D)
        att = self.fc_o(att)
        if hasattr(self, 'dropout'):
            att = self.dropout(att)
        o = x + att
        if hasattr(self, 'ln1'):
            o = self.ln1(o)
        ff = self.ffn2(F.relu(self.ffn1(o)))
        if hasattr(self, 'dropout'):
            ff = self.dropout(ff)
        o = o + ff
        if hasattr(self, 'ln2'):
            o = self.ln2(o)
        return o


class ISAB(nn.Module):
    """Induced Set Attention Block for encoding."""
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=True, dropout=0.0):
        super().__init__()
        self.i = nn.Parameter(torch.empty(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.i)
        self.att_proj = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln, slot_att=True, dropout=dropout)
        self.att_broad = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln, slot_att=False, dropout=dropout)

    def project(self, x):
        # x: (B, N, D) → h: (B, M, D)
        i = self.i.expand(x.shape[0], -1, -1)
        return self.att_proj(i, x)

    def forward(self, x):
        h = self.project(x)
        o = self.att_broad(x, h)
        return o, h


class ElemMLP(nn.Module):
    """Element-wise MLP applied to (B, M, D) tensors."""
    def __init__(self, dim_in, dim_hidden, dim_out, n_layers=2):
        super().__init__()
        layers = []
        in_d = dim_in
        for _ in range(n_layers):
            layers += [nn.Linear(in_d, dim_hidden), nn.ReLU()]
            in_d = dim_hidden
        layers.append(nn.Linear(in_d, dim_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class InitialSet(nn.Module):
    """GMM-based seed set for the decoder. Fibonacci sphere init."""
    def __init__(self, num_points, n_mixtures, dim_seed, dim_out, train_gmm=True):
        super().__init__()
        self.num_points = num_points
        self.n_mixtures = n_mixtures
        self.dim_seed = dim_seed
        self.tau = 1.0

        if n_mixtures == 1:
            self.mu = nn.Parameter(torch.randn(1, 1, dim_seed))
            self.logvar = nn.Parameter(torch.zeros(1, 1, dim_seed))
        else:
            mu = torch.randn(n_mixtures, dim_seed)
            mu = F.normalize(mu, dim=-1)  # spread on sphere
            logits = torch.ones(n_mixtures)
            sig = torch.ones(n_mixtures, dim_seed) * 0.1
            if train_gmm:
                self.logits = nn.Parameter(logits)
                self.mu = nn.Parameter(mu)
                self.sig = nn.Parameter(sig)
            else:
                self.register_buffer('logits', logits)
                self.register_buffer('mu', mu)
                self.register_buffer('sig', sig)
        self.output = nn.Linear(dim_seed, dim_out)

    def forward(self, B, device=None):
        if device is None:
            device = self.output.weight.device
        N = self.num_points

        if self.n_mixtures == 1:
            eps = torch.randn(B, N, self.dim_seed, device=device)
            x = self.mu + (self.logvar / 2).exp() * eps
        else:
            eps = torch.randn(B, N, self.dim_seed, device=device)
            logits = self.logits.reshape(1, 1, self.n_mixtures).expand(B, N, -1)
            onehot = F.gumbel_softmax(logits, tau=self.tau, hard=True)  # (B, N, K)
            mu = self.mu.reshape(1, 1, self.n_mixtures, self.dim_seed)
            sig = self.sig.reshape(1, 1, self.n_mixtures, self.dim_seed).abs().clamp(min=1e-5)
            # select per point
            mu_sel = (mu * onehot.unsqueeze(-1)).sum(2)   # (B, N, D)
            sig_sel = (sig * onehot.unsqueeze(-1)).sum(2)  # (B, N, D)
            x = mu_sel + sig_sel * eps
        return self.output(x)


class DecoderBlock(nn.Module):
    """Stochastic Attentive Bottleneck Layer."""
    def __init__(self, hidden_dim, z_dim, num_heads, num_inds, ln=True, dropout=0.0, slot_att=True, cond_prior=True):
        super().__init__()
        self.num_inds = num_inds
        self.z_dim = z_dim
        self.cond_prior = cond_prior
        D = hidden_dim

        self.i = nn.Parameter(torch.empty(1, num_inds, D))
        nn.init.xavier_uniform_(self.i)
        self.att_proj = MAB(D, D, D, num_heads, ln=ln, slot_att=True, dropout=dropout)
        self.att_broad = MAB(D, D, D, num_heads, ln=ln, slot_att=False, dropout=dropout)

        if cond_prior:
            self.prior = ElemMLP(D, D, 2 * z_dim, n_layers=2)
        else:
            self.prior_param = nn.Parameter(torch.zeros(1, num_inds, 2 * z_dim))

        self.posterior = ElemMLP(D, D, 2 * z_dim, n_layers=2)
        self.fc = nn.Linear(z_dim, D)

    def project(self, o):
        i = self.i.expand(o.shape[0], -1, -1)
        return self.att_proj(i, o)  # (B, M, D)

    def compute_prior(self, h):
        B = h.shape[0]
        if self.cond_prior:
            out = self.prior(h)  # (B, M, 2*z)
        else:
            out = self.prior_param.expand(B, -1, -1)  # (B, M, 2*z)
        mu = out[..., :self.z_dim]
        logvar = out[..., self.z_dim:].clamp(-4., 3.)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps
        return z, mu, logvar

    def compute_posterior(self, mu_p, logvar_p, bu_h, td_h=None):
        B = bu_h.shape[0]
        inp = bu_h + td_h if td_h is not None else bu_h
        out = self.posterior(inp)  # (B, M, 2*z)
        delta_mu = out[..., :self.z_dim]
        delta_logvar = out[..., self.z_dim:].clamp(-4., 3.)
        sigma_p = (logvar_p / 2).exp()
        delta_sigma = (delta_logvar / 2).exp()
        eps = torch.randn_like(mu_p)
        z = (mu_p + delta_mu) + (sigma_p * delta_sigma) * eps
        # KL of N(mu_p+delta_mu, sigma_p*delta_sigma) wrt N(mu_p, sigma_p)
        # = KL of N(delta_mu, delta_sigma) wrt N(0, 1) scaled by sigma_p
        kl = -0.5 * (
            delta_logvar + 1.0
            - delta_mu.pow(2) / sigma_p.pow(2).clamp(min=1e-6)
            - delta_sigma.pow(2)
        ).reshape(B, -1).sum(-1)  # (B,)
        return z, kl

    def forward_prior(self, o):
        h = self.project(o)
        z, _, _ = self.compute_prior(h)
        return self.att_broad(o, self.fc(z))

    def forward_posterior(self, o, bu_h):
        h = self.project(o)
        _, mu_p, logvar_p = self.compute_prior(h)
        td_h = h if self.cond_prior else None
        z, kl = self.compute_posterior(mu_p, logvar_p, bu_h, td_h)
        o = self.att_broad(o, self.fc(z))
        return o, kl


class SetVAE(nn.Module):
    def __init__(
        self,
        input_dim=3,
        num_points=2048,
        hidden_dim=64,
        z_scales=None,
        z_dim=16,
        n_mixtures=4,
        init_dim=32,
        num_heads=4,
        ln=True,
        dropout=0.0,
        slot_att=True,
        train_gmm=True,
    ):
        super().__init__()
        if z_scales is None:
            z_scales = [1, 1, 2, 4, 8, 16, 32]
        self.num_points = num_points
        self.z_dim = z_dim
        self.z_scales = z_scales
        D = hidden_dim

        enc_inds = list(reversed(z_scales))
        dec_inds = z_scales

        self.input_proj = nn.Linear(input_dim, D)
        self.pre_enc = ElemMLP(D, D, D, n_layers=1)
        self.init_set = InitialSet(num_points, n_mixtures, init_dim, D, train_gmm)
        self.pre_dec = ElemMLP(D, D, D, n_layers=1)
        self.post_dec = ElemMLP(D, D, D, n_layers=1)
        self.output_proj = nn.Linear(D, input_dim)

        self.encoder = nn.ModuleList([
            ISAB(D, D, num_heads, m, ln=ln, dropout=dropout)
            for m in enc_inds
        ])
        self.decoder = nn.ModuleList([
            DecoderBlock(D, z_dim, num_heads, m, ln=ln, dropout=dropout,
                         slot_att=slot_att, cond_prior=(i > 0))
            for i, m in enumerate(dec_inds)
        ])

    def encode(self, x):
        feat = self.pre_enc(self.input_proj(x))
        features = []
        for enc in self.encoder:
            feat, h = enc(feat)
            features.append(h)
        return list(reversed(features))  # top-down order

    def decode(self, B, bottom_up=None, device=None):
        if device is None:
            device = next(self.parameters()).device
        o = self.pre_dec(self.init_set(B, device=device))
        kls = []
        for i, dec in enumerate(self.decoder):
            if bottom_up is not None:
                o, kl = dec.forward_posterior(o, bottom_up[i])
                kls.append(kl)
            else:
                o = dec.forward_prior(o)
        xyz = self.output_proj(self.post_dec(o))
        return xyz, kls

    def forward(self, x):
        # x: (B, N, C) — use only xyz
        x_in = normalize_pc(x)[..., :3]
        B = x_in.shape[0]
        features = self.encode(x_in)
        xyz, kls = self.decode(B, bottom_up=features, device=x_in.device)
        return xyz, kls

    def reconstruct(self, x):
        return self.forward(x)[0]

    @torch.no_grad()
    def generate(self, n, device=None):
        if device is None:
            device = next(self.parameters()).device
        xyz, _ = self.decode(n, bottom_up=None, device=device)
        return xyz

    @property
    def z_flat_dim(self) -> int:
        """Total flat latent dimensionality: sum(z_scales) * z_dim."""
        return sum(self.z_scales) * self.z_dim

    @torch.no_grad()
    def encode_latents(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract flattened posterior mean z_flat ∈ ℝ^{sum(z_scales)*z_dim}.
        Uses posterior means (deterministic) — suitable for diffusion training.
        """
        x_in = normalize_pc(x)[..., :3]
        B, device = x_in.shape[0], x_in.device
        features = self.encode(x_in)
        o = self.pre_dec(self.init_set(B, device=device))
        zs = []
        for i, dec in enumerate(self.decoder):
            h = dec.project(o)
            _, mu_p, logvar_p = dec.compute_prior(h)
            td_h = h if dec.cond_prior else None
            # posterior mean: mu_q = mu_p + delta_mu
            inp = features[i] + td_h if td_h is not None else features[i]
            out = dec.posterior(inp)
            delta_mu = out[..., :dec.z_dim]
            mu_q = mu_p + delta_mu          # (B, M_l, z_dim), no sampling
            zs.append(mu_q.reshape(B, -1))  # (B, M_l * z_dim)
            # advance o with posterior mean for next level's context
            o = dec.att_broad(o, dec.fc(mu_q))
        return torch.cat(zs, dim=-1)        # (B, sum(z_scales)*z_dim)

    @torch.no_grad()
    def decode_latents(self, z_flat: torch.Tensor) -> torch.Tensor:
        """
        Decode from a flat latent vector z_flat ∈ ℝ^{sum(z_scales)*z_dim}.
        Inverse of encode_latents — used at diffusion inference time.
        """
        B, device = z_flat.shape[0], z_flat.device
        o = self.pre_dec(self.init_set(B, device=device))
        offset = 0
        for i, (dec, m) in enumerate(zip(self.decoder, self.z_scales)):
            chunk = z_flat[:, offset: offset + m * self.z_dim]   # (B, M_l*z_dim)
            z = chunk.reshape(B, m, self.z_dim)                  # (B, M_l, z_dim)
            offset += m * self.z_dim
            o = dec.att_broad(o, dec.fc(z))
        xyz = self.output_proj(self.post_dec(o))
        return xyz
