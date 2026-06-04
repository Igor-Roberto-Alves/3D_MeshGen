class UpgradedAdaGNBlock(nn.Module):
    """Deep AdaGN Block with a Residual MLP path for cross-channel styling."""
    def __init__(self, channels, latent_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels) # 8 groups for better granularity
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Residual channel adjustment to give the network more expression capacity
        self.conv_res = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x, z):
        norm_x = self.norm(x)
        style = self.fc(z).unsqueeze(-1)
        gamma, beta = torch.chunk(style, 2, dim=1)
        
        # Modulate features and blend with a residual transformation pass
        out = norm_x * (1 + gamma) + beta
        return out + self.conv_res(x)


class MiniLionDecoder(nn.Module):
    """
    Decoder LION otimizado para o Sanity Check.
    Separa as cabeças de predição para evitar o travamento das normais.
    """
    def __init__(self, latent_dim=512, n_out=2048):
        super().__init__()
        self.n_out = n_out
        self.n_seed = 256
        
        self.seed_points = nn.Parameter(torch.randn(1, 3, self.n_seed) * 0.1)
        self.seed_features = nn.Parameter(torch.randn(1, 128, self.n_seed))

        self.layer1_conv = nn.Conv1d(128, 256, kernel_size=1)
        self.layer1_gn   = UpgradedAdaGNBlock(256, latent_dim)
        
        self.layer2_conv = nn.Conv1d(256, 256, kernel_size=1)
        self.layer2_gn   = UpgradedAdaGNBlock(256, latent_dim)

        self.layer3_conv = nn.Conv1d(256, 512, kernel_size=1)
        self.layer3_gn   = UpgradedAdaGNBlock(512, latent_dim)

        self.layer4_conv = nn.Conv1d(512, 512, kernel_size=1)
        self.layer4_gn   = UpgradedAdaGNBlock(512, latent_dim)

        self.layer5_conv = nn.Conv1d(512, 1024, kernel_size=1)
        self.layer5_gn   = UpgradedAdaGNBlock(1024, latent_dim)

        self.expand_ratio = n_out // self.n_seed # Ex: 8
        
        # Corpo do upsampling comum
        self.upsample_base = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=1),
            nn.GELU(),
        )
        
        # ── SEPARAÇÃO DAS CABEÇAS (Dá caminhos de gradiente independentes) ──
        # Cabeça para Coordenadas XYZ
        self.xyz_head = nn.Conv1d(1024, self.expand_ratio * 3, kernel_size=1)
        # Cabeça dedicada para as Normais
        self.normal_head = nn.Conv1d(1024, self.expand_ratio * 3, kernel_size=1)

    def forward(self, z):
        B = z.shape[0]
        
        if z.dim() > 2:
            z = z.view(B, -1)

        feat = self.seed_features.expand(B, -1, -1)

        # Modulação profunda de estilo
        feat = F.gelu(self.layer1_conv(feat))
        feat = self.layer1_gn(feat, z)

        feat = F.gelu(self.layer2_conv(feat))
        feat = self.layer2_gn(feat, z)

        feat = F.gelu(self.layer3_conv(feat))
        feat = self.layer3_gn(feat, z)

        feat = F.gelu(self.layer4_conv(feat))
        feat = self.layer4_gn(feat, z)
        feat = F.gelu(self.layer5_conv(feat))
        feat = self.layer5_gn(feat, z)

        # Upsampling de canais base
        feat = self.upsample_base(feat)
        
        # Projeta separadamente
        out_xyz = self.xyz_head(feat).view(B, 3, self.n_out).transpose(1, 2)
        out_normal = self.normal_head(feat).view(B, 3, self.n_out).transpose(1, 2)

        out_xyz = out_xyz.clamp(-1, 1)  # Limita as coordenadas para evitar explosões
        normals = out_normal / (out_normal.norm(dim=-1, keepdim=True) + 1e-8)

        return {"recon": torch.cat([out_xyz, normals], dim=-1)}
# ── MAIN COMPOSITE VAE SYSTEM ──────────────────────────────────────────────────