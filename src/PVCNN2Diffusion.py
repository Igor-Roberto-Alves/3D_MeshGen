"""
PVD-style raw-point-cloud diffusion backbone (PVCNN2), with class
conditioning added on top, and using the pure-PyTorch modules from
src/pvcnn_modules.py (no CUDA extensions to compile).

This is an alternative to the LION-style latent diffusion pipeline already
documented in ARQUITETURA_DIFFUSION.md: instead of denoising a small latent
(z_g, z_l) produced by a frozen VAE, this model denoises the point cloud
itself, shaped (B, 3 + extra_feature_channels, N).

Kept as new files on purpose: nothing in the existing training scripts
imports this, so the VAE / StyleDenoiser / LatentPointDenoiser pipeline is
untouched.

Class conditioning
-------------------
The base PVCNN2 architecture only conditions on the diffusion timestep `t`
(via `temb`). Conditioning on the object category is added here the same way
ADM / guided-diffusion condition U-Nets on class labels: a learned embedding
table (`self.class_embed`, with one extra "unconditional" row for
classifier-free guidance) is summed into `temb` before it gets broadcast to
every point. `forward(inputs, t, class_idx)` keeps the same positional
signature the project's `CosineSchedule.sample`/`_p_step` already call
(`denoiser(xt, t, cond)`), so it plugs into src/Diffusion.py's existing
sampling loop without changes there:

    model = PVCNN2(num_categories=55, ...)
    uncond = torch.full_like(class_idx, model.uncond_idx)
    schedule.sample(model, shape, condition=class_idx, uncond=uncond, guidance=cfg.guidance)

During training, pass the real `class_idx` each step; a fraction
(`cfg_dropout`) of the batch is automatically replaced by the unconditional
token so the model also learns the unconditional score (needed for CFG).
"""
import functools

import numpy as np
import torch
import torch.nn as nn

from src.pvcnn_modules import (
    SharedMLP,
    PVConv,
    PointNetSAModule,
    PointNetAModule,
    PointNetFPModule,
    Attention,
    Swish,
)


def _linear_gn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels), nn.GroupNorm(8, out_channels), Swish())


def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):
    r = width_multiplier
    block = _linear_gn_relu if dim == 1 else SharedMLP

    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_gn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))
    return layers, (out_channels[-1] if classifier else int(r * out_channels[-1]))


def create_pointnet2_sa_components(sa_blocks, extra_feature_channels, embed_dim=64, use_att=False,
                                    dropout=0.1, with_se=False, normalize=True, eps=0,
                                    width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier
    in_channels = extra_feature_channels + 3

    sa_layers, sa_in_channels = [], []
    c = 0
    for conv_configs, sa_configs in sa_blocks:
        k = 0
        sa_in_channels.append(in_channels)
        stage_blocks = []

        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = (c + 1) % 2 == 0 and use_att and p == 0
                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(
                        PVConv, kernel_size=3, resolution=int(vr * voxel_resolution), attention=attention,
                        dropout=dropout, with_se=with_se, with_se_relu=True, normalize=normalize, eps=eps,
                    )
                if c == 0:
                    stage_blocks.append(block(in_channels, out_channels))
                elif k == 0:
                    stage_blocks.append(block(in_channels + embed_dim, out_channels))
                in_channels = out_channels
                k += 1
            extra_feature_channels = in_channels

        num_centers, radius, num_neighbors, out_channels = sa_configs
        out_channels = [int(r * oc) for oc in out_channels]
        block = PointNetAModule if num_centers is None else functools.partial(
            PointNetSAModule, num_centers=num_centers, radius=radius, num_neighbors=num_neighbors
        )
        stage_blocks.append(
            block(in_channels=extra_feature_channels + (embed_dim if k == 0 else 0),
                  out_channels=out_channels, include_coordinates=True)
        )
        c += 1
        in_channels = extra_feature_channels = stage_blocks[-1].out_channels
        sa_layers.append(stage_blocks[0] if len(stage_blocks) == 1 else nn.Sequential(*stage_blocks))

    return sa_layers, sa_in_channels, in_channels, (1 if num_centers is None else num_centers)


def create_pointnet2_fp_modules(fp_blocks, in_channels, sa_in_channels, embed_dim=64, use_att=False,
                                 dropout=0.1, with_se=False, normalize=True, eps=0,
                                 width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier

    fp_layers = []
    c = 0
    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
        stage_blocks = []
        out_channels = tuple(int(r * oc) for oc in fp_configs)
        stage_blocks.append(
            PointNetFPModule(in_channels=in_channels + sa_in_channels[-1 - fp_idx] + embed_dim, out_channels=out_channels)
        )
        in_channels = out_channels[-1]

        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = (c + 1) % 2 == 0 and use_att and p == 0
                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(
                        PVConv, kernel_size=3, resolution=int(vr * voxel_resolution), attention=attention,
                        dropout=dropout, with_se=with_se, with_se_relu=True, normalize=normalize, eps=eps,
                    )
                stage_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels
        fp_layers.append(stage_blocks[0] if len(stage_blocks) == 1 else nn.Sequential(*stage_blocks))
        c += 1

    return fp_layers, in_channels


class PVCNN2Base(nn.Module):
    """
    num_classes:      output channels of the per-point classifier head (3 to predict
                      xyz noise) — NOT the number of object categories.
    num_categories:   number of object categories for class conditioning (e.g. 55).
    cfg_dropout:      probability of replacing class_idx by the unconditional token
                      during training, so classifier-free guidance works at sampling time.
    """

    def __init__(self, num_classes, embed_dim, num_categories, use_att, dropout=0.1,
                 extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1,
                 cfg_dropout=0.1):
        super().__init__()
        assert extra_feature_channels >= 0
        self.embed_dim = embed_dim
        self.in_channels = extra_feature_channels + 3
        self.num_categories = num_categories
        self.uncond_idx = num_categories
        self.cfg_dropout = cfg_dropout

        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        self.global_att = None if not use_att else Attention(channels_sa_features, 8, D=1)

        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, with_se=True,
            embed_dim=embed_dim, use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[128, dropout, num_classes],
                                           classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

        self.embedf = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        # +1 row for the unconditional token used by classifier-free guidance.
        self.class_embed = nn.Embedding(num_categories + 1, embed_dim)

    def get_timestep_embedding(self, timesteps, device):
        assert len(timesteps.shape) == 1

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb

    def forward(self, inputs, t, class_idx=None):
        B, device = inputs.shape[0], inputs.device

        if class_idx is None:
            class_idx = torch.full((B,), self.uncond_idx, dtype=torch.long, device=device)
        elif self.training and self.cfg_dropout > 0:
            drop = torch.rand(B, device=device) < self.cfg_dropout
            class_idx = torch.where(drop, torch.full_like(class_idx, self.uncond_idx), class_idx)

        temb = self.embedf(self.get_timestep_embedding(t, device)) + self.class_embed(class_idx)
        temb = temb[:, :, None].expand(-1, -1, inputs.shape[-1])

        coords, features = inputs[:, :3, :].contiguous(), inputs
        coords_list, in_features_list = [], []
        for i, sa_blocks in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            if i == 0:
                features, coords, temb = sa_blocks((features, coords, temb))
            else:
                features, coords, temb = sa_blocks((torch.cat([features, temb], dim=1), coords, temb))
        in_features_list[0] = inputs[:, 3:, :].contiguous()

        if self.global_att is not None:
            features = self.global_att(features)

        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords, temb = fp_blocks(
                (coords_list[-1 - fp_idx], coords, torch.cat([features, temb], dim=1),
                 in_features_list[-1 - fp_idx], temb)
            )

        return self.classifier(features)


class PVCNN2(PVCNN2Base):
    """
    Ready-to-instantiate PVCNN2 for 2048-point clouds (matches the project's
    default n_points=2048 in src/dataset.py). Predicts per-point xyz noise
    (num_classes=3), conditioned on timestep + object category.

    If you sample point clouds with a different N, re-tune num_centers in
    sa_blocks (roughly N/2, N/8, N/32, N/128) so the FPS downsampling ratios
    stay sensible.
    """

    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, num_categories=55, embed_dim=64, use_att=True, dropout=0.1,
                 extra_feature_channels=0, width_multiplier=1, voxel_resolution_multiplier=1,
                 cfg_dropout=0.1):
        super().__init__(
            num_classes=3, embed_dim=embed_dim, num_categories=num_categories, use_att=use_att, dropout=dropout,
            extra_feature_channels=extra_feature_channels, width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier, cfg_dropout=cfg_dropout,
        )
