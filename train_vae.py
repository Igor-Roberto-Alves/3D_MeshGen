"""
train_vae_gan.py
────────────────
Training loop for DualBranchPointVAE with the GAN discriminator.

Two separate optimisers are maintained:
  • optimiser_G  — VAE parameters (encoders + decoder)
  • optimiser_D  — discriminator parameters only

Each iteration runs in two phases:
  1. D step  : update discriminator on real vs. reconstructed clouds
  2. G step  : update VAE with Chamfer + KL + adversarial losses
"""

import os
import open3d as o3d
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.Vae import DualBranchPointVAE
from src.dataset import Ds_point_sampled, Ds_point_model


# ── visualisation helper ──────────────────────────────────────────────────────

def save_before_after(original_tensor, reconstructed_tensor, epoch, class_name):
    os.makedirs("reconstructions", exist_ok=True)

    orig  = original_tensor.detach().cpu().numpy()
    recon = reconstructed_tensor.detach().cpu().numpy()

    pcd_orig          = o3d.geometry.PointCloud()
    pcd_orig.points   = o3d.utility.Vector3dVector(orig[:, :3])
    pcd_orig.normals  = o3d.utility.Vector3dVector(orig[:, 3:])

    pcd_recon         = o3d.geometry.PointCloud()
    pcd_recon.points  = o3d.utility.Vector3dVector(recon[:, :3])
    pcd_recon.normals = o3d.utility.Vector3dVector(recon[:, 3:])

    path_orig  = f"reconstructions/epoch_{epoch+1}_{class_name}_ORIGINAL.ply"
    path_recon = f"reconstructions/epoch_{epoch+1}_{class_name}_RECONSTRUCTED.ply"

    o3d.io.write_point_cloud(path_orig,  pcd_orig)
    o3d.io.write_point_cloud(path_recon, pcd_recon)
    print(f"\n[INFO] Saved {path_orig} and {path_recon}")


# ── main training function ────────────────────────────────────────────────────

def train_vae_gan(
    model: DualBranchPointVAE,
    dataset,
    num_epochs: int = 200,
    batch_size: int = 16,
    lr_G: float = 1e-3,
    lr_D: float = 1e-4,           # D usually benefits from a slower lr
    warmup_epochs: int = 10,      # train as plain VAE before enabling GAN
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Separate parameter groups so D and G can have different lr / schedules
    vae_params  = (
        list(model.hier_enc.parameters())
        + list(model.glob_enc.parameters())
        + list(model.fusion.parameters())
        + list(model.bottleneck.parameters())
        + list(model.decoder.parameters())
    )
    disc_params = list(model.discriminator.parameters())

    optimiser_G = torch.optim.Adam(vae_params,  lr=lr_G, betas=(0.9,  0.999))
    optimiser_D = torch.optim.Adam(disc_params, lr=lr_D, betas=(0.5,  0.999))

    for epoch in range(num_epochs):
        model.train()
        use_gan = epoch >= warmup_epochs   # GAN kicks in after warm-up

        totals = dict(G=0.0, cd=0.0, kl=0.0, adv=0.0, D=0.0)

        last_xyz   = None
        last_recon = None
        last_class = "unknown"

        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]{'  GAN' if use_gan else '  warmup'}")
        for class_name, pcd in loop:
            xyz = pcd.to(device)   # (B, N, 6)

            # ── 1. GERAÇÃO (Forward do VAE) ───────────────────────────────────
            out = model(xyz)

            # ── Phase 1: Discriminator step (Atualiza APENAS o D) ─────────────
            if use_gan:
                model.unfreeze_D()
                optimiser_D.zero_grad()

                # .detach() impede que o gradiente do D flua de volta para o VAE/Gerador
                fake_recon_detached = out["recon"].detach()
                
                d_losses = model.loss_discriminator(real=xyz, fake=fake_recon_detached)
                d_losses["loss_D"].backward() # <-- SEM retain_graph=True! Memória limpa.
                optimiser_D.step()

                totals["D"] += d_losses["loss_D"].item()

            # ── Phase 2: Generator step (Atualiza APENAS o VAE) ───────────────
            model.freeze_D()
            optimiser_G.zero_grad()

            if use_gan:
                # Aqui passamos o "out" original (sem detach) para atualizar os pesos do VAE via GAN
                g_losses = model.loss_generator(out=out, target=xyz)
                totals["adv"] += g_losses["loss_adv"].item()
            else:
                g_losses = model.loss(out=out, target=xyz)

            g_losses["total"].backward()
            optimiser_G.step()
        
            totals["G"]  += g_losses["total"].item()
            totals["cd"] += g_losses["cd"].item()
            totals["kl"] += g_losses["kl"].item()

            loop.set_postfix(
                G   = f"{g_losses['total'].item():.4f}",
                cd  = f"{g_losses['cd'].item():.6f}",
                kl  = f"{g_losses['kl'].item():.4f}",
                adv = f"{g_losses.get('loss_adv', torch.tensor(0)).item():.4f}",
            )

            last_xyz   = xyz
            last_recon = out["recon"]
            last_class = class_name[0]

        n = len(dataloader)
        print(
            f"Epoch [{epoch+1}/{num_epochs}] — "
            f"G: {totals['G']/n:.4f}  "
            f"CD: {totals['cd']/n:.6f}  "
            f"KL: {totals['kl']/n:.4f}  "
            f"Adv: {totals['adv']/n:.4f}  "
            f"D: {totals['D']/n:.4f}"
        )

        if last_xyz is not None:
            save_before_after(last_xyz[0], last_recon[0], epoch, last_class)

        os.makedirs("checkpoints", exist_ok=True)

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualBranchPointVAE(
        d_model      = 384,
        latent_dim   = 512 ,
        n_out        = 2048,
        enc_depth    = 4,
        dec_depth    = 4,
        n_heads      = 6,
        beta         = 0,
        lambda_adv   = 0,
        disc_base_ch = 64,
    ).to(device)

    dataset = Ds_point_sampled(Ds_point_model())

    train_vae_gan(
        model,
        dataset,
        num_epochs     = 5000,
        batch_size     = 16,
        lr_G           = 1e-3,
        lr_D           = 1e-4,
        warmup_epochs  = 10,
    )