"""
app.py — Deployee da difusao latente unconditional (ckpt_diff_flat_noclass).

Serve uma UI onde:
  - um seletor escolhe o checkpoint por criterio (menor val loss, maior COV,
    menor MMD ou 1-NNA mais perto de 0.5), usando as metricas de
    checkpoints_metrics.json (gerado por eval_checkpoints.py);
  - o botao Generate amostra um latente via DDIM com o modelo escolhido e
    decodifica com o decoder do VAE congelado;
  - sliders aplicam shifts (em unidades de std do latente de treino) nas
    dimensoes mais ativas do vetor latente e mostram a reconstrucao.

Rodar a partir da raiz do repo:
    .venv/bin/python Deployee/app.py
"""

import json
import random
import sys
import threading
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import open3d as o3d
import torch
from flask import Flask, jsonify, request, send_from_directory

from src.Diffusion import CosineSchedule
from src.UnetDiffusion import UNet1D, ddim_sample
from src.vae_flat import VaeFlat

CKPT_DIR     = ROOT / "ckpt_diff_flat_noclass" / "exp0_noclass"
METRICS_PATH = HERE / "checkpoints_metrics.json"
OUT_DIR      = HERE / "generated_clouds"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TOP_DIMS   = 8

app = Flask(__name__)
OUT_DIR.mkdir(exist_ok=True)

_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# Selecao de checkpoint por criterio
# ─────────────────────────────────────────────────────────────────────────────

METRICS = json.loads(METRICS_PATH.read_text())

CRITERIA = {
    "best_val": {
        "label": "Menor erro de validação (eps-MSE)",
        "key":   lambda m: m["val_loss"],
    },
    "best_cov": {
        "label": "Maior COV",
        "key":   lambda m: -m["cov"],
    },
    "best_mmd": {
        "label": "Menor MMD-CD",
        "key":   lambda m: m["mmd"],
    },
    "best_nna": {
        "label": "1-NNA mais perto de 0.5",
        "key":   lambda m: abs(m["nna_1nn"] - 0.5),
    },
}


def resolve_criterion(criterion: str) -> str:
    """Nome do checkpoint (chave de METRICS) que vence o criterio.
    Empates sao desfeitos pela menor val loss."""
    key = CRITERIA[criterion]["key"]
    return min(METRICS, key=lambda name: (key(METRICS[name]),
                                          METRICS[name]["val_loss"]))


# ─────────────────────────────────────────────────────────────────────────────
# Modelos (VAE congelado + UNet de difusao por checkpoint, com cache)
# ─────────────────────────────────────────────────────────────────────────────

def build_vae_flat(cfg: dict, decoder_norm: str) -> VaeFlat:
    return VaeFlat(
        latent_dim=cfg["latent_dim"], in_channels=cfg["in_channels"],
        n_latent=cfg["n_latent"], n_points=cfg["n_points"],
        seed_dim=cfg["seed_dim"],
        hidden_dim=cfg["decoder_hidden_dim"], n_stages=cfg["decoder_stages"],
        fold_dim=cfg["decoder_fold_dim"], fold_hidden=cfg["decoder_fold_hidden"],
        resolution=cfg["decoder_resolution"],
        encoder_hidden_dim=cfg["encoder_hidden_dim"],
        encoder_out_dim=cfg["encoder_out_dim"],
        encoder_stages=cfg["encoder_stages"],
        encoder_resolution=cfg["encoder_resolution"],
        decoder_norm=decoder_norm,
    ).to(DEVICE)


_vae = None
_diff_cache: dict[str, dict] = {}


def get_vae(vae_ckpt: str) -> VaeFlat:
    global _vae
    if _vae is None:
        ck = torch.load(ROOT / vae_ckpt, map_location=DEVICE, weights_only=False)
        cfg, state = ck["config"], ck["model"]
        norm = "batch" if "decoder.feat_proj.1.running_mean" in state else "group"
        vae = build_vae_flat(cfg, norm)
        vae.load_state_dict(state)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        _vae = vae
        print(f"[✓] VAE loaded: {vae_ckpt} (norm={norm})")
    return _vae


def get_diffusion(name: str) -> dict:
    """Carrega (com cache) o UNet EMA + schedule + stats do latente."""
    if name not in _diff_cache:
        path = CKPT_DIR / METRICS[name]["file"]
        ck = torch.load(path, map_location=DEVICE, weights_only=False)

        model = UNet1D(**ck["model_kwargs"]).to(DEVICE)
        state = ck["ema"] if ck.get("ema") is not None else ck["model"]
        model.load_state_dict(state)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        latent_std = ck["latent_std"].to(DEVICE)
        _diff_cache[name] = {
            "model":       model,
            "schedule":    CosineSchedule(T=ck["args"]["T"]).to(DEVICE),
            "latent_mean": ck["latent_mean"].to(DEVICE),
            "latent_std":  latent_std,
            "latent_dim":  ck["model_kwargs"]["latent_dim"],
            "top_dims":    latent_std.argsort(descending=True)[:N_TOP_DIMS].tolist(),
            "vae_ckpt":    ck["vae_ckpt"],
            "epoch":       int(ck["epoch"]),
        }
        print(f"[✓] Diffusion loaded: {path.name} (epoch {ck['epoch']}, EMA)")
    return _diff_cache[name]


@torch.no_grad()
def decode_points(entry: dict, z_norm: torch.Tensor) -> np.ndarray:
    """z normalizado (1, D) → nuvem (N, 3)."""
    vae = get_vae(entry["vae_ckpt"])
    z = z_norm * entry["latent_std"] + entry["latent_mean"]
    return vae.decoder(z)[0].cpu().numpy()


def points_payload(pts: np.ndarray) -> list:
    return np.round(pts.astype(np.float64), 4).tolist()


def shifted_z(data: dict, entry: dict) -> torch.Tensor:
    """Monta o latente (1, D) a partir de data['z'] + data['shifts'].
    Levanta ValueError com mensagem apropriada para 400."""
    z = torch.tensor(data["z"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    if z.shape[1] != entry["latent_dim"]:
        raise ValueError(f"z must have {entry['latent_dim']} dims")
    for k, v in (data.get("shifts") or {}).items():
        dim = int(k)
        if not 0 <= dim < entry["latent_dim"]:
            raise ValueError(f"dim {dim} out of range")
        z[0, dim] += float(v)
    return z


def poisson_mesh(pts: np.ndarray, depth: int = 8, knn: int = 30,
                 density_q: float = 0.02, max_tris: int = 20000):
    """
    Nuvem (N, 3) → malha via:
      1. normais por PCA da vizinhanca k-NN (autovetor de menor autovalor
         da covariancia local — e o que o estimate_normals do Open3D faz);
      2. orientacao consistente das normais (propagacao por MST);
      3. reconstrucao de Poisson;
      4. corte dos vertices de baixa densidade (extrapolacao do Poisson
         longe dos pontos) + crop na bbox da nuvem;
      5. decimacao se passar de max_tris;
      6. normais por vertice (media ponderada das normais dos triangulos
         adjacentes) para o cliente interpolar no shading (Gouraud).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn))
    pcd.orient_normals_consistent_tangent_plane(knn)

    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, density_q))

    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.scale(1.15, aabb.get_center())
    mesh = mesh.crop(aabb)

    if len(mesh.triangles) > max_tris:
        mesh = mesh.simplify_quadric_decimation(max_tris)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    verts = np.round(np.asarray(mesh.vertices), 4).tolist()
    tris  = np.asarray(mesh.triangles).tolist()
    norms = np.round(np.asarray(mesh.vertex_normals), 4).tolist()
    return verts, tris, norms


# ─────────────────────────────────────────────────────────────────────────────
# Rotas
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(HERE), "index.html")


@app.route("/models", methods=["GET"])
def models():
    """Criterios disponiveis e para qual checkpoint cada um resolve."""
    out = []
    for cid, spec in CRITERIA.items():
        name = resolve_criterion(cid)
        out.append({
            "id":         cid,
            "label":      spec["label"],
            "checkpoint": METRICS[name]["file"],
            "epoch":      METRICS[name]["epoch"],
            "metrics":    {k: METRICS[name][k]
                           for k in ("val_loss", "mmd", "cov", "nna_1nn")},
        })
    return jsonify({"criteria": out, "device": str(DEVICE)}), 200


@app.route("/generate", methods=["POST"])
def generate():
    """
    Amostra 1 latente via DDIM com o checkpoint do criterio escolhido e
    devolve a nuvem decodificada + o latente normalizado (para shifts).

    JSON: {"criterion": "best_val", "seed": 123, "steps": 200}
    """
    try:
        data = request.get_json(force=True) or {}
        criterion = data.get("criterion", "best_val")
        if criterion not in CRITERIA:
            return jsonify({"success": False,
                            "error": f"criterion must be one of {list(CRITERIA)}"}), 400

        seed  = data.get("seed")
        seed  = random.randint(0, 2**31 - 1) if seed in (None, "") else int(seed)
        steps = max(10, min(1000, int(data.get("steps", 200))))

        name  = resolve_criterion(criterion)
        with _lock:
            entry = get_diffusion(name)
            torch.manual_seed(seed)
            cond = torch.zeros((1,), device=DEVICE, dtype=torch.long)
            z_norm = ddim_sample(entry["schedule"], entry["model"],
                                 (entry["latent_dim"],), cond,
                                 uncond=None, guidance=1.0, steps=steps, eta=1.0)
            pts = decode_points(entry, z_norm)

        fname = f"{name}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
        np.save(OUT_DIR / fname, pts)

        return jsonify({
            "success":    True,
            "criterion":  criterion,
            "checkpoint": METRICS[name]["file"],
            "epoch":      entry["epoch"],
            "seed":       seed,
            "steps":      steps,
            "z":          [round(v, 5) for v in z_norm[0].cpu().tolist()],
            "top_dims":   entry["top_dims"],
            "points":     points_payload(pts),
            "filename":   fname,
        }), 200
    except Exception as e:
        print(f"[✗] /generate: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/decode", methods=["POST"])
def decode():
    """
    Decodifica um latente (normalizado) com shifts por dimensao — os
    "shiftizinhos". Shifts em unidades de std do latente de treino.

    JSON: {"criterion": "best_val",
           "z": [256 floats],
           "shifts": {"9": 0.8, "134": -1.2}}
    """
    try:
        data = request.get_json(force=True) or {}
        criterion = data.get("criterion", "best_val")
        if criterion not in CRITERIA:
            return jsonify({"success": False,
                            "error": f"criterion must be one of {list(CRITERIA)}"}), 400

        name  = resolve_criterion(criterion)
        with _lock:
            entry = get_diffusion(name)
            try:
                z = shifted_z(data, entry)
            except ValueError as ve:
                return jsonify({"success": False, "error": str(ve)}), 400
            pts = decode_points(entry, z)

        return jsonify({
            "success":    True,
            "checkpoint": METRICS[name]["file"],
            "shifts":     data.get("shifts") or {},
            "points":     points_payload(pts),
        }), 200
    except Exception as e:
        print(f"[✗] /decode: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/mesh", methods=["POST"])
def mesh():
    """
    Decodifica o latente (com shifts) e reconstroi a malha: normais
    estimadas por PCA da vizinhanca k-NN + reconstrucao de Poisson.

    JSON: {"criterion": "best_val",
           "z": [256 floats],
           "shifts": {"9": 0.8},
           "depth": 8, "knn": 30}
    """
    try:
        data = request.get_json(force=True) or {}
        criterion = data.get("criterion", "best_val")
        if criterion not in CRITERIA:
            return jsonify({"success": False,
                            "error": f"criterion must be one of {list(CRITERIA)}"}), 400

        depth = max(5, min(10, int(data.get("depth", 8))))
        knn   = max(8, min(100, int(data.get("knn", 30))))

        name = resolve_criterion(criterion)
        with _lock:
            entry = get_diffusion(name)
            try:
                z = shifted_z(data, entry)
            except ValueError as ve:
                return jsonify({"success": False, "error": str(ve)}), 400
            pts = decode_points(entry, z)
            verts, tris, norms = poisson_mesh(pts, depth=depth, knn=knn)

        return jsonify({
            "success":    True,
            "checkpoint": METRICS[name]["file"],
            "depth":      depth,
            "knn":        knn,
            "vertices":   verts,
            "triangles":  tris,
            "normals":    norms,
        }), 200
    except Exception as e:
        print(f"[✗] /mesh: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/download/<path:filename>", methods=["GET"])
def download(filename):
    return send_from_directory(str(OUT_DIR), filename, as_attachment=True)


if __name__ == "__main__":
    print("=" * 60)
    print("Latent Diffusion (noclass) — Deployee")
    print(f"Device: {DEVICE}")
    for cid in CRITERIA:
        name = resolve_criterion(cid)
        m = METRICS[name]
        print(f"  {CRITERIA[cid]['label']:<38} → {m['file']} (epoch {m['epoch']})")
    print("🚀 http://localhost:5000")
    print("=" * 60)
    # Pre-carrega o checkpoint default para o primeiro Generate ser rapido
    get_diffusion(resolve_criterion("best_val"))
    app.run(host="0.0.0.0", port=5000, debug=False)
