import open3d as o3d
import numpy as np
import os
import argparse

def str2bool(v):
    """Helper to cleanly parse 'True' or 'False' from the command line."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def verify_and_load_pcd(file_path):
    """Auxiliary function to check logs and load the point cloud safely."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    pcd = o3d.io.read_point_cloud(file_path)
    
    if pcd.is_empty():
        print(f"❌ The PLY file '{file_path}' is completely EMPTY (0 points).")
        return None

    points = np.asarray(pcd.points)
    num_points = len(points)
    has_normals = pcd.has_normals()
    
    print(f"✅ Loaded '{os.path.basename(file_path)}' -> Points: {num_points} | Has normals: {has_normals}")
    
    if np.isnan(points).any() or np.isinf(points).any():
        print(f"   - ⚠️ WARNING: '{os.path.basename(file_path)}' contains NaN or Inf values!")
        
    return pcd

def compare_ply_contents(epoch_num, visualize=False):
    # Gerar os caminhos para ambos os ficheiros
    orig_path = f"reconstructions/epoch_{epoch_num}_03001627_ORIGINAL.ply"
    rec_path  = f"reconstructions/epoch_{epoch_num}_03001627_RECONSTRUCTED.ply"
    
    print(f"\n--- [INFO] Checking Epoch {epoch_num} Data ---")
    pcd_orig = verify_and_load_pcd(orig_path)
    pcd_rec  = verify_and_load_pcd(rec_path)
    
    if pcd_orig is None or pcd_rec is None:
        print("❌ Cannot visualize. One or both files are missing/corrupted.")
        return

    # ─── POP WINDOW IF FLAG IS TRUE ───
    if visualize:
        print(f"\n[INFO] Launching Open3D visualizer side-by-side for epoch {epoch_num}")
        print("[INFO] Legend: GREEN = Original | RED = Reconstructed")
        print("[INFO] Press 'N' inside the window to toggle surface normals display.")
        print("[INFO] Close the window manually to finish script execution.")
        
        # Copiar para não alterar as nuvens originais em futuras operações se necessário
        pcd_orig_vis = o3d.geometry.PointCloud(pcd_orig)
        pcd_rec_vis = o3d.geometry.PointCloud(pcd_rec)
        
        # Pintar com cores distintas para fácil identificação
        pcd_orig_vis.paint_uniform_color([0.1, 0.7, 0.1])  # Verde para Original
        pcd_rec_vis.paint_uniform_color([0.7, 0.1, 0.1])   # Vermelho para Reconstruída
        
        # Calcular deslocamento baseado no bounding box da nuvem original
        bbox = pcd_orig_vis.get_axis_aligned_bounding_box()
        extent_x = bbox.get_extent()[0]
        
        # Deslocar a reconstruída para o lado no eixo X (com folga de 1.5x a sua largura)
        pcd_rec_vis.translate([extent_x * 1.5, 0, 0])
        
        # Renderizar ambas na mesma janela
        o3d.visualization.draw_geometries(
            [pcd_orig_vis, pcd_rec_vis], 
            window_name=f"Comparison Epoch {epoch_num} - Left: Original (Green) | Right: Reconstructed (Red)",
            width=1280,
            height=720
        )

if __name__ == "__main__":
    # Hide verbose Open3D backend warnings/errors during window init
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    parser = argparse.ArgumentParser(description="Check and compare original vs reconstructed point clouds side-by-side.")
    parser.add_argument(
        "-v", "--visualize", 
        type=str2bool, 
        default=False, 
        help="Set to True to pop up an interactive Open3D viewer window"
    )
    parser.add_argument("-e", "--epoch", type=str, default="1", help="Epoch value to inspect")
    
    args = parser.parse_args()
    
    # Executa a comparação direta
    compare_ply_contents(args.epoch, visualize=args.visualize)