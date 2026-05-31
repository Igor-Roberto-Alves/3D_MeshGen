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

def check_ply_contents(file_path, visualize=False):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    # Load the point cloud
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Check if the Open3D object is physically empty
    if pcd.is_empty():
        print(f"❌ The PLY file '{file_path}' is completely EMPTY (0 points).")
        return

    # Gather stats
    points = np.asarray(pcd.points)
    num_points = len(points)
    has_normals = pcd.has_normals()
    
    print(f"✅ The PLY file '{file_path}' is NOT empty.")
    print(f"   - Number of points: {num_points}")
    print(f"   - Has normals:      {has_normals}")
    
    # Check for NaNs or Infinity in spatial points
    if np.isnan(points).any() or np.isinf(points).any():
        print("   - ⚠️ WARNING: Your points contain NaN or Inf values!")
    else:
        print("\n   - First 3 spatial points (X, Y, Z):")
        print(points[:3])

    # Check and print normals
    if has_normals:
        normals = np.asarray(pcd.normals)
        if np.isnan(normals).any() or np.isinf(normals).any():
            print("\n   - ⚠️ WARNING: Your normals contain NaN or Inf values!")
        else:
            print("\n   - First 3 normals (NX, NY, NZ):")
            print(normals[:3])
            
            print("\n   - Combined View [X, Y, Z, NX, NY, NZ] for the first 3 points:")
            # Stack points and normals horizontally for a clean 6D view
            combined = np.hstack((points[:3], normals[:3]))
            print(combined)

    # ─── POP WINDOW IF FLAG IS TRUE ───
    if visualize:
        print(f"\n[INFO] Launching Open3D visualizer for: {file_path}")
        print("[INFO] Close the window manually to finish script execution.")
        
        # If your 3 extra channels are un-trained random values, the shader might make them invisible.
        # Clearing or painting handles this smoothly for inspection.
        pcd.paint_uniform_color([0.0, 0.8, 0.8]) # Paint it a solid teal/cyan color
        
        o3d.visualization.draw_geometries([pcd], window_name=f"Checking: {os.path.basename(file_path)}")

if __name__ == "__main__":
    # Hide verbose Open3D backend warnings/errors during window init
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    parser = argparse.ArgumentParser(description="Check and verify a point cloud PLY file.")
    parser.add_argument(
        "-v", "--visualize", 
        type=str2bool, 
        default=False, 
        help="Set to True to pop up an interactive Open3D viewer window"
    )
    parser.add_argument("-e", "--epoch", type = str, default = 1, help= "epoch value")
    parser.add_argument("-rec", "--reconstruction", type = str2bool, default = False, help = "recontructed")
    args = parser.parse_args()
    if args.reconstruction:
        rec = "RECONSTRUCTED" 
    else:
        rec = "ORIGINAL"

    target_file = "reconstructions/epoch_" + args.epoch + "_02876657_" + rec + ".ply"
    
    # Run verification with command line flag configuration
    check_ply_contents(target_file, visualize=args.visualize)