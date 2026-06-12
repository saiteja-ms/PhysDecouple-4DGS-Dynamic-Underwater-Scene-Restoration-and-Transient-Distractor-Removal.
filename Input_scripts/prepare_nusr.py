import os
import numpy as np
import json
import subprocess
from plyfile import PlyData

def generate_w2c_poses(poses_bounds_path, w2c_path):
    if not os.path.exists(poses_bounds_path):
        print(f"poses_bounds.npy not found at {poses_bounds_path}")
        return
    pb = np.load(poses_bounds_path)
    # Extract first 15 elements, reshape to (N, 3, 5)
    poses = pb[:, :15].reshape(-1, 3, 5)
    
    # Calculate w2c poses
    w2c_poses = np.zeros_like(poses)
    for i in range(poses.shape[0]):
        r_c2w = poses[i, :, :3]
        t_c2w = poses[i, :, 3]
        
        # Coordinate system transformation
        r_w2c_c2w_T = r_c2w.T
        r_w2c = np.stack([r_w2c_c2w_T[1], r_w2c_c2w_T[0], -r_w2c_c2w_T[2]], axis=0)
        t_w2c = -r_w2c @ t_c2w
        
        w2c_poses[i, :, :3] = r_w2c
        w2c_poses[i, :, 3] = t_w2c
        w2c_poses[i, :, 4] = poses[i, :, 4] # Keep H, W, focal
        
    np.save(w2c_path, w2c_poses)
    print(f"Generated {w2c_path}")

def generate_scene_json(ply_path, json_path):
    if not os.path.exists(ply_path):
        # Fallback to dummy values if PLY doesn't exist yet
        scene_json = {
            "scale": 1.0,
            "center": [0.0, 0.0, 0.0],
            "bbox": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
            "near": 0.01,
            "far": 100.0
        }
    else:
        plydata = PlyData.read(ply_path)
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        
        # Filter outliers using percentiles
        min_val = np.percentile(pts, 2.0, axis=0)
        max_val = np.percentile(pts, 98.0, axis=0)
        
        center = np.mean(pts, axis=0)
        bbox = [min_val.tolist(), max_val.tolist()]
        
        # Calculate scale
        max_dist = np.max(np.abs(pts - center))
        scale = 1.0 / max_dist if max_dist > 0 else 1.0
        
        scene_json = {
            "scale": float(scale),
            "center": center.tolist(),
            "bbox": bbox,
            "near": 0.01,
            "far": 100.0
        }
        
    with open(json_path, 'w') as f:
        json.dump(scene_json, f, indent=2)
    print(f"Generated {json_path}")

def main():
    dataset_dir = "/mnt/md0/IITM/ipcv/saiteja/PhysDecouple-4DGS/Data/nusr/underwater dataset"
    scenes = ["composite", "coral", "sardine", "turtle"]
    python_bin = "/mnt/md0/IITM/ipcv/3DGS/code/vggsfm/miniconda/envs/physdecouple/bin/python"
    
    for scene in scenes:
        scene_path = os.path.join(dataset_dir, scene)
        if not os.path.exists(scene_path):
            print(f"Scene path {scene_path} does not exist. Skipping.")
            continue
            
        print(f"\n=========================================")
        print(f"Processing scene: {scene}")
        print(f"=========================================")
        
        # Step 1: Create .jpg symlinks to .png files to match COLMAP images.bin expectations
        image_dir = os.path.join(scene_path, "images")
        if os.path.exists(image_dir):
            jpg_count = 0
            for f in os.listdir(image_dir):
                if f.endswith(".png"):
                    jpg_name = f[:-4] + ".jpg"
                    jpg_path = os.path.join(image_dir, jpg_name)
                    if not os.path.exists(jpg_path):
                        os.symlink(f, jpg_path)
                        jpg_count += 1
            if jpg_count > 0:
                print(f"Created {jpg_count} .jpg symlinks in {image_dir}")
        
        # Step 2: Generate mono depth maps using Depth Anything V2
        print(f"Generating mono_depth for scene {scene}...")
        cmd = [
            python_bin,
            "scripts/generate_depth_anything.py",
            "--data-root", dataset_dir,
            "--scene", scene,
            "--images-dir", "images",
            "--output-dir", "mono_depth"
        ]
        subprocess.run(cmd, check=True)
        print(f"Finished mono_depth generation for scene {scene}")
        
        # Step 3: w2c_poses.npy
        poses_bounds_path = os.path.join(scene_path, "poses_bounds.npy")
        w2c_path = os.path.join(scene_path, "w2c_poses.npy")
        generate_w2c_poses(poses_bounds_path, w2c_path)
        
        # Step 4: Check/create PLY and generate scene.json
        ply_path = os.path.join(scene_path, "sparse/0/points3D.ply")
        bin_path = os.path.join(scene_path, "sparse/0/points3D.bin")
        if not os.path.exists(ply_path) and os.path.exists(bin_path):
            # Try to pre-convert points3D.bin to points3D.ply
            try:
                from scene.colmap_loader import read_points3D_binary
                from scene.dataset_readers import storePly
                xyz, rgb, _ = read_points3D_binary(bin_path)
                storePly(ply_path, xyz, rgb)
                print(f"Converted points3D.bin to points3D.ply for {scene}")
            except Exception as e:
                print(f"Error converting points3D.bin: {e}")
                
        json_path = os.path.join(scene_path, "scene.json")
        generate_scene_json(ply_path, json_path)

if __name__ == "__main__":
    main()
