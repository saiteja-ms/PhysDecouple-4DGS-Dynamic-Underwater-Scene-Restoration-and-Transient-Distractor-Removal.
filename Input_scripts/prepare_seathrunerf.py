import os
import subprocess
from pathlib import Path

def prepare_dataset():
    dataset_dir = "/mnt/md0/IITM/ipcv/saiteja/PhysDecouple-4DGS/Data/seathrunerf/SeathruNeRF_dataset"
    scenes = ["Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama"]
    
    for scene in scenes:
        scene_path = os.path.join(dataset_dir, scene)
        if not os.path.exists(scene_path):
            print(f"Scene path {scene_path} does not exist. Skipping.")
            continue
            
        images_wb_path = os.path.join(scene_path, "images_wb")
        images_path = os.path.join(scene_path, "images")
        
        # Step 1: Symlink images_wb to images if not already done
        if os.path.exists(images_wb_path) and not os.path.exists(images_path):
            os.symlink("images_wb", images_path)
            print(f"Created symlink images -> images_wb for scene {scene}")
        
        # Step 2: Generate mono depth maps using Depth Anything V2
        print(f"Generating mono_depth for scene {scene}...")
        python_bin = "/mnt/md0/IITM/ipcv/saiteja/PhysDecouple-4DGS/.conda/envs/physdecouple/bin/python"
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

if __name__ == "__main__":
    prepare_dataset()
