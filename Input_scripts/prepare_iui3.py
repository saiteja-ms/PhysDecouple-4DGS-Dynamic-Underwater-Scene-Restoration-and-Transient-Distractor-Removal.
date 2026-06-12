import os
import subprocess

def prepare_iui3():
    dataset_dir = "/mnt/md0/IITM/ipcv/saiteja/PhysDecouple-4DGS/Data/seathrunerf/SeathruNeRF_dataset"
    scene = "IUI3-RedSea"
    
    scene_path = os.path.join(dataset_dir, scene)
    images_wb_path = os.path.join(scene_path, "Images_wb")
    images_path = os.path.join(scene_path, "images")
    
    if os.path.exists(images_wb_path) and not os.path.exists(images_path):
        os.symlink("Images_wb", images_path)
        print(f"Created symlink images -> Images_wb for scene {scene}")
        
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
    prepare_iui3()
