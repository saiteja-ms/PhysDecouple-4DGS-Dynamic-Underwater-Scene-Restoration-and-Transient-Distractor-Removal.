import zipfile
import os

def extract_seathrunerf():
    zip_path = "/mnt/md0/IITM/ipcv/saiteja/PhysDecouple-4DGS/Data/seathrunerf.zip"
    dest_dir = "/mnt/md0/IITM/ipcv/saiteja/PhysDecouple-4DGS/Data/seathrunerf"
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Extracting {zip_path} to {dest_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    print("Extraction complete.")

if __name__ == "__main__":
    extract_seathrunerf()
