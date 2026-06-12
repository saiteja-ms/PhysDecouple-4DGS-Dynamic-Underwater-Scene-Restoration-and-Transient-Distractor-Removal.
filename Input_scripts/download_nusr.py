import os
import gdown

def download_nusr():
    file_id = "1MP-zPbCmALwdzS6xbng3rE5XG9j9rff4"
    output_dir = "/mnt/md0/IITM/ipcv/saiteja/PhysDecouple-4DGS/Data"
    os.makedirs(output_dir, exist_ok=True)
    
    zip_path = os.path.join(output_dir, "nusr.zip")
    
    print(f"Downloading NUSR dataset using python gdown to {zip_path}...")
    gdown.download(id=file_id, output=zip_path, quiet=False)
    print("Download complete.")

if __name__ == "__main__":
    download_nusr()
