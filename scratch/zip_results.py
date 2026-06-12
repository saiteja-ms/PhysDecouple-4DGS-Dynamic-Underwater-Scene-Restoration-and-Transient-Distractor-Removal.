import os
import zipfile
import sys

target_models = [
    "output/seathrunerf/Curasao",
    "output/seathrunerf/IUI3-RedSea",
    "output/seathrunerf/JapaneseGradens-RedSea",
    "output/seathrunerf/Panama",
    "output/set4_physdecouple_v14",
    "output/set4_physdecouple_v15",
    "output/set5_physdecouple_v14",
    "output/set5_physdecouple_v15",
    "output/set8_physdecouple_v14",
    "output/set8_physdecouple_v15",
    "output/set9_physdecouple_v14",
    "output/set12_physdecouple_v14",
    "output/set13_physdecouple_v14",
    "output/synthetic_blender_physdecouple_v14"
]

zip_filename = "rendered_results.zip"
print(f"Creating zip file: {zip_filename}...")

count = 0
total_files = 0

# Count total files first
for model_path in target_models:
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} does not exist!")
        continue
    for root, dirs, files in os.walk(model_path):
        if "ours_25000" in root:
            total_files += len(files)

print(f"Total files to zip: {total_files}")

if total_files == 0:
    print("No files found to zip!")
    sys.exit(1)

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_STORED) as zipf:
    for model_path in target_models:
        if not os.path.exists(model_path):
            continue
        print(f"Zipping {model_path}...")
        for root, dirs, files in os.walk(model_path):
            if "ours_25000" in root:
                for file in files:
                    full_path = os.path.join(root, file)
                    # Relative path under output directory
                    archive_name = os.path.relpath(full_path, "output")
                    zipf.write(full_path, arcname=archive_name)
                    count += 1
                    if count % 1000 == 0 or count == total_files:
                        print(f"Processed {count}/{total_files} files ({(count/total_files)*100:.1f}%)")

print("Zipping complete!")
