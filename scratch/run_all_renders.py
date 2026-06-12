import subprocess
import os

# Model path, iteration, skip_train, skip_test, skip_video
models = [
    ("output/seathrunerf/IUI3-RedSea", 25000, False, True, False),                  # Needs train, video
    ("output/seathrunerf/JapaneseGradens-RedSea", 25000, False, True, False),       # Needs train, video
    ("output/seathrunerf/Panama", 25000, False, True, False),                      # Needs train, video
    ("output/set4_physdecouple_v14", 25000, False, True, True),                    # Needs train
    ("output/set5_physdecouple_v14", 25000, False, True, True),                    # Needs train
    ("output/set8_physdecouple_v14", 25000, False, True, False),                   # Needs train, video
    ("output/set8_physdecouple_v15", 25000, False, True, False),                   # Needs train, video
    ("output/set9_physdecouple_v14", 25000, False, False, False),                  # Needs everything
    ("output/set12_physdecouple_v14", 25000, False, False, False),                 # Needs everything
    ("output/set13_physdecouple_v14", 25000, False, False, False),                 # Needs everything
    ("output/synthetic_blender_physdecouple_v14", 25000, False, True, False),      # Needs train, video
]

for model_path, iteration, skip_train, skip_test, skip_video in models:
    cmd = [
        ".conda/envs/physdecouple/bin/python",
        "scratch_render.py",
        "-m", model_path,
        "--iteration", str(iteration)
    ]
    if skip_train:
        cmd.append("--skip_train")
    if skip_test:
        cmd.append("--skip_test")
    if skip_video:
        cmd.append("--skip_video")
        
    print(f"============================================================")
    print(f"Running: {' '.join(cmd)}")
    print(f"============================================================")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed running rendering for {model_path}: {e}")
