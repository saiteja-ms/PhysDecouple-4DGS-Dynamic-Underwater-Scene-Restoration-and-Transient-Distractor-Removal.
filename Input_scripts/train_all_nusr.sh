#!/bin/bash

# Environment Python path
PYTHON_BIN="/mnt/md0/IITM/ipcv/3DGS/code/vggsfm/miniconda/envs/physdecouple/bin/python"

# Outputs directory
OUTPUT_DIR="output/nusr"

# Dataset root directory
DATA_ROOT="Data/nusr/underwater dataset"

# Launch training for turtle on GPU 4 (Port 6020)
echo "Starting turtle training on GPU 4..."
CUDA_VISIBLE_DEVICES=4 $PYTHON_BIN train.py \
  -s "$DATA_ROOT/turtle" \
  --port 6020 \
  --configs arguments/hypernerf/physdecouple_uw.py \
  --model_path "$OUTPUT_DIR/turtle" \
  --expname nusr_turtle &

# Launch training for coral on GPU 4 (Port 6021)
echo "Starting coral training on GPU 4..."
CUDA_VISIBLE_DEVICES=4 $PYTHON_BIN train.py \
  -s "$DATA_ROOT/coral" \
  --port 6021 \
  --configs arguments/hypernerf/physdecouple_uw.py \
  --model_path "$OUTPUT_DIR/coral" \
  --expname nusr_coral &

# Launch training for sardine on GPU 1 (Port 6022)
echo "Starting sardine training on GPU 1..."
CUDA_VISIBLE_DEVICES=1 $PYTHON_BIN train.py \
  -s "$DATA_ROOT/sardine" \
  --port 6022 \
  --configs arguments/hypernerf/physdecouple_uw.py \
  --model_path "$OUTPUT_DIR/sardine" \
  --expname nusr_sardine &

# Launch training for composite on GPU 2 (Port 6023)
echo "Starting composite training on GPU 2..."
CUDA_VISIBLE_DEVICES=2 $PYTHON_BIN train.py \
  -s "$DATA_ROOT/composite" \
  --port 6023 \
  --configs arguments/hypernerf/physdecouple_uw.py \
  --model_path "$OUTPUT_DIR/composite" \
  --expname nusr_composite &

wait
echo "All NUSR trainings completed!"
