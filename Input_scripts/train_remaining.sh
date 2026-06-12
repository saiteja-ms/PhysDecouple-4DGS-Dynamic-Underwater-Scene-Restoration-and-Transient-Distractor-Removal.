#!/bin/bash

# Environment Python path
PYTHON_BIN="/mnt/md0/IITM/ipcv/3DGS/code/vggsfm/miniconda/envs/physdecouple/bin/python"

# Outputs directory
OUTPUT_DIR="output/seathrunerf"

# Launch training for Curasao on GPU 4 (Port 6010)
echo "Starting Curasao training on GPU 4..."
CUDA_VISIBLE_DEVICES=4 $PYTHON_BIN train.py \
  -s Data/seathrunerf/SeathruNeRF_dataset/Curasao \
  --port 6010 \
  --configs arguments/hypernerf/physdecouple_uw.py \
  --model_path $OUTPUT_DIR/Curasao \
  --expname seathrunerf_curasao &

# Launch training for JapaneseGardens on GPU 1 (Port 6011)
echo "Starting JapaneseGradens-RedSea training on GPU 1..."
CUDA_VISIBLE_DEVICES=1 $PYTHON_BIN train.py \
  -s Data/seathrunerf/SeathruNeRF_dataset/JapaneseGradens-RedSea \
  --port 6011 \
  --configs arguments/hypernerf/physdecouple_uw.py \
  --model_path $OUTPUT_DIR/JapaneseGradens-RedSea \
  --expname seathrunerf_japanesegardens &

# Launch training for IUI3-RedSea on GPU 2 (Port 6012, shared GPU with Panama)
echo "Starting IUI3-RedSea training on GPU 2..."
CUDA_VISIBLE_DEVICES=2 $PYTHON_BIN train.py \
  -s Data/seathrunerf/SeathruNeRF_dataset/IUI3-RedSea \
  --port 6012 \
  --configs arguments/hypernerf/physdecouple_uw.py \
  --model_path $OUTPUT_DIR/IUI3-RedSea \
  --expname seathrunerf_iui3 &

wait
echo "Remaining trainings completed!"
