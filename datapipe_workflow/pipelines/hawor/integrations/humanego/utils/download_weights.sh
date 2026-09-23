#!/bin/bash
export HF_HOME=/mnt/data/liuyu/project/HumanEgo/models/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0

mkdir -p ./models

# 1. mediapipe-hand
hf download Leo-TX/mediapipe-hand hand_landmarker.task --local-dir ./models

# 2. hamer 全部文件
hf download Leo-TX/hamer hamer.ckpt --local-dir ./models
hf download Leo-TX/hamer model_config.yaml --local-dir ./models
hf download Leo-TX/hamer dataset_config.yaml --local-dir ./models
hf download Leo-TX/hamer mano_mean_params.npz --local-dir ./models

# 3. WiLoR-mini MANO_RIGHT.pkl
hf download warmshao/WiLoR-mini "pretrained_models/MANO_RIGHT.pkl" --local-dir ./models

# 4. Orient-AnyV2
hf download Viglong/OriAnyV2_ckpt "demo_ckpts/rotmod_realrotaug_best.pt" --local-dir ./models

# 5. ViTPose 权重
hf download JunkyByte/easy_ViTPose "torch/wholebody/vitpose-h-wholebody.pth" --local-dir ./models
hf download JunkyByte/easy_ViTPose "yolov8/yolov8s.pt" --local-dir ./models

hf download facebook/cotracker3 scaled_offline.pth --local-dir /mnt/data/liuyu/project/HumanEgo/models/cotracker3

hf download Carve/LaMa-ONNX lama_fp32.onnx --local-dir /mnt/data/liuyu/project/HumanEgo/models/lama-onnx

echo "All weights downloaded to ./models"