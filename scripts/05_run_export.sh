#!/bin/bash
cd /home/harve/laya-rknn/env-laya
exec /home/harve/.local/bin/uv run --no-sync python /home/harve/laya-rknn/scripts/export_onnx_fixed.py \
  --model /home/harve/laya-rknn/models/laya-multilingual \
  --output /home/harve/laya-rknn/onnx/laya-multilingual.s512.m16.fp32.onnx \
  --seq-len 512 --num-markers 16 --seed 1234
