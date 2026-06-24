#!/bin/bash

CUDA_VISIBLE_DEVICES=GPU-00d31b08-e71c-a0ad-7f0f-62ee482cda42,GPU-d2c7640f-db52-04c5-6d03-6635359d91a9 \

/root/llama.cpp/build/bin/llama-server \
  -m /models/Models/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q6_K.gguf \
  --mmproj /models/Models/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF/Qwen3.6-27B-mmproj-BF16.gguf \
  --n-gpu-layers 999 \
  --tensor-split 6,3 \
  --flash-attn on \
  -c 262144 \
  -b 4096 \
  -ub 512 \
  -np 1 \
  --jinja \
  --host 0.0.0.0 \
  --port 8080 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-n-min 0 \
  --spec-draft-p-min 0.75 \
  --reasoning-budget 6144 \
