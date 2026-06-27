#!/bin/bash
CUDA_VISIBLE_DEVICES=GPU-00d31b08-e71c-a0ad-7f0f-62ee482cda42,GPU-d2c7640f-db52-04c5-6d03-6635359d91a9 \
/root/llama.cpp/build/bin/llama-server \
  -m /models/Models/HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf \
  -md /models/Models/HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP/mtp-gemma-4-26B-A4B-it.gguf \
  --mmproj /models/Models/HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP/mmproj-Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf \
  --n-gpu-layers 999 \
  --tensor-split 7,2 \
  --flash-attn on \
  -c 262144 \
  -b 4096 \
  -ub 512 \
  -np 1 \
  --jinja \
  --host 0.0.0.0 \
  --port 8080 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  --spec-draft-n-min 0 \
  --spec-draft-p-min 0.75 \
  --temp 0.6 \
  --top-k 64 \
  --top-p 0.9 \
  --min-p 0.05 \
  --repeat-penalty 1.1
