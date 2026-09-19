#!/bin/bash
CUDA_VISIBLE_DEVICES=GPU-00d31b08-e71c-a0ad-7f0f-62ee482cda42,GPU-d2c7640f-db52-04c5-6d03-6635359d91a9 \
/root/llama.cpp/build/bin/llama-server \
  -m /models/Models/bartowski/deepreinforce-ai_Ornith-1.0-35B-GGUF/deepreinforce-ai_Ornith-1.0-35B-Q5_K_M.gguf \
  --mmproj /models/Models/bartowski/deepreinforce-ai_Ornith-1.0-35B-GGUF/mmproj-deepreinforce-ai_Ornith-1.0-35B-f16.gguf \
  --n-gpu-layers 999 \
  --tensor-split 5,3 \
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
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20


  # Recommended Agentic/Benchmark usage 
  #--temp 1.0 \
  #--top-p 1.0\
