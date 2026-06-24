#!/bin/bash
/root/llama.cpp/build/bin/llama-server \
  -m /models/Models/llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q3_K_M.gguf \
  --n-gpu-layers 999 \
  --tensor-split 1,3 \
  --flash-attn on \
  -c 90000 \
  -b 4096 \
  -ub 512 \
  -np 1 \
  --jinja \
  --host 0.0.0.0 \
  --port 8080 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0