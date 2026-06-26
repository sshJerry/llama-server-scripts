#!/bin/bash

# ── Ornith (agentive coder) ──────────────────────────────────────────
CUDA_VISIBLE_DEVICES=GPU-00d31b08-e71c-a0ad-7f0f-62ee482cda42,GPU-d2c7640f-db52-04c5-6d03-6635359d91a9 \
/root/llama.cpp/build/bin/llama-server \
  -m /models/Models/bartowski/deepreinforce-ai_Ornith-1.0-35B-GGUF/deepreinforce-ai_Ornith-1.0-35B-Q5_K_M.gguf \
  --n-gpu-layers 999 \
  --tensor-split 5,3 \
  --flash-attn on \
  -c 131072 \
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
  --top-k 20 &

ORNITH_PID=$!

# ── AgentWorld (environment simulator) ───────────────────────────────
sleep 3

CUDA_VISIBLE_DEVICES=GPU-00d31b08-e71c-a0ad-7f0f-62ee482cda42,GPU-d2c7640f-db52-04c5-6d03-6635359d91a9 \
/root/llama.cpp/build/bin/llama-server \
  -m /models/Models/unsloth/Qwen-AgentWorld-35B-A3B-GGUF/Qwen-AgentWorld-35B-A3B-UD-IQ2_M.gguf \
  --n-gpu-layers 999 \
  --tensor-split 5,3 \
  --flash-attn on \
  -c 32768 \
  -b 2048 \
  -ub 512 \
  -np 1 \
  --jinja \
  --host 0.0.0.0 \
  --port 8081 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 &

AGENTWORLD_PID=$!

# ── Keep both alive ──────────────────────────────────────────────────
trap 'kill $ORNITH_PID $AGENTWORLD_PID 2>/dev/null; exit' INT TERM
wait
