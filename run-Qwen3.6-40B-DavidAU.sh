CUDA_VISIBLE_DEVICES=GPU-00d31b08-e71c-a0ad-7f0f-62ee482cda42,GPU-d2c7640f-db52-04c5-6d03-6635359d91a9 \

/root/llama.cpp/build/bin/llama-server \
  -m /models/Models/DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-NEO-CODE-Di-IMatrix-MAX-GGUF/Qwen3.6-40B-Deck-Opus-NEO-CODE-HERE-2T-OT-Q5_K_M.gguf \
  --n-gpu-layers 999 \
  --tensor-split 6,3 \
  --flash-attn on \
  -c 200000 \
  -b 512 \
  -ub 128 \
  -np 1 \
  --jinja \
  --temp 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --host 0.0.0.0 \
  --port 8080 \
  --no-mmap \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --reasoning-budget 32768 \
  #--no-kv-offload \
