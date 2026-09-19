#!/bin/bash
# vLLM 0.29.0 (/opt/vllm29) — lued/Qwen3.8-27B-INT8-W8A16-MTP, 2x RTX 3090 TP=2
# vs the old 0.26.1rc1 pipenv script: interpreter swapped, FLASHINFER_DISABLE_VERSION_CHECK
# dropped (flashinfer 0.6.18, no cubin mismatch), all flags re-verified on 0.29.0.

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
export UVICORN_TIMEOUT_KEEP_ALIVE=300
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"

cleanup() {
    trap - INT TERM
    echo ""; echo "[shutdown] Stopping vLLM process tree..."
    pkill -TERM -f "[v]llm serve" 2>/dev/null
    pkill -TERM -f "[E]ngineCore" 2>/dev/null
    pkill -TERM -f "[W]orker_TP" 2>/dev/null
    sleep 3
    pkill -KILL -f "[E]ngineCore" 2>/dev/null
    pkill -KILL -f "[W]orker_TP" 2>/dev/null
    echo "[shutdown] Done. VRAM released."
    exit 0
}
trap cleanup INT TERM

/opt/vllm29/bin/vllm serve /models/Models/lued/Qwen3.8-27B-INT8-W8A16-MTP \
  --served-model-name qwen3.8-27b-int8-w8a16 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --dtype bfloat16 \
  --performance-mode balanced \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8_e4m3 \
  --long-prefill-token-threshold 4096 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --mamba-block-size 8192 \
  --mamba-cache-mode align \
  --prefix-match-unit 16 \
  --enable-prompt-tokens-details \
  --enable-per-request-metrics \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --disable-custom-all-reduce \
  --trust-remote-code \
  --chat-template /models/Models/froggeric/Qwen-Fixed-Chat-Templates/chat_template.jinja \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}' \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"repetition_penalty":1.0,"presence_penalty":0.0}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --host 0.0.0.0 \
  --port 8080