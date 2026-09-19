#!/bin/bash
# Frozenlock/Qwen3.8-27B-int4-AutoRound — W4A8 int8 activations, spec OFF
# Patches applied to /opt/vllm29: pr48375 (prefix caching) + w4a8-int8-act.
# W4A8 armed via VLLM_MARLIN_INPUT_DTYPE=int8 below. To revert to W4A16: DELETE that
# export line (an empty value crashes vLLM — absence means off).


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
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=201326592
export VLLM_MARLIN_INPUT_DTYPE=int8

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

/opt/vllm29/bin/vllm serve /models/Models/Frozenlock/Qwen3.8-27B-int4-AutoRound \
  --served-model-name qwen3.8-27b-w4a8 \
  --quantization auto_round \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --long-prefill-token-threshold 4096 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASHINFER \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --mamba-cache-mode align \
  --mamba-block-size 8192 \
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
  --host 0.0.0.0 \
  --port 8080