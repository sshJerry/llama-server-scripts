#!/bin/bash
#
# Qwen3.8-Flash-Next: albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE served via the
# DominikBucko pinned-vLLM docker runtime (/opt/flashnext).
#   Topology:  2x RTX 3090, TP2+EP2, UVA expert offload (~114 GB true host RAM
#              residency, ~58 GB of it in NVIDIA driver pages invisible to
#              container level memory counters), hot-84 expert cache on GPU,
#              MTP3 draft, prefix caching on (vllm#48375 patched, see
#              pr48375_mamba_drop_eagle_block.py in this directory).
#   Context:   262144 native (KV_CACHE_MEMORY_BYTES=4429185024 covers the
#              full window: 276,313 tokens, 1.05x one full context request).
#   Endpoint:  0.0.0.0:8080 on the machine's LAN address
#              (PORT=8080 in .env + the 0.0.0.0 bind in scripts/docker_serve.sh,
#              both one-time edits, see GETTING_STARTED.md).
#   Load:      ~129 GB over NFS, ~20 minutes per boot. The runtime caches
#              kernels, not weights.
#
# Foreground launcher: serves until Ctrl+C or the container dies (OOM/crash).
# Either way the container is stopped and removed and both GPUs return to
# base state.

cd /opt/flashnext || exit 1

# Refuse to double-launch; clear stale leftovers
if docker ps --format '{{.Names}}' | grep -q '^qwen38-flash-next$'; then
    echo "[flash-next] already running - Ctrl+C its terminal first:"; docker ps; exit 1
fi
docker rm -f qwen38-flash-next >/dev/null 2>&1

# Refuse if something else holds the GPUs
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
if [ "${used:-0}" -gt 2048 ]; then
    echo "[flash-next] GPUs not idle (${used} MiB used) - stop the running tier first"; exit 1
fi

# Port config must be the LAN-visible form
grep -q '^PORT=8080' .env && grep -q '0\.0\.0\.0' scripts/docker_serve.sh || {
    echo "[flash-next] incomplete: need PORT=8080 in .env and the 0.0.0.0 bind in scripts/docker_serve.sh"; exit 1; }

cleanup() {
    trap - INT TERM
    echo ""
    echo "[shutdown] removing container..."
    docker rm -f qwen38-flash-next >/dev/null 2>&1
    sleep 3
    echo "[shutdown] base state restored. GPU memory (expect ~1 MiB each):"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 0
}
trap cleanup INT TERM

# Foreground: build (cached) -> preflight -> docker run attached.
# Ctrl+C stops the container via signal propagation -> trap cleans up.
# If the container dies on its own (OOM), make returns -> final cleanup below
# restores base state the same way.
make serve
cleanup
