#!/bin/bash
#
# Use case: Thinking mode for general tasks
#   temperature=1.0  top_p=0.95  top_k=20  min_p=0.0  presence_penalty=1.5  repetition_penalty=1.0
#
# Use case: Thinking mode for precise coding tasks (e.g. WebDev)
#   temperature=0.6  top_p=0.95  top_k=20  min_p=0.0  presence_penalty=0.0  repetition_penalty=1.0
#
# Use case: Agentic coding (tool calling + thinking preservation)
#   temperature=0.6  top_p=0.95  top_k=20  min_p=0.0  presence_penalty=0.0  repetition_penalty=1.0
#   --jinja  (chat_template_kwargs: {"preserve_thinking": true})
#   Tool-call parser: --tool-call-parser qwen3_coder  (vLLM/SGLang; llama.cpp supports via jinja template)
#
# Use case: Instruct (non-thinking) mode for general tasks
#   temperature=0.7  top_p=0.8  top_k=20  min_p=0.0  presence_penalty=1.5  repetition_penalty=1.0
#
# Use case: Instruct (non-thinking) mode for reasoning tasks
#   temperature=1.0  top_p=0.95  top_k=20  min_p=0.0  presence_penalty=1.5  repetition_penalty=1.0
#
# To disable thinking via API:
#   extra_body: {"chat_template_kwargs": {"enable_thinking": false}}
#   llama.cpp CLI: --chat-template-kwargs '{"enable_thinking":false}'
#
# Vision (mmproj) disabled — MTP speculative decoding does not support --mmproj.
#
CUDA_VISIBLE_DEVICES=GPU-00d31b08-e71c-a0ad-7f0f-62ee482cda42,GPU-d2c7640f-db52-04c5-6d03-6635359d91a9 \

/root/llama.cpp/build/bin/llama-server \
  -m /models/Models/unsloth/Qwen3.5-122B-A10B-MTP-GGUF/Qwen3.5-122B-A10B-UD-IQ3_XXS.gguf \
  --flash-attn on \
  -c 262144 \
  -b 4096 \
  -ub 512 \
  -np 1 \
  --jinja \
  --temp 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --min-p 0.0 \
  --host 0.0.0.0 \
  --port 8080 \
  --no-mmap \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-n-min 0 \
  --spec-draft-p-min 0.75 \
  --reasoning-budget 32768 \
