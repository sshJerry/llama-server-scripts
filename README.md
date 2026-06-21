# llama-server-scripts

Bash scripts for running llama-server from llama.cpp, tuned for split-GPU setups with mismatched VRAM.

## Hardware

These scripts were tuned on a two-GPU rig:

- NVIDIA RTX 3060 Ti (8 GiB, CUDA 0)
- NVIDIA RTX 5060 Ti (16 GiB, CUDA 1)
- 40 GB system RAM

The goal was to run a 27B dense model with a long context window (up to 80k tokens) while keeping generation speed usable. The same approach works for any split-GPU setup where one card is smaller than the other.

## Scripts

### run-Qwen3.6-27B-MTP.sh

Runs Qwen3.6-27B-MTP (Unsloth UD-Q4_K_XL) with MTP speculative decoding enabled.

```bash
#!/bin/bash
/root/llama.cpp/build/bin/llama-server \
  -m /models/Models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf \
  --n-gpu-layers 999 \
  --tensor-split 1,2 \
  --flash-attn on \
  -c 81920 \
  -b 4096 \
  -ub 512 \
  -np 1 \
  --jinja \
  --host 0.0.0.0 \
  --port 8080 \
  --swa-full \
  --cache-type-k iq4_nl \
  --cache-type-v q4_0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-n-min 0 \
  --spec-draft-p-min 0.75 \
  --reasoning-budget 4096
```

## Flag notes

A few of these are not obvious and took some iteration to get right.

**--tensor-split 1,2** — splits layers across the two GPUs in proportion to their VRAM (8:16 is roughly 1:2). The default 1:1 overloads the smaller card. Pushing it to 1:3 to load more onto the bigger card sounds helpful, but the smaller card then becomes the pipeline bottleneck and generation slows down.

**--swa-full** — Qwen3.6 uses hybrid sliding-window attention. Without this flag, the server invalidates context checkpoints on every new turn and reprocesses the entire prompt from scratch. On a long conversation that adds 15-20 seconds of silence before each reply. With it, follow-up turns process in about a second.

**--cache-type-k iq4_nl / --cache-type-v q4_0** — KV cache quantization. q4_0 is the floor for fitting 80k context into 24 GiB total VRAM. iq4_nl on K is the same size as q4_0 but uses a non-linear mapping that recovers some of the long-range recall lost when dropping from q8_0. V is less sensitive, so it stays on q4_0.

**--spec-type draft-mtp** — enables multi-token prediction using the MTP heads baked into the GGUF. This is the main speed win. The model drafts up to 3 tokens per step and the server verifies them in a single forward pass. No separate draft model is needed.

**--spec-draft-p-min 0.75** — skips drafting when the model's confidence is below 0.75. Without this, MTP can actually be slower than plain generation on long outputs because rejected drafts waste compute. 0.75 is the sweet spot for Qwen3.6-27B.

**--reasoning-budget 4096** — caps thinking tokens. The model runs long reasoning chains by default; if you don't need deep chain-of-thought on every request, this saves several seconds with no measurable quality loss on normal tasks.

## Results

On the hardware above, with roughly a 650-token prompt:

- Prompt processing: ~143 tokens/second
- Generation start: ~27.5 tokens/second
- Sustained (5000+ tokens in context): ~18-22 tokens/second
- MTP draft acceptance: ~92% (3229 accepted / 3518 generated), mean accepted length 2.92

For comparison, the same model without MTP sits around 12-16 tokens/second and decays faster as context grows.

## Context versus quality

Going from q8_0 to q4_0 KV cache trades roughly 5-10% long-range recall (needle-in-haystack style tasks) for 2.5x the context window and roughly 2x the generation speed at long context. For chat, coding, and reasoning this is the right trade. If you regularly feed in 60k-token documents and need to pull a specific detail out of them, consider dropping context to 32k and raising K cache to q5_0 or q8_0.

## Notes

- Paths are hardcoded to `/root/llama.cpp/build/bin/llama-server` and `/models/...`. Adjust to your setup.
- Requires a recent llama.cpp build (May 2026 or later) for the `--spec-type draft-mtp` flag. Older builds used `--spec-type mtp`, which was renamed.
- The MTP GGUF from Unsloth includes the draft heads in the same file. No separate draft model download is needed.
