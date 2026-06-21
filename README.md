# llama-server-scripts

Bash scripts for running `llama-server` from llama.cpp, tuned for split-GPU setups with mismatched VRAM. Each script targets a specific model and includes benchmark results from the hardware it was tested on.

## Hardware

All scripts in this repo were tuned on the same two-GPU rig:

- NVIDIA RTX 3060 Ti (8 GiB, CUDA 0)
- NVIDIA RTX 5060 Ti (16 GiB, CUDA 1)
- 40 GB system RAM

The recurring problem this repo solves: fitting large models with long context windows into 24 GiB of total VRAM split across two cards of unequal size, while keeping generation speed usable. The notes under each script explain the flag choices that made that work.

---

## Qwen3.6-27B-MTP

Script: [`run-Qwen3.6-27B-MTP.sh`](run-Qwen3.6-27B-MTP.sh)
Stats: [`Qwen3.6-27B-MTP-Stats/`](Qwen3.6-27B-MTP-Stats/)
Model: `unsloth/Qwen3.6-27B-MTP-GGUF` (UD-Q4_K_XL)
Quant: UD-Q4_K_XL (Unsloth importance-weighted Q4)
Context: 80k (81920)
KV cache: q4_0 / q4_0

### Measured performance

| Metric | Value |
|---|---|
| Prompt processing | 206 tokens/second |
| Generation (sustained, 8192 tokens) | 35.0 tokens/second |
| Generation (short, 2048 tokens) | 33.8 tokens/second |
| MTP draft acceptance | 90.6% (4524 accepted / 4993 generated) |
| Mean accepted draft length | 2.88 of 3 |
| Acceptance per position | 0.938 / 0.575 / 0.373 |

No measurable throughput decay across 8000+ generated tokens. The q4_0 KV cache plus `--swa-full` eliminated the spill-and-reprocess pattern that earlier configs hit at long context.

### Flag notes

A few of the flags in the script are not obvious and took iteration to get right. The script itself is one click away, so these notes focus on the why rather than restating the what.

**`--tensor-split 1,2`** — splits layers across the two GPUs in proportion to their VRAM (8:16 is roughly 1:2). The default 1:1 overloads the smaller card. Pushing it to 1:3 to load more onto the bigger card sounds helpful, but the smaller card then becomes the pipeline bottleneck and generation slows down. 1:2 keeps both cards finishing their slices at roughly the same time.

**`--swa-full`** — Qwen3.6 uses hybrid sliding-window attention. Without this flag, the server invalidates context checkpoints on every new turn and reprocesses the entire prompt from scratch. On a long conversation that adds 15-20 seconds of silence before each reply. With it, follow-up turns process in about a second. This is the single biggest UX win in the script.

**`--cache-type-k q4_0 --cache-type-v q4_0`** — the floor for fitting 80k context into 24 GiB total VRAM. Going from q8_0 to q4_0 trades roughly 5-10% long-range recall (needle-in-haystack style tasks) for 2.5x the context window and roughly 2x the generation speed at long context. For chat, coding, and reasoning this is the right trade. If you regularly feed in 60k-token documents and need to pull a specific detail out of them, drop context to 32k and raise K cache to q5_0 or q8_0.

**`--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75`** — enables multi-token prediction using the MTP heads baked into the GGUF. The model drafts up to 3 tokens per step and the server verifies them in a single forward pass. No separate draft model is needed. `--spec-draft-p-min 0.75` skips drafting when the model's confidence is below 0.75; without it, MTP can actually be slower than plain generation on long outputs because rejected drafts waste compute. 0.75 is the sweet spot for Qwen3.6-27B.

**`--reasoning-budget 4096`** — caps thinking tokens. The model runs long reasoning chains by default; if you don't need deep chain-of-thought on every request, this saves several seconds with no measurable quality loss on normal tasks.

### Notes

- Requires a recent llama.cpp build (May 2026 or later) for the `--spec-type draft-mtp` flag. Older builds used `--spec-type mtp`, which was renamed.
- The MTP GGUF from Unsloth includes the draft heads in the same file. No separate draft model download is needed.
- `llama-bench` cannot load MTP GGUFs and does not support speculative decoding flags. Benchmarking was done against the running server via the OpenAI-compatible API. See the stats directory for the raw JSON and draft acceptance logs.

---

## Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved

Script: *(pending)*
Stats: *(pending)*
Model: `Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF` (Q3_K_M)

*Notes and benchmark results to be filled in once the script is tuned.*

---
