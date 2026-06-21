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

Script: [`run-Qwen3.6-35B-A3B-MTP-heretic.sh`](run-Qwen3.6-35B-A3B-MTP-heretic.sh)
Stats: [`Qwen3.6-35B-A3B-uncensored-heretic-MTP-Stats/`](Qwen3.6-35B-A3B-uncensored-heretic-MTP-Stats/)
Model: `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF` (Q4_K_S)
Context: 90k (90000)
KV cache: q8_0 / q8_0

This is a Mixture-of-Experts model with only 3B active parameters out of 35B total. That changes the optimization math significantly compared to the 27B dense model above. Generation is cheap because only 3B parameters fire per token, but the full 35B weight set must still reside in VRAM. The result is a model that is fast by default and bandwidth-bound only at the margins, which limits how much MTP can help.

The stats directory contains two subdirectories: `baseline/` (the original Q3_K_M config with no MTP) and `q4_ks_mtp/` (the current Q4_K_S config with MTP enabled). Comparing them tells the story of what each change bought.

### Measured performance

| Metric | Baseline (Q3_K_M, no MTP) | Final (Q4_K_S + MTP) | Change |
|---|---|---|---|
| Generation (short, 2048 tok) | 114.8 t/s | 121.3 t/s | +5.7% |
| Generation (long, ~7000 tok) | 112.2 t/s | 124.9 t/s | +11.3% |
| Prompt processing (warm) | 307.1 t/s | 414.7 t/s | +35% |
| Throughput decay over 7k tokens | ~3% | ~0% (flat) | Improved |
| MTP draft acceptance | N/A | 90.3% (3867 / 4283) | — |
| Mean accepted draft length | N/A | 2.84 of 3 | — |
| Acceptance per position | N/A | 0.932 / 0.549 / 0.359 | — |
| VRAM headroom (idle) | 5.3 GiB | 1.8 GiB | -3.5 GiB |

The throughput curve is flat across 7000+ generated tokens. No spill, no decay. The model fits cleanly in VRAM at 90k context.

### What the numbers mean

The Q3 to Q4 quant upgrade is the real win here, not the tps delta. A full quant level higher means measurably better output quality, less degradation on reasoning and code, and fewer of the subtle errors that Q3 models produce under stress. The speed improvement on top of that is a bonus.

MTP gives a smaller relative speedup on this MoE than on the 27B dense model. On the 27B, MTP roughly doubled throughput because the dense model is bandwidth-bound and MTP amortizes memory reads. On this MoE, generation is already fast at 112 t/s baseline (only 3B active params per token), so there is less bandwidth pressure for MTP to relieve. The +11% is real but modest. The draft heads cost 625 MiB of VRAM at startup and about 4 MiB of runtime state, which is the price of admission.

Prompt processing saw the largest jump at +35%. The warm prompt processing rate of 414 t/s means that even without `--swa-full`, follow-up turns that reprocess the prompt cost well under a second for typical conversation lengths. The SWA invalidation warnings still appear in the log, but their practical impact is negligible at this speed.

### Flag notes

**`--tensor-split 1,2`** — the baseline used `1,3` which puts 75% of the model on the 5060 Ti. That works at Q3_K_M (17.3 GB) but overflows the 5060 Ti at Q4_K_S (20.4 GB). Rebalancing to 1:2 (33% / 67%) matches the 8:16 VRAM ratio and leaves headroom on both cards. This is the same ratio used on the 27B dense model.

**`--cache-type-k q8_0 --cache-type-v q8_0`** — unlike the 27B dense model, which dropped to q4_0 KV to fit 80k context, this model keeps q8_0 KV at 90k. The reason is architectural: Qwen3.6-35B-A3B uses hybrid sliding-window attention plus recurrent layers, so the KV cache does not grow linearly with context. The sliding window caps the attention KV, and the recurrent layers use fixed-size state. Context is nearly free in VRAM terms, so there is no pressure to quantize the KV cache further. q8_0 preserves full long-range recall at no cost.

**`--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75`** — same MTP configuration as the 27B. The model has MTP heads preserved in the GGUF and they activate automatically. Acceptance is 90.3%, slightly lower than the 27B's 90.6%, with a mean accepted length of 2.84 versus 2.88. The per-position acceptance drops faster (0.932, 0.549, 0.359 vs the 27B's 0.938, 0.575, 0.373), which is expected for a MoE where expert routing introduces more variability in the next-token distribution.

**`--reasoning-budget 4096`** — same cap as the 27B. The model's chat template has thinking enabled by default, and without a budget it will run long reasoning chains on every request. 4096 is enough for most tasks and saves several seconds per request.

**No `--swa-full`** — omitted deliberately. On the 27B dense model, `--swa-full` was the single biggest UX win because it stopped 15-20 second prompt reprocessing on follow-up turns. On this MoE, prompt processing is so fast (414 t/s warm) that reprocessing a typical conversation prompt takes under a second. Adding `--swa-full` at 90k context would force full-context KV for the SWA layers, adding several GiB of KV that the 1.8 GiB of headroom cannot absorb. The cost-benefit is inverted compared to the 27B.

### Notes

- The stats directory is organized into `baseline/` and `q4_ks_mtp/` subdirectories, each containing the server script, benchmark JSON, command files, server log, and VRAM snapshot for that configuration.
- Requires a recent llama.cpp build (May 2026 or later) for the `--spec-type draft-mtp` flag. Older builds used `--spec-type mtp`, which was renamed.
- The MTP heads are preserved in the same GGUF by llmfan46. No separate draft model download is needed.
- Higher quants (Q4_K_M, Q5_K_S, Q5_K_M) were considered and rejected. Q4_K_M fits with under 1 GiB headroom and no room for MTP growth. Q5 and above do not fit at 90k context without dropping to q4_0 KV, which defeats the purpose of the upgrade. Q4_K_S is the sweet spot for this hardware.
- The model is uncensored via the Heretic abliteration method. Output behavior differs from base Qwen3.6-35B-A3B. Benchmark numbers reflect inference performance only and are independent of the ablitation.

---

## Cross-model insights: 27B dense vs 35B-A3B MoE

Tuning both models on the same hardware revealed how different architectures respond to the same optimization toolkit. The takeaways generalize to other dense vs MoE pairs on split-GPU rigs.

### MTP speculative decoding behaves differently by architecture

| | 27B dense | 35B-A3B MoE |
|---|---|---|
| Baseline t/s (no MTP) | ~12-16 | ~112 |
| With MTP | ~35 | ~125 |
| Speedup from MTP | ~2x | +11% |
| Draft acceptance | 90.6% | 90.3% |
| Mean accepted length | 2.88 / 3 | 2.84 / 3 |

MTP acceptance rates are nearly identical between the two models, but the throughput payoff is vastly different. The 27B dense model is memory-bandwidth-bound: every token requires reading all 27B weights, so drafting 3 tokens and verifying them in one pass roughly doubles effective throughput. The 35B-A3B MoE is compute-cheap per token (only 3B active params), so it is already fast without speculation. MTP's amortization of bandwidth matters less when bandwidth is not the bottleneck. The lesson: MTP is high-impact on dense models, modest on small-active-param MoEs.

### KV cache strategy is driven by attention architecture, not context size

The 27B dense uses standard attention, so KV grows linearly with context. At 80k tokens, q8_0 KV would consume ~10 GiB — more than the model itself. Dropping to q4_0 was mandatory to fit the context window, at the cost of 5-10% long-range recall.

The 35B-A3B uses hybrid sliding-window attention plus recurrent layers. The sliding window caps attention KV at a fixed size regardless of context length, and recurrent layers use fixed-size state. Context is nearly free in VRAM. q8_0 KV at 90k costs roughly the same as q8_0 at 4k on a standard attention model. There was never a reason to quantize the KV cache further.

The lesson: check the attention architecture before choosing KV quant. Standard attention forces a context-vs-quality trade at long context. Hybrid SWA + recurrent models sidestep that trade entirely.

### `--swa-full` helps when prompt processing is slow, hurts when VRAM is tight

On the 27B dense, `--swa-full` was the single biggest UX win. Without it, follow-up turns reprocessed the entire prompt from scratch at ~140 t/s, adding 15-20 seconds of silence before each reply on long conversations. Adding it dropped that to about a second.

On the 35B-A3B MoE, `--swa-full` is omitted. Prompt processing is 414 t/s warm — reprocessing a typical conversation prompt takes under a second even without it. Meanwhile the 1.8 GiB of VRAM headroom cannot absorb the several GiB of full-context SWA KV that the flag would allocate. The cost-benefit is inverted.

The lesson: `--swa-full` is worth it when prompt processing is the bottleneck and VRAM has room. When prompt processing is already fast and VRAM is tight, skip it.

### Tensor split must track the model size, not just the card ratio

Both models landed on `--tensor-split 1,2`, but for different reasons. The 27B at Q4_K_XL is ~16.5 GB, and 1:2 balances the pipeline so neither card finishes early and waits. The 35B-A3B at Q4_K_S is 20.4 GB, and 1:2 is the only ratio that fits — 1:3 overflows the 5060 Ti, 1:1 overflows the 3060 Ti.

The baseline 35B script used 1:3, which worked at Q3_K_M (17.3 GB) but broke at Q4_K_S. The lesson: when changing quants, recheck the tensor split. A split that worked at one model size may overflow a card at the next size up.

### What generalizes

- Match tensor split to VRAM ratio, not to a number that worked on a different model size.
- Drop KV quant only when context pressure forces it. If the architecture gives you context for free, keep KV quality high.
- Try MTP on everything that supports it, but expect the biggest wins on dense models and modest wins on small-active-param MoEs.
- `--spec-draft-p-min 0.75` and `--spec-draft-n-max 3` worked identically well across both architectures. These are safe defaults for Qwen3.6-family MTP models.
- `--reasoning-budget 4096` is a free win on any Qwen3.6 model with thinking enabled. Saves time, no measurable quality loss on normal tasks.

---

## Acknowledgments

Research and benchmarking for this repo was conducted with assistance from GLM-5.2 (opencode-go/glm-5.2), which helped analyze llama.cpp flag behavior, interpret server logs, and model the VRAM sizing math for split-GPU configurations.
