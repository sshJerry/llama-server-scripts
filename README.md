# llama-server-scripts

Bash scripts for running `llama-server` from llama.cpp, tuned for split-GPU setups with mismatched VRAM. Each script targets a specific model and includes benchmark results from the hardware it was tested on.

## Hardware

All scripts in this repo were tuned on the same two-GPU rig, running llama.cpp inside an LXC container on Proxmox:

- NVIDIA RTX 3090 (24 GiB, CUDA 0, compute 8.6) — PCIe 4.0 x16 (via CPU, PCI_E1)
- NVIDIA RTX 5060 Ti (16 GiB, CUDA 1, compute 12.0) — PCIe 4.0 x4 (via chipset, PCI_E3)
- 48 GB DDR4 3600 MHz system RAM
- Intel Core i5-12400 (6C/12T), MSI PRO Z790-P WIFI DDR4
- Debian 13 (trixie), NVIDIA driver 595.84, CUDA 13.2
- llama.cpp build 9733, GNU 14.2.0

The recurring problem this repo solves: fitting large models with long context windows into 40 GiB of total VRAM split across two cards of unequal size, while keeping generation speed usable. The notes under each script explain the flag choices that made that work.

### Power management

Persistent mode and per-GPU power limits are applied at boot via cron:

```
@reboot nvidia-smi -pm 1 && nvidia-smi -i 0 -pl 250 && nvidia-smi -i 1 -pl 155
```

| GPU | Power limit |
|-----|------------|
| RTX 3090 (CUDA 0) | 250W |
| RTX 5060 Ti (CUDA 1) | 155W |

---

## Models

Each model has a launch script, a stats directory with benchmark results, and notes on the flag choices. Benchmarks were run with [`bench.sh`](bench.sh) — see the [Benchmarking](#benchmarking) section below for usage.

---

### Ornith-1.0-35B Q5_K_M

Script: [`run-Ornith-1.0-35B.sh`](run-Ornith-1.0-35B.sh)
Stats: [`Ornith-1.0-35B-Stats/q5_k_m/`](Ornith-1.0-35B-Stats/q5_k_m/)
Model: `bartowski/deepreinforce-ai_Ornith-1.0-35B-GGUF` (Q5_K_M)
Architecture: 35B MoE (~3B active)
Context: 256k (262144)
KV cache: q8_0 / q8_0
MTP: No (Ornith has no draft heads)

Ornith is a reinforcement-learning-trained agent model. It is a 35B MoE like Qwen3.6-35B-A3B below, but without MTP speculative decoding heads. The Q5_K_M quant balances quality against VRAM — Q6_K and above didn't fit alongside AgentWorld in the tandem setup.

#### Measured performance

| Metric | Value |
|---|---|
| Prompt processing (bulk, 5610 tok) | 3,893 tokens/second |
| Generation (short, 2048 tok) | 127.5 tokens/second |
| Generation (long, 7125 tok) | 124.3 tokens/second |
| Throughput decay over 7K tokens | −3.1% |
| MTP draft acceptance | N/A (no MTP heads) |
| VRAM idle — RTX 3090 | 19,700 / 24,576 MiB (4,876 MiB free) |
| VRAM idle — RTX 5060 Ti | 11,466 / 16,311 MiB (4,845 MiB free) |
| VRAM load — RTX 3090 | 248.5 W, 59°C |
| VRAM load — RTX 5060 Ti | 86.6 W, 48°C |

Throughput is essentially flat across context — the −3.1% drop from token 385 to token 7098 is near-monotonic with no anomalies. The MoE architecture means generation is compute-cheap (only ~3B active params per token), so KV cache growth doesn't drag on speed.

#### Flag notes

**`--tensor-split 5,3`** — puts 62.5% of layers on the 3090. At Q5_K_M the model is ~22 GB, and 5:3 fits both cards with ~4.8 GiB headroom each.

**`--cache-type-k q8_0 --cache-type-v q8_0`** — hybrid SWA architecture makes context nearly free in VRAM. No need to drop KV quant.

**No `--spec-type`** — Ornith lacks MTP heads. The model is already fast enough at 124 t/s that the missing speculative speedup is not a practical concern.

**`--temp 0.6 --top-p 0.95 --top-k 20`** — recommended agent sampling settings from the Ornith model card. Tighter than the defaults.

#### No previous comparison

Ornith is new to this hardware. There is no 3060 Ti baseline. The most relevant comparison is against Qwen3.6-35B-A3B below — same architecture, same hardware, with vs. without MTP.

---

### Qwen3.6-27B-MTP-heretic Q6_K

Script: [`run-Qwen3.6-27B-MTP-heretic.sh`](run-Qwen3.6-27B-MTP-heretic.sh)
Stats: [`Qwen3.6-27B-MTP-heretic-Stats/q6_k/`](Qwen3.6-27B-MTP-heretic-Stats/q6_k/)
Model: `llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF` (Q6_K)
Architecture: 27B dense
Context: 256k (262144)
KV cache: q4_0 / q4_0
MTP: Yes — draft heads preserved, `--spec-type draft-mtp`

This is a dense 27B model with MTP speculative decoding. The Q6_K quant is a two-level upgrade from the Q4_K_S benchmarked on the old 3060 Ti rig. The 3090's extra VRAM makes it possible — Q6_K wouldn't have fit the 3060 Ti at any context.

#### Measured performance

| Metric | Value |
|---|---|
| Prompt processing (bulk, 5610 tok) | 983 tokens/second |
| Generation (short, 2048 tok) | 38.0 tokens/second |
| Generation (long, 6855 tok) | 37.6 tokens/second |
| MTP draft acceptance (short) | 90.9% (1,118 / 1,230) |
| MTP draft acceptance (long) | 91.9% (3,792 / 4,125) |
| Mean accepted draft length | 2.90 of 3 |
| Acceptance per position | 0.940 / 0.581 / 0.381 |
| VRAM idle — RTX 3090 | 20,552 / 24,576 MiB (4,024 MiB free) |
| VRAM idle — RTX 5060 Ti | 13,610 / 16,311 MiB (2,701 MiB free) |
| VRAM load — RTX 3090 | 225.7 W, 62°C |
| VRAM load — RTX 5060 Ti | 105.3 W, 58°C |

Generation is steady at ~37-38 t/s with <1% variation between the 2048-token and 6855-token runs. The model stopped early at 6855/8192 tokens (it completed the essay) — not a performance issue.

#### Previous vs current

| Metric | Q4_K_S (3060 Ti + 5060 Ti) | Q6_K (3090 + 5060 Ti) | Change |
|---|---|---|---|
| Generation (short) | 36.7 t/s | 38.0 t/s | +3.5% |
| Generation (long) | 37.6 t/s | 37.6 t/s | — |
| Prompt processing (bulk) | — | 983 t/s | new metric |
| MTP acceptance | 91.8% | 90.9–91.9% | comparable |
| Context | 153k | 256k | +67% |
| VRAM free (3090) | N/A (3060 Ti: 0.5 GiB) | 4,024 MiB | — |
| VRAM free (5060 Ti) | ~1.1 GiB | 2,701 MiB | +1.5 GiB |
| Quant | Q4_K_S (~4.5 bpw) | Q6_K (~6.5 bpw) | +2 bpw |

The generation tps is nearly identical — the 27B dense model at Q6_K on the 3090 runs at the same speed as Q4_K_S on the 3060 Ti. The real wins are context (+67%), VRAM headroom (+1.6 GiB on the 5060 Ti), and a two-level quant upgrade that measurably improves output quality. The 3090 doesn't make the 27B faster — it makes room for higher precision and longer context.

#### Throughput decay

The decay curve shows a ramp-down from ~49 t/s to ~38 t/s over the first 900 tokens, then flat at ~37-38 for the remaining 5,900 tokens. Two mechanisms: KV cache growth (attention over 33 keys at token 1 vs. ~900 at the knee) tapers off once the marginal cost per additional KV entry becomes negligible, and the reasoning budget expires early in the window. The flat portion confirms the KV cache and bandwidth configuration are not degrading over long generations.

#### GPU bottleneck

The 5060 Ti at 63% utilization is the pipeline bottleneck — the 3090 at 54% is stalling, waiting on PCIe 4.0 x4 transfers and the 5060 Ti's layer slices. This is the same bottleneck that limited the old 3060 Ti rig. Upgrading to the 3090 bought VRAM capacity and context headroom but didn't shift the throughput ceiling — the 5060 Ti's 128-bit bus and x4 link are the limiter. A second 3090 (or moving the 5060 Ti to an x16 slot) would unlock more of the first 3090's potential. For practical throughput gains, Q5_K_M or Q4_K_S on this hardware would likely reach 42-48 t/s by reducing per-token weight traffic.

---

### Qwen3.6-35B-A3B-MTP-heretic Q6_K

Script: [`run-Qwen3.6-35B-A3B-MTP-heretic.sh`](run-Qwen3.6-35B-A3B-MTP-heretic.sh)
Stats: [`Qwen3.6-35B-A3B-MTP-heretic-Stats/q6_k/`](Qwen3.6-35B-A3B-MTP-heretic-Stats/q6_k/)
Model: `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF` (Q6_K)
Architecture: 35B MoE (~3B active)
Context: 256k (262144)
KV cache: q8_0 / q8_0
MTP: Yes — draft heads preserved, `--spec-type draft-mtp`

This is the same architecture as Ornith above (35B MoE, ~3B active params), but with MTP heads and at Q6_K quant. Q6_K on a 35B model is demanding — it pushes VRAM to the limit on this hardware. The old rig (3060 Ti) couldn't fit Q5_K, let alone Q6_K.

#### Measured performance

| Metric | Value |
|---|---|
| Prompt processing (bulk, 5610 tok) | 3,117 tokens/second |
| Generation (short, 2048 tok) | 115.5 tokens/second |
| Generation (long, 8192 tok) | 119.9 tokens/second |
| MTP draft acceptance (short) | 91.8% (1,080 / 1,176) |
| MTP draft acceptance (long) | 90.4% (4,409 / 4,876) |
| Mean accepted draft length | 2.88 of 3 |
| Acceptance per position | 0.932 / 0.578 / 0.372 |
| VRAM idle — RTX 3090 | 22,706 / 24,576 MiB (1,870 MiB free) |
| VRAM idle — RTX 5060 Ti | 14,342 / 16,311 MiB (1,969 MiB free) |
| VRAM load — RTX 3090 | 209.6 W, 58°C |
| VRAM load — RTX 5060 Ti | 82.8 W, 48°C |

The model generated the full 8192 tokens without stopping early — consistent with the MoE architecture's tendency to fill the token budget. Throughput is excellent at 116-120 t/s.

#### Previous vs current

| Metric | Q4_K_S (3060 Ti + 5060 Ti) | Q6_K (3090 + 5060 Ti) | Change |
|---|---|---|---|
| Generation (short) | 121.3 t/s | 115.5 t/s | −4.8% |
| Generation (long) | 124.9 t/s | 119.9 t/s | −4.0% |
| Prompt processing (bulk) | — | 3,117 t/s | new metric |
| MTP acceptance | 90.3% | 90.4–91.8% | comparable |
| Context | 90k | 256k | +184% |
| VRAM free (3090) | N/A (3060 Ti: <1 GiB) | 1,870 MiB | — |
| VRAM free (5060 Ti) | ~1.4 GiB | 1,969 MiB | +0.5 GiB |
| Quant | Q4_K_S (~4.5 bpw) | Q6_K (~6.5 bpw) | +2 bpw |

Generation tps is 4-5% *slower* at Q6_K despite the hardware upgrade. This is expected: Q6_K is a much larger file than Q4_K_S (~24 GB vs ~20 GB), and the denser weight reads cost bandwidth. The 3090's additional VRAM absorbs this — the old 3060 Ti couldn't load the model at all — but the memory bandwidth ceiling is still there. The payoff is output quality: Q6_K is a two-level quant upgrade over Q4_K_S, delivering measurably better reasoning and code generation.

Context jumped from 90k to 256k (+184%). The hybrid SWA architecture means this costs virtually nothing in KV cache VRAM.

#### Throughput decay

The decay curve for the MoE shows a U-shape: 131 t/s → 115 t/s → 120 t/s. The initial 131 t/s is prompt-influenced (the model processes the last cached tokens at higher effective speed). It settles to ~115-116 t/s after ~2,000 tokens, then gradually recovers to ~120 t/s toward the end. The recovery is content-driven: the prompt asks for a technical guide with code examples, and the transition from prose (higher entropy, lower MTP acceptance) to structured code (predictable syntax, higher MTP acceptance) lifts effective throughput. Net variation is ~4% peak-to-trough — negligible in practical terms.

#### GPU bottleneck

GPU utilization is low (38% on the 3090, 42% on the 5060 Ti) despite 120 t/s throughput. Both memory controllers are pegged at maximum frequency (9501 / 13801 MHz), while SMs sit mostly idle — the classic signature of a memory-bandwidth-bound workload. The 3090 reads 22.7 GiB of weights every forward pass; with only 936 GB/s of bandwidth, that's the ceiling. The 5060 Ti's x4 PCIe link adds inter-GPU transfer latency after every layer, keeping the 3090 stalled. Like the 27B, this rig's throughput is limited by the 5060 Ti and PCIe, not the 3090's compute.

#### VRAM headroom

At 1,870 MiB free on the 3090 and 1,969 MiB free on the 5060 Ti, Q6_K is at the limit. There is no room for `--swa-full`, no room for higher context, and no room for KV cache growth with large prompts. This config works for the benchmarked use case but leaves zero safety margin. If you hit OOM, the first lever is dropping context from 256k to 128k, which frees several hundred MiB of KV cache pre-allocation.

---

### Cross-model: Ornith 35B vs Qwen3.6 35B-A3B

Both are 35B Mixture-of-Experts models with ~3B active parameters, running on the same hardware. Ornith has no MTP heads; Qwen has MTP heads and a higher quant. This comparison isolates the effect of MTP speculative decoding on a MoE.

| Metric | Ornith Q5_K_M (no MTP) | Qwen Q6_K (MTP) | Delta |
|---|---|---|---|
| Generation (short) | 127.5 t/s | 115.5 t/s | Ornith +10.4% |
| Generation (long) | 124.3 t/s | 119.9 t/s | Ornith +3.7% |
| Prompt processing (bulk) | 3,893 t/s | 3,117 t/s | Ornith +25% |
| MTP acceptance | N/A | 90.4–91.8% | — |
| VRAM used (3090) | 19,700 MiB | 22,706 MiB | Qwen +3,006 MiB |
| VRAM free (3090) | 4,876 MiB | 1,870 MiB | Qwen −3,006 MiB |
| Quant | Q5_K_M (~5.5 bpw) | Q6_K (~6.5 bpw) | Qwen +1 bpw |
| Power load (3090) | 248.5 W | 209.6 W | Ornith +39W |
| GPU util (3090) | 48% | 38% | Ornith +10pp |

Ornith is faster across the board despite having no speculative decoding. Three factors explain it:

1. **Lower quant.** Q5_K_M is a smaller file than Q6_K. Less weight data per forward pass means less bandwidth pressure. On a MoE that's already compute-light, bandwidth is the dominant constraint — a smaller quant directly translates to higher tps.

2. **No MTP overhead.** MTP draft heads consume ~625 MiB of VRAM at startup and cost a small per-token overhead for draft generation and verification. On a dense model this overhead is dwarfed by the bandwidth savings. On a MoE, the savings are smaller and the overhead can be a net negative at high quants.

3. **Prompt processing gap.** Ornith processes bulk prompts 25% faster than Qwen Q6_K. This is entirely the quant size difference — prompt processing reads the full weight set, and Ornith's Q5_K_M (~22 GB) is lighter than Qwen's Q6_K (~24 GB).

**The trade:** Qwen Q6_K gives you a full quant level higher quality plus MTP speculative decoding. Ornith Q5_K_M gives you 10% more tps, 3 GiB more VRAM headroom, and no speculative decoding dependency — at the cost of one quant level and no MTP. For agent workloads where speed and VRAM headroom matter more than the last quant level, Ornith Q5_K_M is the stronger pick on this hardware. For maximum output quality on reasoning and code tasks, Qwen Q6_K is the choice.

---

### Cross-model: Hardware bottlenecks

Both oracles independently identified the same limiting factor across all three models:

**The RTX 5060 Ti is the throughput ceiling.** Its 128-bit memory bus, 16 GiB capacity, and PCIe 4.0 x4 link make it the slower card in every tensor-parallel configuration. The 3090 sits partially idle (38-54% utilization depending on the model), stalling on PCIe transfers and waiting for the 5060 Ti to finish its layer slices. This was also true on the old rig — the 3060 Ti was the bottleneck then, and the upgrade to a 3090 bought VRAM headroom and context length but didn't shift the tps ceiling because the 5060 Ti didn't change.

**What would unlock the 3090:**
- A second 3090 (or any 24 GiB card with ≥384-bit bus) on an x8 or x16 slot
- Moving the 5060 Ti to a PCIe 4.0 x16 slot (currently in x4 via chipset)
- Running models that fit entirely on one GPU, avoiding tensor-parallel overhead

**The quant sweet spot:** On this hardware, Q5_K_M consistently delivers the best ratio of throughput to quality. Q6_K pushes VRAM to the limit (1.8 GiB free on the 35B MoE) with minimal throughput gain. Q4_K_S runs ~10-25% faster but loses measurable output quality. Q5_K_M on the 35B MoE (Ornith) achieves 124 t/s with 4.9 GiB headroom — the ideal balance.

---

## Legacy hardware

### [`RTX3060Ti-8GiB-RTX5060Ti-16GiB/`](RTX3060Ti-8GiB-RTX5060Ti-16GiB/)

Scripts and benchmarks from the previous rig (RTX 3060 Ti 8 GiB + RTX 5060 Ti 16 GiB). The 3060 Ti has since been retired and replaced with an RTX 3090. These remain for reference — the tensor splits, KV cache choices, and MTP tuning notes still generalize. Detailed cross-model analysis (27B dense vs 35B MoE, Unsloth vs heretic recipes) lives in that subdirectory's README.

Models covered:
- Qwen3.6-27B-MTP (Unsloth UD-Q4_K_XL)
- Qwen3.6-27B-uncensored-heretic-MTP (llmfan Q4_K_S)
- Qwen3.6-35B-A3B-uncensored-heretic-MTP (llmfan Q4_K_S)

### [`ornith-agentworld-tandem/`](ornith-agentworld-tandem/)

Two 35B MoE models running side-by-side — an agent (Ornith) and a world simulator (AgentWorld) — in an autonomous task loop across seven domains: terminal, swe, web, os, android, mcp, and search.

---

## Benchmarking

[`bench.sh`](bench.sh) is the standard benchmarking script for this repo. It runs a warm-up, short and long generation benchmarks, a prompt-processing benchmark, captures VRAM at idle and under load, and extracts MTP draft stats and throughput decay from the server log — then writes a single summary JSON.

**Requires `jq`.**

### Usage

Start the server with its output piped through `tee` so the benchmark script can extract draft stats and throughput decay:

```
./run-Ornith-1.0-35B.sh 2>&1 | tee /tmp/llama-server.log
```

Then run the benchmark:

```
./bench.sh <output_dir> <prefix> [port] [server_log]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `output_dir` | *(required)* | Directory for result files (created if missing) |
| `prefix` | *(required)* | Filename prefix, e.g. `bench` or `q4_ks` |
| `port` | `8080` | llama-server API port |
| `server_log` | `/tmp/llama-server.log` | Path to the tee'd server log |

Example:

```
./bench.sh RTX3090-24GiB-RTX5060Ti-16GiB/Qwen3.6-27B-uncensored-heretic-MTP-Stats/default bench
```

### Output files

| File | Contents |
|------|----------|
| `<prefix>_short.json` | 2048-token generation benchmark (`timings` + `usage`) |
| `<prefix>_long.json` | 8192-token generation benchmark |
| `<prefix>_prompt.json` | Prompt-processing benchmark (~5600 tokens in, 1 token out) |
| `<prefix>_vram-idle.txt` | `nvidia-smi` at idle |
| `<prefix>_vram-load.txt` | `nvidia-smi` during active generation |
| `<prefix>_draft-stats.txt` | MTP draft acceptance lines from server log |
| `<prefix>_decay.csv` | Throughput-over-time CSV (`tokens,tps`) from server log |
| `<prefix>_summary.json` | All key metrics in one file, also printed to terminal |

The server log path must match between the `tee` command and `bench.sh`. If you use a different log path, pass it as the fourth argument:

```
./run-some-model.sh 2>&1 | tee ~/my-server.log
./bench.sh stats/my-model default 8080 ~/my-server.log
```

---

## Long-term goals

- **Evaluate higher quants on the 3090 + 5060 Ti.** The extra 16 GiB on the 3090 (vs. the retired 3060 Ti) re-opens design space on tensor split ratios, KV cache quality, and context windows above 150k.
- **Set up vLLM and evaluate whether it offers benefits over llama-server on this hardware.** The current scripts are tuned for llama.cpp specifically. vLLM has different scheduling, paged attention, and tensor-parallel assumptions that may or may not help on a two-card consumer rig.
- **Adopt the sampler settings recommended by the distributors of each quantized model.** The scripts here currently use llama-server defaults for temperature, top-p, and top-k. The model distributors (Unsloth, llmfan) publish recommended sampler parameters per checkpoint, and running close to those settings rather than the defaults can measurably improve output quality without affecting tps.

---

## Acknowledgments

Research and benchmarking for this repo was conducted with assistance from GLM-5.2 (opencode-go/glm-5.2), which helped analyze llama.cpp flag behavior, interpret server logs, and model the VRAM sizing math for split-GPU configurations.
