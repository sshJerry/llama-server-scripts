# llama-server-scripts

Bash scripts for running `llama-server` from llama.cpp, scripts for vLLM tuned for 2x 3090s. Each script targets a specific model and includes benchmark results from the hardware it was tested on.


## Hardware


All scripts in this repo were tuned on the same two-GPU rig, running llama.cpp inside an LXC container on Proxmox:


- NVIDIA RTX 3090 (24 GiB, CUDA 0, compute 8.6) — 265 W power cap
- NVIDIA RTX 3090 (24 GiB, CUDA 1, compute 8.6) — 265 W power cap
- Interconnect: PCIe only. No NVLink, no P2P on the stock driver, so every script runs `NCCL_P2P_DISABLE=1` + `--disable-custom-all-reduce`
- LXC container on Proxmox VE (kernel 7.0.14-4-pve): 10 cores, 115 GiB RAM; GPUs pinned to host cores 22–31 (NUMA 0) for vLLM instances. llamacpp instances are pinned to host cores 0-21. 
- Boot disk: 34 GiB LXC root — deliberately small; nothing model-related lives on it
- `/models` — NFS mount from the LAN media server
- Debian 13 (trixie), NVIDIA driver 595.84, CUDA 13.2 (system toolkit at `/usr/local/cuda-13.2`)

The recurring problem this repo solves: serving a 27B-class hybrid-GDN dense model
(Qwen3.8-27B) with a 262K context window on two 24 GiB Ampere cards through vLLM.
On this architecture the hard problems are prefill latency, prefix caching on the
mamba-hybrid KV path, KV pool sizing, and concurrency


### Power management

Persistent mode and per-GPU power limits are applied at boot via cron, same
pattern as the sibling rig:

```
@reboot nvidia-smi -pm 1 && nvidia-smi -i 0 -pl 265 && nvidia-smi -i 1 -pl 265
```

| GPU | Power limit |
|---|---|
| RTX 3090 (CUDA 0) | 265 W |
| RTX 3090 (CUDA 1) | 265 W |

All benchmarks in this repo were measured under these caps.

I've eventually turned to using this:
https://github.com/sshJerry/gpu-undervolt

### Stack as of September 19th 2026

| Component | Version | Notes |
|---|---|---|
| vLLM | 0.29.0 | plain venv at `/opt/vllm29`; migrated from a 0.26.1rc1 pipenv nightly |
| Python | 3.13.5 | |
| torch | 2.13.0+cu130 | NCCL 2.29.7 |
| flashinfer-python | 0.6.18 | no `flashinfer-cubin` exists past 0.6.13 — kernels JIT-compile via ninja against the system toolkit (`CUDA_HOME=/usr/local/cuda`) |
| ninja-build | 1.12.1 (apt) | required for the FlashInfer JIT; without it, boot dies at CUDA-graph capture with `FileNotFoundError: 'ninja'` |
| transformers | 5.17.0 | |
| compressed-tensors | 0.17.0 | |
| triton | 3.7.1 | |


### Engine patches

Three patches are applied to `/opt/vllm29`'s site-packages, vendored from
[club-3090](https://github.com/noonghunna/club-3090). All are idempotent and
anchor-checked. **All three must be re-run after any vLLM upgrade into
`/opt/vllm29`.**

| Patch | Origin | What it does |
|---|---|---|
| `vllm-pr48375-mamba-drop-eagle-block` | vllm#48375 (open) | makes `MambaManager` honor `drop_eagle_block` — required for prefix caching on the hybrid GDN architecture; without it, cached recurrent state corrupts silently (wrong tool calls / long-context output, no error) |
| `patch-inc-w4a8` | club-3090 #609 | threads `VLLM_MARLIN_INPUT_DTYPE` through vLLM's auto-round route |
| `patch-negscale-fold` | club-3090 #609 | folds AutoRound's negative group scales into the weight codes for the int8 Marlin kernel — without it, W4A8 serves garbage with no error |

The W4A8 patches are no-ops until `VLLM_MARLIN_INPUT_DTYPE=int8` is exported (the
concurrency script does this). To revert to W4A16, delete that export line

### Compile-cache layout

The 34 GiB boot disk can't absorb multi-GB compile caches, so they're split by
persistence:

| Path | Backing | Survives reboot? |
|---|---|---|
| `/root/.cache/vllm/torch_compile_cache`, `torch_aot_compile` | tmpfs, 16 GiB each (fstab) | no — first boot after every reboot recompiles for a few minutes |
| `/root/.triton` | disk | yes |
| `/root/.cache/flashinfer` | disk | yes — FlashInfer kernels JIT once, ever |

### Serving tiers (one at a time, port 8080)

| Tier | Weights | Drafter | Prefill @16K | Solo decode | KV pool |
|---|---|---|---|---|---|
| Fidelity | lued W8A16 (`INT8-W8A16-MTP`) | MTP n=3 | 1,882 tok/s | 76–110 tok/s | ~266K tok |
| Prefill | Freaksterz W8A8 (`SmoothQuant-W8A8-INT8`) | MTP n=3 | 2,773 tok/s | ~65–72 tok/s | ~266K tok |
| Concurrency | Frozenlock W4A8 (`int4-AutoRound`, int8 activations) | off | 3,301 tok/s | ~59 tok/s | ~483K tok @ 8 lanes |

Under 8 concurrent streams the concurrency tier holds ~43 tok/s per stream for a
342 tok/s aggregate. Prefix caching is patched and enabled on all tiers: on the
spec-off tier reuse is exact to the 16-token `--prefix-match-unit` (99.9% of a
repeated 12.5K prompt served from cache); with the drafter on, reuse snaps to
1600-token blocks (76.5%).

# Qwen3.8-27B INT8-W8A8 12 Minute Agentic Performance

Live opencode session: 4 subagents fanned out at once against the Prefill tier
(`run-Qwen3.8-27B-INT8-W8A8-MTP.sh`, `max-num-seqs 2`). Window 05:30:10 to
05:41:20, sampled every 10s.

- The 4-way fan-out hit the `max-num-seqs 2` cap. The engine held 2 running and queued the rest (`Waiting: 1`) from 05:30:50 to 05:37:50.
- Prefix cache carried the prefill load. Hit rate stayed 87.4% to 79.5% because every subagent and the main session share one system prompt and tool schema. Prefill spikes reached 3.5K to 8.2K tok/s on a 262K context.
- KV cache was the binding constraint. It peaked at 95.6% at 05:37:10 with two long-context streams live. A third stream would have hit eviction or OOM. The 2-seq cap kept the run safe.
- Chunked prefill stalls decode. The 1 to 2 tok/s dips (05:32:30, 05:33:20, 05:35:00, 05:37:00, 05:38:00) are running streams pausing while a new subagent prompt prefills in 4096-token chunks.
- Two-stream decode averaged 140 to 177 tok/s. Solo decode was 49 to 97 tok/s, in line with the 65 to 72 prefill-tier benchmark.
- MTP drafting added roughly 1.5x. Mean acceptance length was 3.0 to 3.8 of 4 tokens.
- The run ended clean. KV fell to 0% at 05:41:10 with no dropped requests.

| Time | Prefill (tok/s) | Decode (tok/s) | Run/Wait | KV cache (%) | dKV (pts) | Prefix hit (%) | MTP accept |
|---|---|---|---|---|---|---|---|
| 05:30:10 | 228.3 | 49.0 | 1/0 | 16.0 | n/a | 87.4 | 2.59 |
| 05:30:20 | 0.0 | 69.0 | 1/0 | 16.6 | +0.6 | 87.4 | 2.56 |
| 05:30:30 | 0.0 | 78.6 | 2/0 | 24.3 | +7.7 | 87.4 | 2.92 |
| 05:30:40 | 870.0 | 95.6 | 2/0 | 27.6 | +3.3 | 87.3 | 3.49 |
| 05:30:50 | 637.8 | 118.0 | 2/1 | 28.7 | +1.1 | 87.1 | 3.47 |
| 05:31:00 | 410.3 | 121.7 | 2/0 | 21.0 | -7.7 | 87.0 | 3.32 |
| 05:31:10 | 848.8 | 74.1 | 2/0 | 22.1 | +1.1 | 86.9 | 3.17 |
| 05:31:20 | 150.6 | 112.8 | 2/1 | 20.4 | -1.7 | 86.9 | 2.85 |
| 05:31:30 | 386.7 | 128.1 | 2/0 | 24.3 | +3.9 | 86.8 | 3.56 |
| 05:31:40 | 290.0 | 163.2 | 2/1 | 24.9 | +0.6 | 86.8 | 3.26 |
| 05:31:50 | 0.0 | 157.0 | 2/1 | 25.4 | +0.5 | 86.8 | 3.07 |
| 05:32:00 | 300.1 | 139.3 | 2/1 | 23.2 | -2.2 | 86.8 | 3.17 |
| 05:32:10 | 601.2 | 112.2 | 2/1 | 27.6 | +4.4 | 86.7 | 3.21 |
| 05:32:20 | 0.0 | 177.1 | 2/1 | 28.2 | +0.6 | 86.7 | 3.50 |
| 05:32:30 | 0.0 | 4.1 | 2/1 | 39.2 | +11.0 | 85.8 | 3.15 |
| 05:32:40 | 0.0 | 1.7 | 2/1 | 53.0 | +13.8 | 85.8 | 3.40 |
| 05:32:50 | 4351.4 | 64.6 | 2/1 | 58.6 | +5.6 | 85.8 | 3.15 |
| 05:33:00 | 0.0 | 170.3 | 2/0 | 29.3 | -29.3 | 85.8 | 3.79 |
| 05:33:10 | 540.7 | 48.2 | 2/0 | 49.2 | +19.9 | 85.3 | 3.65 |
| 05:33:20 | 0.0 | 1.5 | 2/1 | 63.0 | +13.8 | 85.3 | 3.00 |
| 05:33:30 | 3088.1 | 18.0 | 2/1 | 70.7 | +7.7 | 85.3 | 3.25 |
| 05:33:40 | 0.0 | 50.1 | 2/1 | 70.2 | -0.5 | 84.7 | 3.15 |
| 05:33:50 | 0.0 | 2.3 | 2/1 | 86.7 | +16.5 | 84.7 | 3.83 |
| 05:34:00 | 3366.5 | 105.3 | 2/1 | 89.0 | +2.3 | 84.7 | 3.01 |
| 05:34:10 | 0.0 | 140.2 | 2/1 | 89.0 | 0.0 | 84.7 | 3.12 |
| 05:34:20 | 0.0 | 64.1 | 2/1 | 59.1 | -29.9 | 84.2 | 3.62 |
| 05:34:30 | 2826.0 | 24.6 | 2/1 | 69.6 | +10.5 | 84.2 | 2.92 |
| 05:34:40 | 0.0 | 155.3 | 2/1 | 70.2 | +0.6 | 84.2 | 3.24 |
| 05:34:50 | 0.0 | 154.0 | 2/0 | 50.3 | -19.9 | 82.6 | 3.69 |
| 05:35:00 | 0.0 | 1.4 | 2/1 | 66.9 | +16.6 | 82.6 | 2.33 |
| 05:35:10 | 0.0 | 2.2 | 2/1 | 83.4 | +16.5 | 82.6 | 3.67 |
| 05:35:20 | 0.0 | 1.4 | 2/1 | 94.5 | +11.1 | 82.6 | 3.50 |
| 05:35:30 | 0.0 | 2.4 | 2/1 | 73.5 | -21.0 | 82.6 | 4.00 |
| 05:35:40 | 0.0 | 1.6 | 2/1 | 84.5 | +11.0 | 82.6 | 4.00 |
| 05:35:50 | 8234.3 | 66.5 | 2/1 | 87.3 | +2.8 | 82.6 | 3.19 |
| 05:36:00 | 0.0 | 155.2 | 2/1 | 87.8 | +0.5 | 82.6 | 3.48 |
| 05:36:10 | 0.0 | 162.7 | 2/1 | 88.4 | +0.6 | 82.6 | 3.65 |
| 05:36:20 | 0.0 | 35.5 | 2/0 | 70.2 | -18.2 | 82.0 | 3.58 |
| 05:36:30 | 0.0 | 2.2 | 2/1 | 85.6 | +15.4 | 82.0 | 3.67 |
| 05:36:40 | 3500.4 | 135.2 | 2/1 | 85.1 | -0.5 | 82.0 | 3.34 |
| 05:36:50 | 0.0 | 78.7 | 2/1 | 67.4 | -17.7 | 80.8 | 3.59 |
| 05:37:00 | 0.0 | 1.8 | 2/1 | 81.8 | +14.4 | 80.8 | 3.00 |
| 05:37:10 | 0.0 | 1.0 | 2/1 | 95.6 | +13.8 | 80.8 | 2.00 |
| 05:37:20 | 0.0 | 1.9 | 2/1 | 80.7 | -14.9 | 80.8 | 3.17 |
| 05:37:30 | 6561.0 | 34.6 | 2/1 | 88.4 | +7.7 | 80.8 | 2.97 |
| 05:37:40 | 0.0 | 137.6 | 2/1 | 89.0 | +0.6 | 80.8 | 3.19 |
| 05:37:50 | 0.0 | 119.4 | 2/0 | 50.8 | -38.2 | 80.2 | 3.32 |
| 05:38:00 | 0.0 | 1.8 | 2/0 | 67.4 | +16.6 | 80.2 | 3.00 |
| 05:38:10 | 3711.9 | 29.7 | 2/0 | 78.5 | +11.1 | 80.2 | 3.75 |
| 05:38:20 | 0.0 | 171.1 | 2/0 | 79.0 | +0.5 | 80.2 | 3.72 |
| 05:38:30 | 0.0 | 147.4 | 1/0 | 38.7 | -40.3 | 80.2 | 3.40 |
| 05:38:40 | 0.0 | 4.9 | 2/0 | 79.6 | +40.9 | 80.2 | 3.27 |
| 05:38:50 | 1536.3 | 108.0 | 2/0 | 80.1 | +0.5 | 80.2 | 3.24 |
| 05:39:00 | 0.0 | 151.0 | 2/0 | 81.2 | +1.1 | 80.2 | 3.36 |
| 05:39:10 | 0.0 | 155.5 | 2/0 | 81.2 | 0.0 | 80.2 | 3.44 |
| 05:39:20 | 0.0 | 151.2 | 2/0 | 84.0 | +2.8 | 80.2 | 3.36 |
| 05:39:30 | 0.0 | 141.1 | 2/0 | 82.3 | -1.7 | 80.2 | 3.15 |
| 05:39:40 | 0.0 | 139.9 | 2/0 | 82.9 | +0.6 | 80.2 | 3.11 |
| 05:39:50 | 0.0 | 143.7 | 2/0 | 83.4 | +0.5 | 80.2 | 3.21 |
| 05:40:00 | 0.0 | 150.8 | 2/0 | 84.0 | +0.6 | 80.2 | 3.38 |
| 05:40:10 | 0.0 | 97.7 | 1/0 | 43.6 | -40.4 | 80.2 | 3.22 |
| 05:40:20 | 0.0 | 78.5 | 1/0 | 44.2 | +0.6 | 80.2 | 3.18 |
| 05:40:30 | 0.0 | 52.6 | 1/0 | 11.0 | -33.2 | 79.5 | 3.33 |
| 05:40:40 | 0.0 | 0.0 | 1/0 | 27.6 | +16.6 | 79.5 | n/a |
| 05:40:50 | 3857.8 | 9.8 | 1/0 | 38.7 | +11.1 | 79.5 | 2.77 |
| 05:41:00 | 0.0 | 81.1 | 1/0 | 38.7 | 0.0 | 79.5 | 3.08 |
| 05:41:10 | 0.0 | 6.9 | 0/0 | 0.0 | -38.7 | 79.5 | 3.04 |
| 05:41:20 | 0.0 | 0.0 | 0/0 | 0.0 | 0.0 | 79.5 | n/a |

Raw log: [Qwen3.8-27B-Benchmarks/INT8-W8A8-12-Minute-Agentic-Session/vLLM-API-Server.log](Qwen3.8-27B-Benchmarks/INT8-W8A8-12-Minute-Agentic-Session/vLLM-API-Server.log)
