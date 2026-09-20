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

# Benchmarks will be added soon
