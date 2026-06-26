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
@reboot nvidia-smi -pm 1 && nvidia-smi -i 0 -pl 250 && nvidia-smi -i 1 -pl 170
```

| GPU | Power limit |
|-----|------------|
| RTX 3090 (CUDA 0) | 250W |
| RTX 5060 Ti (CUDA 1) | 170W |

---

## Models

Scripts and benchmark results live in subdirectories organized by GPU configuration.

### [`RTX3060Ti-8GiB-RTX5060Ti-16GiB/`](RTX3060Ti-8GiB-RTX5060Ti-16GiB/)

Scripts and benchmarks from the previous rig (RTX 3060 Ti 8 GiB + RTX 5060 Ti 16 GiB). The 3060 Ti has since been retired and replaced with an RTX 3090. These remain for reference — the tensor splits, KV cache choices, and MTP tuning notes still generalize.

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
| `<prefix>_prompt.json` | Prompt-processing benchmark (~4000 tokens in, 1 token out) |
| `<prefix>_vram-idle.txt` | `nvidia-smi` at idle |
| `<prefix>_vram-load.txt` | `nvidia-smi` during active generation |
| `<prefix>_draft-stats.txt` | MTP draft acceptance lines from server log |
| `<prefix>_decay.txt` | Throughput-over-time (`n_decoded` + `tg`) from server log |
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
