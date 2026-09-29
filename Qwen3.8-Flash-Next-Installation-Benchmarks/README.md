# Qwen3.8-Flash-Next on 2x RTX 3090 (Proxmox, 128 GB dual channel)

Serving [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (125B total with 6B active, plus a 51.2B parameter n-gram/PLE table and a 4B MTP draft on top of it, per the model card: the 125B is the transformer side, the PLE and MTP are additive) on two RTX 3090 24 GB cards, using the [albucino W4A16 + FP8 PLE checkpoint](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE) and the [DominikBucko pinned vLLM runtime](https://github.com/DominikBucko/qwen38-flash-next-2x3090). The full native 262,144 token context window works on this hardware, one request at a time, with the endpoint on port 8080 at the machine's LAN address.

From-scratch setup, including every roadblock and its fix: [GETTING_STARTED.md](GETTING_STARTED.md).

## Hardware

The base rig is described in the [upper level hardware section](https://github.com/sshJerry/llama-server-scripts/blob/master/README.md). This tier adds its own constraints and qualifications:

- **128 GB of RAM on dual channel.** This is the headline qualification. The [DominikBucko/albucino](https://github.com/DominikBucko/qwen38-flash-next-2x3090) reference machine is also 128 GB, but on an 8 channel DDR4-3200 platform, roughly 190 GB/s of bandwidth feeding the offloaded expert pool. This rig is 4x 32 GB DDR4-3600 on a consumer AM4 board, which is two channels, roughly 57 GB/s. Same GPU pair, same 24 GB per card. About a quarter of the CPU side bandwidth, and the serving numbers below pay for it.
- **Proxmox hypervisor.** The runtime serves from an unprivileged LXC (nesting on, unlimited memlock via `lxc.prlimit`). The host has 128 GiB installed (4x 32 GB), of which the OS sees 125 GiB, and the container is capped at about 118 GiB of that plus a 24 GiB swap allowance backed by host swap devices. The model's true residency is about 114 GiB (see the memory note below), which is why the reference spec asks for 128 GB (decimal, 119.2 GiB) and why the load spike leans on swap.
- **Swap on a SATA SSD**, not NVMe. For this workload swap is a load spike absorber, and the spike is mostly sequential and once per load, so SATA is adequate. Steady state paging measured at zero.
- **Model weights on NFS.** First load pulls about 129 GB over the network, 20 to 24 minutes on gigabit. After load, nothing re-reads the checkpoint while serving.
- Both cards power capped at 265 W.

## Measured performance (this rig)

All numbers client measured on the running endpoint, default profile (no P2P, see the contrast below). Method: TTFT from a streaming first token probe, decode from single stream fixed length runs cross checked against the engine's own status windows.

| Probe | Result |
|---|---|
| Prefill, ~16K prompt, cold expert cache | 1,595 tok/s (TTFT 10.0 s) |
| Prefill, ~128K prompt, warm expert cache | 1,807 tok/s (TTFT 72.5 s, ~131K tokens) |
| Prefill, full ~256K window, warm expert cache | 1,833 tok/s (TTFT 139.7 s) |
| Decode, single stream, cold 256 token run | 31.2 tok/s |
| Decode, single stream, warm 256 token run | 39.4 tok/s |
| Decode, single stream, 2048 token run | 52.3 tok/s steady, engine windows 50 to 61 |
| MTP draft acceptance (sampled, temp 1.0) | 41 to 44%, mean accepted length ~2.2 |
| Prefix cache hit rate, real agent sessions | 7.6% to 21.6% cumulative in the first session, 47.6% peak in a later one, an 18.7% trough while the zero-hit probe sequence dilutes the cumulative counter, recovery to 44.1% by the end of the captured tail, 107,200 tokens served from cache at the captured differential |
| First load over NFS | ~24 min, ~22 GiB spills to swap during the spike, drains to ~0 after |
| KV cache | 276,313 tokens, 1.05x the 262,144 window (one full context request) |
| GPU memory at idle | 23.7 of 24.0 GiB per card (weights ~7.5, hot-84 expert cache ~9.7, KV 4.1, pinned and workspace the rest) |

Zero swap activity (vmstat si/so) was observed through the full window prefill and the decode runs. The working set stays resident. Decode across the captured agent traffic runs in the 46 to 80 tok/s band, consistent with the 52.3 tok/s single stream bench figure.

Warmup is real on this tier: the only probe on a cold expert cache was also the first request of the boot (1,595 tok/s at 16K), and every later probe ran faster per token at both lengths, 1,807 at ~128K and 1,833 across the 256K window. That is the cache warming, not prefill speeding up with context, prefill is flat with length (the contrast section separates the two variables). The dynamic expert cache follows the sequence, which matches the reference project's own hillclimb finding.

## Against the reference numbers

The reference machine has the same GPU pair with 8 channel RAM and working bidirectional CUDA P2P, and it runs the fast 256K profile (which requires P2P). Numbers from the [DominikBucko/albucino](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE) release (same person behind both the GitHub runtime and the HF checkpoint):

| Metric | Reference (fast 256K profile, P2P, 8 channel) | This rig (default profile) |
|---|---|---|
| Prefill, ~128K prompt | 2,752 tok/s, TTFT 47.6 s | 1,807 tok/s, TTFT 72.5 s |
| Prefill, 256K window | 2,654 tok/s, TTFT 98.0 s | 1,833 tok/s, TTFT 139.7 s |
| Decode | 104.5 tok/s at 128K, 103.1 at 256K | 52.3 tok/s steady |
| MTP acceptance | 86 to 91% (greedy probes) | 41 to 44% (sampled at temp 1.0) |

Both prefill pairs are like-for-like now: 66% at ~128K (1,807 tok/s against 2,752) and 69% at 256K (1,833 against 2,654). Prefill is flat with context on both machines, the reference falls 3.6% from 128K to 256K (2,752 down to 2,654) while this rig moves up 1.4% (1,807 to 1,833), so the only figure that does not travel between rigs is the 16K probe (1,595 tok/s), which was the first request on a cold expert cache. The 128K number here was taken on a warm engine, the expert cache already populated by earlier requests plus the bench's own warmup pass, which matches the reference's best-of-3 release-image protocol.

So this rig holds about 66 to 69% of reference prefill and about 50% of reference decode. The gap is hardware bound in three ways:

1. **No P2P.** The fast 256K profile requires bidirectional CUDA peer to peer. A stock driver on a PCIe through the host bridge topology does not have it, so this rig runs the default profile, and every draft verify step's all reduce round trips through host memory.
2. **Dual channel RAM.** Expert misses stream from the host pool at roughly a quarter of the reference bandwidth. Prefill holds up better (it is staged DMA, cold experts copied to the GPU ahead of the prefill chunks, plus GPU compute). Decode pays the bandwidth on every step.
3. **Sampled acceptance.** The reference acceptance numbers are greedy probes. Serving at the checkpoint's default temp 1.0 lowers draft acceptance, 41 to 44% sampled here against the reference's 86 to 91% greedy, so each verify step advances about 2.2 tokens on this rig. Acceptance is also content dependent, so real workload numbers can land above the nonsense text used for the probe.

To sum it up, a 125B class model with the full 262K window runs on this hardware at usable speed, roughly half the reference clock. Whether that trade is worth it versus a faster 27B class tier depends on the workload.

## Correctness: the vllm#48375 bug and the patch

The runtime's pinned vLLM build ships with prefix caching enabled by default, and it carries [vllm#48375](https://github.com/vllm-project/vllm/pull/48375), an open, unmerged bug: `MambaManager.find_longest_cache_hit` takes the `drop_eagle_block` flag and never reads it. On hybrid GDN models running MTP with prefix caching in align mode, the final matched mamba block can hold a recurrent state snapshot written over draft tokens that verification later rejected. That stale snapshot is restored on a cache hit and **silently corrupts output**: tool calls go wrong, no error, no log signal, and the poisoned block keeps serving later requests sharing the prefix. Upstream [#43559](https://github.com/vllm-project/vllm/issues/43559) measured tool accuracy dropping from about 90% to about 50%.

This applies to Qwen3.8-Flash-Next specifically, not only to the model the PR was written against. The PR thread carries two independent Flash-Next confirmations, including a production PP4 + MTP4 + align deployment running a variant of the fix, and a 13.5 hour soak that ran clean only with the patch applied. One reporter's summary: the fix is a dependency for a working Flash-Next spec decode + prefix caching path.

The bug has been verified in this build's container (the flag appears in the method signature and never in the body), the image has been patched, and the fix is confirmed in the running container at the source level: the flag is now honored in the body. In production traffic the cache works and the patch is engaged on every hit: a real multi turn agent session drove the cumulative prefix cache hit rate from 7.6% up to 21.6%, a later session (hitting blocks the first one cached twenty minutes earlier) reached 47.6% peak, then the zero-hit probe sequence dilutes the cumulative counter down to an 18.7% trough before later sessions lift it back to 44.1% at the end of the capture, 107,200 tokens served from cache at the captured differential, and every one of those hits passes through the patched path, dropping the final matched mamba block instead of restoring a snapshot written over rejected draft tokens. Under the unpatched build, those same hits would have been the corruption trigger. One caveat, stated honestly: a synthetic identical-repeat probe (the same 12.5K prompt sent twice back to back) measures zero hits on this runtime, so the probe's fixed-versus-broken band verdicts do not travel to this tier. Verify the patch with the source level check, and judge cache health from the hit rate lines in the server log (captured in [extra/vLLM-Server.log](extra/vLLM-Server.log)). The patch is [`pr48375_mamba_drop_eagle_block.py`](pr48375_mamba_drop_eagle_block.py) in this directory, wired into the image through two lines in `docker/Dockerfile` plus one line in `.dockerignore`. The full procedure, including how to check any build for the bug, is in [GETTING_STARTED.md, the vllm#48375 patch](GETTING_STARTED.md).

## A memory note that changes how you read the gauges

About 58 GiB of the model's residency lives in NVIDIA driver UVM pages backing the offloaded expert pool. No container level counter sees them, and they are charged to the root cgroup on the host, not the container's. The container's `free` and `docker stats` report about 56 GiB, which reads like a huge amount of headroom, but the true residency is about 114 GiB.

The arithmetic, for anyone reconciling it: about 48.5 GiB of PLE table in the offload worker (process memory, container visible), about 58 GiB of expert pool in driver pages (container invisible), and about 8 GiB of process overhead. The GPU side holds about 15 GiB of dense weights plus a 19.5 GiB hot-84 expert cache, and that cache is a copy of the hot experts, the host pool keeps the complete set, so nothing is subtracted from the host side for it. The on-disk checkpoint, about 120 GiB, splits the same way: PLE and experts land on the host, dense weights and the MTP draft land on the GPUs. The parameter arithmetic closes the same way: about 121B of the model's parameters are routed experts (512 experts x 48 layers x 3 matrices x 2560 x 640), which at 4 bits plus group scales is the ~58 GiB host pool, and the model card's 125B total is the transformer side with the 51.2B PLE table and the 4B MTP draft on top of it.

So the host has 128 GiB installed and 125 GiB visible, and the load spike (loader temporaries on top of residency, plus host level reclaim under the pressure those driver pages create) lands against that, not against the container's 118 GiB. The reference spec's 128 GB is decimal, 119.2 GiB, which is exactly what the preflight gate encodes. For this tier, the host's `free` is the capacity gauge, not the container's. The full walk, including why the load never OOMs at the container's own limit, is in [GETTING_STARTED.md](GETTING_STARTED.md).

## A real agent session, in the logs

[extra/vLLM-Server-2.log](extra/vLLM-Server-2.log) captures a friend's agentic coding session against the endpoint, a single-file HTML canvas simulation generated and then iterated on repeatedly. The capture opens mid-session with the cumulative prefix hit rate already at 86.9%, and it only erodes to 78.4% across the following fifty minutes. That is the strongest prefix reuse this tier has shown, well above the 47.6% peak of the earlier capture, and it is what a loop that resends a long shared context looks like when the cache serves nearly all of it back.

Three behaviors worth reading off the numbers. First, the one-request design in the wild: `Waiting: 1 reqs` sits behind the running request for most of the session, follow-up turns queueing politely behind each long generation, exactly as `max_num_seqs` 1 intends. Second, the cached-burst reading gets its evidence: prompt throughput spikes of roughly 2,000 to 5,714 tok/s recur through the capture, each one a re-queried shared prefix landing almost entirely on cache, each nudging the cumulative hit rate down by only 0.1 to 1.0 points. The first capture has bursts of the same shape, and step 9 of GETTING_STARTED attributes those to the ~131K bench prefills. Here the mechanism is the cache-side form of the same accounting, bursts riding on eighty percent reuse, re-queried prefixes served from cache rather than recomputed. Third, acceptance tracks content as the caveat says: across 273 spec decoding windows the mean draft acceptance rate is 59.8% and the mean accepted length 2.8, sitting between the 41 to 44% nonsense-text probe and the reference's greedy 86 to 91%. Code generation is easier to draft than gibberish, which is the expected direction.

Throughput held up the whole session. Decode stayed between roughly 47 and 82 tok/s, mostly 55 to 65, and KV usage peaked at 38%, a ~105K token context on a 276K pool, unstressed. The last clean stretch, 01:49 to 01:57, is about 27K tokens of uninterrupted decode at ~57 tok/s, the final full HTML file written out in one pass, roughly eight minutes for a single response. The practical read for interactive users: iteration is slow per turn at these decode rates, but retries are nearly free because the shared context is cache-served.

## Operational notes

- **One active request** is the design point. `max_num_seqs` is 1 and the KV reservation covers 1.05 full context requests.
- **Text only by default.** The base model is a VL, but this runtime ships with multimodal limits set to 0.
- **Tool parsing is the `qwen3_coder` flavor**, not `qwen3_xml` used on the 27B tiers in this repo.
- **Prefix reuse works on real traffic.** Multi turn agent sessions hit the cache hard: 7.6% to 21.6% cumulative within one captured session, up to 47.6% peak in a later one that reused blocks the first session cached twenty minutes earlier, an 18.7% trough while the zero-hit probe sequence dilutes the running average, then recovery to 44.1% by the end of the captured tail. One quirk: a synthetic identical-repeat probe reads zero on this runtime, so judge cache health from the engine log's hit rate lines, not from that probe. The two ~10K tok/s prompt bursts in the capture are the two ~131K bench prefills landing as interval-metric artifacts, not prefill speed, step 9 of GETTING_STARTED and the notes in the extra README have the accounting.
- Launcher: [`run-Qwen-Flash-Next.sh`](run-Qwen-Flash-Next.sh), foreground until Ctrl+C, cleanup trap returns both GPUs to base state.

## Credits

Thanks to DominikBucko/albucino for the startup script making things streamlined and catered to 2x 3090 owners.

https://github.com/DominikBucko/qwen38-flash-next-2x3090/tree/main

https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE

Thank you vLLM for the inference engine.

https://github.com/vllm-project/vllm

Thank you Paul Otto (@potto007) for the work on the prefix caching corruption fix.

https://github.com/potto007

Thanks to @noonghunna of club-3090 for the discussion posts and constant work updating curated slugs for 3090 owners.

https://github.com/noonghunna

https://github.com/noonghunna/club-3090
