# Getting started: Qwen3.8-Flash-Next on 2x RTX 3090 under Proxmox

Everything from a bare setup to a vLLM endpoint on port 8080 that other machines on the LAN can hit, written after doing it once. This includes every roadblock hit along the way and how each one was resolved. Performance figures and the comparison against the reference machine live in the [README](README.md).

Starting point assumed here: a Proxmox VE host with an unprivileged LXC container that already has both GPUs passed through and working (`nvidia-smi` runs inside the LXC), the NVIDIA driver and a CUDA toolkit installed inside the LXC, and NFS or local storage for model weights. Nothing else is assumed. If you do not have the LXC GPU passthrough yet, sort that first, it is its own project.

Two files ship next to this guide:

- [`run-Qwen-Flash-Next.sh`](run-Qwen-Flash-Next.sh), the foreground launcher. Copy it into the LXC, for example to `/root/`.
- [`pr48375_mamba_drop_eagle_block.py`](pr48375_mamba_drop_eagle_block.py), the prefix caching correctness patch. It gets wired into the image build in step 6.

Order of operations:

1. [Proxmox host: memory, swap, memlock, disk](#1-proxmox-host-memory-swap-memlock-disk)
2. [LXC: Docker and the NVIDIA toolkit](#2-lxc-docker-and-the-nvidia-toolkit)
3. [Checkpoint download and verification](#3-checkpoint-download-and-verification)
4. [The runtime repo and .env](#4-the-runtime-repo-and-env)
5. [Serve on port 8080 from the LAN address](#5-serve-on-port-8080-from-the-lan-address)
6. [The vllm#48375 patch](#6-the-vllm48375-patch)
7. [Build the image](#7-build-the-image)
8. [First boot and what to expect](#8-first-boot-and-what-to-expect)
9. [Verify](#9-verify)
10. [Day two operations](#10-day-two-operations)
11. [Roadblocks quick reference](#11-roadblocks-quick-reference)

## 1) Proxmox host: memory, swap, memlock, disk

All of these run on the Proxmox host as root. Replace `<vmid>` with your container's ID.

**Container memory and swap allowance.** The runtime wants approximately 128 GB of RAM (decimal, 119.2 GiB). This rig ran it at ~118 GiB visible to the container plus a 24 GiB swap allowance, which works, with the load spike (more below) absorbed by swap. Proxmox containers do not have their own swap devices, they use host swap up to an allowance, and `lxcfs` presents that allowance as `SwapTotal` inside the container. So the memory work happens in two places: raise the allowance in the container config, and make sure the host actually has enough swap behind it.

```bash
pct set <vmid> --memory 121000 --swap 24576
```

**Host swap.** The weight load spikes past physical RAM by about 22 GiB on this hardware (measured once; this is documented behavior, not a malfunction: "the loader can exceed physical RAM even though steady serving fits much more comfortably"). Steady state touches no swap at all. Two knobs have to agree: the host needs the swap devices, and the container needs the swap allowance, which is the hard cap on how much of it the container may use. Host swap beyond the allowance is unusable by the container. If your volume group has free extents, a thick LV outside the thin pool is the clean way to add swap:

```bash
lvcreate -l 100%FREE -n flashswap pve
mkswap /dev/pve/flashswap && swapon /dev/pve/flashswap
echo '/dev/pve/flashswap none swap sw 0 0' >> /etc/fstab
```

Never put swap on a thin-provisioned LV (`local-lvm` is a thin pool). A regular LV outside the pool, or a file on an ext4 filesystem, is fine. NVMe is not required here. The spike is mostly sequential and happens once per load, and a SATA SSD absorbs it fine. Measured the full spike draining back in about a minute with zero steady state paging afterward.

**Unlimited memlock.** This caused a failed launch (roadblock R2 in the quick reference). The docker launch requests `--ulimit memlock=-1` for its pinned memory expert pool, and raising the hard memlock limit requires a capability that unprivileged LXC root does not have. The fix is on the host: give the container's init process an unlimited memlock hard limit at spawn time.

```bash
echo 'lxc.prlimit.memlock: unlimited' >> /etc/pve/lxc/<vmid>.conf
pct reboot <vmid>
```

After the reboot, verify inside the LXC with `ulimit -Hl`, it should print `unlimited`.

**Container root disk.** The pinned vLLM docker image plus runtime caches needs room. A 34 GB root disk is too small (roadblock territory), 60 GB is comfortable:

```bash
pct resize <vmid> rootfs 60G
```

This is online and instant, the filesystem grows with it.

## 2) LXC: Docker and the NVIDIA toolkit

Inside the LXC, the container must have `features: nesting=1` in its config for Docker to run. With that in place:

```bash
apt-get update && apt-get install -y docker.io

apt-get install -y nvidia-container-toolkit || {
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/debian/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update && apt-get install -y nvidia-container-toolkit
}

nvidia-ctk runtime configure --runtime=docker
nvidia-ctk config --in-place --set nvidia-container-cli.no-cgroups=true
systemctl restart docker
```

The `no-cgroups` line matters inside an LXC: the container cannot delegate cgroup device rules, so the toolkit must not try. This is the standard fix for the "error setting rlimit type 8" and friends class of failures.

Smoke test, this must print both 3090s from inside a container:

```bash
docker run --rm --gpus all debian:trixie nvidia-smi
```

## 3) Checkpoint download and verification

The checkpoint is the pinned revision, ~120 GiB total:

```bash
hf download albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE \
  --revision ef554143369a706525336f6b42a09094835dc077 \
  --local-dir /models/Models/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE

cd /models/Models/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE
sha256sum -c --quiet SHA256SUMS && echo "ALL FILES OK"
```

Two things that are not problems, so they do not waste your time:

- **Shard numbers 00002 and 00016 are intentionally absent.** The layout is 15 target shards plus 10 `model-plefp8-*` shards, 25 safetensors files total. The hybrid build replaced the BF16 n-gram shards of the source layout with the FP8 PLE table files. A missing 02 and 16 is correct, not a failed download.
- **`./README.md: FAILED` from the checksum run is expected.** The model card was updated after the sums file was written. Everything else should verify OK. If a weight shard fails, re-download that shard.

## 4) The runtime repo and .env

```bash
git clone https://github.com/DominikBucko/qwen38-flash-next-2x3090.git /opt/flashnext
cd /opt/flashnext
cp .env.example .env
```

Set these in `.env`:

| Key | Value | Why                                                                                                                                                                                                                                                                                                                                                 |
|---|---|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `MODEL_DIR` | your download path from step 3 | where the weights live                                                                                                                                                                                                                                                                                                                              |
| `PORT` | `8080` | the example ships `PORT=8000` on line 3. **Edit that line, do not append a second PORT line.** I appended by mistake at first and removed the duplicate before it ever served. Make resolves an included .env file last line wins, but nothing guarantees every reader of the file does, so keep it to one line. See step 5 for how the port flows. |
| `MAX_MODEL_LEN` | `262144` | the native window. Context is nearly free on this architecture: KV costs about 13.5 KB per token plus ~0.46 GiB fixed, so the whole 262K window fits in the default KV reservation.                                                                                                                                                                 |
| `KV_CACHE_MEMORY_BYTES` | `4429185024` | the default, sized for the full window. This reservation is GPU memory, not host RAM. Do not shrink it for RAM reasons, that was a roadblock R3.                                                                                                                                                                                                    |
| `VLLM_WNA16_STATIC_HOT_CACHE_SIZE` | leave at `84` | the GPU resident expert cache                                                                                                                                                                                                                                                                                                                       |
| `MAX_NUM_BATCHED_TOKENS` | leave at `4096` | default                                                                                                                                                                                                                                                                                                                                             |

Do **not** append `configs/fast-256k.env`. That profile requires working bidirectional CUDA peer to peer between the cards, which a stock driver on a PCIe topology does not have. The default profile is what this guide runs.

## 5) Serve on port 8080 from the LAN address

Out of the box the endpoint is `127.0.0.1:8000`, localhost only. Two one-time edits open it to the LAN on 8080.

The launch script builds everything from one variable (check `grep -n '[Pp]ort' scripts/docker_serve.sh` to see all three sites):

- line 6: `port=${PORT:-8000}`, reads the port from the environment, which the Makefile exports from `.env`
- line 20: `-e "PORT=$port"`, injects it into the container so vLLM inside listens on the same port
- line 80: `-p "0.0.0.0:$port:$port"` **after** the edit below, publishes it on all interfaces

So with `PORT=8080` in `.env` (step 4), the only file edit left is the bind address:

```bash
cd /opt/flashnext
grep -n '127.0.0.1' scripts/docker_serve.sh          # expect the -p line: 127.0.0.1:$port:$port
sed -i 's|127\.0\.0\.1|0.0.0.0|g' scripts/docker_serve.sh
grep -n '0\.0\.0\.0' scripts/docker_serve.sh         # expect -p "0.0.0.0:$port:$port"
```

The sed is global. At this revision the file holds exactly one `127.0.0.1`, the `-p` publish line, which the first grep confirms. If your checkout shows any other `127.0.0.1` (a healthcheck, a bind address), edit only the `-p` line.

Result: the container publishes `0.0.0.0:8080` and vLLM inside listens on 8080. The endpoint becomes `http://<LXC_IP>:8080/v1` from any machine on the LAN.

You can confirm the mapping seconds after a launch, before the 20 minute load finishes:

```bash
docker ps --format '{{.Ports}}'      # expect 0.0.0.0:8080->8080/tcp
```

## 6) The vllm#48375 patch

**The bug.** [vllm#48375](https://github.com/vllm-project/vllm/pull/48375) (open, unmerged) fixes a silent correctness bug in `MambaManager.find_longest_cache_hit`: the method takes a `drop_eagle_block` flag and never reads it. On a hybrid GDN model running MTP speculative decoding with prefix caching in align mode, the final matched mamba block can hold a recurrent state snapshot that was written over draft tokens that verification later rejected. On a cache hit that stale snapshot is restored and reused, and the output goes wrong with **no error and no log signal**, and the poisoned block keeps serving later requests that share the prefix. Upstream issue [#43559](https://github.com/vllm-project/vllm/issues/43559) measured tool accuracy dropping from about 90% to about 50%.

**This applies to Qwen3.8-Flash-Next specifically, not just to the Qwen3.6 model the PR was written against.** The PR thread contains two independent confirmations on Flash-Next hardware and configs: one 13.5 hour soak that ran clean with the patch after the unpatched build crashed with an Xid 31 in the align path within 20 minutes, and one production deployment (PP4 + MTP4 + align mode) running a variant of the fix. One of those reporters states it plainly: the fix is a dependency for a working Flash-Next spec-decode + prefix-caching path.

**Verify your build actually has the bug before patching** (mine did, the pinned base is a recent dev build that predates any fix landing upstream):

```bash
docker exec -i qwen38-flash-next python3 - <<'PY'
import inspect
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
print("params:", list(inspect.signature(MambaManager.find_longest_cache_hit).parameters))
for i, l in enumerate(inspect.getsource(MambaManager.find_longest_cache_hit).splitlines()):
    if "drop_eagle_block" in l:
        print(f"{i:4d}: {l}")
PY
```

If the only hit is the signature line, the body ignores the flag and the bug is present. The rest of this section patches it into the image build so it survives every rebuild.

**The patch script.** It lowers `max_length` by one mamba block when `drop_eagle_block` is set, which bounds both the coarse block search and the fine-grained partial-unit search (the production variant reported in the PR thread). It locates the method by introspection rather than by a text anchor, because the pinned build's file contains a second `MambaSpec` assert elsewhere and a plain text anchor refuses on that (that was roadblock R7, first version did exactly that).

```bash
mkdir -p /opt/flashnext/runtime/patches
cat >/opt/flashnext/runtime/patches/pr48375_mamba_drop_eagle_block.py <<'PY'
#!/usr/bin/env python3
"""[pr48375] Honor drop_eagle_block in MambaManager.find_longest_cache_hit.

Upstream: https://github.com/vllm-project/vllm/pull/48375 (open, unmerged).
On hybrid GDN models with MTP + prefix caching (align mode), the final matched
mamba block may hold a recurrent-state snapshot taken over draft tokens that
verification later rejected; restoring it silently corrupts output (vllm#43559).
Fix: lower `max_length` by one mamba block when drop_eagle_block is set. This
bounds both the coarse block search and the fine-grained partial-unit search
(the production variant reported on Qwen3.8-Flash-Next PP4+MTP4 in the PR
thread). Locates the method via introspection (the MambaSpec assert is not
unique in the pinned build's file), so the patch lands in exactly the method
verified buggy. Idempotent; refuses loudly on drift.
"""
import inspect
import sys

from vllm.v1.core.single_type_kv_cache_manager import MambaManager

TARGET = inspect.getsourcefile(MambaManager)
MARK = "[pr48375]"
ANCHOR = "        block_size = kv_cache_spec.block_size\n"

src = open(TARGET).read()
if MARK in src:
    print("[pr48375] already applied - no-op")
    sys.exit(0)

method = inspect.getsource(MambaManager.find_longest_cache_hit)
if method.count(ANCHOR) != 1:
    sys.exit(f"[pr48375] REFUSE: anchor not unique inside the method: {method.count(ANCHOR)}")

patch = (
    "        if drop_eagle_block:\n"
    "            # [pr48375] the final matched block may hold a recurrent-state\n"
    "            # snapshot taken over draft tokens later rejected by verification;\n"
    "            # lower the search ceiling by one mamba block so it is recomputed\n"
    "            # rather than restored (bounds the coarse and partial-unit paths).\n"
    "            max_length = max(0, max_length - block_size)\n"
)
patched = method.replace(ANCHOR, ANCHOR + patch, 1)

if src.count(method) != 1:
    sys.exit(f"[pr48375] REFUSE: method source not unique in file: {src.count(method)}")

open(TARGET, "w").write(src.replace(method, patched, 1))
print("[pr48375] applied: MambaManager now honors drop_eagle_block")
PY
```

The same script ships as [`pr48375_mamba_drop_eagle_block.py`](pr48375_mamba_drop_eagle_block.py) next to this guide, so you can copy the file instead of pasting.

**Wire it into the image.** Open `/opt/flashnext/docker/Dockerfile` and add these two lines immediately after the `RUN python3 /opt/qwen38/runtime/install_overlay.py ...` line:

```dockerfile
COPY runtime/patches/ /opt/qwen38/runtime/patches/
RUN python3 /opt/qwen38/runtime/patches/pr48375_mamba_drop_eagle_block.py
```

Everything before that line stays cached, so the rebuild costs seconds.

**The .dockerignore.** Slight roadblock with this one (roadblock R6). The repo's `.dockerignore` is an allowlist: `**` excludes everything, then `!` lines re-include specific files. A new directory is not on the list, so it never enters the build context and the `COPY` fails with a confusing `"/runtime/patches": not found`. One line fixes it:

```bash
echo '!runtime/patches/**' >> /opt/flashnext/.dockerignore
```

## 7) Build the image

```bash
cd /opt/flashnext
python3 scripts/validate_repo.py     # sanity, it should pass with the patch file counted
make build-image
```

Watch the `RUN` step for the patch output. `[pr48375] applied: MambaManager now honors drop_eagle_block` means it landed. A `REFUSE` line means anchor drift, paste it somewhere and adjust, do not force past it.

`make build-image` does not run the preflight, that happens on the first serve, in the next step.

## 8) First boot and what to expect

Copy the launcher into the LXC (for example `/root/run-Qwen-Flash-Next.sh`, see [day two](#10-day-two-operations)) or run `make serve` directly the first time. Either way, the preflight gate runs first.

**The preflight gate (roadblock R1).** `make serve` runs `make preflight` before anything loads, and the preflight script errors unless `MemTotal` is at least 125000000 kB inside the container. That number is exactly 128 GB decimal, 119.2 GiB, and a container at ~118 GiB fails the gate even though the model runs fine on this much with the swap behind it. Lower the threshold, a documented deviation that has run stable here since:

```bash
sed -i 's|125000000|120000000|' scripts/preflight.sh
```

Restore `125000000` if this ever moves to a host with the full amount in the container.

Then the load itself:

- **The load takes about 20 to 24 minutes.** ~129 GB over NFS into RAM, with the PLE table loading in parallel over 25 shards. On gigabit LAN the network is the ceiling, so a warm server side cache does not speed it up much. A faster LAN link is the upgrade that shortens this.
- **Swap fills by ~22 GiB during the load, then drains to ~0.** That is the documented load spike, not a leak. The runtime's own numbers say the same thing.
- **Nothing re-reads the checkpoint while serving.** After load, the PLE table and expert pool sit in memory and the only disk traffic is the runtime's kernel caches.

Watch it from a second shell if you want to see the spike happen:

```bash
watch -n 5 'free -h; swapon --show'
vmstat 1 20          # si/so columns: nonzero during the spike, ~0 after
```

## 9) Verify

Three checks, in order of how much they prove:

**The endpoint, from another machine:**

```bash
docker ps --format '{{.Ports}}'                          # 0.0.0.0:8080->8080/tcp
curl -s http://127.0.0.1:8080/v1/models | head -c 200    # from the LXC
curl -s http://<LXC_IP>:8080/v1/models | head -c 200     # from any LAN machine, the goal
```

**The patch, in the running container** (re-run the introspection from step 6, it should now show `if drop_eagle_block:` plus `max_length = max(0, max_length - block_size)` in the body, not just the signature).

**The cache hit behavior.** A fixed build holds back one mamba block on a repeat request. A broken build returns a full length hit. The block size is 1600 tokens on this runtime, and that is measured, not assumed: the boot log prints "Setting attention block size to 1600 tokens" (the mamba page size in align mode), and the 27B tiers in this repo measured the same page size (9,600 of 12,552 tokens cached, which is (floor(N/1600) - 1) x 1600). It is runtime derived, so if a future build logs a different number, re-derive the bands in the verdict line below from it. This is the quick empirical test from the PR thread itself, with one caveat covered below the probe: on this runtime the probe reads zero, so the bands apply to the 27B tiers and the source level check carries the verification here:

```bash
cat >/root/qfn-prefix-probe.py <<'PY'
#!/usr/bin/env python3
import json, urllib.request, random, string
def counters():
    q = h = None
    with urllib.request.urlopen("http://127.0.0.1:8080/metrics") as r:
        for l in r.read().decode().splitlines():
            if l.startswith("vllm:prefix_cache_queries_total"):
                q = float(l.rsplit(" ", 1)[1])
            elif l.startswith("vllm:prefix_cache_hits_total"):
                h = float(l.rsplit(" ", 1)[1])
    return q, h
def ask(p):
    b = json.dumps({"model": "Qwen3.8-Flash-Next", "max_tokens": 32, "stream": False,
                    "messages": [{"role": "user", "content": p}]}).encode()
    urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions", data=b,
        headers={"Content-Type": "application/json"}), timeout=600).read()
para = ("The quick brown fox jumps over the lazy dog while the sun sets behind "
        "the hills and the river carries leaves toward the bridge. ")
p = "".join(random.choices(string.ascii_letters, k=32)) + "\n" + para * 500   # ~12.5K, fixed text
q0, h0 = counters(); ask(p); q1, h1 = counters(); ask(p); q2, h2 = counters()
print(f"first send:  queries +{q1-q0:.0f}, hits +{h1-h0:.0f} (expect ~0 hits)")
print(f"repeat send: queries +{q2-q1:.0f}, hits +{h2-h1:.0f}")
print(f"counters now: queries {q2:.0f}, hits {h2:.0f}")
print("verdict: ~9.6K hits = one 1600-token block held back (FIXED) | ~11.2K hits = full length (BROKEN)")
print("         ~0 hits with queries advancing = the caveat below (observed on this runtime)")
PY
python3 /root/qfn-prefix-probe.py
```

The probe reads zero on this runtime, and the server log proves that is a property of the probe, not of the cache. The differential settles the counter question: the queries counter advances by the prompt size on every probe send while hits stays flat, and yet that same hits counter held 107,200 tokens accumulated from real traffic before the probe ever ran. The log tells the real story, and it is the opposite of the probe: during real multi turn agent sessions the prefix cache hit rate climbs turn over turn, 7.6% to 11.1% to 21.6% cumulative in one session, a later session that reused blocks the first one cached twenty minutes earlier peaked at 47.6%. The tail is not a clean climb though, check the capture ([extra/vLLM-Server.log](extra/vLLM-Server.log)): the zero-hit queries of the probe sequence dilute the cumulative counter from the 47.6% peak to an 18.7% trough, then the following sessions lift it back through 30.0% to 44.1% at the end of the capture. That shape is what a cumulative ratio does under miss-only queries, so read the recovery, not just the peak. On the cache heavy turns the engine's prompt throughput drops to a few hundred tok/s because the rest of the prompt came from cache.

One more entry in the capture needs a note so it cannot be misread as prefill speed. Two windows at 23:19 and 23:20 print prompt throughput near 10,240 tok/s with generation flat at 0.5, and the capture also holds single-window bursts of 3,340 and 2,192 tok/s at 23:41 and 23:42. Nothing on this hardware recomputes at those rates, and the windows are isolated with 0.0 on either side. The attribution on the two big ones is settled: they are the two ~131K bench prefills, warmup pass and measured run, and the engine's interval metric does not reconcile to wall time on them, the excerpt is also trimmed through the middle of the bench. The accounting is in the log notes of the extra [README](extra/README.md). The cache-side form of the same pattern is in a second capture, [extra/vLLM-Server-2.log](extra/vLLM-Server-2.log), a fifty minute agent session running at roughly eighty percent prefix reuse where prompt spikes of 2,000 to 5,714 tok/s are re-queried shared prefixes landing on cache and move the cumulative rate by around a point each. The agent session section of the [README](README.md) breaks that one down. None of it moves the performance tables. Those are client measured, 72.5 s TTFT and 1,807 tok/s for exactly these prefill requests, and the interval lines in this file should be read as traffic evidence, not as speed.

So the reading on this tier:

- The cache works, and the patch is engaged on every hit. Each hit passes through the patched path and drops the final matched mamba block, exactly the block the bug would have restored. Under the unpatched build, the hits that real sessions take would have been the corruption trigger.
- The identical back to back repeat reads zero here for reasons unconfirmed. The PR thread carries related draft work on state block hits behind commit fences (#56524, #56525), which is the right neighborhood, but nothing is proven. Do not use the probe bands to judge this runtime. Judge cache health from the engine log's hit rate lines during real multi turn traffic, and judge the patch from the source level check in step 6.

The script prints both counter deltas and the running totals, so one run answers the counter question by itself. Queries advancing with hits flat on the repeat, while the hits total is nonzero from earlier real traffic, is the observed signature on this rig: the probe measures no reuse, the cache itself is alive and serving.

For scale on what to expect from this hardware: prefill around 1600 tok/s at 16K on a cold expert cache, 1807 tok/s at ~128K on a warm one (TTFT 72.5 s for ~131K tokens), and around 1830 tok/s across a full 256K window, decode around 52 tok/s single stream. The [README](README.md) has the full table and the comparison against the reference machine.

## 10) Day two operations

**The launcher.** [`run-Qwen-Flash-Next.sh`](run-Qwen-Flash-Next.sh) is the day two entry point. Copy it into the LXC, `chmod +x`, and run it from an interactive shell. It serves in the foreground until Ctrl+C, then stops and removes the container and both GPUs return to base state (~1 MiB each). It also refuses to start if something else holds the GPUs, if the container is already running, or if the port edits from steps 4 and 5 are missing.

One behavior worth knowing (roadblock R4): `make serve` runs the container attached, and the container is `--rm`. Closing the terminal or the SSH session that launched it kills the container and the `--rm` removes it, so it looks like it "terminated by itself and vanished". The launcher fixes the workflow part of this by being the single foreground process you keep a terminal for, and its cleanup trap makes the teardown deliberate.

**Every boot pays the weight load.** The runtime persists kernel caches between starts, not weights. Budget the ~20 minutes.

**Trust host free, not container free.** This is the most important operational note on this tier. About 58 GiB of the model's residency lives in NVIDIA driver UVM pages that no container level counter sees and that are charged to the root cgroup, not the container's. The container's own `free` reports ~56 GiB used and a huge amount "available", which is fiction for capacity planning. The host's `free` is the real budget.

The full accounting, for anyone reconciling the numbers. The host has 128 GiB installed (4x 32 GB), the OS sees 125 GiB, and the container is capped at ~118 GiB plus a 24 GiB swap allowance. Host side residency is about 114 GiB: ~48.5 GiB of PLE table in the offload worker (process memory, container visible), ~58 GiB of expert pool in driver UVM pages (container invisible), and ~8 GiB of process overhead. The GPU side holds ~15 GiB of dense weights plus a ~19.5 GiB hot-84 expert cache, and that cache is a copy of the hot experts, the host pool keeps the complete set, so nothing is subtracted from the host side for it. The on-disk checkpoint, ~120 GiB, splits the same way: PLE and experts land on the host, dense weights and the MTP draft land on the GPUs.

**Why the load never OOMs at the container's 118 GiB limit.** Two mechanisms. First, the container's real budget is memory.max plus the swap allowance, about 142 GiB total, and pages above 118 GiB go to swap before anything OOMs. Second, the ~58 GiB of driver pages never touch the container's budget at all, they are charged to the root cgroup. The ~22 GiB spike itself is loader temporaries on top of residency plus host level reclaim: with the driver pages resident, the host runs close to full through the load (~117 of 125 GiB in the post-load measurement, the mid-load peak was never captured), and the kernel swaps out cold container pages under that global pressure. That is why the spike lands in the container's swap counters while the container never hits its own ceiling, and it is why the reference spec asks for 128 GB (decimal, 119.2 GiB, exactly what the preflight gate encodes).

**Optional hygiene: drain the spike residue.** After a load, the ~22 GiB that spilled to swap sits there until touched. It is harmless (steady state paging measured at zero), but a clean drain makes the first serving hours tidy and keeps `free` honest. It must run on the Proxmox host, the container has no power over host swap:

```bash
swapoff -a && swapon -a
```

Only do this while the server is idle and the host has more available RAM than the swap in use.

**This is a one request tier.** `max_num_seqs` is 1 by design, the KV reservation covers 1.05 full context requests, and while it runs nothing else fits on the cards or in RAM. It is also **text only** by default (the base model is a VL, but the runtime ships with multimodal limits set to 0), and its tool parser is the `qwen3_coder` flavor, not `qwen3_xml`.

## 11) Roadblocks quick reference

| # | Symptom | Cause                                                                                                                                    | Fix |
|---|---|------------------------------------------------------------------------------------------------------------------------------------------|---|
| R1 | `make preflight` errors: "needs approximately 128 GiB RAM" | preflight gate wants MemTotal >= 125000000 kB (128 GB decimal, 119.2 GiB), container sees ~118 GiB | lower the threshold in `scripts/preflight.sh` to 120000000 (step 8), restore on a bigger host |
| R2 | `error setting rlimit type 8: operation not permitted` at container start | docker wants `--ulimit memlock=-1`, unprivileged LXC root cannot raise the hard limit                                                    | `lxc.prlimit.memlock: unlimited` in the container config on the host, then reboot the container (step 1) |
| R3 | `ValueError: ... 0.67 GiB KV cache is needed, which is larger than the available KV cache memory (0.5 GiB)` | shrank `KV_CACHE_MEMORY_BYTES` for "RAM margin", but KV is GPU memory and the shrunk pool failed the min request check                   | keep the default `4429185024`, it covers the full 262K window (step 4) |
| R4 | the container "terminated by itself" and vanished | docker run is attached to the terminal, `--rm` removes the container on exit, so a closed SSH session or Ctrl+C on make ends it silently | use the launcher for a deliberate lifecycle (step 10) |
| R5 | endpoint unreachable from other machines, `docker ps` shows `127.0.0.1:8000->8000` | the launch script publishes localhost only, port built from the `PORT` variable                                                          | `PORT=8080` in `.env` (edit, do not duplicate) plus the `0.0.0.0` bind sed in `scripts/docker_serve.sh` (step 5) |
| R6 | `docker build` fails: `"/runtime/patches": not found` | `.dockerignore` is an allowlist and the new patch directory is not on it                                                                 | append `!runtime/patches/**` (step 6) |
| R7 | patch script prints `REFUSE: expected exactly 1 MambaSpec assert, found 2` | the pinned build's file has a second `MambaSpec` assert, so a text anchor is not unique                                                  | the introspection anchored patch (this is already the version in step 6) |
| R8 | load dies OOM mid-load | the loader spikes ~22 GiB past physical RAM and the swap behind it comes up short | raise both knobs together: the container swap allowance (the hard cap, host swap beyond it is unusable by the container) and the host swap behind it (step 1). The 24 GiB allowance here covers the measured 22 GiB spike with ~2.5 GiB of margin |
| R9 | container `free` shows plenty available but the host is tight | ~58 GiB of the expert pool lives in NVIDIA driver UVM pages invisible to container accounting                                            | plan capacity from the host's `free` (step 10) |
