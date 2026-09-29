# Extra: the swap drain script and the captured server logs

Three files live here: `drain-flashnext-swap.sh`, an optional hygiene script for the host side of the Flash-Next setup, `vLLM-Server.log`, the captured server log backing the prefix cache claims and the real traffic figures in the main [README](../README.md), and `vLLM-Server-2.log`, a second capture backing the agent session section there. The drain script is not part of the install, the runtime does not need it, and skipping it changes nothing about correctness or steady state performance.

## What it does

The Flash-Next weight load spikes past physical RAM by design (about 22 GiB on this hardware, measured, and documented behavior of the runtime). After the load finishes, those pages sit in swap until something touches them. This script waits for the endpoint to come up, gives it a minute to settle, then runs a `swapoff -a && swapon -a` cycle on the host, which pulls all of the residue back into RAM in one pass.

## Who it is for

Proxmox users only. The model serves from inside an LXC container, but swap devices belong to the host kernel, and the container has no power over them. That split is the entire reason this script exists: the load happens in the container, the swap lives on the host, so the drain has to run on the host, and the script polls the endpoint so you do not have to watch the clock for the ~20 minute load yourself.

On a bare metal install of the same runtime, none of this plumbing is needed. The equivalent is one command after startup:

```bash
sudo swapoff -a && sudo swapon -a
```

## Is it necessary

No, it is mostly personal preference. Measured on this rig:

- Steady state swap paging is zero. vmstat si/so sat flat through the full window prefill and the decode runs.
- The load residue is inert. The only observed effect without draining was a trickle of a few pages per second faulting back in, which is the PLE worker's busy loop touching its pool. Negligible.
- The model card's own warning about paging ("sustained paging can hurt performance") is about sustained pressure, which this rig never hit while serving.

What draining buys: a tidy first serving hour with everything resident from the start, and an honest `free` readout, which matters if you like watching the memory gauges mean what they say.

## Usage

Copy the script to the Proxmox host and run it alongside launching the tier:

```bash
nohup LXC_IP=<your container address> ./drain-flashnext-swap.sh > /tmp/drain.log 2>&1 &
tail -f /tmp/drain.log
```

It is guarded on both ends:

- Under 1 GiB in swap: no-op, warm reloads spill little or nothing.
- Not enough available RAM to absorb the drain (residue plus a 2 GiB margin): it skips with a message rather than pushing the host into pressure.

So it is safe to fire on every boot of the tier, including the boots where it does nothing.

## Where the accounting details live

Why the load spikes, why the container's `free` under-reports the model (the NVIDIA driver UVM pages), and the full memory walk are in [../GETTING_STARTED.md](../GETTING_STARTED.md), sections 8 and 10.

## The captured server logs

### vLLM-Server.log

An excerpt of the vLLM server log from the patched build, kept as the evidence file for the prefix cache claims and the real traffic figures in the main [README](../README.md).

What it contains:

- The server start banner (22:31:58), from `Starting vLLM server on http://0.0.0.0:8080` through `Started server process [1]` and `Application startup complete`. The engine init section (weight loading, PLE offload registration, and the `Setting attention block size to 1600 tokens` line quoted in [../GETTING_STARTED.md](../GETTING_STARTED.md)) runs before this point and is not part of the excerpt.
- Real agent traffic from a LAN client across several sessions, 22:36 through 23:52. The cumulative prefix cache hit rate climbs session over session: 7.6% to 21.6% in the first, a 47.6% peak in the second (which reused blocks the first session cached twenty minutes earlier), and 44.1% by the end of the last. Decode across the long stretches runs in the 46 to 80 tok/s band, consistent with the 52.3 tok/s bench figure in the README.
- The zero-hit probe and the ~131K bench runs as localhost traffic (172.17.0.1) from 23:17 onward.

Two artifacts worth knowing before reading it:

- The ~10,243 tok/s prompt throughput windows at 23:19:48 and 23:20:58 are the two ~131K bench prefills, not a prefill ceiling. The client-measured TTFT for those same requests is 72.5 s (1,807 tok/s), the same log's generation windows swing 2.1 to 65.7 tok/s between adjacent intervals, and the excerpt is trimmed through the middle of the bench, so the engine's interval metric does not reconcile to wall time in this file. The documented numbers are the client-measured ones. Related, the 1,806.2 tok/s interval at 23:17:08 resembles the 1,807 client figure but belongs to the second session's agent traffic, close in value, not the same request.
- The 107,200 cached-tokens figure quoted in the README came from the `vllm:prefix_cache_hits_total` counter read at the metrics endpoint, not from this file. The log carries the percentage lines only.

### vLLM-Server-2.log

The second capture, a friend's agentic coding session from 01:08 through 01:57, the workload a single-file HTML canvas simulation generated and then iterated on repeatedly. It is the evidence file for the agent session section of the main [README](../README.md). The file opens mid-session and its first line is a fragment cut mid-line inside a SpecDecoding entry, that is the excerpt point, not corruption. What it shows: the cumulative prefix hit rate opens at 86.9% and erodes to 78.4% across fifty minutes, the prompt spikes of 2,000 to 5,714 tok/s are re-queried shared prefixes landing on cache, the `Waiting: 1 reqs` queue shows the `max_num_seqs` 1 design in the wild, the mean draft acceptance across 273 spec decoding windows is 59.8% with mean accepted length 2.8, and the closing stretch is about 27K tokens of uninterrupted decode at roughly 57 tok/s.

