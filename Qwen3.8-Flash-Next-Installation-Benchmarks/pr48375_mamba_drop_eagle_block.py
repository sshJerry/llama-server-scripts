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

Usage: applied inside the qwen38-flash-next-2x3090:locked image as a Dockerfile
RUN step, after the runtime overlay install. See GETTING_STARTED.md.
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
