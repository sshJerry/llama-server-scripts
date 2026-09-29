#!/bin/bash
# Drain the swap residue left by the Qwen3.8-Flash-Next weight load.
# Runs ON THE PROXMOX HOST. The container cannot manage host swap devices,
# which is why this script exists at all (see extra/README.md).
#
# Usage:
#   LXC_IP=<your container address> ./drain-flashnext-swap.sh
# or:
#   ENDPOINT=http://<ip>:8080 ./drain-flashnext-swap.sh
#
# Waits for the endpoint to come up (the ~20 minute weight load), settles
# 60 seconds, then drains. Guarded: no-op if under 1 GiB is in swap, and
# skips rather than pushes the host into pressure if there is not enough
# available RAM to absorb the drain.

if [ -z "${LXC_IP:-}" ] && [ -z "${ENDPOINT:-}" ]; then
    echo "usage: LXC_IP=<container address> $0"
    echo "   or: ENDPOINT=http://<ip>:8080 $0"
    exit 1
fi
ENDPOINT="${ENDPOINT:-http://${LXC_IP}:8080}"

echo "[drain] waiting for ${ENDPOINT}/v1/models ..."
until curl -sf "${ENDPOINT}/v1/models" >/dev/null 2>&1; do sleep 15; done
echo "[drain] endpoint up, settling 60s"
sleep 60

used=$(free -m | awk '/^Swap:/{print $3}')
avail=$(free -m | awk '/^Mem:/{print $7}')

if [ "${used:-0}" -lt 1024 ]; then
    echo "[drain] only ${used} MiB in swap - nothing to do"
    exit 0
fi
if [ "${avail:-0}" -lt $((used + 2048)) ]; then
    echo "[drain] insufficient headroom (used=${used}, avail=${avail} MiB) - skipping"
    exit 1
fi

swapoff -a && swapon -a
echo "[drain] pulled ${used} MiB back to RAM"
free -h | sed -n '1,3p'
