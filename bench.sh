#!/bin/bash
# bench.sh — llama-server benchmarking suite
#
# Usage:  ./bench.sh <output_dir> <prefix> [port] [server_log]
#
#   output_dir   Directory to write results into (created if missing)
#   prefix       Filename prefix for all output files
#   port         llama-server API port (default: 8080)
#   server_log   Path to the tee'd server log for draft/decay extraction (default: /tmp/llama-server.log)
#
# Output files:
#   <prefix>_short.json      2048-token generation benchmark
#   <prefix>_long.json       8192-token generation benchmark
#   <prefix>_prompt.json     Prompt-processing benchmark (~4000 tokens in, 1 token out)
#   <prefix>_vram-idle.txt   nvidia-smi at idle
#   <prefix>_vram-load.txt   nvidia-smi during active generation
#   <prefix>_draft-stats.txt MTP draft acceptance lines from server log
#   <prefix>_decay.txt       Throughput-over-time from server log (n_decoded lines)
#   <prefix>_summary.json    Single-file summary of all key metrics

set -euo pipefail

OUTPUT_DIR="${1:?Usage: $0 <output_dir> <prefix> [port] [server_log]}"
PREFIX="${2:?Missing prefix}"
PORT="${3:-8080}"
SERVER_LOG="${4:-/tmp/llama-server.log}"

BASE="http://localhost:${PORT}/v1/chat/completions"
mkdir -p "$OUTPUT_DIR"

# ── Helpers ────────────────────────────────────────────────────────────────

api() {
  # $1 = output path, $2 = max_tokens, $3 = prompt text
  local out="$1" max_tok="$2" prompt="$3"
  curl -s "$BASE" \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg content "$prompt" --argjson mt "$max_tok" '{
      messages: [{role: "user", content: $content}],
      max_tokens: $mt,
      stream: false
    }')" \
    | jq '{timings, usage}' > "$out"
}

log_section() { echo; echo "── $* ──"; }

# ── Warm-up ────────────────────────────────────────────────────────────────

log_section "Warm-up"
curl -s "$BASE" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Write a short paragraph about linux."}],"max_tokens":128,"stream":false}' \
  > /dev/null
echo "Done."
sleep 2

# ── Short bench (2048 tokens) ──────────────────────────────────────────────

log_section "Short bench (2048 tok)"
api "${OUTPUT_DIR}/${PREFIX}_short.json" 2048 \
  "Write a detailed 1000-word essay about the history of computing, from Babbage to modern AI."
echo "Wrote ${OUTPUT_DIR}/${PREFIX}_short.json"

# ── Long bench (8192 tokens) + VRAM under load ─────────────────────────────

log_section "Long bench (8192 tok) + VRAM under load"
api "${OUTPUT_DIR}/${PREFIX}_long.json" 8192 \
  "Write a comprehensive 5000-word technical guide on implementing distributed systems, covering consensus algorithms, leader election, log replication, snapshotting, and failure recovery. Include code examples." &
LONG_PID=$!

# Wait for generation to ramp up, then snapshot VRAM
sleep 4
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv \
  > "${OUTPUT_DIR}/${PREFIX}_vram-load.txt"
echo "Captured VRAM under load."

wait $LONG_PID
echo "Wrote ${OUTPUT_DIR}/${PREFIX}_long.json"

# ── Prompt-processing bench ────────────────────────────────────────────────

log_section "Prompt-processing bench (~4000 tok prompt)"
# Generate ~4000 tokens of repeated natural text (no python dependency)
LONG_PROMPT=$(printf 'The quick brown fox jumps over the lazy dog near the riverbank. '%.0s $(seq 1 400))
api "${OUTPUT_DIR}/${PREFIX}_prompt.json" 1 "$LONG_PROMPT"
echo "Wrote ${OUTPUT_DIR}/${PREFIX}_prompt.json"

# ── VRAM idle ──────────────────────────────────────────────────────────────

log_section "VRAM idle"
sleep 4  # let GPU clocks settle
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv \
  > "${OUTPUT_DIR}/${PREFIX}_vram-idle.txt"
echo "Wrote ${OUTPUT_DIR}/${PREFIX}_vram-idle.txt"

# ── Draft stats from server log ────────────────────────────────────────────

log_section "Draft stats (from server log)"
if [ -f "$SERVER_LOG" ]; then
  grep -E 'draft acceptance =|statistics\s+draft-mtp' "$SERVER_LOG" \
    > "${OUTPUT_DIR}/${PREFIX}_draft-stats.txt" 2>/dev/null || {
    echo "(no draft stats found in log)" > "${OUTPUT_DIR}/${PREFIX}_draft-stats.txt"
  }
  echo "Wrote ${OUTPUT_DIR}/${PREFIX}_draft-stats.txt"
else
  echo "(server log not found at $SERVER_LOG)" > "${OUTPUT_DIR}/${PREFIX}_draft-stats.txt"
  echo "WARNING: server log not found at $SERVER_LOG — draft stats skipped."
fi

# ── Throughput decay from server log ───────────────────────────────────────

log_section "Throughput decay (from server log)"
if [ -f "$SERVER_LOG" ]; then
  grep 'n_decoded' "$SERVER_LOG" \
    | grep -oP 'n_decoded =\s+\d+,\s+tg =\s+[\d.]+' \
    > "${OUTPUT_DIR}/${PREFIX}_decay.txt" 2>/dev/null || {
    echo "(no decay data found in log)" > "${OUTPUT_DIR}/${PREFIX}_decay.txt"
  }
  echo "Wrote ${OUTPUT_DIR}/${PREFIX}_decay.txt"
else
  echo "(server log not found)" > "${OUTPUT_DIR}/${PREFIX}_decay.txt"
fi

# ── Summary ────────────────────────────────────────────────────────────────

log_section "Summary"

jq -n \
  --arg prefix "$PREFIX" \
  --slurpfile short "${OUTPUT_DIR}/${PREFIX}_short.json" \
  --slurpfile long  "${OUTPUT_DIR}/${PREFIX}_long.json" \
  --slurpfile prompt "${OUTPUT_DIR}/${PREFIX}_prompt.json" \
  '
  def pct($accepted; $generated):
    if $generated > 0 then ($accepted / $generated * 10000 | round) / 100 else 0 end;

  {
    model: $prefix,
    short: {
      prompt_tps:             $short[0].timings.prompt_per_second,
      generation_tps:         $short[0].timings.predicted_per_second,
      prompt_tokens:          $short[0].timings.prompt_n,
      generated_tokens:       $short[0].timings.predicted_n,
      draft_acceptance_pct:   pct($short[0].timings.draft_n_accepted; $short[0].timings.draft_n),
      drafts_accepted:        $short[0].timings.draft_n_accepted,
      drafts_generated:       $short[0].timings.draft_n
    },
    long: {
      prompt_tps:             $long[0].timings.prompt_per_second,
      generation_tps:         $long[0].timings.predicted_per_second,
      prompt_tokens:          $long[0].timings.prompt_n,
      generated_tokens:       $long[0].timings.predicted_n,
      draft_acceptance_pct:   pct($long[0].timings.draft_n_accepted; $long[0].timings.draft_n),
      drafts_accepted:        $long[0].timings.draft_n_accepted,
      drafts_generated:       $long[0].timings.draft_n
    },
    prompt_processing: {
      tps:    $prompt[0].timings.prompt_per_second,
      tokens: $prompt[0].timings.prompt_n
    }
  }' \
  > "${OUTPUT_DIR}/${PREFIX}_summary.json"

echo "Wrote ${OUTPUT_DIR}/${PREFIX}_summary.json"

# ── Print summary ──────────────────────────────────────────────────────────

echo
echo "══════════════════════════════════════════════════"
echo "  Results: ${OUTPUT_DIR}/${PREFIX}_*"
echo "══════════════════════════════════════════════════"
jq -r '
  "  Short  | prompt: \(.short.prompt_tps | .*100 | round/100) t/s  |  gen: \(.short.generation_tps | .*100 | round/100) t/s  |  draft: \(.short.draft_acceptance_pct)%",
  "  Long   | prompt: \(.long.prompt_tps | .*100 | round/100) t/s  |  gen: \(.long.generation_tps | .*100 | round/100) t/s  |  draft: \(.long.draft_acceptance_pct)%",
  "  Prompt | \(.prompt_processing.tokens) tok in  |  \(.prompt_processing.tps | .*100 | round/100) t/s",
  ""
' "${OUTPUT_DIR}/${PREFIX}_summary.json"
echo "Done."
