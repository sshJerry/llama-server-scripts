#!/bin/bash
# bench.sh — llama-server benchmarking suite
#
# Usage:  ./bench.sh <output_dir> <prefix> [port] [server_log] [server_cmd]
#
#   output_dir   Directory to write results into (created if missing)
#   prefix       Filename prefix for all output files
#   port         llama-server API port (default: 8080)
#   server_log   Path to the tee'd server log for draft/decay extraction (default: /tmp/llama-server.log)
#   server_cmd   The command used to start the server (saved for reproducibility)
#
# Output files:
#   <prefix>_short.json        2048-token generation benchmark (timings + usage)
#   <prefix>_short-prompt.txt  Prompt text used for the short bench
#   <prefix>_long.json         8192-token generation benchmark (timings + usage)
#   <prefix>_long-prompt.txt   Prompt text used for the long bench
#   <prefix>_prompt.json       Prompt-processing benchmark (~4000 tokens in, 1 token out)
#   <prefix>_vram-idle.txt     nvidia-smi at idle
#   <prefix>_vram-load.txt     nvidia-smi during active generation
#   <prefix>_draft-stats.txt   MTP draft acceptance lines from server log (long bench window)
#   <prefix>_decay.csv         Throughput-over-time CSV (tokens,tps) from server log (long bench window)
#   <prefix>_server-cmd.sh     Server launch command (if provided)
#   <prefix>_summary.json      Single-file summary of all key metrics
#
# Requirements: bash >= 4.0, curl, jq, nvidia-smi, GNU grep, GNU sed, awk, seq, GNU coreutils

set -euo pipefail

OUTPUT_DIR="${1:?Usage: $0 <output_dir> <prefix> [port] [server_log] [server_cmd]}"
PREFIX="${2:?Missing prefix}"
PORT="${3:-8080}"
SERVER_LOG="${4:-/tmp/llama-server.log}"
SERVER_CMD="${5:-}"

SHORT_TOKENS=2048
LONG_TOKENS=8192
CURL_TIMEOUT=600
VRAM_POLL_DELAY=4

BASE="http://localhost:${PORT}/v1/chat/completions"
mkdir -p "$OUTPUT_DIR"

# ── Cleanup trap ────────────────────────────────────────────────────────────

LONG_PID=""
cleanup() {
  if [ -n "${LONG_PID:-}" ]; then
    kill "$LONG_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ── Prerequisites ──────────────────────────────────────────────────────────

for tool in curl jq nvidia-smi awk grep sed seq wc tail head; do
  if ! command -v "$tool" &>/dev/null; then
    echo "ERROR: $tool not found in PATH" >&2
    exit 1
  fi
done

if ! grep -P 'test' <<< "test" &>/dev/null; then
  echo "ERROR: grep does not support -P (Perl regex). GNU grep required." >&2
  exit 1
fi

# ── Server health check ────────────────────────────────────────────────────

if ! curl -sf --max-time 10 "$BASE" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"ping"}],"max_tokens":1,"stream":false}' \
  > /dev/null 2>&1; then
  echo "ERROR: llama-server not responding at ${BASE}" >&2
  echo "       Check that the server is running and port ${PORT} is correct." >&2
  exit 1
fi
echo "Server health check: OK"

# ── Save server command ────────────────────────────────────────────────────

if [ -n "$SERVER_CMD" ]; then
  echo "#!/bin/bash" > "${OUTPUT_DIR}/${PREFIX}_server-cmd.sh"
  echo "# Server launched with:" >> "${OUTPUT_DIR}/${PREFIX}_server-cmd.sh"
  echo "$SERVER_CMD" >> "${OUTPUT_DIR}/${PREFIX}_server-cmd.sh"
  echo "Saved server command to ${OUTPUT_DIR}/${PREFIX}_server-cmd.sh"
fi

# ── Helpers ────────────────────────────────────────────────────────────────

api() {
  # $1 = output path, $2 = max_tokens, $3 = prompt text
  local out="$1" max_tok="$2" prompt="$3" http_code body
  body=$(curl -s -w '\n%{http_code}' --max-time "$CURL_TIMEOUT" "$BASE" \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg content "$prompt" --argjson mt "$max_tok" '{
      messages: [{role: "user", content: $content}],
      max_tokens: $mt,
      stream: false
    }')")
  http_code=$(echo "$body" | tail -1)
  body=$(echo "$body" | sed '$d')
  if [ "$http_code" != "200" ]; then
    echo "ERROR: API returned HTTP $http_code on $out" >&2
    echo '{"error":"HTTP '"$http_code"'"}' > "$out"
    return 1
  fi
  echo "$body" | jq '{timings, usage}' > "$out"
}

log_section() { echo; echo "── $* ──"; }

# ── VRAM idle (captured first — before any GPU activity) ───────────────────

log_section "VRAM idle"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv \
  > "${OUTPUT_DIR}/${PREFIX}_vram-idle.txt"
echo "Wrote ${OUTPUT_DIR}/${PREFIX}_vram-idle.txt"

# ── Warm-up ────────────────────────────────────────────────────────────────

log_section "Warm-up"
curl -s --max-time 120 "$BASE" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Write a short paragraph about linux."}],"max_tokens":128,"stream":false}' \
  > /dev/null
echo "Done."
sleep 2

# ── Short bench ────────────────────────────────────────────────────────────

SHORT_PROMPT="Write a detailed 1000-word essay about the history of computing, from Babbage to modern AI."

log_section "Short bench (${SHORT_TOKENS} tok)"
echo "$SHORT_PROMPT" > "${OUTPUT_DIR}/${PREFIX}_short-prompt.txt"
api "${OUTPUT_DIR}/${PREFIX}_short.json" "$SHORT_TOKENS" "$SHORT_PROMPT"
SHORT_N=$(jq '.timings.predicted_n' "${OUTPUT_DIR}/${PREFIX}_short.json")
echo "Wrote ${OUTPUT_DIR}/${PREFIX}_short.json  ($SHORT_N tokens generated)"

# ── Long bench + VRAM under load ───────────────────────────────────────────

LONG_PROMPT="Write a comprehensive 5000-word technical guide on implementing distributed systems, covering consensus algorithms, leader election, log replication, snapshotting, and failure recovery. Include code examples."

log_section "Long bench (${LONG_TOKENS} tok) + VRAM under load"
echo "$LONG_PROMPT" > "${OUTPUT_DIR}/${PREFIX}_long-prompt.txt"

# Record log position before the long bench (for windowed decay/draft extraction)
LOG_START=0
if [ -f "$SERVER_LOG" ]; then
  LOG_START=$(wc -l < "$SERVER_LOG")
fi

api "${OUTPUT_DIR}/${PREFIX}_long.json" "$LONG_TOKENS" "$LONG_PROMPT" &
LONG_PID=$!

# Wait for generation to ramp up, then snapshot VRAM
sleep "$VRAM_POLL_DELAY"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv \
  > "${OUTPUT_DIR}/${PREFIX}_vram-load.txt"
echo "Captured VRAM under load."

wait $LONG_PID
LONG_PID=""  # cleared so trap won't try to kill an already-finished process

LOG_END=0
if [ -f "$SERVER_LOG" ]; then
  LOG_END=$(wc -l < "$SERVER_LOG")
fi
LONG_N=$(jq '.timings.predicted_n' "${OUTPUT_DIR}/${PREFIX}_long.json")
echo "Wrote ${OUTPUT_DIR}/${PREFIX}_long.json  ($LONG_N tokens generated)"

# ── Prompt-processing bench ────────────────────────────────────────────────

log_section "Prompt-processing bench (~4000 tok prompt)"
# Generate ~4000 tokens of repeated natural text
BULK_PROMPT=$(printf 'The quick brown fox jumps over the lazy dog near the riverbank. %.0s' $(seq 1 400))
api "${OUTPUT_DIR}/${PREFIX}_prompt.json" 1 "$BULK_PROMPT"
PROMPT_N=$(jq '.timings.prompt_n' "${OUTPUT_DIR}/${PREFIX}_prompt.json")
echo "Wrote ${OUTPUT_DIR}/${PREFIX}_prompt.json  ($PROMPT_N prompt tokens)"

# ── Draft stats from server log (long bench window only) ───────────────────

log_section "Draft stats (from server log, long bench window)"
if [ -f "$SERVER_LOG" ] && [ "$LOG_END" -gt "$LOG_START" ]; then
  tail -n +$((LOG_START + 1)) "$SERVER_LOG" | head -n $((LOG_END - LOG_START)) \
    | grep -E 'draft acceptance[[:space:]]*=|statistics[[:space:]]+draft-mtp' \
    > "${OUTPUT_DIR}/${PREFIX}_draft-stats.txt" || true
  if [ -s "${OUTPUT_DIR}/${PREFIX}_draft-stats.txt" ]; then
    echo "Wrote ${OUTPUT_DIR}/${PREFIX}_draft-stats.txt"
  else
    echo "(model has no MTP heads — no draft stats in log)" > "${OUTPUT_DIR}/${PREFIX}_draft-stats.txt"
    echo "No MTP draft stats found (model likely has no MTP heads)."
  fi
else
  echo "(server log not found at $SERVER_LOG)" > "${OUTPUT_DIR}/${PREFIX}_draft-stats.txt"
  echo "WARNING: server log not found at $SERVER_LOG — draft stats skipped."
fi

# ── Throughput decay from server log (long bench window only) ───────────────

log_section "Throughput decay (from server log, long bench window)"
echo "tokens,tps" > "${OUTPUT_DIR}/${PREFIX}_decay.csv"
if [ -f "$SERVER_LOG" ] && [ "$LOG_END" -gt "$LOG_START" ]; then
  tail -n +$((LOG_START + 1)) "$SERVER_LOG" | head -n $((LOG_END - LOG_START)) \
    | grep 'n_decoded' \
    | grep -oP 'n_decoded =\s+\d+,\s+tg =\s+[\d.]+' \
    | sed -E 's/n_decoded =\s+([0-9]+),\s+tg =\s+([0-9.]+)/\1,\2/' \
    >> "${OUTPUT_DIR}/${PREFIX}_decay.csv" || true
fi
POINTS=$(tail -n +2 "${OUTPUT_DIR}/${PREFIX}_decay.csv" 2>/dev/null | wc -l)
if [ "${POINTS:-0}" -gt 0 ]; then
  echo "Wrote ${OUTPUT_DIR}/${PREFIX}_decay.csv  ($POINTS data points)"
else
  echo "No decay data found."
fi

# ── Summary ────────────────────────────────────────────────────────────────

log_section "Summary"

# Parse nvidia-smi CSV into JSON arrays.
# Handles N/A fields (emits null) and whitespace-padded values.
parse_nvsmi() {
  awk -F',' 'NR>1 {
    gsub(/^[ \t]+|[ \t]+$/, "", $3);
    gsub(/^[ \t]+|[ \t]+$/, "", $4);
    gsub(/^[ \t]+|[ \t]+$/, "", $8);
    gsub(/^[ \t]+|[ \t]+$/, "", $9);

    used  = ($3 ~ /^[0-9]/) ? $3+0 : "null";
    total = ($4 ~ /^[0-9]/) ? $4+0 : "null";
    power = ($8 ~ /^[0-9]/) ? $8+0 : "null";
    temp  = ($9 ~ /^[0-9]/) ? $9+0 : "null";
    free  = (used != "null" && total != "null") ? total - used : "null";

    printf "{\"gpu\":%s,\"name\":\"%s\",\"used_mib\":%s,\"total_mib\":%s,\"free_mib\":%s,\"power_w\":%s,\"temp_c\":%s}\n",
      $1, $2, used, total, free,
      (power == "null" ? "null" : sprintf("%.1f", power)),
      temp
  }' "$1" | jq -s '.'
}

VRAM_IDLE_JSON=$(parse_nvsmi "${OUTPUT_DIR}/${PREFIX}_vram-idle.txt")
VRAM_IDLE_JSON="${VRAM_IDLE_JSON:-[]}"

VRAM_LOAD_JSON=$(parse_nvsmi "${OUTPUT_DIR}/${PREFIX}_vram-load.txt")
VRAM_LOAD_JSON="${VRAM_LOAD_JSON:-[]}"

jq -n \
  --arg prefix "$PREFIX" \
  --argjson short_expected "$SHORT_TOKENS" \
  --argjson long_expected "$LONG_TOKENS" \
  --argjson vram_idle "$VRAM_IDLE_JSON" \
  --argjson vram_load "$VRAM_LOAD_JSON" \
  --slurpfile short "${OUTPUT_DIR}/${PREFIX}_short.json" \
  --slurpfile long  "${OUTPUT_DIR}/${PREFIX}_long.json" \
  --slurpfile prompt "${OUTPUT_DIR}/${PREFIX}_prompt.json" \
  '
  def pct($accepted; $generated):
    if $accepted and $generated and $generated > 0
    then ($accepted / $generated * 10000 | round) / 100
    else null end;

  def early_stop($generated; $expected):
    if $generated < $expected then
      "stopped at \($generated)/\($expected) tokens"
    else
      null end;

  def draft_obj:
    {
      acceptance_pct:  pct(.timings.draft_n_accepted; .timings.draft_n),
      accepted:        .timings.draft_n_accepted,
      generated:       .timings.draft_n
    };

  {
    model: $prefix,
    short: {
      prompt_tps:             $short[0].timings.prompt_per_second,
      generation_tps:         $short[0].timings.predicted_per_second,
      prompt_tokens:          $short[0].timings.prompt_n,
      generated_tokens:       $short[0].timings.predicted_n,
      requested_tokens:       $short_expected,
      early_stop:             early_stop($short[0].timings.predicted_n; $short_expected),
      draft:                  ($short[0] | draft_obj)
    },
    long: {
      prompt_tps:             $long[0].timings.prompt_per_second,
      generation_tps:         $long[0].timings.predicted_per_second,
      prompt_tokens:          $long[0].timings.prompt_n,
      generated_tokens:       $long[0].timings.predicted_n,
      requested_tokens:       $long_expected,
      early_stop:             early_stop($long[0].timings.predicted_n; $long_expected),
      draft:                  ($long[0] | draft_obj)
    },
    prompt_processing: {
      tps:    $prompt[0].timings.prompt_per_second,
      tokens: $prompt[0].timings.prompt_n
    },
    vram_idle: $vram_idle,
    vram_load: $vram_load
  }' \
  > "${OUTPUT_DIR}/${PREFIX}_summary.json"

echo "Wrote ${OUTPUT_DIR}/${PREFIX}_summary.json"

# ── Print summary ──────────────────────────────────────────────────────────

echo
echo "══════════════════════════════════════════════════"
echo "  Results: ${OUTPUT_DIR}/${PREFIX}_*"
echo "══════════════════════════════════════════════════"
jq -r '
  def r: if . then (. * 100 | round / 100) else "?" end;
  def note:
    if . then "  ⚠ \(.)" else "" end;
  def gpu_line($arr; $label):
    "  \($label)  | " + (
      [$arr | to_entries[] | "GPU\(.key): \(.value.used_mib // "?")/\(.value.total_mib // "?") MiB (\(.value.free_mib // "?") MiB free)  \(.value.power_w // "?")W  \(.value.temp_c // "?")°C"]
      | join("  |  ")
    );

  "  Short  | prompt: \(.short.prompt_tps | r) t/s  |  gen: \(.short.generation_tps | r) t/s  |  draft: \(.short.draft.acceptance_pct // "N/A")%\(.short.early_stop | note)",
  "  Long   | prompt: \(.long.prompt_tps | r) t/s  |  gen: \(.long.generation_tps | r) t/s  |  draft: \(.long.draft.acceptance_pct // "N/A")%\(.long.early_stop | note)",
  "  Prompt | \(.prompt_processing.tokens) tok in  |  \(.prompt_processing.tps | r) t/s",
  "",
  gpu_line(.vram_idle; "VRAM idle "),
  gpu_line(.vram_load; "VRAM load "),
  ""
' "${OUTPUT_DIR}/${PREFIX}_summary.json"
echo "Done."
