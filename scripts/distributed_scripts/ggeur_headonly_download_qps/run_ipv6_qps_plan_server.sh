#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_PREFIX="${RUN_PREFIX:-qps_ipv6_plan_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-exp/headonly_download_qps}"
LISTEN_HOST="${LISTEN_HOST:-::}"
BASE_PORT="${BASE_PORT:-39301}"
ACK_MODE="${ACK_MODE:-first-chunk}"
FIRST_CHUNK_BYTES="${FIRST_CHUNK_BYTES:-4096}"
OVERHEAD_MULTIPLIER="${OVERHEAD_MULTIPLIER:-1.0}"
PAYLOAD_FILE="${PAYLOAD_FILE:-}"

# Format: name:subservers:clients_per_subserver:ready_timeout:ack_timeout
QPS_PLAN="${QPS_PLAN:-smoke_1x10:1:10:120:120,flat_1x500:1:500:240:240,h2_2x500:2:500:300:300,h4_4x500:4:500:360:360,h8_8x500:8:500:480:480,h4_4x2500:4:2500:720:720,h8_8x1250:8:1250:720:720}"

cd "$REPO_DIR"

echo "repo_dir=$REPO_DIR"
echo "run_prefix=$RUN_PREFIX"
echo "listen_host=$LISTEN_HOST"
echo "base_port=$BASE_PORT"
echo "ack_mode=$ACK_MODE"
echo "first_chunk_bytes=$FIRST_CHUNK_BYTES"
echo "qps_plan=$QPS_PLAN"

IFS=',' read -r -a cases <<< "$QPS_PLAN"
for item in "${cases[@]}"; do
  IFS=':' read -r name subservers clients ready_timeout ack_timeout <<< "$item"
  run_id="${RUN_PREFIX}_${name}_server"
  echo
  echo "===== SERVER CASE $name subservers=$subservers clients_per_subserver=$clients ====="
  RUN_ID="$run_id" \
  OUT_ROOT="$OUT_ROOT" \
  LISTEN_HOST="$LISTEN_HOST" \
  BASE_PORT="$BASE_PORT" \
  SUBSERVERS="$subservers" \
  CLIENTS_PER_SUBSERVER="$clients" \
  ACK_MODE="$ACK_MODE" \
  FIRST_CHUNK_BYTES="$FIRST_CHUNK_BYTES" \
  READY_TIMEOUT="$ready_timeout" \
  ACK_TIMEOUT="$ack_timeout" \
  OVERHEAD_MULTIPLIER="$OVERHEAD_MULTIPLIER" \
  PAYLOAD_FILE="$PAYLOAD_FILE" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash "$SCRIPT_DIR/run_download_qps_server.sh"
done

"$PYTHON_BIN" scripts/summarize_headonly_qps_runs.py \
  "$OUT_ROOT" \
  --prefix "$RUN_PREFIX" \
  --output "$OUT_ROOT/${RUN_PREFIX}_summary.tsv"

echo "$OUT_ROOT/${RUN_PREFIX}_summary.tsv"
