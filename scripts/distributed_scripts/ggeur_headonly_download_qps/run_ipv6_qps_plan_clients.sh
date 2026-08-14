#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_PREFIX="${RUN_PREFIX:?Set RUN_PREFIX to the same value used by the server script}"
OUT_ROOT="${OUT_ROOT:-exp/headonly_download_qps}"
CONNECT_HOST="${CONNECT_HOST:?Set CONNECT_HOST to the server IPv6 address without brackets}"
BASE_PORT="${BASE_PORT:-39301}"
CONNECT_PORTS="${CONNECT_PORTS:-}"
SOURCE_HOST="${SOURCE_HOST:-}"
SOURCE_HOSTS="${SOURCE_HOSTS:-}"
ACK_MODE="${ACK_MODE:-first-chunk}"

# Must match the server plan. Format:
# name:subservers:clients_per_subserver:ready_timeout:ack_timeout
QPS_PLAN="${QPS_PLAN:-smoke_1x10:1:10:120:120,flat_1x500:1:500:240:240,h2_2x500:2:500:300:300,h4_4x500:4:500:360:360,h8_8x500:8:500:480:480,h4_4x2500:4:2500:720:720,h8_8x1250:8:1250:720:720}"

take_csv_prefix() {
  local value="$1"
  local count="$2"
  if [ -z "$value" ]; then
    echo ""
    return
  fi
  IFS=',' read -r -a items <<< "$value"
  if [ "${#items[@]}" -lt "$count" ]; then
    echo "Need at least $count comma-separated values, got ${#items[@]}: $value" >&2
    exit 1
  fi
  local result=""
  local idx
  for idx in $(seq 0 $((count - 1))); do
    if [ -n "$result" ]; then
      result+=","
    fi
    result+="${items[$idx]}"
  done
  echo "$result"
}

cd "$REPO_DIR"

echo "repo_dir=$REPO_DIR"
echo "run_prefix=$RUN_PREFIX"
echo "connect_host=$CONNECT_HOST"
echo "base_port=$BASE_PORT"
echo "ack_mode=$ACK_MODE"
echo "qps_plan=$QPS_PLAN"

IFS=',' read -r -a cases <<< "$QPS_PLAN"
for item in "${cases[@]}"; do
  IFS=':' read -r name subservers clients ready_timeout ack_timeout <<< "$item"
  run_id="${RUN_PREFIX}_${name}_clients"
  case_connect_ports="$(take_csv_prefix "$CONNECT_PORTS" "$subservers")"
  case_source_hosts="$(take_csv_prefix "$SOURCE_HOSTS" "$subservers")"
  echo
  echo "===== CLIENT CASE $name subservers=$subservers clients_per_subserver=$clients ====="
  RUN_ID="$run_id" \
  OUT_ROOT="$OUT_ROOT" \
  CONNECT_HOST="$CONNECT_HOST" \
  BASE_PORT="$BASE_PORT" \
  CONNECT_PORTS="$case_connect_ports" \
  SOURCE_HOST="$SOURCE_HOST" \
  SOURCE_HOSTS="$case_source_hosts" \
  SUBSERVERS="$subservers" \
  CLIENTS_PER_SUBSERVER="$clients" \
  ACK_MODE="$ACK_MODE" \
  CONNECT_TIMEOUT="$ready_timeout" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash "$SCRIPT_DIR/run_download_qps_clients.sh"
done
