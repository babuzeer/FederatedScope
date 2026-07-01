#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_DIR="${REPO_DIR:-$DEFAULT_REPO_DIR}"
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x /root/miniconda3/envs/fs/bin/python ]; then
    PYTHON_BIN="/root/miniconda3/envs/fs/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
RUN_ID="${RUN_ID:-headonly_dispatch_qps_server_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-exp/headonly_download_qps}"

LISTEN_HOST="${LISTEN_HOST:-0.0.0.0}"
BASE_PORT="${BASE_PORT:-39301}"
SUBSERVERS="${SUBSERVERS:-4}"
CLIENTS_PER_SUBSERVER="${CLIENTS_PER_SUBSERVER:-250}"
SERVER_MODE="${SERVER_MODE:-server-mp}"
ACK_MODE="${ACK_MODE:-first-chunk}"
FIRST_CHUNK_BYTES="${FIRST_CHUNK_BYTES:-4096}"
MODEL_VERSION="${MODEL_VERSION:-global_mlp_v1}"
OVERHEAD_MULTIPLIER="${OVERHEAD_MULTIPLIER:-1.0}"
PAYLOAD_FILE="${PAYLOAD_FILE:-}"
READY_TIMEOUT="${READY_TIMEOUT:-300}"
ACK_TIMEOUT="${ACK_TIMEOUT:-300}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-60}"
MP_START_METHOD="${MP_START_METHOD:-spawn}"
NOFILE_LIMIT="${NOFILE_LIMIT:-1048576}"
SOMAXCONN="${SOMAXCONN:-65535}"

cd "$REPO_DIR"

RUN_DIR="$OUT_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
ulimit -n "$NOFILE_LIMIT" 2>/dev/null || true
sysctl -w "net.core.somaxconn=$SOMAXCONN" >/dev/null 2>&1 || true

{
  echo "role=server"
  echo "run_id=$RUN_ID"
  echo "repo_dir=$REPO_DIR"
  echo "server_mode=$SERVER_MODE"
  echo "listen_host=$LISTEN_HOST"
  echo "base_port=$BASE_PORT"
  echo "subservers=$SUBSERVERS"
  echo "clients_per_subserver=$CLIENTS_PER_SUBSERVER"
  echo "ack_mode=$ACK_MODE"
  echo "first_chunk_bytes=$FIRST_CHUNK_BYTES"
  echo "model_version=$MODEL_VERSION"
  echo "overhead_multiplier=$OVERHEAD_MULTIPLIER"
  echo "payload_file=$PAYLOAD_FILE"
  echo "ready_timeout=$READY_TIMEOUT"
  echo "ack_timeout=$ACK_TIMEOUT"
  echo "request_timeout=$REQUEST_TIMEOUT"
  echo "nofile_limit=$NOFILE_LIMIT"
  echo "somaxconn=$SOMAXCONN"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo
  echo "[network]"
  for iface in /sys/class/net/*; do
    name="$(basename "$iface")"
    [ "$name" = "lo" ] && continue
    printf '%s speed_mbps=' "$name"
    cat "$iface/speed" 2>/dev/null || echo unknown
    printf '%s state=' "$name"
    cat "$iface/operstate" 2>/dev/null || echo unknown
  done
} | tee "$RUN_DIR/run_info.log"

"$PYTHON_BIN" scripts/benchmark_headonly_mlp_download_window.py "$SERVER_MODE" \
  --listen-host "$LISTEN_HOST" \
  --base-port "$BASE_PORT" \
  --subservers "$SUBSERVERS" \
  --clients-per-subserver "$CLIENTS_PER_SUBSERVER" \
  --ack-mode "$ACK_MODE" \
  --first-chunk-bytes "$FIRST_CHUNK_BYTES" \
  --model-version "$MODEL_VERSION" \
  --overhead-multiplier "$OVERHEAD_MULTIPLIER" \
  --payload-file "$PAYLOAD_FILE" \
  --ready-timeout "$READY_TIMEOUT" \
  --ack-timeout "$ACK_TIMEOUT" \
  --request-timeout "$REQUEST_TIMEOUT" \
  --mp-start-method "$MP_START_METHOD" \
  --output "$RUN_DIR/download_qps_summary.json" \
  2>&1 | tee "$RUN_DIR/server.log"

{
  echo
  echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
} | tee -a "$RUN_DIR/run_info.log"

echo "$RUN_DIR"
