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
RUN_ID="${RUN_ID:-headonly_dispatch_qps_clients_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-exp/headonly_download_qps}"

CONNECT_HOST="${CONNECT_HOST:?Set CONNECT_HOST to the server public host/IP}"
BASE_PORT="${BASE_PORT:-39301}"
CONNECT_PORTS="${CONNECT_PORTS:-}"
SOURCE_HOST="${SOURCE_HOST:-}"
SOURCE_HOSTS="${SOURCE_HOSTS:-}"
SUBSERVERS="${SUBSERVERS:-4}"
CLIENTS_PER_SUBSERVER="${CLIENTS_PER_SUBSERVER:-250}"
CLIENT_MODE="${CLIENT_MODE:-client-mp}"
ACK_MODE="${ACK_MODE:-first-chunk}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-60}"
MP_START_METHOD="${MP_START_METHOD:-spawn}"
NOFILE_LIMIT="${NOFILE_LIMIT:-1048576}"

cd "$REPO_DIR"

RUN_DIR="$OUT_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
ulimit -n "$NOFILE_LIMIT" 2>/dev/null || true

{
  echo "role=clients"
  echo "run_id=$RUN_ID"
  echo "repo_dir=$REPO_DIR"
  echo "client_mode=$CLIENT_MODE"
  echo "connect_host=$CONNECT_HOST"
  echo "base_port=$BASE_PORT"
  echo "connect_ports=$CONNECT_PORTS"
  echo "source_host=$SOURCE_HOST"
  echo "source_hosts=$SOURCE_HOSTS"
  echo "subservers=$SUBSERVERS"
  echo "clients_per_subserver=$CLIENTS_PER_SUBSERVER"
  echo "ack_mode=$ACK_MODE"
  echo "connect_timeout=$CONNECT_TIMEOUT"
  echo "nofile_limit=$NOFILE_LIMIT"
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

"$PYTHON_BIN" scripts/benchmark_headonly_mlp_download_window.py "$CLIENT_MODE" \
  --connect-host "$CONNECT_HOST" \
  --base-port "$BASE_PORT" \
  --connect-ports "$CONNECT_PORTS" \
  --source-host "$SOURCE_HOST" \
  --source-hosts "$SOURCE_HOSTS" \
  --subservers "$SUBSERVERS" \
  --clients-per-subserver "$CLIENTS_PER_SUBSERVER" \
  --ack-mode "$ACK_MODE" \
  --connect-timeout "$CONNECT_TIMEOUT" \
  --mp-start-method "$MP_START_METHOD" \
  2>&1 | tee "$RUN_DIR/clients.log"

{
  echo
  echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
} | tee -a "$RUN_DIR/run_info.log"

echo "$RUN_DIR"
