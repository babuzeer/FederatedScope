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

RUN_ID="${RUN_ID:-headonly_loopback_multiip_qps_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-exp/headonly_download_qps_loopback_multiip}"

SERVER_HOST="${SERVER_HOST:-127.77.0.10}"
CLIENT_IP_BASE="${CLIENT_IP_BASE:-127.77.0}"
CLIENT_IP_START="${CLIENT_IP_START:-101}"
BASE_PORT="${BASE_PORT:-39301}"
SUBSERVERS="${SUBSERVERS:-4}"
CLIENT_GROUPS="${CLIENT_GROUPS:-$SUBSERVERS}"
CLIENTS_PER_GROUP="${CLIENTS_PER_GROUP:-250}"
if [ $((CLIENT_GROUPS % SUBSERVERS)) -ne 0 ]; then
  echo "CLIENT_GROUPS must be divisible by SUBSERVERS for balanced subserver load" >&2
  exit 1
fi
GROUPS_PER_SUBSERVER=$((CLIENT_GROUPS / SUBSERVERS))
CLIENTS_PER_SUBSERVER_EXPECTED=$((CLIENTS_PER_GROUP * GROUPS_PER_SUBSERVER))

OVERHEAD_MULTIPLIER="${OVERHEAD_MULTIPLIER:-1.0}"
PAYLOAD_FILE="${PAYLOAD_FILE:-}"
READY_TIMEOUT="${READY_TIMEOUT:-300}"
ACK_TIMEOUT="${ACK_TIMEOUT:-300}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-60}"
NOFILE_LIMIT="${NOFILE_LIMIT:-1048576}"
SOMAXCONN="${SOMAXCONN:-65535}"

client_ip() {
  local idx="$1"
  echo "${CLIENT_IP_BASE}.$((CLIENT_IP_START + idx - 1))"
}

cd "$REPO_DIR"
RUN_DIR="$OUT_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
ulimit -n "$NOFILE_LIMIT" 2>/dev/null || true
sysctl -w "net.core.somaxconn=$SOMAXCONN" >/dev/null 2>&1 || true

{
  echo "role=loopback_multiip_single_host"
  echo "run_id=$RUN_ID"
  echo "repo_dir=$REPO_DIR"
  echo "python_bin=$PYTHON_BIN"
  echo "server_host=$SERVER_HOST"
  echo "client_ip_base=$CLIENT_IP_BASE"
  echo "client_ip_start=$CLIENT_IP_START"
  echo "base_port=$BASE_PORT"
  echo "subservers=$SUBSERVERS"
  echo "client_groups=$CLIENT_GROUPS"
  echo "groups_per_subserver=$GROUPS_PER_SUBSERVER"
  echo "clients_per_group=$CLIENTS_PER_GROUP"
  echo "clients_per_subserver_expected=$CLIENTS_PER_SUBSERVER_EXPECTED"
  echo "total_logical_clients=$((CLIENT_GROUPS * CLIENTS_PER_GROUP))"
  echo "overhead_multiplier=$OVERHEAD_MULTIPLIER"
  echo "payload_file=$PAYLOAD_FILE"
  echo "nofile_limit=$NOFILE_LIMIT"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo
  echo "[client_group_ips]"
  for idx in $(seq 1 "$CLIENT_GROUPS"); do
    subserver_idx=$(( (idx - 1) % SUBSERVERS ))
    echo "group${idx} source_ip=$(client_ip "$idx") subserver_port=$((BASE_PORT + subserver_idx))"
  done
} | tee "$RUN_DIR/run_info.log"

PYTHON_BIN="$PYTHON_BIN" \
REPO_DIR="$REPO_DIR" \
RUN_ID="$RUN_ID" \
OUT_ROOT="$OUT_ROOT" \
LISTEN_HOST="$SERVER_HOST" \
BASE_PORT="$BASE_PORT" \
SUBSERVERS="$SUBSERVERS" \
CLIENTS_PER_SUBSERVER="$CLIENTS_PER_SUBSERVER_EXPECTED" \
OVERHEAD_MULTIPLIER="$OVERHEAD_MULTIPLIER" \
PAYLOAD_FILE="$PAYLOAD_FILE" \
READY_TIMEOUT="$READY_TIMEOUT" \
ACK_TIMEOUT="$ACK_TIMEOUT" \
bash "$REPO_DIR/scripts/distributed_scripts/ggeur_headonly_download_qps/run_download_qps_server.sh" \
>"$RUN_DIR/server_outer.log" 2>&1 &
server_pid="$!"

cleanup() {
  kill "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

sleep 2

client_pids=()
for idx in $(seq 1 "$CLIENT_GROUPS"); do
  subserver_idx=$(( (idx - 1) % SUBSERVERS ))
  port=$((BASE_PORT + subserver_idx))
  source_ip="$(client_ip "$idx")"
  PYTHON_BIN="$PYTHON_BIN" \
  REPO_DIR="$REPO_DIR" \
  RUN_ID="${RUN_ID}_clients_group${idx}" \
  OUT_ROOT="$OUT_ROOT" \
  CONNECT_HOST="$SERVER_HOST" \
  SOURCE_HOST="$source_ip" \
  BASE_PORT="$port" \
  SUBSERVERS=1 \
  CLIENTS_PER_SUBSERVER="$CLIENTS_PER_GROUP" \
  CONNECT_TIMEOUT="$CONNECT_TIMEOUT" \
  bash "$REPO_DIR/scripts/distributed_scripts/ggeur_headonly_download_qps/run_download_qps_clients.sh" \
  >"$RUN_DIR/clients_group${idx}.log" 2>&1 &
  client_pids+=("$!")
done

status=0
for pid in "${client_pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if ! wait "$server_pid"; then
  status=1
fi

trap - EXIT

{
  echo
  echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "status=$status"
} | tee -a "$RUN_DIR/run_info.log"

echo "$RUN_DIR"
exit "$status"
