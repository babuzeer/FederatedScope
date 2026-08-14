#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <case_dir>"
  exit 1
fi

CASE_DIR="$(cd "$1" && pwd)"
ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PID_DIR="$CASE_DIR/pids"
LOG_DIR="$CASE_DIR/logs"
SYSTEM_DIR="$CASE_DIR/system"

mkdir -p "$PID_DIR" "$LOG_DIR" "$SYSTEM_DIR"
rm -f "$PID_DIR"/client_*.pid 2>/dev/null || true
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

{
  echo "role=clients"
  echo "case_dir=$CASE_DIR"
  echo "root_dir=$ROOT_DIR"
  echo "python_bin=$PYTHON_BIN"
  echo "client_start_gap=${CLIENT_START_GAP:-2}"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo
  echo "[git]"
  git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || true
  git -C "$ROOT_DIR" status --short 2>/dev/null || true
  echo
  echo "[system]"
  uname -a || true
  nvidia-smi || true
} | tee "$SYSTEM_DIR/clients_run_info.log"

shopt -s nullglob
configs=("$CASE_DIR"/configs/client_*.yaml)
if [ "${#configs[@]}" -eq 0 ]; then
  echo "No client configs under $CASE_DIR/configs"
  exit 1
fi

for cfg in "${configs[@]}"; do
  base="$(basename "$cfg" .yaml)"
  client_id="${base#client_}"
  nohup "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" --cfg "$cfg" \
    > "$LOG_DIR/client_${client_id}.log" 2>&1 &
  pid="$!"
  echo "$pid" > "$PID_DIR/client_${client_id}.pid"
  echo "client_${client_id}_pid=$pid"
  sleep "${CLIENT_START_GAP:-2}"
done
