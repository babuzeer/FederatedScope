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
CFG="$CASE_DIR/configs/server.yaml"
SYSTEM_DIR="$CASE_DIR/system"

mkdir -p "$PID_DIR" "$LOG_DIR" "$SYSTEM_DIR"
rm -f "$PID_DIR/server.pid"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [ ! -f "$CFG" ]; then
  echo "Missing server config: $CFG"
  exit 1
fi

{
  echo "role=server"
  echo "case_dir=$CASE_DIR"
  echo "root_dir=$ROOT_DIR"
  echo "python_bin=$PYTHON_BIN"
  echo "cfg=$CFG"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo
  echo "[git]"
  git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || true
  git -C "$ROOT_DIR" status --short 2>/dev/null || true
  echo
  echo "[system]"
  uname -a || true
  nvidia-smi || true
} | tee "$SYSTEM_DIR/server_run_info.log"

nohup "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" --cfg "$CFG" \
  > "$LOG_DIR/server.log" 2>&1 &
pid="$!"
echo "$pid" > "$PID_DIR/server.pid"
echo "server_pid=$pid"
echo "server_log=$LOG_DIR/server.log"
