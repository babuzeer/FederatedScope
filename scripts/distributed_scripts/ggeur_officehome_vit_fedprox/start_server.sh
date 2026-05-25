#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
TASK_DIR="$ROOT_DIR/scripts/distributed_scripts/ggeur_officehome_vit_fedprox"
PID_DIR="${PID_DIR:-$TASK_DIR/pids}"
LOG_DIR="${LOG_DIR:-$TASK_DIR/logs/manual_$(date +%Y%m%d_%H%M%S)}"
CFG="$TASK_DIR/officehome_vit_ggeur_fedprox_server.yaml"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

mkdir -p "$PID_DIR" "$LOG_DIR"

echo "Starting OfficeHome GGEUR FedProx distributed server..."
nohup "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" --cfg "$CFG" \
    > "$LOG_DIR/server.log" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$PID_DIR/server.pid"

echo "Server PID: $SERVER_PID"
echo "Log: $LOG_DIR/server.log"
