#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <client_id: 1|2|3|4>"
    exit 1
fi

CLIENT_ID="$1"
case "$CLIENT_ID" in
    1|2|3|4) ;;
    *)
        echo "Invalid client_id: $CLIENT_ID"
        exit 1
        ;;
esac

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
TASK_DIR="$ROOT_DIR/scripts/distributed_scripts/ggeur_officehome_vit_fedavg"
PID_DIR="${PID_DIR:-$TASK_DIR/pids}"
LOG_DIR="${LOG_DIR:-$TASK_DIR/logs/manual_$(date +%Y%m%d_%H%M%S)}"
CFG="$TASK_DIR/officehome_vit_ggeur_fedavg_client_${CLIENT_ID}.yaml"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

mkdir -p "$PID_DIR" "$LOG_DIR"

echo "Starting OfficeHome distributed sample client $CLIENT_ID..."
nohup "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" --cfg "$CFG" \
    > "$LOG_DIR/client_${CLIENT_ID}.log" 2>&1 &

CLIENT_PID=$!
echo "$CLIENT_PID" > "$PID_DIR/client_${CLIENT_ID}.pid"

echo "Client $CLIENT_ID PID: $CLIENT_PID"
echo "Log: $LOG_DIR/client_${CLIENT_ID}.log"
