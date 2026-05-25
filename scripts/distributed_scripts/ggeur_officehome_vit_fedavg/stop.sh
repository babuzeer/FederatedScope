#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
TASK_DIR="$ROOT_DIR/scripts/distributed_scripts/ggeur_officehome_vit_fedavg"
PID_DIR="${PID_DIR:-$TASK_DIR/pids}"

echo "Stopping OfficeHome distributed sample processes..."

if [ ! -d "$PID_DIR" ]; then
    echo "No PID directory found."
    exit 0
fi

for pid_file in "$PID_DIR"/*.pid; do
    if [ ! -f "$pid_file" ]; then
        continue
    fi

    pid="$(cat "$pid_file")"
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Killing process $pid from $(basename "$pid_file")"
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
done

echo "Stop complete."
