#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
TASK_DIR="$ROOT_DIR/scripts/distributed_scripts/ggeur_officehome_vit_fedavg"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
PID_DIR="$TASK_DIR/pids"
LOG_DIR="$TASK_DIR/logs/$RUN_ID"
export PID_DIR
export LOG_DIR

mkdir -p "$PID_DIR" "$LOG_DIR"
rm -f "$PID_DIR"/*.pid 2>/dev/null || true

cat > "$LOG_DIR/run_info.txt" <<EOF
run_id: $RUN_ID
started_at: $(date '+%Y-%m-%d %H:%M:%S %z')
root_dir: $ROOT_DIR
task_dir: $TASK_DIR
EOF

cleanup() {
    echo ""
    echo "Stopping OfficeHome distributed sample..."
    bash "$TASK_DIR/stop.sh" || true
}

trap cleanup EXIT INT TERM

echo "Starting OfficeHome distributed sample on one machine..."
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"

bash "$TASK_DIR/start_server.sh"
sleep 5

for client_id in 1 2 3 4; do
    bash "$TASK_DIR/start_client.sh" "$client_id"
    sleep 3
done

echo ""
echo "All sample processes started."
echo "Logs:"
echo "  $LOG_DIR/server.log"
echo "  $LOG_DIR/client_1.log"
echo "  $LOG_DIR/client_2.log"
echo "  $LOG_DIR/client_3.log"
echo "  $LOG_DIR/client_4.log"
echo ""
echo "Press Ctrl+C to stop all tracked processes."

while true; do
    alive=0
    for pid_file in "$PID_DIR"/*.pid; do
        if [ ! -f "$pid_file" ]; then
            continue
        fi
        pid="$(cat "$pid_file")"
        if ps -p "$pid" > /dev/null 2>&1; then
            alive=1
            break
        fi
    done

    if [ "$alive" -eq 0 ]; then
        echo "All tracked processes exited."
        break
    fi

    sleep 5
done
