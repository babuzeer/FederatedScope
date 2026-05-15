#!/bin/bash

set -e

PID_FILE="/tmp/federatedscope_distributed_pids.txt"

cleanup() {
    echo ""
    echo "============================================"
    echo "Stopping all FederatedScope processes..."
    echo "============================================"

    if [ -f "$PID_FILE" ]; then
        while read pid; do
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "Killing process $pid"
                kill -9 "$pid" 2>/dev/null || true
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi

    pkill -9 -f "federatedscope/main.py.*distributed" 2>/dev/null || true

    echo "All processes stopped."
    exit 0
}

trap cleanup EXIT INT TERM

echo "============================================"
echo "Starting FederatedScope Distributed Training"
echo "============================================"
echo "PID file: $PID_FILE"
echo ""

rm -f "$PID_FILE"

echo "Generating data..."
python scripts/distributed_scripts/gen_data.py

echo ""
echo "Starting processes..."
echo ""

echo "1. Starting server..."
python federatedscope/main.py \
    --cfg scripts/distributed_scripts/distributed_configs/distributed_server.yaml &
SERVER_PID=$!
echo $SERVER_PID >> "$PID_FILE"
echo "   Server PID: $SERVER_PID"

sleep 3

echo "2. Starting client 1..."
python federatedscope/main.py \
    --cfg scripts/distributed_scripts/distributed_configs/distributed_client_1.yaml &
CLIENT1_PID=$!
echo $CLIENT1_PID >> "$PID_FILE"
echo "   Client 1 PID: $CLIENT1_PID"

sleep 2

echo "3. Starting client 2..."
python federatedscope/main.py \
    --cfg scripts/distributed_scripts/distributed_configs/distributed_client_2.yaml &
CLIENT2_PID=$!
echo $CLIENT2_PID >> "$PID_FILE"
echo "   Client 2 PID: $CLIENT2_PID"

sleep 2

echo "4. Starting client 3..."
python federatedscope/main.py \
    --cfg scripts/distributed_scripts/distributed_configs/distributed_client_3.yaml &
CLIENT3_PID=$!
echo $CLIENT3_PID >> "$PID_FILE"
echo "   Client 3 PID: $CLIENT3_PID"

echo ""
echo "============================================"
echo "All processes started successfully!"
echo "============================================"
echo ""
echo "Logs are saved to the exp/ directory."
echo ""
echo "To stop all processes:"
echo "  - Press Ctrl+C in this terminal, or"
echo "  - Run: ./scripts/distributed_scripts/stop_distributed.sh"
echo ""
echo "Waiting for processes to complete (Press Ctrl+C to stop)..."
echo ""

wait
