#!/bin/bash

# Distributed GGEUR training on Office-Home with process management
# This script can be easily stopped with Ctrl+C or by running the stop script

set -e

# PID file to track all spawned processes
PID_FILE="/tmp/federatedscope_ggeur_officehome_pids.txt"
SHARD_ROOT="data/officehome/shards"
MANIFEST_PATH="$SHARD_ROOT/manifest.json"
CLIENT_SHARDS=(
    "$SHARD_ROOT/client_1"
    "$SHARD_ROOT/client_2"
    "$SHARD_ROOT/client_3"
    "$SHARD_ROOT/client_4"
)

# Cleanup function to kill all spawned processes
cleanup() {
    echo ""
    echo "============================================"
    echo "Stopping all GGEUR Office-Home processes..."
    echo "============================================"

    if [ -f "$PID_FILE" ]; then
        while read pid; do
            if ps -p $pid > /dev/null 2>&1; then
                echo "Killing process $pid"
                kill -9 $pid 2>/dev/null || true
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi

    # Also kill any remaining federatedscope processes related to GGEUR
    pkill -9 -f "federatedscope/main.py.*ggeur" 2>/dev/null || true

    echo "All processes stopped."
    exit 0
}

# Set trap to call cleanup on script exit or Ctrl+C
trap cleanup EXIT INT TERM

echo "============================================"
echo "Starting GGEUR Distributed Training"
echo "Dataset: Office-Home (4 clients)"
echo "============================================"
echo "PID file: $PID_FILE"
echo ""

# Clear old PID file
rm -f "$PID_FILE"

# Check manifest file
if [ ! -f "$MANIFEST_PATH" ]; then
    echo "Manifest not found at $MANIFEST_PATH"
    echo "Please run the following command first:"
    echo "  python scripts/tools/prepare_officehome_shards.py \\"
    echo "    --root /root/OfficeHomeDataset_10072016 \\"
    echo "    --output data/officehome/shards \\"
    echo "    --clients 4 \\"
    echo "    --lds-alpha 0.1"
    exit 1
fi

# Check shard directories
for shard in "${CLIENT_SHARDS[@]}"; do
    if [ ! -d "$shard" ]; then
        echo "Missing shard directory: $shard"
        echo "Please run scripts/tools/prepare_officehome_shards.py to generate shards."
        exit 1
    fi
    if [ ! -f "$shard/train.json" ]; then
        echo "Shard $shard is incomplete (train.json missing)."
        exit 1
    fi
done

echo "Starting processes..."
echo ""

# Start server
echo "1. Starting server..."
python federatedscope/main.py \
    --cfg scripts/distributed_scripts/distributed_configs/distributed_ggeur_officehome_server.yaml &
SERVER_PID=$!
echo $SERVER_PID >> "$PID_FILE"
echo "   Server PID: $SERVER_PID"

# Wait for server to initialize
sleep 5

# Start client 1 (Art domain)
echo "2. Starting client 1 (Art domain)..."
python federatedscope/main.py \
    --cfg scripts/distributed_scripts/distributed_configs/distributed_ggeur_officehome_client_1.yaml &
CLIENT1_PID=$!
echo $CLIENT1_PID >> "$PID_FILE"
echo "   Client 1 PID: $CLIENT1_PID"

sleep 2

# Start client 2 (Clipart domain)
echo "3. Starting client 2 (Clipart domain)..."
python federatedscope/main.py \
    --cfg scripts/distributed_scripts/distributed_configs/distributed_ggeur_officehome_client_2.yaml &
CLIENT2_PID=$!
echo $CLIENT2_PID >> "$PID_FILE"
echo "   Client 2 PID: $CLIENT2_PID"

sleep 2

# Start client 3 (Product domain)
echo "4. Starting client 3 (Product domain)..."
python federatedscope/main.py \
    --cfg scripts/distributed_scripts/distributed_configs/distributed_ggeur_officehome_client_3.yaml &
CLIENT3_PID=$!
echo $CLIENT3_PID >> "$PID_FILE"
echo "   Client 3 PID: $CLIENT3_PID"

sleep 2

# Start client 4 (Real World domain)
echo "5. Starting client 4 (Real World domain)..."
python federatedscope/main.py \
    --cfg scripts/distributed_scripts/distributed_configs/distributed_ggeur_officehome_client_4.yaml &
CLIENT4_PID=$!
echo $CLIENT4_PID >> "$PID_FILE"
echo "   Client 4 PID: $CLIENT4_PID"

echo ""
echo "============================================"
echo "All processes started successfully!"
echo "============================================"
echo ""
echo "Logs are saved to the exp/ directory (configured in YAML files)"
echo ""
echo "Cache structure:"
echo "  clip_feature_cache/"
echo "    client_1/  ← Client 1 (Art)"
echo "    client_2/  ← Client 2 (Clipart)"
echo "    client_3/  ← Client 3 (Product)"
echo "    client_4/  ← Client 4 (Real World)"
echo "    server/    ← Server cache"
echo ""
echo "To stop all processes:"
echo "  - Press Ctrl+C in this terminal, or"
echo "  - Run: ./scripts/distributed_scripts/stop_distributed_ggeur_officehome.sh"
echo ""
echo "Waiting for processes to complete (Press Ctrl+C to stop)..."
echo ""

# Wait for all background processes
wait
