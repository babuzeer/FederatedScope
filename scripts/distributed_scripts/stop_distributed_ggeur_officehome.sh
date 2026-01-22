#!/bin/bash

# Stop script for GGEUR Office-Home distributed training

PID_FILE="/tmp/federatedscope_ggeur_officehome_pids.txt"

echo "============================================"
echo "Stopping GGEUR Office-Home processes..."
echo "============================================"

if [ -f "$PID_FILE" ]; then
    echo "Reading PIDs from $PID_FILE"
    while read pid; do
        if ps -p $pid > /dev/null 2>&1; then
            echo "Killing process $pid"
            kill -9 $pid 2>/dev/null || true
        else
            echo "Process $pid is not running"
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
    echo "PID file removed"
else
    echo "No PID file found at $PID_FILE"
fi

# Also kill any remaining federatedscope processes related to GGEUR
echo "Killing any remaining GGEUR processes..."
pkill -9 -f "federatedscope/main.py.*ggeur" 2>/dev/null || true

echo ""
echo "============================================"
echo "All GGEUR Office-Home processes stopped"
echo "============================================"
