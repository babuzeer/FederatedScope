#!/bin/bash

PID_FILE="/tmp/federatedscope_distributed_pids.txt"

echo "============================================"
echo "Stopping FederatedScope Distributed Training"
echo "============================================"

if [ -f "$PID_FILE" ]; then
    echo "Found PID file, stopping tracked processes..."
    killed_count=0
    while read pid; do
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "  Killing process $pid"
            kill -9 "$pid" 2>/dev/null || true
            killed_count=$((killed_count + 1))
        else
            echo "  Process $pid already stopped"
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
    echo "Stopped $killed_count tracked processes"
else
    echo "No PID file found"
fi

echo ""
echo "Checking for remaining FederatedScope processes..."
remaining=$(ps aux | grep "federatedscope/main.py.*distributed" | grep -v grep | wc -l)

if [ "$remaining" -gt 0 ]; then
    echo "Found $remaining remaining processes, cleaning up..."
    pkill -9 -f "federatedscope/main.py.*distributed" 2>/dev/null || true
    sleep 1
    echo "Cleanup complete"
else
    echo "No remaining processes found"
fi

echo ""
echo "============================================"
echo "All FederatedScope processes stopped"
echo "============================================"
