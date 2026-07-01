#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <case_dir>"
  exit 1
fi

CASE_DIR="$(cd "$1" && pwd)"
PID_DIR="$CASE_DIR/pids"
LOG_DIR="$CASE_DIR/logs"

echo "case_dir=$CASE_DIR"
if [ -d "$PID_DIR" ]; then
  for pid_file in "$PID_DIR"/*.pid; do
    [ -f "$pid_file" ] || continue
    name="$(basename "$pid_file" .pid)"
    pid="$(cat "$pid_file")"
    if ps -p "$pid" >/dev/null 2>&1; then
      echo "$name pid=$pid alive"
    else
      echo "$name pid=$pid exited"
    fi
  done
fi

if [ -d "$LOG_DIR" ]; then
  for log in "$LOG_DIR"/*.log; do
    [ -f "$log" ] || continue
    echo "--- $(basename "$log") tail ---"
    tail -n 8 "$log" || true
  done
fi
