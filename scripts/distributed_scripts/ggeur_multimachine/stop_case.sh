#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <case_dir>"
  exit 1
fi

CASE_DIR="$(cd "$1" && pwd)"
PID_DIR="$CASE_DIR/pids"

if [ ! -d "$PID_DIR" ]; then
  echo "No pid dir: $PID_DIR"
  exit 0
fi

for pid_file in "$PID_DIR"/*.pid; do
  [ -f "$pid_file" ] || continue
  pid="$(cat "$pid_file")"
  if ps -p "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
  fi
done

sleep 3

for pid_file in "$PID_DIR"/*.pid; do
  [ -f "$pid_file" ] || continue
  pid="$(cat "$pid_file")"
  if ps -p "$pid" >/dev/null 2>&1; then
    kill -9 "$pid" >/dev/null 2>&1 || true
  fi
done

rm -f "$PID_DIR"/*.pid 2>/dev/null || true
echo "stopped $CASE_DIR"
