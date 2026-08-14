#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_remote_case.sh <case_dir> <server_ssh> <client_ssh>

Environment:
  REPO_DIR              Remote repository path. Default: /root/autodl-tmp/FederatedScope
  PYTHON_BIN            Remote Python. Default: /root/miniconda3/envs/fs/bin/python
  SERVER_READY_TIMEOUT  Seconds to wait for server listen log. Default: 300
  CASE_TIMEOUT          Seconds to wait for all processes. Default: 10800
  CLIENT_START_GAP      Seconds between client launches. Default: 2

This script assumes the generated case_dir exists under REPO_DIR on both
machines. Use rsync/scp/git to sync the repo and generated configs first.
EOF
}

if [ "$#" -ne 3 ]; then
  usage
  exit 1
fi

CASE_DIR="$1"
SERVER_SSH="$2"
CLIENT_SSH="$3"
REPO_DIR="${REPO_DIR:-/root/autodl-tmp/FederatedScope}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/fs/bin/python}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-300}"
CASE_TIMEOUT="${CASE_TIMEOUT:-10800}"
CLIENT_START_GAP="${CLIENT_START_GAP:-2}"

SCRIPT_DIR="scripts/distributed_scripts/ggeur_multimachine"
REMOTE_CASE="$REPO_DIR/$CASE_DIR"

remote() {
  local host="$1"
  shift
  ssh -o BatchMode=yes "$host" "$@"
}

cleanup() {
  echo "Cleaning up remote processes..."
  remote "$CLIENT_SSH" "cd '$REPO_DIR' && bash '$SCRIPT_DIR/stop_case.sh' '$REMOTE_CASE'" || true
  remote "$SERVER_SSH" "cd '$REPO_DIR' && bash '$SCRIPT_DIR/stop_case.sh' '$REMOTE_CASE'" || true
}

trap cleanup EXIT INT TERM

echo "Starting server on $SERVER_SSH"
remote "$SERVER_SSH" "cd '$REPO_DIR' && PYTHON_BIN='$PYTHON_BIN' bash '$SCRIPT_DIR/launch_server.sh' '$REMOTE_CASE'"

echo "Waiting for server listen log..."
remote "$SERVER_SSH" "cd '$REPO_DIR' && '$PYTHON_BIN' '$SCRIPT_DIR/wait_for_log.py' '$REMOTE_CASE/logs/server.log' 'Listen to' '$SERVER_READY_TIMEOUT'"

echo "Starting clients on $CLIENT_SSH"
remote "$CLIENT_SSH" "cd '$REPO_DIR' && PYTHON_BIN='$PYTHON_BIN' CLIENT_START_GAP='$CLIENT_START_GAP' bash '$SCRIPT_DIR/launch_clients.sh' '$REMOTE_CASE'"

echo "Waiting for completion..."
remote "$SERVER_SSH" "cd '$REPO_DIR' && '$PYTHON_BIN' '$SCRIPT_DIR/wait_case.py' '$REMOTE_CASE' '$CASE_TIMEOUT'"
remote "$CLIENT_SSH" "cd '$REPO_DIR' && '$PYTHON_BIN' '$SCRIPT_DIR/wait_case.py' '$REMOTE_CASE' '$CASE_TIMEOUT'"

echo "Collecting server summary..."
remote "$SERVER_SSH" "cd '$REPO_DIR' && '$PYTHON_BIN' '$SCRIPT_DIR/summarize_case.py' '$REMOTE_CASE'"

echo "Collecting client summary..."
remote "$CLIENT_SSH" "cd '$REPO_DIR' && '$PYTHON_BIN' '$SCRIPT_DIR/summarize_case.py' '$REMOTE_CASE'"

trap - EXIT INT TERM
echo "Remote case completed. Processes should already be exited."
