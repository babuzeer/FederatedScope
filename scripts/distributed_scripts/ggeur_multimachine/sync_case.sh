#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  cat <<'EOF'
Usage:
  sync_case.sh <case_or_run_dir> <remote_ssh> [remote_repo_dir]

Example:
  bash sync_case.sh runs/demo office-client /root/autodl-tmp/FederatedScope
EOF
  exit 1
fi

LOCAL_PATH="$1"
REMOTE_SSH="$2"
REMOTE_REPO="${3:-/root/autodl-tmp/FederatedScope}"

if [ ! -e "$LOCAL_PATH" ]; then
  echo "Local path does not exist: $LOCAL_PATH"
  exit 1
fi

rel_path="$(python - "$LOCAL_PATH" <<'PY'
import os
import sys
print(os.path.relpath(os.path.abspath(sys.argv[1]), os.getcwd()).replace('\\', '/'))
PY
)"

remote_parent="$REMOTE_REPO/$(dirname "$rel_path")"
ssh -o BatchMode=yes "$REMOTE_SSH" "mkdir -p '$remote_parent'"
scp -r "$LOCAL_PATH" "$REMOTE_SSH:$remote_parent/"
echo "synced $LOCAL_PATH -> $REMOTE_SSH:$remote_parent/"
