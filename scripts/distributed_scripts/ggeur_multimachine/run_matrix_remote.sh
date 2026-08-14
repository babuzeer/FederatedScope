#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  cat <<'EOF'
Usage:
  run_matrix_remote.sh <run_dir> <server_ssh> <client_ssh>

Runs every case listed in <run_dir>/matrix_manifest.json sequentially.
EOF
  exit 1
fi

RUN_DIR="$1"
SERVER_SSH="$2"
CLIENT_SSH="$3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$RUN_DIR/matrix_manifest.json" ]; then
  echo "Missing matrix manifest: $RUN_DIR/matrix_manifest.json"
  exit 1
fi

python - "$RUN_DIR/matrix_manifest.json" <<'PY' | while read -r case_dir; do
import json
import sys
from pathlib import Path
data = json.load(open(sys.argv[1], encoding='utf-8'))
for case in data['cases']:
    print(Path(case['server_config']).parents[1])
PY
  rel_case="$(python - "$case_dir" <<'PY'
import os, sys
print(os.path.relpath(os.path.abspath(sys.argv[1]), os.getcwd()).replace('\\', '/'))
PY
)"
  echo "===== running $rel_case ====="
  bash "$SCRIPT_DIR/run_remote_case.sh" "$rel_case" "$SERVER_SSH" "$CLIENT_SSH"
done
