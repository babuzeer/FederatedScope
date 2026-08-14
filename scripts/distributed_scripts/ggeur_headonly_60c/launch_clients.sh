#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
CASE_DIR="${1:?Usage: $0 <case_dir>}"

cd "$REPO_DIR"
exec bash scripts/distributed_scripts/ggeur_multimachine/launch_clients.sh \
  "$CASE_DIR"
