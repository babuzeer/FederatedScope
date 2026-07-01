#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/root/autodl-tmp/FederatedScope"
PYTHON_BIN="/root/miniconda3/envs/fs/bin/python"
OUTPUT_ROOT="exp/ggeur_accuracy_reruns"
RUN_ID=""
WAIT_PATTERN="federatedscope/main.py --cfg scripts/attack_exp_scripts/privacy_attack"
WAIT_INTERVAL=300

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --wait-pattern)
      WAIT_PATTERN="$2"
      shift 2
      ;;
    --wait-interval)
      WAIT_INTERVAL="$2"
      shift 2
      ;;
    --no-wait)
      WAIT_PATTERN=""
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="accuracy_rerun_2d3m6m_$(date +%Y%m%d_%H%M%S)"
fi

cd "$REPO_DIR"
RUN_DIR="$REPO_DIR/$OUTPUT_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"

echo "launcher_started $(date --iso-8601=seconds)"
echo "run_id=$RUN_ID"
echo "run_dir=$RUN_DIR"

if [[ -n "$WAIT_PATTERN" ]]; then
  while pgrep -af "$WAIT_PATTERN" >/dev/null; do
    echo "waiting_existing_training $(date --iso-8601=seconds) pattern=$WAIT_PATTERN"
    sleep "$WAIT_INTERVAL"
  done
fi

echo "queue_start $(date --iso-8601=seconds)"
"$PYTHON_BIN" scripts/run_ggeur_accuracy_matrix_queue.py \
  --repo-dir "$REPO_DIR" \
  --python-bin "$PYTHON_BIN" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$RUN_ID"
echo "queue_finished $(date --iso-8601=seconds)"
