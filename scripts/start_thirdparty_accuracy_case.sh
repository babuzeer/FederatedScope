#!/usr/bin/env bash
# Start one third-party accuracy case in the background.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/start_thirdparty_accuracy_case.sh <case-name> [runner args...]" >&2
  echo "example: bash scripts/start_thirdparty_accuracy_case.sh officehome__cnn__ggeur" >&2
  exit 2
fi

CASE_NAME="$1"
shift

REPO_DIR="${REPO_DIR:-$(pwd)}"
MATRIX_FILE="${MATRIX_FILE:-scripts/thirdparty_accuracy_cases.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/fs/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-exp/ggeur_accuracy_reruns}"
RUN_ID="${RUN_ID:-thirdparty_accuracy_manual}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_DIR="${REPO_DIR}/${OUTPUT_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/launcher_logs"
mkdir -p "${LOG_DIR}"

SAFE_CASE="${CASE_NAME//[^A-Za-z0-9_.-]/_}"
LOG_PATH="${LOG_DIR}/${SAFE_CASE}.nohup.log"
PID_PATH="${LOG_DIR}/${SAFE_CASE}.pid"

cd "${REPO_DIR}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

nohup "${PYTHON_BIN}" scripts/run_thirdparty_accuracy_case.py \
  --matrix "${MATRIX_FILE}" \
  --repo-dir "${REPO_DIR}" \
  --python-bin "${PYTHON_BIN}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
  --case "${CASE_NAME}" \
  --plot-group \
  "$@" > "${LOG_PATH}" 2>&1 &

echo "$!" > "${PID_PATH}"
echo "case=${CASE_NAME}"
echo "pid=$(cat "${PID_PATH}")"
echo "run_dir=${RUN_DIR}"
echo "log=${LOG_PATH}"
