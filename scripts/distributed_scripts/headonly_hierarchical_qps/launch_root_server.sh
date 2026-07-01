#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CASE_DIR="${1:-${CASE_DIR:-}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -z "${CASE_DIR}" ]]; then
  echo "usage: $0 <case_dir>" >&2
  exit 2
fi

mkdir -p "${CASE_DIR}/system"
CONFIG="${CASE_DIR}/configs/root_server.json"
LOG="${CASE_DIR}/system/root_server.stdout.log"
PID_FILE="${CASE_DIR}/system/root_server.pid"

if [[ ! -f "${CONFIG}" ]]; then
  echo "missing root config: ${CONFIG}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
nohup "${PYTHON_BIN}" "${SCRIPT_DIR}/headonly_hierarchical_qps.py" root \
  --config "${CONFIG}" > "${LOG}" 2>&1 &
echo "$!" > "${PID_FILE}"
echo "root_server_pid=$(cat "${PID_FILE}")"
echo "root_server_log=${LOG}"
