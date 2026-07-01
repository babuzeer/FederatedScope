#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CASE_DIR="${CASE_DIR:-scripts/distributed_scripts/headonly_hierarchical_qps/runs/headonly60c-1sub-paramqps-20260625}"
PYTHON_BIN="${PYTHON_BIN:-/root/.local/share/mamba/envs/GGEUR/bin/python}"
RESTART="${RESTART:-1}"

cd "${REPO_ROOT}"

if [[ "${RESTART}" == "1" && -x "${SCRIPT_DIR}/stop_case.sh" ]]; then
  bash "${SCRIPT_DIR}/stop_case.sh" "${CASE_DIR}" || true
fi

"${PYTHON_BIN}" -m py_compile "${SCRIPT_DIR}/headonly_hierarchical_qps.py"

grep '"total_rounds": 20' "${CASE_DIR}/configs/root_server.json"
for cfg in "${CASE_DIR}"/configs/subserver_*.json; do
  [[ -f "${cfg}" ]] || continue
  grep '"total_rounds": 20' "${cfg}"
done

bash "${SCRIPT_DIR}/launch_root_server.sh" "${CASE_DIR}"
for cfg in "${CASE_DIR}"/configs/subserver_*.json; do
  [[ -f "${cfg}" ]] || continue
  name="$(basename "${cfg}")"
  sub_id="${name#subserver_}"
  sub_id="${sub_id%.json}"
  SUBSERVER_ID="${sub_id}" bash "${SCRIPT_DIR}/launch_subserver.sh" "${CASE_DIR}"
done
bash "${SCRIPT_DIR}/status_case.sh" "${CASE_DIR}"
