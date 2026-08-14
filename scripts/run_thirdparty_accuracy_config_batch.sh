#!/usr/bin/env bash
# Run all enabled third-party accuracy cases from the YAML matrix.
#
# This is the batch helper for producing the full result set.  The acceptance
# workflow should still use scripts/run_thirdparty_accuracy_case.py or
# scripts/start_thirdparty_accuracy_case.sh to launch one configured case at a
# time.

set -uo pipefail

if [[ $# -gt 0 && "${1}" != --* ]]; then
  REPO_DIR="${1}"
  shift
else
  REPO_DIR="${REPO_DIR:-$(pwd)}"
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/fs/bin/python}"
MATRIX_FILE="${MATRIX_FILE:-scripts/thirdparty_accuracy_cases.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-exp/ggeur_accuracy_reruns}"
RUN_ID="${RUN_ID:-thirdparty_accuracy_config_$(date +%Y%m%d_%H%M%S)}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_DIR="${REPO_DIR}/${OUTPUT_ROOT}/${RUN_ID}"
LOG_PATH="${RUN_DIR}/thirdparty_accuracy_config_batch.log"
PLOT_DIR="${RUN_DIR}/plots"

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[thirdparty-batch] started_at=$(date --iso-8601=seconds)"
echo "[thirdparty-batch] repo_dir=${REPO_DIR}"
echo "[thirdparty-batch] matrix=${MATRIX_FILE}"
echo "[thirdparty-batch] run_id=${RUN_ID}"
echo "[thirdparty-batch] python_bin=${PYTHON_BIN}"
echo "[thirdparty-batch] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

cd "${REPO_DIR}" || exit 2
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" scripts/run_thirdparty_accuracy_case.py \
  --matrix "${MATRIX_FILE}" \
  --repo-dir "${REPO_DIR}" \
  --python-bin "${PYTHON_BIN}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
  --allow-multiple \
  "$@"
RUN_RC=$?

mkdir -p "${PLOT_DIR}"
"${PYTHON_BIN}" - "${MATRIX_FILE}" "${RUN_DIR}" "${PYTHON_BIN}" <<'PY'
import subprocess
import sys
from pathlib import Path

import yaml

matrix_path = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
python_bin = sys.argv[3]

with matrix_path.open("r", encoding="utf-8") as f:
    matrix = yaml.safe_load(f) or {}

methods = [str(item) for item in matrix.get("methods", [])]
enabled_cases = [
    item for item in matrix.get("cases", [])
    if item.get("enabled", True)
]
plot_groups = matrix.get("plot_groups") or []
plot_rc = 0

for group in plot_groups:
    dataset = str(group["dataset"])
    model = str(group["model"])
    available = {
        str(item["method"])
        for item in enabled_cases
        if str(item["dataset"]) == dataset and str(item["model"]) == model
    }
    group_methods = [method for method in methods if method in available]
    if not group_methods:
        print(f"[thirdparty-batch] skip_plot dataset={dataset} model={model}: no methods")
        continue
    cmd = [
        python_bin,
        "scripts/plot_ggeur_accuracy_curves.py",
        "--input-dir",
        str(run_dir / "cases"),
        "--output-dir",
        str(run_dir / "plots"),
        "--dataset",
        dataset,
        "--model",
        model,
        "--methods",
        *group_methods,
        "--combined",
        "--allow-missing",
    ]
    print(f"[thirdparty-batch] plotting dataset={dataset} model={model}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(
            f"[thirdparty-batch] plot_failed dataset={dataset} "
            f"model={model} rc={proc.returncode}"
        )
        plot_rc = 1

raise SystemExit(plot_rc)
PY
PLOT_RC=$?

ARCHIVE_LIST="${RUN_DIR}/archive_files.txt"
: > "${ARCHIVE_LIST}"
for name in manifest.json state.json summary.json summary.csv experiment_log.md thirdparty_accuracy_config_batch.log; do
  if [[ -f "${RUN_DIR}/${name}" ]]; then
    echo "${name}" >> "${ARCHIVE_LIST}"
  fi
done
if [[ -d "${PLOT_DIR}" ]]; then
  find "${PLOT_DIR}" -type f -printf '%P\n' | sed 's#^#plots/#' >> "${ARCHIVE_LIST}"
fi
if [[ -d "${RUN_DIR}/cases" ]]; then
  find "${RUN_DIR}/cases" -type f \( \
    -name 'stdout.log' -o \
    -name 'exp_print.log' -o \
    -name '*.csv' -o \
    -name '*.json' \
  \) -printf '%P\n' | sed 's#^#cases/#' >> "${ARCHIVE_LIST}"
fi

RESULTS_ARCHIVE="${RUN_DIR}/thirdparty_accuracy_results_${RUN_ID}.tar.gz"
tar -czf "${RESULTS_ARCHIVE}" -C "${RUN_DIR}" -T "${ARCHIVE_LIST}"

echo "[thirdparty-batch] run_rc=${RUN_RC}"
echo "[thirdparty-batch] plot_rc=${PLOT_RC}"
echo "[thirdparty-batch] results_archive=${RESULTS_ARCHIVE}"
echo "[thirdparty-batch] finished_at=$(date --iso-8601=seconds)"

if [[ ${RUN_RC} -ne 0 ]]; then
  exit "${RUN_RC}"
fi
exit "${PLOT_RC}"
