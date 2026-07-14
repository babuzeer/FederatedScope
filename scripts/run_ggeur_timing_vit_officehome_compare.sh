#!/usr/bin/env bash
# Run two prioritized GGEUR timing experiments:
# OfficeHome + ViT + 60 clients, generated amount 20 and 50.

set -euo pipefail

if [[ $# -gt 0 && "${1}" != --* ]]; then
  REPO_DIR="${1}"
  shift
else
  REPO_DIR="${REPO_DIR:-$(pwd)}"
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/fs/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-exp/ggeur_timing}"
RUN_PREFIX="${RUN_PREFIX:-ggeur_timing_vit_officehome_$(date +%Y%m%d_%H%M%S)}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CASE_NAME="${CASE_NAME:-officehome__vit__ggeur}"
MATRIX_FILE="${MATRIX_FILE:-scripts/thirdparty_accuracy_cases.yaml}"
TOTAL_ROUNDS="${TOTAL_ROUNDS:-1}"
TIMING_DIR="${REPO_DIR}/${OUTPUT_ROOT}/${RUN_PREFIX}"
LAUNCH_LOG="${TIMING_DIR}/timing_compare.log"

mkdir -p "${TIMING_DIR}"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

echo "[ggeur-timing] started_at=$(date --iso-8601=seconds)"
echo "[ggeur-timing] repo_dir=${REPO_DIR}"
echo "[ggeur-timing] run_prefix=${RUN_PREFIX}"
echo "[ggeur-timing] python_bin=${PYTHON_BIN}"
echo "[ggeur-timing] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[ggeur-timing] total_rounds=${TOTAL_ROUNDS}"

cd "${REPO_DIR}" || exit 2
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

run_one() {
  local gen="$1"
  shift
  local run_id="${RUN_PREFIX}_gen${gen}"
  local run_dir="${REPO_DIR}/${OUTPUT_ROOT}/${run_id}"
  echo "[ggeur-timing] run gen=${gen} run_id=${run_id} started_at=$(date --iso-8601=seconds)"

  "${PYTHON_BIN}" scripts/run_thirdparty_accuracy_case.py \
    --matrix "${MATRIX_FILE}" \
    --repo-dir "${REPO_DIR}" \
    --python-bin "${PYTHON_BIN}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-id "${run_id}" \
    --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
    --case "${CASE_NAME}" \
    --no-resume \
    --opt "federate.total_round_num=${TOTAL_ROUNDS}" \
    --opt "ggeur.num_generated_per_sample=${gen}" \
    --opt "ggeur.num_generated_per_prototype=${gen}" \
    --opt "ggeur.target_size_per_class=${gen}" \
    --opt "ggeur.reuse_augmented_feature_cache=False" \
    --opt "ggeur.save_augmented_feature_cache=True" \
    --opt "ggeur.augmented_feature_cache_version=timing_vit_officehome_gen${gen}_v1" \
    "$@"

  local stdout_log="${run_dir}/cases/${CASE_NAME}/stdout.log"
  mkdir -p "${TIMING_DIR}/parsed"
  "${PYTHON_BIN}" scripts/parse_ggeur_timing.py \
    --log "${stdout_log}" \
    --output-dir "${TIMING_DIR}/parsed" \
    --run-label "gen${gen}"
  echo "[ggeur-timing] run gen=${gen} finished_at=$(date --iso-8601=seconds)"
}

run_one 20 "$@"
run_one 50 "$@"

"${PYTHON_BIN}" - "${TIMING_DIR}/parsed" "${TIMING_DIR}/timing_compare_aggregate.csv" <<'PY'
import csv
import sys
from pathlib import Path

parsed_dir = Path(sys.argv[1])
output = Path(sys.argv[2])
rows = []
for path in sorted(parsed_dir.glob("*_timing_aggregate.csv")):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

if rows:
    fields = sorted({key for row in rows for key in row.keys()})
    preferred = ["run_label", "source", "stage", "events"]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
print(f"compare={output}")
PY

echo "[ggeur-timing] finished_at=$(date --iso-8601=seconds)"
echo "[ggeur-timing] output_dir=${TIMING_DIR}"
