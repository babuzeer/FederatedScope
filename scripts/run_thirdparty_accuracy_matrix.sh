#!/usr/bin/env bash
# Run the full third-party accuracy matrix, plot per dataset/model comparisons,
# and package the lightweight artifacts for copying back to the local machine.

set -uo pipefail

if [[ $# -gt 0 && "${1}" != --* ]]; then
  REPO_DIR="${1}"
  shift
else
  REPO_DIR="${REPO_DIR:-$(pwd)}"
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/fs/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-exp/ggeur_accuracy_reruns}"
RUN_ID="${RUN_ID:-thirdparty_accuracy_$(date +%Y%m%d_%H%M%S)}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE_OVERRIDE="${DEVICE_OVERRIDE:-0}"
THIRDPARTY_DATASETS="${THIRDPARTY_DATASETS:-officehome domainnet}"

RUN_DIR="${REPO_DIR}/${OUTPUT_ROOT}/${RUN_ID}"
LOG_PATH="${RUN_DIR}/thirdparty_accuracy_matrix.log"
PLOT_DIR="${RUN_DIR}/plots"

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[thirdparty] started_at=$(date --iso-8601=seconds)"
echo "[thirdparty] repo_dir=${REPO_DIR}"
echo "[thirdparty] run_id=${RUN_ID}"
echo "[thirdparty] python_bin=${PYTHON_BIN}"
echo "[thirdparty] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[thirdparty] device_override=${DEVICE_OVERRIDE}"
echo "[thirdparty] datasets=${THIRDPARTY_DATASETS}"

cd "${REPO_DIR}" || exit 2
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

QUEUE_ARGS=(
  --repo-dir "${REPO_DIR}"
  --python-bin "${PYTHON_BIN}"
  --output-root "${OUTPUT_ROOT}"
  --run-id "${RUN_ID}"
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}"
)
if [[ -n "${DEVICE_OVERRIDE}" ]]; then
  QUEUE_ARGS+=(--opt "device=${DEVICE_OVERRIDE}")
fi
for dataset in ${THIRDPARTY_DATASETS}; do
  QUEUE_ARGS+=(--dataset "${dataset}")
done
QUEUE_ARGS+=(
  --opt "ggeur.use_feature_cache=True"
  --opt "ggeur.reuse_augmented_feature_cache=True"
  --opt "ggeur.save_augmented_feature_cache=True"
)

"${PYTHON_BIN}" scripts/run_ggeur_accuracy_matrix_queue.py \
  "${QUEUE_ARGS[@]}" \
  "$@"
QUEUE_RC=$?

mkdir -p "${PLOT_DIR}"
PLOT_RC=0
COMBOS=(
  "officehome cnn"
  "officehome mixer"
  "officehome vit"
  "domainnet cnn"
  "domainnet mlp"
  "domainnet vit"
  "pacs cnn"
  "pacs mixer"
  "pacs vit"
)

SELECTED_COMBOS=()
for combo in "${COMBOS[@]}"; do
  read -r dataset model <<< "${combo}"
  for selected_dataset in ${THIRDPARTY_DATASETS}; do
    if [[ "${dataset}" == "${selected_dataset}" ]]; then
      SELECTED_COMBOS+=("${combo}")
      break
    fi
  done
done

for combo in "${SELECTED_COMBOS[@]}"; do
  read -r dataset model <<< "${combo}"
  echo "[thirdparty] plotting dataset=${dataset} model=${model}"
  "${PYTHON_BIN}" scripts/plot_ggeur_accuracy_curves.py \
    --input-dir "${RUN_DIR}/cases" \
    --output-dir "${PLOT_DIR}" \
    --dataset "${dataset}" \
    --model "${model}" \
    --combined \
    --allow-missing
  rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "[thirdparty] plot_failed dataset=${dataset} model=${model} rc=${rc}"
    PLOT_RC=1
  fi
done

ARCHIVE_LIST="${RUN_DIR}/archive_files.txt"
: > "${ARCHIVE_LIST}"
for name in manifest.json state.json summary.json summary.csv experiment_log.md thirdparty_accuracy_matrix.log; do
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

echo "[thirdparty] queue_rc=${QUEUE_RC}"
echo "[thirdparty] plot_rc=${PLOT_RC}"
echo "[thirdparty] results_archive=${RESULTS_ARCHIVE}"
echo "[thirdparty] finished_at=$(date --iso-8601=seconds)"

if [[ ${QUEUE_RC} -ne 0 ]]; then
  exit "${QUEUE_RC}"
fi
exit "${PLOT_RC}"
