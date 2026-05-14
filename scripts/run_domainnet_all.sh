#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNNER="${RUNNER:-federatedscope/main.py}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-exp/domainnet_4domains/_batch_runner/${TIMESTAMP}}"
MODE="${1:-all}"
ON_ERROR="${ON_ERROR:-continue}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$LOG_ROOT"

VIT_CFGS=(
  "scripts/example_configs/ggeur_baseline_vit_domainnet/domainnet_4domains_vit_ggeur_fedavg.yaml"
  "scripts/example_configs/ggeur_baseline_vit_domainnet/domainnet_4domains_vit_fedavg.yaml"
  "scripts/example_configs/ggeur_baseline_vit_domainnet/domainnet_4domains_vit_fedprox.yaml"
  "scripts/example_configs/ggeur_baseline_vit_domainnet/domainnet_4domains_vit_fedproto.yaml"
  "scripts/example_configs/ggeur_baseline_vit_domainnet/domainnet_4domains_vit_fedopt.yaml"
  "scripts/example_configs/ggeur_baseline_vit_domainnet/domainnet_4domains_vit_moon.yaml"
)

CNN_CFGS=(
  "scripts/example_configs/ggeur_baseline_cnn_domainnet/domainnet_4domains_cnn_ggeur_fedavg.yaml"
  "scripts/example_configs/ggeur_baseline_cnn_domainnet/domainnet_4domains_cnn_fedavg.yaml"
  "scripts/example_configs/ggeur_baseline_cnn_domainnet/domainnet_4domains_cnn_fedprox.yaml"
  "scripts/example_configs/ggeur_baseline_cnn_domainnet/domainnet_4domains_cnn_fedproto.yaml"
  "scripts/example_configs/ggeur_baseline_cnn_domainnet/domainnet_4domains_cnn_fedopt.yaml"
  "scripts/example_configs/ggeur_baseline_cnn_domainnet/domainnet_4domains_cnn_moon.yaml"
)

MLP_CFGS=(
  "scripts/example_configs/ggeur_baseline_mlp_domainnet/domainnet_4domains_mlp_ggeur_fedavg.yaml"
  "scripts/example_configs/ggeur_baseline_mlp_domainnet/domainnet_4domains_mlp_fedavg.yaml"
  "scripts/example_configs/ggeur_baseline_mlp_domainnet/domainnet_4domains_mlp_fedprox.yaml"
  "scripts/example_configs/ggeur_baseline_mlp_domainnet/domainnet_4domains_mlp_fedproto.yaml"
  "scripts/example_configs/ggeur_baseline_mlp_domainnet/domainnet_4domains_mlp_fedopt.yaml"
  "scripts/example_configs/ggeur_baseline_mlp_domainnet/domainnet_4domains_mlp_moon.yaml"
)

ALL_CFGS=()

append_group() {
  local group_name="$1"
  shift
  local cfg
  for cfg in "$@"; do
    ALL_CFGS+=("${group_name}|${cfg}")
  done
}

case "$MODE" in
  vit)
    append_group "vit" "${VIT_CFGS[@]}"
    ;;
  cnn)
    append_group "cnn" "${CNN_CFGS[@]}"
    ;;
  mlp)
    append_group "mlp" "${MLP_CFGS[@]}"
    ;;
  all)
    append_group "vit" "${VIT_CFGS[@]}"
    append_group "cnn" "${CNN_CFGS[@]}"
    append_group "mlp" "${MLP_CFGS[@]}"
    ;;
  *)
    echo "Usage: bash scripts/run_domainnet_all.sh [all|vit|cnn|mlp]"
    echo "Optional env vars:"
    echo "  PYTHON_BIN=python"
    echo "  RUNNER=federatedscope/main.py"
    echo "  LOG_ROOT=exp/domainnet_4domains/_batch_runner/xxx"
    echo "  ON_ERROR=continue|stop"
    echo "  DRY_RUN=0|1"
    exit 1
    ;;
esac

SUCCESS_COUNT=0
FAIL_COUNT=0
SUMMARY_FILE="${LOG_ROOT}/summary.txt"
PLAN_FILE="${LOG_ROOT}/run_plan.txt"
TOTAL_COUNT="${#ALL_CFGS[@]}"

echo "Batch start: $(date)" | tee "$SUMMARY_FILE"
echo "Mode: ${MODE}" | tee -a "$SUMMARY_FILE"
echo "Logs: ${LOG_ROOT}" | tee -a "$SUMMARY_FILE"
echo "On error: ${ON_ERROR}" | tee -a "$SUMMARY_FILE"
echo "Dry run: ${DRY_RUN}" | tee -a "$SUMMARY_FILE"
echo "Total configs: ${TOTAL_COUNT}" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

: > "$PLAN_FILE"
for item in "${ALL_CFGS[@]}"; do
  group_name="${item%%|*}"
  cfg_path="${item#*|}"
  echo "${group_name} ${cfg_path}" >> "$PLAN_FILE"
done

echo "Run plan saved to: ${PLAN_FILE}" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

run_one() {
  local index="$1"
  local group_name="$2"
  local cfg_path="$3"
  local cfg_name
  local log_file
  cfg_name="$(basename "$cfg_path" .yaml)"
  log_file="${LOG_ROOT}/${group_name}__${cfg_name}.log"

  if [[ ! -f "${cfg_path}" ]]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "[$(date '+%F %T')] FAIL  ${cfg_path} (missing config file)" | tee -a "$SUMMARY_FILE"
    if [[ "${ON_ERROR}" == "stop" ]]; then
      echo "Stopping on missing config." | tee -a "$SUMMARY_FILE"
      exit 1
    fi
    return
  fi

  echo "[$(date '+%F %T')] START [${index}/${TOTAL_COUNT}] ${cfg_path}" | tee -a "$SUMMARY_FILE"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[$(date '+%F %T')] DRY   ${PYTHON_BIN} ${RUNNER} --cfg ${cfg_path}" | tee -a "$SUMMARY_FILE"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    return
  fi

  ${PYTHON_BIN} "${RUNNER}" --cfg "${cfg_path}" > "${log_file}" 2>&1
  local status=$?

  if [[ $status -eq 0 ]]; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    echo "[$(date '+%F %T')] OK    [${index}/${TOTAL_COUNT}] ${cfg_path}" | tee -a "$SUMMARY_FILE"
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "[$(date '+%F %T')] FAIL  [${index}/${TOTAL_COUNT}] ${cfg_path} (exit=${status}) log=${log_file}" | tee -a "$SUMMARY_FILE"
    if [[ "${ON_ERROR}" == "stop" ]]; then
      echo "Stopping on first failure." | tee -a "$SUMMARY_FILE"
      exit "$status"
    fi
  fi
}

item=""
INDEX=0
for item in "${ALL_CFGS[@]}"; do
  INDEX=$((INDEX + 1))
  group_name="${item%%|*}"
  cfg_path="${item#*|}"
  run_one "${INDEX}" "${group_name}" "${cfg_path}"
done

echo "" | tee -a "$SUMMARY_FILE"
echo "Batch end: $(date)" | tee -a "$SUMMARY_FILE"
echo "Success: ${SUCCESS_COUNT}" | tee -a "$SUMMARY_FILE"
echo "Fail: ${FAIL_COUNT}" | tee -a "$SUMMARY_FILE"

if [[ $FAIL_COUNT -gt 0 ]]; then
  exit 1
fi

exit 0
