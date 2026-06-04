#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/FederatedScope}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/fs/bin/python}"
RUN_ID="${RUN_ID:-officehome_vit_headonly_system_60c_100r_gen20_$(date +%Y%m%d_%H%M%S)}"
CLIENT_NUM="${CLIENT_NUM:-60}"
TOTAL_ROUNDS="${TOTAL_ROUNDS:-100}"
GEN_NUM="${GEN_NUM:-20}"
SKIP_ROUND0_IF_AUG_CACHE="${SKIP_ROUND0_IF_AUG_CACHE:-0}"
case "${SKIP_ROUND0_IF_AUG_CACHE,,}" in
  1|true|yes|y|on)
    SKIP_ROUND0_IF_AUG_CACHE_CFG="True"
    ;;
  *)
    SKIP_ROUND0_IF_AUG_CACHE_CFG="False"
    ;;
esac

cd "$REPO_DIR"

CFG="scripts/example_configs/ggeur_headonly_system/officehome_lds_vit_headonly_fedavg_gen20.yaml"
RUN_ROOT="exp/headonly_system/runs"
RUN_DIR="$RUN_ROOT/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
METRIC_DIR="$RUN_DIR/metrics"
CONFIG_DIR="$RUN_DIR/configs"
SYSTEM_DIR="$RUN_DIR/system"

mkdir -p "$LOG_DIR" "$METRIC_DIR" "$CONFIG_DIR" "$SYSTEM_DIR"
cp "$CFG" "$CONFIG_DIR/input_config.yaml"

{
  echo "run_id=$RUN_ID"
  echo "repo_dir=$REPO_DIR"
  echo "cfg=$CFG"
  echo "run_dir=$RUN_DIR"
  echo "client_num=$CLIENT_NUM"
  echo "total_rounds=$TOTAL_ROUNDS"
  echo "gen_num=$GEN_NUM"
  echo "skip_round0_if_aug_cache=$SKIP_ROUND0_IF_AUG_CACHE_CFG"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo
  echo "[git]"
  git rev-parse HEAD || true
  git status --short || true
  echo
  echo "[gpu_before]"
  nvidia-smi || true
} | tee "$SYSTEM_DIR/run_info.log"

set +e
"$PYTHON_BIN" federatedscope/main.py \
  --cfg "$CFG" \
  federate.client_num "$CLIENT_NUM" \
  federate.sample_client_num "$CLIENT_NUM" \
  federate.total_round_num "$TOTAL_ROUNDS" \
  ggeur.num_generated_per_sample "$GEN_NUM" \
  ggeur.num_generated_per_prototype "$GEN_NUM" \
  ggeur.target_size_per_class "$GEN_NUM" \
  ggeur.headonly_skip_round0_if_augmented_cache_exists "$SKIP_ROUND0_IF_AUG_CACHE_CFG" \
  outdir "$RUN_DIR/fs_out" \
  expname "$RUN_ID" \
  > "$LOG_DIR/server.log" 2>&1
STATUS=$?
set -e

{
  echo "[gpu_after]"
  nvidia-smi || true
  echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "exit_status=$STATUS"
} | tee -a "$SYSTEM_DIR/run_info.log"

"$PYTHON_BIN" scripts/parse_headonly_system_metrics.py \
  "$LOG_DIR" \
  --output "$METRIC_DIR/metrics_summary.json" \
  > "$METRIC_DIR/metrics_summary.pretty.json" || true

echo "$RUN_DIR"
exit "$STATUS"
