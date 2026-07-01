#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/FederatedScope}"
PYTHON_BIN="${PYTHON_BIN:-/root/.local/share/mamba/envs/GGEUR/bin/python}"
CLIENT_NUM="${CLIENT_NUM:-120}"
TOTAL_ROUNDS="${TOTAL_ROUNDS:-20}"
GEN_NUM="${GEN_NUM:-20}"
SEED="${SEED:-42}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/OfficeHomeDataset_10072016}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-/root/autodl-tmp/models/open_clip_vitb16.bin}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-64}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-0}"
STAGE_TIMEOUT="${STAGE_TIMEOUT:-3600}"
SKIP_ROUND0_IF_AUG_CACHE="${SKIP_ROUND0_IF_AUG_CACHE:-0}"

if ! [[ "$CLIENT_NUM" =~ ^[0-9]+$ ]]; then
  echo "CLIENT_NUM must be a positive integer" >&2
  exit 2
fi
if (( CLIENT_NUM <= 0 )); then
  echo "CLIENT_NUM must be positive" >&2
  exit 2
fi
if (( CLIENT_NUM % 4 != 0 )); then
  echo "CLIENT_NUM must be divisible by 4 for OfficeHome LDS client split" >&2
  exit 2
fi

case "${SKIP_ROUND0_IF_AUG_CACHE,,}" in
  1|true|yes|y|on)
    SKIP_ROUND0_IF_AUG_CACHE_CFG="True"
    ;;
  *)
    SKIP_ROUND0_IF_AUG_CACHE_CFG="False"
    ;;
esac

DATE_TAG="$(date +%Y%m%d_%H%M%S)"
CACHE_VERSION="${CACHE_VERSION:-officehome_vitb16_${CLIENT_NUM}c_gen${GEN_NUM}_fcache_v1}"
CACHE_ROOT="${CACHE_ROOT:-exp/ggeur_headonly_real_cache}"
CACHE_DIR="${CACHE_DIR:-${CACHE_ROOT}/${CACHE_VERSION}}"
RUN_ROOT="${RUN_ROOT:-exp/headonly_cache_generation/runs}"
RUN_ID="${RUN_ID:-officehome_vit_cachegen_${CLIENT_NUM}c_${TOTAL_ROUNDS}r_gen${GEN_NUM}_${DATE_TAG}}"

cd "$REPO_DIR"

CFG="${CFG:-scripts/headonly_cache_generation/officehome_vit_cachegen_gen20.yaml}"
RUN_DIR="$RUN_ROOT/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
METRIC_DIR="$RUN_DIR/metrics"
CONFIG_DIR="$RUN_DIR/configs"
SYSTEM_DIR="$RUN_DIR/system"
CACHE_OUTPUT_DIR="$CACHE_DIR/headonly_augmented/$CACHE_VERSION"

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
  echo "seed=$SEED"
  echo "data_root=$DATA_ROOT"
  echo "clip_model_path=$CLIP_MODEL_PATH"
  echo "cache_version=$CACHE_VERSION"
  echo "cache_dir=$CACHE_DIR"
  echo "cache_output_dir=$CACHE_OUTPUT_DIR"
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
  seed "$SEED" \
  federate.client_num "$CLIENT_NUM" \
  federate.sample_client_num "$CLIENT_NUM" \
  federate.total_round_num "$TOTAL_ROUNDS" \
  data.root "$DATA_ROOT" \
  dataloader.num_workers "$DATALOADER_WORKERS" \
  ggeur.clip_model_path "$CLIP_MODEL_PATH" \
  ggeur.extract_batch_size "$EXTRACT_BATCH_SIZE" \
  ggeur.num_generated_per_sample "$GEN_NUM" \
  ggeur.num_generated_per_prototype "$GEN_NUM" \
  ggeur.target_size_per_class "$GEN_NUM" \
  ggeur.distributed_stage_timeout "$STAGE_TIMEOUT" \
  ggeur.feature_cache_dir "$CACHE_DIR" \
  ggeur.headonly_cache_version "$CACHE_VERSION" \
  ggeur.headonly_skip_round0_if_augmented_cache_exists "$SKIP_ROUND0_IF_AUG_CACHE_CFG" \
  outdir "$RUN_DIR/fs_out" \
  expname "$RUN_ID" \
  > "$LOG_DIR/server.log" 2>&1
STATUS=$?
set -e

CACHE_COUNT="0"
if [[ -d "$CACHE_OUTPUT_DIR" ]]; then
  find "$CACHE_OUTPUT_DIR" -maxdepth 1 -type f -name 'office-home_client_*.pt' \
    | sort > "$SYSTEM_DIR/cache_manifest.txt"
  CACHE_COUNT="$(wc -l < "$SYSTEM_DIR/cache_manifest.txt" | tr -d ' ')"
else
  : > "$SYSTEM_DIR/cache_manifest.txt"
fi

{
  echo
  echo "[gpu_after]"
  nvidia-smi || true
  echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "exit_status=$STATUS"
  echo "cache_file_count=$CACHE_COUNT"
  echo "cache_manifest=$SYSTEM_DIR/cache_manifest.txt"
} | tee -a "$SYSTEM_DIR/run_info.log"

"$PYTHON_BIN" scripts/parse_headonly_system_metrics.py \
  "$LOG_DIR" \
  --output "$METRIC_DIR/metrics_summary.json" \
  > "$METRIC_DIR/metrics_summary.pretty.json" || true

echo "$RUN_DIR"
exit "$STATUS"
