#!/usr/bin/env bash
set -euo pipefail

if [ "${HEADONLY_ALLOW_PROTOTYPE:-0}" != "1" ]; then
  cat <<'EOF'
This is the old HeadOnly prototype QPS runner. It does not exercise the full
GGEUR server/client round-0 message flow and must not be used as the formal
system validation entry.

Use instead:
  bash scripts/run_headonly_system_officehome_vit_60c_100r_gen20.sh

To run this prototype deliberately, set HEADONLY_ALLOW_PROTOTYPE=1.
EOF
  exit 2
fi

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/FederatedScope}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/fs/bin/python}"
RUN_ID="${RUN_ID:-officehome_vitb16_headonly_60c_100r_gen20_$(date +%Y%m%d_%H%M%S)}"

cd "$REPO_DIR"

RUN_ROOT="exp/ggeur_headonly_real_cache/runs"
RUN_DIR="$RUN_ROOT/$RUN_ID"
CACHE_DIR="exp/ggeur_headonly_real_cache/officehome_vitb16_60c_gen20_fcache_v1"
mkdir -p "$RUN_DIR"

COMMON_OPTS=(
  --cfg scripts/example_configs/ggeur_headonly_hierarchical_local.yaml
  federate.client_num 60
  federate.sample_client_num 60
  ggeur_headonly.client_total 60
  ggeur_headonly.sample_clients_per_round 60
  ggeur_headonly.num_sub_servers 6
  ggeur_headonly.process_workers 16
  ggeur.num_generated_per_sample 20
  ggeur.num_generated_per_prototype 20
  ggeur.target_size_per_class 20
  ggeur_headonly.feature_cache_dir "$CACHE_DIR"
  ggeur_headonly.feature_cache_version officehome_vitb16_60c_gen20_fcache_v1
  ggeur_headonly.run_id "$RUN_ID"
)

{
  echo "run_id=$RUN_ID"
  echo "repo_dir=$REPO_DIR"
  echo "run_dir=$RUN_DIR"
  echo "cache_dir=$CACHE_DIR"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo
  echo "[gpu_before]"
  nvidia-smi || true
} | tee "$RUN_DIR/driver.log"

echo "[stage] generation $(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "$RUN_DIR/status.log" "$RUN_DIR/driver.log"
"$PYTHON_BIN" scripts/headonly_hierarchical_train.py \
  "${COMMON_OPTS[@]}" \
  federate.total_round_num 0 \
  ggeur_headonly.overwrite_feature_cache True \
  ggeur_headonly.output_json "$RUN_DIR/generation_metrics.json" \
  > "$RUN_DIR/generation.log" 2>&1

echo "[stage] training $(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "$RUN_DIR/status.log" "$RUN_DIR/driver.log"
"$PYTHON_BIN" scripts/headonly_hierarchical_train.py \
  "${COMMON_OPTS[@]}" \
  federate.total_round_num 100 \
  ggeur_headonly.overwrite_feature_cache False \
  ggeur_headonly.output_json "$RUN_DIR/training_metrics.json" \
  > "$RUN_DIR/training.log" 2>&1

{
  echo "[gpu_after]"
  nvidia-smi || true
  echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "[done]"
} | tee -a "$RUN_DIR/driver.log" "$RUN_DIR/status.log"

echo "$RUN_DIR"
