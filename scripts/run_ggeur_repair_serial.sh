#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/FederatedScope}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/fs/bin/python}"
RUN_ID="${RUN_ID:-repair_serial_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-exp/ggeur_repair_reruns}"
STAGE_TIMEOUT="${STAGE_TIMEOUT:-86400}"

cd "$REPO_DIR"
RUN_DIR="$REPO_DIR/$OUTPUT_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR/cases"

cat > "$RUN_DIR/manifest.tsv" <<EOF
slug	config	extra_opts
officehome__cnn__ggeur	scripts/example_configs/ggeur_baseline_cnn/officehome_lds_cnn_ggeur_fedavg.yaml	device=0 ggeur.distributed_stage_timeout=$STAGE_TIMEOUT
domainnet__cnn__ggeur	scripts/example_configs/ggeur_baseline_cnn_domainnet/domainnet_4domains_cnn_ggeur_fedavg.yaml	device=0 ggeur.distributed_stage_timeout=$STAGE_TIMEOUT
domainnet__cnn__fedproto	scripts/example_configs/ggeur_baseline_cnn_domainnet/domainnet_4domains_cnn_fedproto.yaml	device=0 ggeur.distributed_stage_timeout=$STAGE_TIMEOUT
domainnet__mlp__ggeur	scripts/example_configs/ggeur_baseline_mlp_domainnet/domainnet_4domains_mlp_ggeur_fedavg.yaml	device=0 ggeur.distributed_stage_timeout=$STAGE_TIMEOUT
domainnet__mlp__fedproto	scripts/example_configs/ggeur_baseline_mlp_domainnet/domainnet_4domains_mlp_fedproto.yaml	device=0 ggeur.distributed_stage_timeout=$STAGE_TIMEOUT
EOF

echo "repair_started $(date --iso-8601=seconds)"
echo "run_dir=$RUN_DIR"
echo "stage_timeout=$STAGE_TIMEOUT"

write_summary() {
  {
    echo "slug,status,returncode,last_accuracy,started_at,finished_at,stdout_log"
    for status in "$RUN_DIR"/cases/*/status.json; do
      [[ -e "$status" ]] || continue
      "$PYTHON_BIN" - "$status" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
r = json.loads(p.read_text())
print(",".join([
    str(r.get("slug", "")),
    str(r.get("status", "")),
    str(r.get("returncode", "")),
    str(r.get("last_accuracy", "")),
    str(r.get("started_at", "")),
    str(r.get("finished_at", "")),
    str(p.parent / "stdout.log"),
]))
PY
    done
  } > "$RUN_DIR/summary.csv"
}

run_one() {
  local slug="$1"
  local cfg="$2"
  shift 2
  local case_dir="$RUN_DIR/cases/$slug"
  mkdir -p "$case_dir"
  local log="$case_dir/stdout.log"
  local status="$case_dir/status.json"
  local start_ts
  start_ts="$(date --iso-8601=seconds)"
  echo "{\"slug\":\"$slug\",\"status\":\"running\",\"started_at\":\"$start_ts\"}" > "$status"
  write_summary
  echo "case_start $slug $start_ts"

  local cmd=("$PYTHON_BIN" federatedscope/main.py --cfg "$cfg" outdir "$case_dir" expname run)
  local opt
  for opt in "$@"; do
    cmd+=("${opt%%=*}" "${opt#*=}")
  done

  set +e
  "${cmd[@]}" > "$log" 2>&1
  local rc=$?
  set -e

  local end_ts
  end_ts="$(date --iso-8601=seconds)"
  local final_acc
  final_acc="$(grep -E 'average: final=|Round [0-9]+ MLP Test Accuracy' "$log" | tail -n 1 | sed -E 's/.*average(: final=|: )([0-9.]+).*/\2/' || true)"
  if [[ "$rc" -eq 0 ]]; then
    echo "{\"slug\":\"$slug\",\"status\":\"success\",\"returncode\":$rc,\"started_at\":\"$start_ts\",\"finished_at\":\"$end_ts\",\"last_accuracy\":\"$final_acc\"}" > "$status"
    echo "case_success $slug $end_ts final=$final_acc"
  else
    echo "{\"slug\":\"$slug\",\"status\":\"failed\",\"returncode\":$rc,\"started_at\":\"$start_ts\",\"finished_at\":\"$end_ts\",\"last_accuracy\":\"$final_acc\"}" > "$status"
    echo "case_failed $slug $end_ts rc=$rc final=$final_acc"
  fi
  write_summary
}

tail -n +2 "$RUN_DIR/manifest.tsv" > "$RUN_DIR/manifest.body.tsv"
while IFS=$'\t' read -r slug cfg opts; do
  # shellcheck disable=SC2086
  run_one "$slug" "$cfg" $opts
done < "$RUN_DIR/manifest.body.tsv"

write_summary
echo "repair_finished $(date --iso-8601=seconds)"
echo "summary=$RUN_DIR/summary.csv"
