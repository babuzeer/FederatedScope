#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
TASK_DIR="$ROOT_DIR/scripts/distributed_scripts/ggeur_headonly_system_officehome_vit"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/fs/bin/python}"
RUN_ID="${RUN_ID:-officehome_vit_headonly_system_distributed_short_$(date +%Y%m%d_%H%M%S)}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-51251}"
CLIENT_NUM="${CLIENT_NUM:-4}"
TOTAL_ROUNDS="${TOTAL_ROUNDS:-5}"
GEN_NUM="${GEN_NUM:-20}"

RUN_DIR="$ROOT_DIR/exp/headonly_system/runs/$RUN_ID"
CONFIG_DIR="$RUN_DIR/configs"
LOG_DIR="$RUN_DIR/logs"
PID_DIR="$RUN_DIR/pids"
METRIC_DIR="$RUN_DIR/metrics"
SYSTEM_DIR="$RUN_DIR/system"

mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$PID_DIR" "$METRIC_DIR" "$SYSTEM_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

"$PYTHON_BIN" - "$CONFIG_DIR" "$SERVER_HOST" "$SERVER_PORT" "$CLIENT_NUM" "$TOTAL_ROUNDS" "$GEN_NUM" <<'PY'
import sys
from pathlib import Path

config_dir = Path(sys.argv[1])
server_host = sys.argv[2]
server_port = int(sys.argv[3])
client_num = int(sys.argv[4])
total_rounds = int(sys.argv[5])
gen_num = int(sys.argv[6])

common = f"""
use_gpu: True
device: 0
seed: 42
verbose: 1

federate:
  method: 'ggeur'
  mode: 'distributed'
  client_num: {client_num}
  total_round_num: {total_rounds}
  sample_client_num: {client_num}
  make_global_eval: False
  online_aggr: False

data:
  type: 'office-home'
  root: '/root/autodl-tmp/datasets/OfficeHomeDataset_10072016'
  splits: [0.7, 0.0, 0.3]

dataloader:
  batch_size: 32
  num_workers: 0

model:
  type: 'ggeur_mlp'
  num_classes: 65

train:
  local_update_steps: 1
  optimizer:
    type: 'Adam'
    lr: 0.0001

eval:
  freq: 1
  metrics: ['acc']
  split: ['test']
  best_res_update_round_wise_key: 'test_acc'

trainer:
  type: 'ggeur'

ggeur:
  use: True
  head_only_mode: True
  head_only_after_round0: True
  headonly_cache_version: 'officehome_vit_headonly_distributed_short'
  headonly_eval_mode: 'server'
  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  clip_model_path: '/root/autodl-tmp/models/open_clip_vitb16.bin'
  embedding_dim: 512
  freeze_backbone: True
  use_feature_cache: True
  unload_extractor_after_cache: True
  use_fp16_extraction: True
  extract_batch_size: 64
  feature_cache_dir: 'exp/headonly_system/cache/officehome_vit_distributed_short'
  num_generated_per_sample: {gen_num}
  num_generated_per_prototype: {gen_num}
  target_size_per_class: {gen_num}
  mlp_hidden_dim: 0
  mlp_dropout: 0.0
  use_cross_client_prototypes: True
  statistics_round: 0
  distributed_stage_timeout: 3600
  use_fedproto: False
  use_lds: True
  lds_alpha: 0.1
  lds_seed: 42
  use_cnn_distillation: False
  use_feature_alignment: False
  use_separated_training: False
  use_end_to_end_finetune: False
  use_promptfl: False
"""

server = common + f"""
distribute:
  use: True
  role: 'server'
  server_host: '{server_host}'
  server_port: {server_port}

outdir: 'exp/headonly_system/distributed_short/server'
expname: 'headonly_system_distributed_short_server'
"""
config_dir.joinpath('server.yaml').write_text(server, encoding='utf-8')

for client_id in range(1, client_num + 1):
    client = common + f"""
distribute:
  use: True
  role: 'client'
  server_host: '{server_host}'
  server_port: {server_port}
  client_host: '127.0.0.1'
  client_port: {server_port + client_id}
  data_idx: {client_id}

outdir: 'exp/headonly_system/distributed_short/client_{client_id}'
expname: 'headonly_system_distributed_short_client_{client_id}'
"""
    config_dir.joinpath(f'client_{client_id}.yaml').write_text(
        client, encoding='utf-8')
PY

{
  echo "run_id=$RUN_ID"
  echo "root_dir=$ROOT_DIR"
  echo "run_dir=$RUN_DIR"
  echo "server_host=$SERVER_HOST"
  echo "server_port=$SERVER_PORT"
  echo "client_num=$CLIENT_NUM"
  echo "total_rounds=$TOTAL_ROUNDS"
  echo "gen_num=$GEN_NUM"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  nvidia-smi || true
} | tee "$SYSTEM_DIR/run_info.log"

cleanup() {
  for pid_file in "$PID_DIR"/*.pid; do
    [ -f "$pid_file" ] || continue
    pid="$(cat "$pid_file")"
    kill "$pid" >/dev/null 2>&1 || true
  done
}
trap cleanup INT TERM

nohup "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" --cfg "$CONFIG_DIR/server.yaml" \
  > "$LOG_DIR/server.log" 2>&1 &
echo "$!" > "$PID_DIR/server.pid"
sleep 8

for client_id in $(seq 1 "$CLIENT_NUM"); do
  nohup "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" --cfg "$CONFIG_DIR/client_${client_id}.yaml" \
    > "$LOG_DIR/client_${client_id}.log" 2>&1 &
  echo "$!" > "$PID_DIR/client_${client_id}.pid"
  sleep 3
done

deadline=$(( $(date +%s) + 7200 ))
while true; do
  alive=0
  for pid_file in "$PID_DIR"/*.pid; do
    [ -f "$pid_file" ] || continue
    pid="$(cat "$pid_file")"
    if ps -p "$pid" >/dev/null 2>&1; then
      alive=1
      break
    fi
  done
  [ "$alive" -eq 0 ] && break
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "timeout" | tee -a "$SYSTEM_DIR/run_info.log"
    cleanup
    break
  fi
  sleep 5
done

"$PYTHON_BIN" scripts/parse_headonly_system_metrics.py \
  "$LOG_DIR" \
  --output "$METRIC_DIR/metrics_summary.json" \
  > "$METRIC_DIR/metrics_summary.pretty.json" || true

{
  echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  nvidia-smi || true
} | tee -a "$SYSTEM_DIR/run_info.log"

grep -q "Training finished" "$LOG_DIR/server.log"
echo "$RUN_DIR"
