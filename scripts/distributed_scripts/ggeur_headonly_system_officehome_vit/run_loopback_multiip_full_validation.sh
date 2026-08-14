#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/fs/bin/python}"

RUN_ID="${RUN_ID:-officehome_vit_headonly_loopback_multiip_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/exp/headonly_system/loopback_multiip_runs}"
RUN_DIR="$RUN_ROOT/$RUN_ID"
CONFIG_DIR="$RUN_DIR/configs"
LOG_DIR="$RUN_DIR/logs"
PID_DIR="$RUN_DIR/pids"
METRIC_DIR="$RUN_DIR/metrics"
SYSTEM_DIR="$RUN_DIR/system"

# Avoid an address ending in ".10": some config dumps in this codebase have
# displayed it as ".1", which makes log-based validation ambiguous.
SERVER_HOST="${SERVER_HOST:-127.77.0.11}"
CLIENT_IP_BASE="${CLIENT_IP_BASE:-127.77.0}"
CLIENT_IP_START="${CLIENT_IP_START:-101}"
SERVER_PORT="${SERVER_PORT:-51251}"
CLIENT_PORT_BASE="${CLIENT_PORT_BASE:-52251}"

CLIENT_NUM="${CLIENT_NUM:-4}"
LAUNCH_CLIENT_NUM="${LAUNCH_CLIENT_NUM:-$CLIENT_NUM}"
TOTAL_ROUNDS="${TOTAL_ROUNDS:-5}"
SAMPLE_CLIENT_NUM="${SAMPLE_CLIENT_NUM:-$CLIENT_NUM}"
GEN_NUM="${GEN_NUM:-20}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/OfficeHomeDataset_10072016}"
OFFICEHOME_MANIFEST_BASE="${OFFICEHOME_MANIFEST_BASE:-}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-/root/autodl-tmp/models/open_clip_vitb16.bin}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-exp/headonly_system/cache/officehome_lds_vit_gen20}"
HEADONLY_CACHE_VERSION="${HEADONLY_CACHE_VERSION:-officehome_vit_headonly_loopback_${RUN_ID}_gen${GEN_NUM}}"
HEADONLY_EVAL_MODE="${HEADONLY_EVAL_MODE:-server}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-64}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
STAGE_TIMEOUT="${STAGE_TIMEOUT:-3600}"
JOIN_TIMEOUT_SECONDS="${JOIN_TIMEOUT_SECONDS:-60}"
MIN_STATISTICS_CLIENTS="${MIN_STATISTICS_CLIENTS:-0}"
MIN_AUGMENTATION_CLIENTS="${MIN_AUGMENTATION_CLIENTS:-0}"
MIN_TRAIN_UPDATES="${MIN_TRAIN_UPDATES:-0}"
CASE_TIMEOUT="${CASE_TIMEOUT:-7200}"
CLIENT_START_GAP="${CLIENT_START_GAP:-3}"
SERVER_START_WAIT="${SERVER_START_WAIT:-8}"
DEVICE="${DEVICE:-0}"
EXTRA_CLIENTS="${EXTRA_CLIENTS:-0}"
CLIENT_FAIL_SPECS="${CLIENT_FAIL_SPECS:-}"
CLIENT_START_DELAYS="${CLIENT_START_DELAYS:-}"

RUN_QPS="${RUN_QPS:-1}"
QPS_SUBSERVERS="${QPS_SUBSERVERS:-4}"
QPS_CLIENT_GROUPS="${QPS_CLIENT_GROUPS:-$QPS_SUBSERVERS}"
QPS_CLIENTS_PER_GROUP="${QPS_CLIENTS_PER_GROUP:-250}"
QPS_BASE_PORT="${QPS_BASE_PORT:-39451}"

mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$PID_DIR" "$METRIC_DIR" "$SYSTEM_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

client_ip() {
  local idx="$1"
  echo "${CLIENT_IP_BASE}.$((CLIENT_IP_START + idx - 1))"
}

cleanup() {
  for pid_file in "$PID_DIR"/*.pid; do
    [ -f "$pid_file" ] || continue
    pid="$(cat "$pid_file")"
    kill "$pid" >/dev/null 2>&1 || true
  done
}
trap cleanup INT TERM

"$PYTHON_BIN" - "$CONFIG_DIR" "$RUN_DIR" "$SERVER_HOST" "$SERVER_PORT" \
  "$CLIENT_IP_BASE" "$CLIENT_IP_START" "$CLIENT_PORT_BASE" "$CLIENT_NUM" \
  "$LAUNCH_CLIENT_NUM" "$EXTRA_CLIENTS" \
  "$TOTAL_ROUNDS" "$SAMPLE_CLIENT_NUM" "$GEN_NUM" "$DATA_ROOT" \
  "$OFFICEHOME_MANIFEST_BASE" "$CLIP_MODEL_PATH" "$FEATURE_CACHE_DIR" "$HEADONLY_CACHE_VERSION" \
  "$EXTRACT_BATCH_SIZE" "$BATCH_SIZE" "$NUM_WORKERS" "$STAGE_TIMEOUT" \
  "$JOIN_TIMEOUT_SECONDS" "$MIN_STATISTICS_CLIENTS" \
  "$MIN_AUGMENTATION_CLIENTS" "$MIN_TRAIN_UPDATES" "$DEVICE" \
  "$CLIENT_FAIL_SPECS" "$HEADONLY_EVAL_MODE" <<'PY'
import sys
from pathlib import Path

config_dir = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
server_host = sys.argv[3]
server_port = int(sys.argv[4])
client_ip_base = sys.argv[5]
client_ip_start = int(sys.argv[6])
client_port_base = int(sys.argv[7])
client_num = int(sys.argv[8])
launch_client_num = int(sys.argv[9])
extra_clients = int(sys.argv[10])
total_rounds = int(sys.argv[11])
sample_client_num = int(sys.argv[12])
gen_num = int(sys.argv[13])
data_root = sys.argv[14]
officehome_manifest_base = sys.argv[15]
clip_model_path = sys.argv[16]
feature_cache_dir = sys.argv[17]
headonly_cache_version = sys.argv[18]
extract_batch_size = int(sys.argv[19])
batch_size = int(sys.argv[20])
num_workers = int(sys.argv[21])
stage_timeout = int(sys.argv[22])
join_timeout_seconds = int(sys.argv[23])
min_statistics_clients = int(sys.argv[24])
min_augmentation_clients = int(sys.argv[25])
min_train_updates = int(sys.argv[26])
device = int(sys.argv[27])
client_fail_specs = sys.argv[28]
headonly_eval_mode = sys.argv[29]

fail_specs = {}
for item in client_fail_specs.split(','):
    item = item.strip()
    if not item:
        continue
    parts = item.split(':')
    if len(parts) != 3:
        raise ValueError(
            f"Invalid CLIENT_FAIL_SPECS item {item!r}; expected "
            "client_id:stage:round")
    fail_specs[int(parts[0])] = (parts[1], int(parts[2]))

def client_ip(idx):
    return f"{client_ip_base}.{client_ip_start + idx - 1}"

common = f"""
use_gpu: True
device: {device}
seed: 42
verbose: 1

federate:
  method: 'ggeur'
  mode: 'distributed'
  client_num: {client_num}
  total_round_num: {total_rounds}
  sample_client_num: {sample_client_num}
  make_global_eval: False
  online_aggr: False

data:
  type: 'office-home'
  root: '{data_root}'
  splits: [0.7, 0.0, 0.3]

dataloader:
  batch_size: {batch_size}
  num_workers: {num_workers}

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
  headonly_cache_version: '{headonly_cache_version}'
  headonly_eval_mode: '{headonly_eval_mode}'
  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  clip_model_path: '{clip_model_path}'
  embedding_dim: 512
  freeze_backbone: True
  use_feature_cache: True
  unload_extractor_after_cache: True
  use_fp16_extraction: True
  extract_batch_size: {extract_batch_size}
  feature_cache_dir: '{feature_cache_dir}'
  num_generated_per_sample: {gen_num}
  num_generated_per_prototype: {gen_num}
  target_size_per_class: {gen_num}
  mlp_hidden_dim: 0
  mlp_dropout: 0.0
  use_cross_client_prototypes: True
  statistics_round: 0
  distributed_stage_timeout: {stage_timeout}
  use_fedproto: False
  use_lds: True
  lds_alpha: 0.1
  lds_seed: 42
  use_cnn_distillation: False
  use_feature_alignment: False
  use_separated_training: False
  use_end_to_end_finetune: False
  use_promptfl: False
  min_statistics_clients: {min_statistics_clients}
  min_augmentation_clients: {min_augmentation_clients}
  min_train_updates: {min_train_updates}
  officehome_manifest_path: ''

ggeur_headonly:
  use: True
  client_total: {client_num}
  sample_clients_per_round: {sample_client_num}
  num_sub_servers: 1
  feature_cache_version: '{headonly_cache_version}'
"""

server = common + f"""
distribute:
  use: True
  join_timeout_seconds: {join_timeout_seconds}
  grpc_max_send_message_length: 314572800
  grpc_max_receive_message_length: 314572800
  grpc_enable_http_proxy: False
  role: 'server'
  server_host: '{server_host}'
  server_port: {server_port}

outdir: '{run_dir}/server_out'
expname: 'headonly_loopback_multiip_server'
"""
config_dir.joinpath('server.yaml').write_text(server, encoding='utf-8')

for client_id in range(1, launch_client_num + extra_clients + 1):
    host = client_ip(client_id)
    fail_stage, fail_round = fail_specs.get(client_id, ('', -1))
    # Extra clients are used to validate late/unknown join behavior. Reuse a
    # valid data split so the client reaches the communication path instead of
    # failing earlier because data_idx exceeds the configured client_num.
    data_idx = ((client_id - 1) % client_num) + 1
    client_data_root = data_root
    manifest_path = ''
    if officehome_manifest_base:
        client_data_root = str(
            Path(officehome_manifest_base) / f'client_{data_idx:06d}')
        manifest_path = str(Path(client_data_root) / 'client_manifest.json')
    client = common + f"""
distribute:
  use: True
  join_timeout_seconds: {join_timeout_seconds}
  grpc_max_send_message_length: 314572800
  grpc_max_receive_message_length: 314572800
  grpc_enable_http_proxy: False
  role: 'client'
  server_host: '{server_host}'
  server_port: {server_port}
  client_host: '{host}'
  client_port: {client_port_base + client_id}
  data_idx: {data_idx}

outdir: '{run_dir}/client_{client_id}_out'
expname: 'headonly_loopback_multiip_client_{client_id}'
"""
    if officehome_manifest_base:
        client = client.replace(
            f"root: '{data_root}'",
            f"root: '{client_data_root}'")
        client = client.replace(
            "  officehome_manifest_path: ''",
            f"  officehome_manifest_path: '{manifest_path}'")
    if fail_stage:
        client = client.replace(
            "\nggeur_headonly:\n",
            f"  fail_after_stage: '{fail_stage}'\n"
            f"  fail_on_round: {fail_round}\n"
            "\nggeur_headonly:\n",
        )
    config_dir.joinpath(f'client_{client_id}.yaml').write_text(
        client, encoding='utf-8')
PY

{
  echo "run_id=$RUN_ID"
  echo "run_dir=$RUN_DIR"
  echo "root_dir=$ROOT_DIR"
  echo "python_bin=$PYTHON_BIN"
  echo "server_host=$SERVER_HOST"
  echo "server_port=$SERVER_PORT"
  echo "client_ip_base=$CLIENT_IP_BASE"
  echo "client_ip_start=$CLIENT_IP_START"
  echo "client_port_base=$CLIENT_PORT_BASE"
  echo "client_num=$CLIENT_NUM"
  echo "launch_client_num=$LAUNCH_CLIENT_NUM"
  echo "extra_clients=$EXTRA_CLIENTS"
  echo "sample_client_num=$SAMPLE_CLIENT_NUM"
  echo "total_rounds=$TOTAL_ROUNDS"
  echo "gen_num=$GEN_NUM"
  echo "join_timeout_seconds=$JOIN_TIMEOUT_SECONDS"
  echo "min_statistics_clients=$MIN_STATISTICS_CLIENTS"
  echo "min_augmentation_clients=$MIN_AUGMENTATION_CLIENTS"
  echo "min_train_updates=$MIN_TRAIN_UPDATES"
  echo "client_fail_specs=$CLIENT_FAIL_SPECS"
  echo "client_start_delays=$CLIENT_START_DELAYS"
  echo "data_root=$DATA_ROOT"
  echo "officehome_manifest_base=$OFFICEHOME_MANIFEST_BASE"
  echo "headonly_eval_mode=$HEADONLY_EVAL_MODE"
  echo "feature_cache_dir=$FEATURE_CACHE_DIR"
  echo "headonly_cache_version=$HEADONLY_CACHE_VERSION"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo
  echo "[client_ips]"
  for client_id in $(seq 1 "$((LAUNCH_CLIENT_NUM + EXTRA_CLIENTS))"); do
    echo "client_${client_id}=$(client_ip "$client_id"):$((CLIENT_PORT_BASE + client_id))"
  done
  echo
  echo "[gpu]"
  nvidia-smi || true
} | tee "$SYSTEM_DIR/run_info.log"

nohup "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" --cfg "$CONFIG_DIR/server.yaml" \
  > "$LOG_DIR/server.log" 2>&1 &
echo "$!" > "$PID_DIR/server.pid"
sleep "$SERVER_START_WAIT"

IFS=',' read -r -a start_delays <<< "$CLIENT_START_DELAYS"
for client_id in $(seq 1 "$((LAUNCH_CLIENT_NUM + EXTRA_CLIENTS))"); do
  nohup "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" --cfg "$CONFIG_DIR/client_${client_id}.yaml" \
    > "$LOG_DIR/client_${client_id}.log" 2>&1 &
  echo "$!" > "$PID_DIR/client_${client_id}.pid"
  delay="${start_delays[$((client_id - 1))]:-$CLIENT_START_GAP}"
  sleep "$delay"
done

deadline=$(( $(date +%s) + CASE_TIMEOUT ))
timed_out=0
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
    timed_out=1
    echo "timeout" | tee -a "$SYSTEM_DIR/run_info.log"
    cleanup
    break
  fi
  sleep 5
done

"$PYTHON_BIN" "$ROOT_DIR/scripts/parse_headonly_system_metrics.py" \
  "$LOG_DIR" \
  --output "$METRIC_DIR/training_metrics_summary.json" \
  > "$METRIC_DIR/training_metrics_summary.pretty.json" || true

if [ "$RUN_QPS" = "1" ]; then
  OUT_ROOT="$RUN_DIR/qps" \
  RUN_ID="download_qps_loopback_multiip" \
  SERVER_HOST="$SERVER_HOST" \
  SUBSERVERS="$QPS_SUBSERVERS" \
  CLIENT_GROUPS="$QPS_CLIENT_GROUPS" \
  CLIENTS_PER_GROUP="$QPS_CLIENTS_PER_GROUP" \
  BASE_PORT="$QPS_BASE_PORT" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash "$ROOT_DIR/scripts/distributed_scripts/ggeur_headonly_download_qps/run_loopback_multiip_download_qps.sh" \
    > "$LOG_DIR/qps.log" 2>&1 || true
fi

{
  echo
  echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "timed_out=$timed_out"
  echo
  echo "[gpu_after]"
  nvidia-smi || true
} | tee -a "$SYSTEM_DIR/run_info.log"

if [ "$timed_out" -ne 0 ]; then
  exit 124
fi

grep -q "Training finished" "$LOG_DIR/server.log"
echo "$RUN_DIR"
