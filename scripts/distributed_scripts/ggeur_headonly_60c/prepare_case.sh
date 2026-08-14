#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RUN_ID="${RUN_ID:-officehome_vit_headonly_60c_$(date +%Y%m%d_%H%M%S)}"
CASE_ROOT="${CASE_ROOT:-scripts/distributed_scripts/ggeur_multimachine/runs/$RUN_ID}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/OfficeHomeDataset_10072016}"
MANIFEST_ROOT="${MANIFEST_ROOT:-exp/headonly_system/manifests/$RUN_ID}"
MANIFEST_MODE="${MANIFEST_MODE:-manifest-only}"
PREPARE_MANIFESTS="${PREPARE_MANIFESTS:-1}"

SERVER_HOST="${SERVER_HOST:?Set SERVER_HOST to the public server host/IP that clients connect to}"
SERVER_BIND_HOST="${SERVER_BIND_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-55051}"

CLIENT_HOST="${CLIENT_HOST:-}"
CLIENT_HOSTS="${CLIENT_HOSTS:-}"
if [ -z "$CLIENT_HOST" ] && [ -z "$CLIENT_HOSTS" ]; then
  echo "Set CLIENT_HOST or CLIENT_HOSTS to the public client callback host/IP reachable by the server." >&2
  exit 1
fi
CLIENT_BIND_HOST="${CLIENT_BIND_HOST:-0.0.0.0}"
CLIENT_BIND_HOSTS="${CLIENT_BIND_HOSTS:-}"
CLIENT_HOST_ASSIGNMENT="${CLIENT_HOST_ASSIGNMENT:-block}"
CLIENT_PORT_BASE="${CLIENT_PORT_BASE:-56000}"
CLIENT_BIND_PORT_BASE="${CLIENT_BIND_PORT_BASE:-$CLIENT_PORT_BASE}"
CLIENT_ADVERTISE_PORT_BASE="${CLIENT_ADVERTISE_PORT_BASE:-$CLIENT_PORT_BASE}"
CLIENT_ADVERTISE_PORTS="${CLIENT_ADVERTISE_PORTS:-}"
CLIENT_BIND_PORTS="${CLIENT_BIND_PORTS:-}"

CLIENT_NUM="${CLIENT_NUM:-60}"
DOMAINS="${DOMAINS:-Art,Clipart,Product,Real_World}"
CLIENTS_PER_DOMAIN="${CLIENTS_PER_DOMAIN:-15}"
TOTAL_ROUNDS="${TOTAL_ROUNDS:-100}"
SAMPLE_CLIENTS="${SAMPLE_CLIENTS:-$CLIENT_NUM}"
GEN_NUM="${GEN_NUM:-20}"
NUM_GENERATED_PER_SAMPLE="${NUM_GENERATED_PER_SAMPLE:-$GEN_NUM}"
NUM_GENERATED_PER_PROTOTYPE="${NUM_GENERATED_PER_PROTOTYPE:-$GEN_NUM}"
TARGET_SIZE_PER_CLASS="${TARGET_SIZE_PER_CLASS:-$GEN_NUM}"
USE_LDS="${USE_LDS:-1}"
LDS_ALPHA="${LDS_ALPHA:-0.1}"
LDS_SEED="${LDS_SEED:-42}"
SEED="${SEED:-42}"

FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-exp/ggeur_headonly_real_cache/officehome_vitb16_60c_gen20_fcache_v1}"
HEADONLY_CACHE_VERSION="${HEADONLY_CACHE_VERSION:-officehome_vitb16_60c_gen20_fcache_v1}"
SKIP_ROUND0_IF_AUG_CACHE="${SKIP_ROUND0_IF_AUG_CACHE:-1}"
HEADONLY_EVAL_MODE="${HEADONLY_EVAL_MODE:-client}"
STAGE_TIMEOUT="${STAGE_TIMEOUT:-7200}"
JOIN_TIMEOUT_SECONDS="${JOIN_TIMEOUT_SECONDS:-900}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-64}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-0}"
USE_GPU="${USE_GPU:-1}"

if [ "$((CLIENTS_PER_DOMAIN * 4))" -ne "$CLIENT_NUM" ]; then
  echo "This final OfficeHome entry expects 4 domains x CLIENTS_PER_DOMAIN = CLIENT_NUM." >&2
  echo "Got CLIENTS_PER_DOMAIN=$CLIENTS_PER_DOMAIN CLIENT_NUM=$CLIENT_NUM" >&2
  exit 1
fi

cd "$REPO_DIR"
mkdir -p "$(dirname "$CASE_ROOT")" "$MANIFEST_ROOT"

if [ "$PREPARE_MANIFESTS" = "1" ]; then
  manifest_args=(
    --source-root "$DATA_ROOT"
    --output-root "$MANIFEST_ROOT"
    --client-num "$CLIENT_NUM"
    --splits 0.7,0.0,0.3
    --seed "$SEED"
    --lds-alpha "$LDS_ALPHA"
    --lds-seed "$LDS_SEED"
    --domains "$DOMAINS"
    --mode "$MANIFEST_MODE"
  )
  if [ "$USE_LDS" = "1" ]; then
    manifest_args+=(--use-lds)
  fi
  "$PYTHON_BIN" scripts/prepare_officehome_client_manifests.py \
    "${manifest_args[@]}"
fi

gen_args=(
  --output-root "$CASE_ROOT"
  --run-id "$RUN_ID"
  --dataset officehome
  --model vit
  --method fedavg
  --head-only
  --server-host "$SERVER_HOST"
  --server-bind-host "$SERVER_BIND_HOST"
  --server-port "$SERVER_PORT"
  --client-host "${CLIENT_HOST:-$CLIENT_HOSTS}"
  --client-hosts "$CLIENT_HOSTS"
  --client-host-assignment "$CLIENT_HOST_ASSIGNMENT"
  --client-bind-host "$CLIENT_BIND_HOST"
  --client-bind-hosts "$CLIENT_BIND_HOSTS"
  --client-port-base "$CLIENT_PORT_BASE"
  --client-bind-port-base "$CLIENT_BIND_PORT_BASE"
  --client-advertise-port-base "$CLIENT_ADVERTISE_PORT_BASE"
  --client-advertise-ports "$CLIENT_ADVERTISE_PORTS"
  --client-bind-ports "$CLIENT_BIND_PORTS"
  --clients "$CLIENT_NUM"
  --sample-clients "$SAMPLE_CLIENTS"
  --rounds "$TOTAL_ROUNDS"
  --stage-timeout "$STAGE_TIMEOUT"
  --join-timeout-seconds "$JOIN_TIMEOUT_SECONDS"
  --device "$DEVICE"
  --seed "$SEED"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --extract-batch-size "$EXTRACT_BATCH_SIZE"
  --num-generated-per-sample "$NUM_GENERATED_PER_SAMPLE"
  --num-generated-per-prototype "$NUM_GENERATED_PER_PROTOTYPE"
  --target-size-per-class "$TARGET_SIZE_PER_CLASS"
  --lds-alpha "$LDS_ALPHA"
  --feature-cache-dir "$FEATURE_CACHE_DIR"
  --headonly-cache-version "$HEADONLY_CACHE_VERSION"
  --officehome-manifest-base "$MANIFEST_ROOT"
  --officehome-domains "$DOMAINS"
  --headonly-eval-mode "$HEADONLY_EVAL_MODE"
)
if [ "$USE_LDS" = "1" ]; then
  gen_args+=(--use-lds)
else
  gen_args+=(--no-use-lds)
fi
if [ "$USE_GPU" = "0" ]; then
  gen_args+=(--no-use-gpu)
fi
if [ "$SKIP_ROUND0_IF_AUG_CACHE" = "0" ]; then
  gen_args+=(--no-headonly-skip-round0-if-augmented-cache-exists)
fi

"$PYTHON_BIN" scripts/distributed_scripts/ggeur_multimachine/generate_configs.py \
  "${gen_args[@]}" | tee "$CASE_ROOT/generate_output.json"

cat > "$CASE_ROOT/final_headonly_60c_run_info.txt" <<EOF
run_id=$RUN_ID
case_root=$CASE_ROOT
data_root=$DATA_ROOT
manifest_root=$MANIFEST_ROOT
client_num=$CLIENT_NUM
domains=$DOMAINS
clients_per_domain=$CLIENTS_PER_DOMAIN
total_rounds=$TOTAL_ROUNDS
gen_num=$GEN_NUM
feature_cache_dir=$FEATURE_CACHE_DIR
headonly_cache_version=$HEADONLY_CACHE_VERSION
skip_round0_if_aug_cache=$SKIP_ROUND0_IF_AUG_CACHE
server_host=$SERVER_HOST
server_bind_host=$SERVER_BIND_HOST
server_port=$SERVER_PORT
client_host=$CLIENT_HOST
client_hosts=$CLIENT_HOSTS
client_bind_host=$CLIENT_BIND_HOST
client_port_base=$CLIENT_PORT_BASE
client_bind_port_base=$CLIENT_BIND_PORT_BASE
client_advertise_port_base=$CLIENT_ADVERTISE_PORT_BASE
created_at=$(date '+%Y-%m-%d %H:%M:%S %z')
EOF

echo "$CASE_ROOT/officehome_vit_fedavg_headonly"
