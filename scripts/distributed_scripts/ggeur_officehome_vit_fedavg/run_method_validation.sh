#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    cat <<'EOF'
Usage: run_method_validation.sh <method>

Methods:
  fedprox
  fedopt
  moon
  fedproto
  promptfl
  all
EOF
    exit 1
fi

METHOD="$1"
ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
TASK_DIR="$ROOT_DIR/scripts/distributed_scripts/ggeur_officehome_vit_fedavg"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$METHOD"
PID_DIR="$TASK_DIR/pids_method_validation_$METHOD"
LOG_DIR="$TASK_DIR/logs/method_validation_$RUN_ID"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export PID_DIR
export LOG_DIR

mkdir -p "$PID_DIR" "$LOG_DIR"
rm -f "$PID_DIR"/*.pid 2>/dev/null || true

cleanup() {
    for pid_file in "$PID_DIR"/*.pid; do
        if [ ! -f "$pid_file" ]; then
            continue
        fi
        pid="$(cat "$pid_file")"
        if ps -p "$pid" >/dev/null 2>&1; then
            kill "$pid" >/dev/null 2>&1 || true
        fi
    done
}

trap cleanup EXIT INT TERM

method_port() {
    case "$1" in
        fedprox) echo 51251 ;;
        fedopt) echo 51261 ;;
        moon) echo 51271 ;;
        fedproto) echo 51281 ;;
        promptfl) echo 51291 ;;
        *) echo "Unknown method: $1" >&2; return 1 ;;
    esac
}

expect_log_contains() {
    local file="$1"
    local pattern="$2"
    if ! grep -q "$pattern" "$file"; then
        echo "Expected pattern not found: $pattern" | tee -a "$LOG_DIR/scenario.log"
        echo "Log file: $file" | tee -a "$LOG_DIR/scenario.log"
        return 1
    fi
}

wait_log_contains() {
    local file="$1"
    local pattern="$2"
    local max_wait="${3:-300}"
    local start_ts
    start_ts="$(date +%s)"

    while true; do
        if [ -f "$file" ] && grep -q "$pattern" "$file"; then
            return 0
        fi

        if [ $(( $(date +%s) - start_ts )) -gt "$max_wait" ]; then
            echo "Timed out waiting for pattern: $pattern" | tee -a "$LOG_DIR/scenario.log"
            echo "Log file: $file" | tee -a "$LOG_DIR/scenario.log"
            return 1
        fi

        sleep 2
    done
}

wait_for_all_tracked_processes() {
    local max_wait="${1:-7200}"
    local start_ts
    start_ts="$(date +%s)"

    while true; do
        local alive=0
        for pid_file in "$PID_DIR"/*.pid; do
            if [ ! -f "$pid_file" ]; then
                continue
            fi
            pid="$(cat "$pid_file")"
            if ps -p "$pid" >/dev/null 2>&1; then
                alive=1
                break
            fi
        done

        if [ "$alive" -eq 0 ]; then
            return 0
        fi

        if [ $(( $(date +%s) - start_ts )) -gt "$max_wait" ]; then
            echo "Method validation timed out after ${max_wait}s" | tee -a "$LOG_DIR/scenario.log"
            return 1
        fi

        sleep 5
    done
}

assert_port_free() {
    local port="$1"
    "$PYTHON_BIN" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', $port)); s.close()"
}

create_method_configs() {
    local method="$1"
    local port="$2"
    local method_dir="$LOG_DIR/$method/configs"
    mkdir -p "$method_dir"

    "$PYTHON_BIN" - "$method" "$port" "$TASK_DIR" "$method_dir" <<'PY'
import copy
import os
import sys
import yaml

method, port, task_dir, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]

with open(os.path.join(task_dir, 'officehome_vit_ggeur_fedavg_server.yaml'), 'r', encoding='utf-8') as f:
    server = yaml.safe_load(f)
with open(os.path.join(task_dir, 'officehome_vit_ggeur_fedavg_client_1.yaml'), 'r', encoding='utf-8') as f:
    client_template = yaml.safe_load(f)

def apply_common(cfg, role, client_id=None):
    cfg['federate']['client_num'] = 4
    cfg['federate']['sample_client_num'] = 4
    cfg['federate']['total_round_num'] = 1 if method == 'promptfl' else 2
    cfg.setdefault('train', {})['local_update_steps'] = 1
    cfg.setdefault('dataloader', {})['batch_size'] = 16
    cfg.setdefault('ggeur', {})['distributed_stage_timeout'] = 1800
    cfg['ggeur']['num_generated_per_sample'] = 2
    cfg['ggeur']['num_generated_per_prototype'] = 2
    cfg['ggeur']['target_size_per_class'] = 2
    cfg['ggeur']['use_cross_client_prototypes'] = True
    cfg['ggeur']['use_fedproto'] = False
    cfg['ggeur']['use_moon'] = False
    cfg['ggeur']['use_promptfl'] = False
    cfg['ggeur']['prompt_samples_per_proto'] = 2
    cfg['ggeur']['prompt_length'] = 4
    cfg['ggeur']['prompt_local_epochs'] = 1
    cfg['distribute']['server_port'] = port
    cfg['distribute']['role'] = role
    if role == 'client':
        cfg['distribute']['client_port'] = port + client_id
        cfg['distribute']['data_idx'] = client_id
    cfg['outdir'] = f'exp/distributed_samples/officehome_vit_ggeur_{method}/{role}{("_" + str(client_id)) if client_id else ""}'
    cfg['expname'] = f'officehome_vit_ggeur_{method}_{role}{("_" + str(client_id)) if client_id else ""}'

def apply_method(cfg):
    if method == 'fedprox':
        cfg.setdefault('fedprox', {})['use'] = True
        cfg['fedprox']['mu'] = 0.1
    elif method == 'fedopt':
        cfg.setdefault('fedopt', {})['use'] = True
        cfg['fedopt']['optimizer'] = {'type': 'SGD', 'lr': 0.01}
        cfg['fedopt']['annealing'] = False
    elif method == 'moon':
        cfg['ggeur']['use_moon'] = True
        cfg['ggeur']['moon_mu'] = 1.0
        cfg['ggeur']['moon_temperature'] = 0.5
    elif method == 'fedproto':
        cfg['ggeur']['use_fedproto'] = True
        cfg['ggeur']['proto_weight'] = 0.1
        cfg['ggeur']['proto_distance'] = 'cosine'
        cfg['ggeur']['proto_temperature'] = 0.1
    elif method == 'promptfl':
        cfg['ggeur']['use_promptfl'] = True
        cfg['ggeur']['num_generated_per_sample'] = 1
        cfg['ggeur']['num_generated_per_prototype'] = 1
        cfg['ggeur']['target_size_per_class'] = 1
        cfg['ggeur']['prompt_proximal_mu'] = 0.0
        cfg['ggeur']['prompt_lr'] = 0.002
        cfg['ggeur']['prompt_temperature'] = 0.07
    else:
        raise ValueError(f'Unknown method: {method}')

apply_common(server, 'server')
apply_method(server)
server_path = os.path.join(out_dir, f'{method}_server.yaml')
with open(server_path, 'w', encoding='utf-8') as f:
    yaml.safe_dump(server, f, sort_keys=False)

for client_id in range(1, 5):
    client = copy.deepcopy(client_template)
    apply_common(client, 'client', client_id)
    apply_method(client)
    client_path = os.path.join(out_dir, f'{method}_client_{client_id}.yaml')
    with open(client_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(client, f, sort_keys=False)

print(server_path)
PY
}

validate_method_logs() {
    local method="$1"
    expect_log_contains "$LOG_DIR/$method/server.log" "Training finished"

    case "$method" in
        fedprox)
            expect_log_contains "$LOG_DIR/$method/client_1.log" "FedProx enabled"
            ;;
        fedopt)
            expect_log_contains "$LOG_DIR/$method/server.log" "FedOpt enabled"
            expect_log_contains "$LOG_DIR/$method/server.log" "Performing FedOpt aggregation"
            ;;
        moon)
            expect_log_contains "$LOG_DIR/$method/client_1.log" "MOON enabled"
            ;;
        fedproto)
            expect_log_contains "$LOG_DIR/$method/client_1.log" "FedProto settings - use_fedproto=True"
            ;;
        promptfl)
            expect_log_contains "$LOG_DIR/$method/client_1.log" "PromptLearner ready"
            expect_log_contains "$LOG_DIR/$method/client_1.log" "prompt_loader built"
            ;;
    esac
}

run_one_method() {
    local method="$1"
    local port
    port="$(method_port "$method")"
    local method_log_dir="$LOG_DIR/$method"
    local method_cfg_dir="$method_log_dir/configs"
    mkdir -p "$method_log_dir"

    echo "===== method validation: $method =====" | tee -a "$LOG_DIR/scenario.log"
    echo "port: $port" | tee -a "$LOG_DIR/scenario.log"
    assert_port_free "$port"

    create_method_configs "$method" "$port" >/dev/null

    nohup "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" --cfg "$method_cfg_dir/${method}_server.yaml" \
        > "$method_log_dir/server.log" 2>&1 &
    echo "$!" > "$PID_DIR/${method}_server.pid"

    wait_log_contains "$method_log_dir/server.log" "Listen to 127.0.0.1:${port}" 120

    for client_id in 1 2 3 4; do
        nohup "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" --cfg "$method_cfg_dir/${method}_client_${client_id}.yaml" \
            > "$method_log_dir/client_${client_id}.log" 2>&1 &
        echo "$!" > "$PID_DIR/${method}_client_${client_id}.pid"
        sleep 3
    done

    wait_for_all_tracked_processes 7200
    validate_method_logs "$method"
    assert_port_free "$port"
    rm -f "$PID_DIR"/*.pid 2>/dev/null || true
    echo "Method passed: $method" | tee -a "$LOG_DIR/scenario.log"
}

run_all() {
    local methods=(fedprox fedopt moon fedproto promptfl)
    for method in "${methods[@]}"; do
        run_one_method "$method"
    done
}

cat > "$LOG_DIR/run_info.txt" <<EOF
method: $METHOD
run_id: $RUN_ID
started_at: $(date '+%Y-%m-%d %H:%M:%S %z')
root_dir: $ROOT_DIR
task_dir: $TASK_DIR
python_bin: $PYTHON_BIN
EOF

case "$METHOD" in
    fedprox|fedopt|moon|fedproto|promptfl)
        run_one_method "$METHOD"
        ;;
    all)
        run_all
        ;;
    *)
        echo "Unknown method: $METHOD"
        exit 1
        ;;
esac

echo "Method validation passed: $METHOD" | tee -a "$LOG_DIR/scenario.log"
echo "Logs: $LOG_DIR"
