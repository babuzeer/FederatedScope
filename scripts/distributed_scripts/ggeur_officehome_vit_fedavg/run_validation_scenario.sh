#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    cat <<'EOF'
Usage: run_validation_scenario.sh <scenario>

Scenarios:
  normal                 Run the verified normal server + 4 clients path
  delayed_client         Start clients with larger delays, still within join timeout
  missing_client_timeout Start only 3/4 clients and expect server join timeout
  port_conflict          Occupy the server port and expect server startup failure
  client_wrong_server_port
                         Start one client with a wrong server port and expect connection failure logs
  client_crash_after_join
                         Kill a joined client and expect server statistics-stage timeout
  client_crash_during_round
                         Kill a client during round training and expect server model-update timeout
  repeat_start_stop      Repeatedly start/stop server and verify tracked processes/port cleanup
  system_short           Run short infrastructure checks: port_conflict, client_wrong_server_port, client_crash_after_join, client_crash_during_round, repeat_start_stop
EOF
    exit 1
fi

SCENARIO="$1"
ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
TASK_DIR="$ROOT_DIR/scripts/distributed_scripts/ggeur_officehome_vit_fedavg"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$SCENARIO"
PID_DIR="$TASK_DIR/pids_validation_$SCENARIO"
LOG_DIR="$TASK_DIR/logs/validation_$RUN_ID"
PYTHON_BIN="${PYTHON_BIN:-python}"
SERVER_CFG="$TASK_DIR/officehome_vit_ggeur_fedavg_server.yaml"
SERVER_PORT="${SERVER_PORT:-51051}"
BAD_SERVER_PORT="${BAD_SERVER_PORT:-51999}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export PID_DIR
export LOG_DIR

mkdir -p "$PID_DIR" "$LOG_DIR"
rm -f "$PID_DIR"/*.pid 2>/dev/null || true

cat > "$LOG_DIR/run_info.txt" <<EOF
scenario: $SCENARIO
run_id: $RUN_ID
started_at: $(date '+%Y-%m-%d %H:%M:%S %z')
root_dir: $ROOT_DIR
task_dir: $TASK_DIR
python_bin: $PYTHON_BIN
EOF

cleanup() {
    bash "$TASK_DIR/stop.sh" >/dev/null 2>&1 || true
    if [ -f "$PID_DIR/port_holder.pid" ]; then
        holder_pid="$(cat "$PID_DIR/port_holder.pid")"
        kill "$holder_pid" >/dev/null 2>&1 || true
        rm -f "$PID_DIR/port_holder.pid"
    fi
}

trap cleanup EXIT INT TERM

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
            local pid
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
            echo "Scenario timed out after ${max_wait}s" | tee -a "$LOG_DIR/scenario.log"
            return 1
        fi

        sleep 5
    done
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

expect_log_matches() {
    local file="$1"
    local pattern="$2"
    if ! grep -Eq "$pattern" "$file"; then
        echo "Expected regex not found: $pattern" | tee -a "$LOG_DIR/scenario.log"
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

assert_no_tracked_processes_alive() {
    local failed=0
    for pid_file in "$PID_DIR"/*.pid; do
        if [ ! -f "$pid_file" ]; then
            continue
        fi
        local pid
        pid="$(cat "$pid_file")"
        if ps -p "$pid" >/dev/null 2>&1; then
            echo "Tracked process still alive: $pid from $pid_file" | tee -a "$LOG_DIR/scenario.log"
            failed=1
        fi
    done
    return "$failed"
}

assert_port_free() {
    local port="$1"
    "$PYTHON_BIN" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', $port)); s.close()"
}

run_normal() {
    echo "Running normal validation scenario..." | tee -a "$LOG_DIR/scenario.log"
    bash "$TASK_DIR/start_server.sh"
    sleep 5
    for client_id in 1 2 3 4; do
        bash "$TASK_DIR/start_client.sh" "$client_id"
        sleep 3
    done
    wait_for_all_tracked_processes 7200
    expect_log_contains "$LOG_DIR/server.log" "Training finished"
}

run_delayed_client() {
    echo "Running delayed client validation scenario..." | tee -a "$LOG_DIR/scenario.log"
    bash "$TASK_DIR/start_server.sh"
    sleep 15
    for client_id in 1 2 3 4; do
        bash "$TASK_DIR/start_client.sh" "$client_id"
        sleep 10
    done
    wait_log_contains "$LOG_DIR/server.log" "Assigned ID #4" 180
    wait_log_contains "$LOG_DIR/server.log" "Starting training (Round #0)" 180
    echo "Delayed clients joined and training started; stopping validation processes." | tee -a "$LOG_DIR/scenario.log"
}

run_missing_client_timeout() {
    echo "Running missing client timeout validation scenario..." | tee -a "$LOG_DIR/scenario.log"
    bash "$TASK_DIR/start_server.sh"
    sleep 5
    for client_id in 1 2 3; do
        bash "$TASK_DIR/start_client.sh" "$client_id"
        sleep 3
    done
    wait_for_all_tracked_processes 120 || true
    expect_log_contains "$LOG_DIR/server.log" "Timeout waiting for clients to join"
    expect_log_contains "$LOG_DIR/server.log" "Only 3/4 clients joined"
}

run_port_conflict() {
    echo "Running server port conflict validation scenario..." | tee -a "$LOG_DIR/scenario.log"
    "$PYTHON_BIN" -c "import socket,time; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('127.0.0.1', $SERVER_PORT)); s.listen(1); time.sleep(60)" &
    echo "$!" > "$PID_DIR/port_holder.pid"
    sleep 2

    set +e
    timeout 20 "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" --cfg "$SERVER_CFG" \
        > "$LOG_DIR/server_port_conflict.log" 2>&1
    local status=$?
    set -e

    if [ "$status" -eq 0 ]; then
        echo "Expected server startup to fail, but it exited with status 0" | tee -a "$LOG_DIR/scenario.log"
        return 1
    fi

    if [ "$status" -eq 124 ]; then
        echo "Expected immediate port conflict, but server command timed out" | tee -a "$LOG_DIR/scenario.log"
        return 1
    fi

    expect_log_contains "$LOG_DIR/server_port_conflict.log" "Failed to bind to address"

    if [ -f "$PID_DIR/port_holder.pid" ]; then
        holder_pid="$(cat "$PID_DIR/port_holder.pid")"
        kill "$holder_pid" >/dev/null 2>&1 || true
        rm -f "$PID_DIR/port_holder.pid"
    fi
    assert_port_free "$SERVER_PORT"
}

run_client_wrong_server_port() {
    echo "Running client wrong server port validation scenario..." | tee -a "$LOG_DIR/scenario.log"

    local bad_cfg="$LOG_DIR/client_wrong_server_port.yaml"
    "$PYTHON_BIN" - "$TASK_DIR/officehome_vit_ggeur_fedavg_client_1.yaml" "$bad_cfg" "$BAD_SERVER_PORT" <<'PY'
import sys
import yaml

src, dst, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
cfg['distribute']['server_port'] = port
cfg['distribute']['client_port'] = port + 1
cfg['outdir'] = 'exp/distributed_samples/officehome_vit_ggeur_fedavg/client_wrong_server_port'
cfg['expname'] = 'officehome_vit_ggeur_fedavg_client_wrong_server_port'
with open(dst, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

    set +e
    timeout 45 "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" --cfg "$bad_cfg" \
        > "$LOG_DIR/client_wrong_server_port.log" 2>&1
    local status=$?
    set -e

    if [ "$status" -eq 0 ]; then
        echo "Expected wrong-port client to fail or time out, but it exited with status 0" | tee -a "$LOG_DIR/scenario.log"
        return 1
    fi

    expect_log_matches "$LOG_DIR/client_wrong_server_port.log" \
        "UNAVAILABLE|failed to connect|Connection refused|Socket closed|StatusCode"
}

run_client_crash_after_join() {
    echo "Running client crash after join validation scenario..." | tee -a "$LOG_DIR/scenario.log"

    local crash_server_cfg="$LOG_DIR/server_client_crash_after_join.yaml"
    local crash_client_cfg="$LOG_DIR/client_crash_after_join.yaml"
    "$PYTHON_BIN" - "$SERVER_CFG" "$TASK_DIR/officehome_vit_ggeur_fedavg_client_1.yaml" \
        "$crash_server_cfg" "$crash_client_cfg" "$SERVER_PORT" <<'PY'
import sys
import yaml

server_src, client_src, server_dst, client_dst, port = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
with open(server_src, 'r', encoding='utf-8') as f:
    server = yaml.safe_load(f)
with open(client_src, 'r', encoding='utf-8') as f:
    client = yaml.safe_load(f)

for cfg in (server, client):
    cfg['federate']['client_num'] = 1
    cfg['federate']['sample_client_num'] = 1
    cfg.setdefault('ggeur', {})['distributed_stage_timeout'] = 20

server['distribute']['server_port'] = port
server['outdir'] = 'exp/distributed_samples/officehome_vit_ggeur_fedavg/server_client_crash_after_join'
server['expname'] = 'officehome_vit_ggeur_fedavg_server_client_crash_after_join'

client['distribute']['server_port'] = port
client['distribute']['client_port'] = port + 1
client['outdir'] = 'exp/distributed_samples/officehome_vit_ggeur_fedavg/client_crash_after_join'
client['expname'] = 'officehome_vit_ggeur_fedavg_client_crash_after_join'

with open(server_dst, 'w', encoding='utf-8') as f:
    yaml.safe_dump(server, f, sort_keys=False)
with open(client_dst, 'w', encoding='utf-8') as f:
    yaml.safe_dump(client, f, sort_keys=False)
PY

    nohup "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" --cfg "$crash_server_cfg" \
        > "$LOG_DIR/server_client_crash_after_join.log" 2>&1 &
    echo "$!" > "$PID_DIR/server_crash_after_join.pid"

    wait_log_contains "$LOG_DIR/server_client_crash_after_join.log" "Listen to 127.0.0.1:${SERVER_PORT}" 60

    nohup "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" --cfg "$crash_client_cfg" \
        > "$LOG_DIR/client_crash_after_join.log" 2>&1 &
    local client_pid="$!"
    echo "$client_pid" > "$PID_DIR/client_crash_after_join.pid"

    wait_log_contains "$LOG_DIR/server_client_crash_after_join.log" "Assigned ID #1" 60
    kill -9 "$client_pid" >/dev/null 2>&1 || true
    rm -f "$PID_DIR/client_crash_after_join.pid"

    wait_log_contains "$LOG_DIR/server_client_crash_after_join.log" "Timeout in distributed stage 'statistics'" 90
    expect_log_contains "$LOG_DIR/server_client_crash_after_join.log" "GGEUR distributed stage timeout: statistics"

    bash "$TASK_DIR/stop.sh" >/dev/null 2>&1 || true
    sleep 2
    assert_port_free "$SERVER_PORT"
}

run_client_crash_during_round() {
    echo "Running client crash during training round validation scenario..." | tee -a "$LOG_DIR/scenario.log"

    local crash_server_cfg="$LOG_DIR/server_client_crash_during_round.yaml"
    local crash_client_cfg="$LOG_DIR/client_crash_during_round.yaml"
    "$PYTHON_BIN" - "$SERVER_CFG" "$TASK_DIR/officehome_vit_ggeur_fedavg_client_1.yaml" \
        "$crash_server_cfg" "$crash_client_cfg" "$SERVER_PORT" <<'PY'
import sys
import yaml

server_src, client_src, server_dst, client_dst, port = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
with open(server_src, 'r', encoding='utf-8') as f:
    server = yaml.safe_load(f)
with open(client_src, 'r', encoding='utf-8') as f:
    client = yaml.safe_load(f)

for cfg in (server, client):
    cfg['federate']['client_num'] = 1
    cfg['federate']['sample_client_num'] = 1
    cfg['federate']['total_round_num'] = 2
    cfg['train']['local_update_steps'] = 100000
    cfg.setdefault('ggeur', {})['distributed_stage_timeout'] = 20
    cfg['ggeur']['num_generated_per_sample'] = 0
    cfg['ggeur']['num_generated_per_prototype'] = 0
    cfg['ggeur']['target_size_per_class'] = 0
    cfg['ggeur']['use_fedproto'] = False
    cfg['ggeur']['use_cnn_distillation'] = False
    cfg['ggeur']['use_feature_alignment'] = False
    cfg['ggeur']['use_promptfl'] = False

server['distribute']['server_port'] = port
server['outdir'] = 'exp/distributed_samples/officehome_vit_ggeur_fedavg/server_client_crash_during_round'
server['expname'] = 'officehome_vit_ggeur_fedavg_server_client_crash_during_round'

client['distribute']['server_port'] = port
client['distribute']['client_port'] = port + 1
client['outdir'] = 'exp/distributed_samples/officehome_vit_ggeur_fedavg/client_crash_during_round'
client['expname'] = 'officehome_vit_ggeur_fedavg_client_crash_during_round'

with open(server_dst, 'w', encoding='utf-8') as f:
    yaml.safe_dump(server, f, sort_keys=False)
with open(client_dst, 'w', encoding='utf-8') as f:
    yaml.safe_dump(client, f, sort_keys=False)
PY

    nohup "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" --cfg "$crash_server_cfg" \
        > "$LOG_DIR/server_client_crash_during_round.log" 2>&1 &
    echo "$!" > "$PID_DIR/server_crash_during_round.pid"

    wait_log_contains "$LOG_DIR/server_client_crash_during_round.log" "Listen to 127.0.0.1:${SERVER_PORT}" 60

    nohup "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" --cfg "$crash_client_cfg" \
        > "$LOG_DIR/client_crash_during_round.log" 2>&1 &
    local client_pid="$!"
    echo "$client_pid" > "$PID_DIR/client_crash_during_round.pid"

    wait_log_contains "$LOG_DIR/server_client_crash_during_round.log" "Entered distributed stage 'round_1_model_updates'" 240
    kill -9 "$client_pid" >/dev/null 2>&1 || true
    rm -f "$PID_DIR/client_crash_during_round.pid"

    wait_log_contains "$LOG_DIR/server_client_crash_during_round.log" "Timeout in distributed stage 'round_1_model_updates'" 90

    bash "$TASK_DIR/stop.sh" >/dev/null 2>&1 || true
    sleep 2
    assert_port_free "$SERVER_PORT"
}

run_repeat_start_stop() {
    echo "Running repeated start/stop validation scenario..." | tee -a "$LOG_DIR/scenario.log"

    for idx in 1 2; do
        echo "Cycle $idx: start server" | tee -a "$LOG_DIR/scenario.log"
        bash "$TASK_DIR/start_server.sh"
        wait_log_contains "$LOG_DIR/server.log" "Listen to 127.0.0.1:${SERVER_PORT}" 60

        echo "Cycle $idx: stop server" | tee -a "$LOG_DIR/scenario.log"
        bash "$TASK_DIR/stop.sh"
        sleep 2
        assert_no_tracked_processes_alive
        assert_port_free "$SERVER_PORT"

        rm -f "$PID_DIR"/*.pid 2>/dev/null || true
        : > "$LOG_DIR/server.log"
    done
}

run_system_short() {
    local scenarios=(port_conflict client_wrong_server_port client_crash_after_join client_crash_during_round repeat_start_stop)
    for scenario in "${scenarios[@]}"; do
        echo "===== system_short: $scenario =====" | tee -a "$LOG_DIR/scenario.log"
        case "$scenario" in
            port_conflict)
                run_port_conflict
                ;;
            client_wrong_server_port)
                run_client_wrong_server_port
                ;;
            client_crash_after_join)
                run_client_crash_after_join
                ;;
            client_crash_during_round)
                run_client_crash_during_round
                ;;
            repeat_start_stop)
                run_repeat_start_stop
                ;;
        esac
    done
}

case "$SCENARIO" in
    normal)
        run_normal
        ;;
    delayed_client)
        run_delayed_client
        ;;
    missing_client_timeout)
        run_missing_client_timeout
        ;;
    port_conflict)
        run_port_conflict
        ;;
    client_wrong_server_port)
        run_client_wrong_server_port
        ;;
    client_crash_after_join)
        run_client_crash_after_join
        ;;
    client_crash_during_round)
        run_client_crash_during_round
        ;;
    repeat_start_stop)
        run_repeat_start_stop
        ;;
    system_short)
        run_system_short
        ;;
    *)
        echo "Unknown scenario: $SCENARIO"
        exit 1
        ;;
esac

echo "Scenario passed: $SCENARIO" | tee -a "$LOG_DIR/scenario.log"
echo "Logs: $LOG_DIR"
