#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
FULL_VALIDATION="$SCRIPT_DIR/run_loopback_multiip_full_validation.sh"

SCENARIO_RUN_ID="${SCENARIO_RUN_ID:-distributed_scenarios_$(date +%Y%m%d_%H%M%S)}"
SCENARIO_ROOT="${SCENARIO_ROOT:-$ROOT_DIR/exp/headonly_system/loopback_multiip_scenario_runs/$SCENARIO_RUN_ID}"
SUMMARY_TSV="$SCENARIO_ROOT/scenario_summary.tsv"

mkdir -p "$SCENARIO_ROOT"
printf "scenario\texpected\tstatus\texit_code\trun_dir\n" > "$SUMMARY_TSV"

run_case() {
  local name="$1"
  local expected="$2"
  shift 2

  local case_dir="$SCENARIO_ROOT/$name"
  local log_file="$case_dir/scenario.log"
  mkdir -p "$case_dir"

  local env_args=(
    "RUN_ROOT=$case_dir"
    "RUN_ID=$name"
    "RUN_QPS=0"
    "GEN_NUM=${GEN_NUM:-1}"
    "TOTAL_ROUNDS=${TOTAL_ROUNDS:-2}"
    "CASE_TIMEOUT=${CASE_TIMEOUT:-900}"
    "CLIENT_START_GAP=${CLIENT_START_GAP:-0}"
    "HEADONLY_CACHE_VERSION=headonly_${SCENARIO_RUN_ID}_${name}_gen${GEN_NUM:-1}"
  )
  env_args+=("$@")

  set +e
  env "${env_args[@]}" bash "$FULL_VALIDATION" > "$log_file" 2>&1
  local exit_code=$?
  set -e

  local status="fail"
  if [ "$expected" = "success" ] && [ "$exit_code" -eq 0 ]; then
    status="pass"
  elif [ "$expected" = "failure" ] && [ "$exit_code" -ne 0 ]; then
    status="pass"
  fi

  printf "%s\t%s\t%s\t%s\t%s\n" \
    "$name" "$expected" "$status" "$exit_code" "$case_dir/$name" \
    | tee -a "$SUMMARY_TSV"

  if [ "$status" != "pass" ]; then
    echo "Scenario $name failed expectation. See $log_file" >&2
    return 1
  fi
}

run_case normal_all_clients success \
  CLIENT_NUM=2 \
  LAUNCH_CLIENT_NUM=2 \
  SAMPLE_CLIENT_NUM=2 \
  SERVER_PORT=51261 \
  CLIENT_PORT_BASE=52261 \
  QPS_BASE_PORT=39461

run_case delayed_client_join success \
  CLIENT_NUM=2 \
  LAUNCH_CLIENT_NUM=2 \
  SAMPLE_CLIENT_NUM=2 \
  CLIENT_START_DELAYS=12,0 \
  SERVER_START_WAIT=8 \
  SERVER_PORT=51271 \
  CLIENT_PORT_BASE=52271 \
  QPS_BASE_PORT=39471

run_case missing_client_join_timeout failure \
  CLIENT_NUM=3 \
  LAUNCH_CLIENT_NUM=2 \
  SAMPLE_CLIENT_NUM=2 \
  JOIN_TIMEOUT_SECONDS=15 \
  CASE_TIMEOUT=120 \
  SERVER_PORT=51281 \
  CLIENT_PORT_BASE=52281 \
  QPS_BASE_PORT=39481

run_case extra_client_late_rejected success \
  CLIENT_NUM=2 \
  LAUNCH_CLIENT_NUM=2 \
  EXTRA_CLIENTS=1 \
  SAMPLE_CLIENT_NUM=2 \
  CLIENT_START_DELAYS=0,10,0 \
  SERVER_PORT=51291 \
  CLIENT_PORT_BASE=52291 \
  QPS_BASE_PORT=39491

run_case client_exit_before_train_quorum success \
  CLIENT_NUM=2 \
  LAUNCH_CLIENT_NUM=2 \
  SAMPLE_CLIENT_NUM=2 \
  MIN_TRAIN_UPDATES=1 \
  CLIENT_FAIL_SPECS=2:before_train_round:1 \
  SERVER_PORT=51301 \
  CLIENT_PORT_BASE=52301 \
  QPS_BASE_PORT=39501

echo "$SUMMARY_TSV"
