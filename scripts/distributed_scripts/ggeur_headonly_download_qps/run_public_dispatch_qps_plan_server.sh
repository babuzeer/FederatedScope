#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_PREFIX="${RUN_PREFIX:-public_dispatch_qps_$(date +%Y%m%d_%H%M%S)}"
export LISTEN_HOST="${LISTEN_HOST:-0.0.0.0}"
export ACK_MODE="${ACK_MODE:-first-chunk}"

exec bash "$SCRIPT_DIR/run_ipv6_qps_plan_server.sh"
