#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ACK_MODE="${ACK_MODE:-first-chunk}"

exec bash "$SCRIPT_DIR/run_ipv6_qps_plan_clients.sh"
