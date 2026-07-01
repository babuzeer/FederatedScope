#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_DIR="${REPO_DIR:-$DEFAULT_REPO_DIR}"
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x /root/miniconda3/envs/fs/bin/python ]; then
    PYTHON_BIN="/root/miniconda3/envs/fs/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

RUN_ID="${RUN_ID:-headonly_netns_download_qps_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-exp/headonly_download_qps_netns}"

BASE_PORT="${BASE_PORT:-39301}"
SUBSERVERS="${SUBSERVERS:-4}"
CLIENT_GROUPS="${CLIENT_GROUPS:-$SUBSERVERS}"
CLIENTS_PER_GROUP="${CLIENTS_PER_GROUP:-250}"
if [ $((CLIENT_GROUPS % SUBSERVERS)) -ne 0 ]; then
  echo "CLIENT_GROUPS must be divisible by SUBSERVERS for balanced subserver load" >&2
  exit 1
fi
GROUPS_PER_SUBSERVER=$((CLIENT_GROUPS / SUBSERVERS))
CLIENTS_PER_SUBSERVER_EXPECTED=$((CLIENTS_PER_GROUP * GROUPS_PER_SUBSERVER))
OVERHEAD_MULTIPLIER="${OVERHEAD_MULTIPLIER:-1.0}"
PAYLOAD_FILE="${PAYLOAD_FILE:-}"
READY_TIMEOUT="${READY_TIMEOUT:-300}"
ACK_TIMEOUT="${ACK_TIMEOUT:-300}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-60}"

BRIDGE_NAME="${BRIDGE_NAME:-hoqps-br0}"
SERVER_NS="${SERVER_NS:-hoqps-srv}"
CLIENT_NS_PREFIX="${CLIENT_NS_PREFIX:-hoqps-cli}"
SERVER_IP="${SERVER_IP:-10.77.0.10}"
BRIDGE_IP_CIDR="${BRIDGE_IP_CIDR:-10.77.0.1/24}"
CLIENT_IP_BASE="${CLIENT_IP_BASE:-10.77.0}"
CLIENT_IP_START="${CLIENT_IP_START:-101}"
MTU="${MTU:-1500}"

# Examples: LINK_RATE=10gbit, LINK_RATE=25gbit. Empty means unlimited veth.
LINK_RATE="${LINK_RATE:-}"
# Examples: LINK_DELAY=0.2ms, LINK_DELAY=1ms. Empty means no delay qdisc.
LINK_DELAY="${LINK_DELAY:-}"

KEEP_NETNS="${KEEP_NETNS:-0}"
CLEANUP_ONLY="${CLEANUP_ONLY:-0}"
NOFILE_LIMIT="${NOFILE_LIMIT:-1048576}"
SOMAXCONN="${SOMAXCONN:-65535}"

SERVER_VETH_HOST="${SERVER_VETH_HOST:-hoqps-vs-h}"
SERVER_VETH_NS="${SERVER_VETH_NS:-eth0}"

require_root() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "run_netns_download_qps.sh requires root because it creates network namespaces" >&2
    exit 1
  fi
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing command: $1. Install iproute2 before running this script." >&2
    exit 1
  fi
}

require_net_admin() {
  local probe="hoqps-probe-$$"
  if ! ip link add "$probe" type dummy >/dev/null 2>&1; then
    cat >&2 <<'EOF'
missing CAP_NET_ADMIN: this container cannot create network namespaces,
bridges, veth devices, or tc qdisc rules.

Use run_loopback_multiip_download_qps.sh on restricted containers, or run this
script on a privileged VM/container with NET_ADMIN enabled.
EOF
    exit 1
  fi
  ip link del "$probe" >/dev/null 2>&1 || true
}

cleanup_netns() {
  set +e
  for idx in $(seq 1 "$CLIENT_GROUPS"); do
    ip netns pids "${CLIENT_NS_PREFIX}${idx}" 2>/dev/null | xargs -r kill 2>/dev/null
  done
  ip netns pids "$SERVER_NS" 2>/dev/null | xargs -r kill 2>/dev/null
  for idx in $(seq 1 "$CLIENT_GROUPS"); do
    ip netns del "${CLIENT_NS_PREFIX}${idx}" 2>/dev/null
  done
  ip netns del "$SERVER_NS" 2>/dev/null
  ip link del "$BRIDGE_NAME" 2>/dev/null
  set -e
}

client_ip() {
  local idx="$1"
  echo "${CLIENT_IP_BASE}.$((CLIENT_IP_START + idx - 1))"
}

create_ns_link() {
  local ns="$1"
  local host_veth="$2"
  local ns_veth="$3"
  local ip_cidr="$4"

  ip netns add "$ns"
  ip link add "$host_veth" type veth peer name "$ns_veth"
  ip link set "$host_veth" master "$BRIDGE_NAME"
  ip link set "$host_veth" mtu "$MTU"
  ip link set "$host_veth" up
  ip link set "$ns_veth" netns "$ns"
  ip netns exec "$ns" ip link set lo up
  ip netns exec "$ns" ip link set "$ns_veth" name eth0
  ip netns exec "$ns" ip link set eth0 mtu "$MTU"
  ip netns exec "$ns" ip addr add "$ip_cidr" dev eth0
  ip netns exec "$ns" ip link set eth0 up
}

apply_server_qdisc() {
  if [ -z "$LINK_RATE" ] && [ -z "$LINK_DELAY" ]; then
    return
  fi
  local args=()
  if [ -n "$LINK_DELAY" ]; then
    args+=(delay "$LINK_DELAY")
  fi
  if [ -n "$LINK_RATE" ]; then
    args+=(rate "$LINK_RATE")
  fi
  ip netns exec "$SERVER_NS" tc qdisc replace dev eth0 root netem "${args[@]}"
}

setup_netns() {
  cleanup_netns
  ip link add "$BRIDGE_NAME" type bridge
  ip addr add "$BRIDGE_IP_CIDR" dev "$BRIDGE_NAME"
  ip link set "$BRIDGE_NAME" mtu "$MTU"
  ip link set "$BRIDGE_NAME" up

  create_ns_link "$SERVER_NS" "$SERVER_VETH_HOST" "$SERVER_VETH_NS" "$SERVER_IP/24"
  for idx in $(seq 1 "$CLIENT_GROUPS"); do
    create_ns_link \
      "${CLIENT_NS_PREFIX}${idx}" \
      "hoqps-vc${idx}-h" \
      "hoqps-vc${idx}-n" \
      "$(client_ip "$idx")/24"
  done
  apply_server_qdisc
}

write_run_info() {
  {
    echo "role=netns_single_host"
    echo "run_id=$RUN_ID"
    echo "repo_dir=$REPO_DIR"
    echo "python_bin=$PYTHON_BIN"
    echo "base_port=$BASE_PORT"
    echo "subservers=$SUBSERVERS"
    echo "client_groups=$CLIENT_GROUPS"
    echo "groups_per_subserver=$GROUPS_PER_SUBSERVER"
    echo "clients_per_group=$CLIENTS_PER_GROUP"
    echo "clients_per_subserver_expected=$CLIENTS_PER_SUBSERVER_EXPECTED"
    echo "total_logical_clients=$((CLIENT_GROUPS * CLIENTS_PER_GROUP))"
    echo "overhead_multiplier=$OVERHEAD_MULTIPLIER"
    echo "payload_file=$PAYLOAD_FILE"
    echo "bridge_name=$BRIDGE_NAME"
    echo "bridge_ip_cidr=$BRIDGE_IP_CIDR"
    echo "server_ns=$SERVER_NS"
    echo "server_ip=$SERVER_IP"
    echo "client_ns_prefix=$CLIENT_NS_PREFIX"
    echo "link_rate=$LINK_RATE"
    echo "link_delay=$LINK_DELAY"
    echo "mtu=$MTU"
    echo "nofile_limit=$NOFILE_LIMIT"
    echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo
    echo "[namespaces]"
    ip netns list || true
    echo
    echo "[addresses]"
    ip -br addr show "$BRIDGE_NAME" || true
    ip netns exec "$SERVER_NS" ip -br addr || true
    for idx in $(seq 1 "$CLIENT_GROUPS"); do
      ip netns exec "${CLIENT_NS_PREFIX}${idx}" ip -br addr || true
    done
    echo
    echo "[qdisc]"
    ip netns exec "$SERVER_NS" tc qdisc show dev eth0 || true
  } | tee "$RUN_DIR/run_info.log"
}

run_experiment() {
  mkdir -p "$RUN_DIR"
  ulimit -n "$NOFILE_LIMIT" 2>/dev/null || true
  sysctl -w "net.core.somaxconn=$SOMAXCONN" >/dev/null 2>&1 || true
  write_run_info

  ip netns exec "$SERVER_NS" env \
    PYTHON_BIN="$PYTHON_BIN" \
    REPO_DIR="$REPO_DIR" \
    RUN_ID="$RUN_ID" \
    OUT_ROOT="$OUT_ROOT" \
    LISTEN_HOST="$SERVER_IP" \
    BASE_PORT="$BASE_PORT" \
    SUBSERVERS="$SUBSERVERS" \
    CLIENTS_PER_SUBSERVER="$CLIENTS_PER_SUBSERVER_EXPECTED" \
    OVERHEAD_MULTIPLIER="$OVERHEAD_MULTIPLIER" \
    PAYLOAD_FILE="$PAYLOAD_FILE" \
    READY_TIMEOUT="$READY_TIMEOUT" \
    ACK_TIMEOUT="$ACK_TIMEOUT" \
    bash "$REPO_DIR/scripts/distributed_scripts/ggeur_headonly_download_qps/run_download_qps_server.sh" \
    >"$RUN_DIR/server_outer.log" 2>&1 &
  local server_pid="$!"

  sleep 2

  local client_pids=()
  for idx in $(seq 1 "$CLIENT_GROUPS"); do
    local subserver_idx=$(( (idx - 1) % SUBSERVERS ))
    local port=$(( BASE_PORT + subserver_idx ))
    ip netns exec "${CLIENT_NS_PREFIX}${idx}" env \
      PYTHON_BIN="$PYTHON_BIN" \
      REPO_DIR="$REPO_DIR" \
      RUN_ID="${RUN_ID}_clients_group${idx}" \
      OUT_ROOT="$OUT_ROOT" \
      CONNECT_HOST="$SERVER_IP" \
      BASE_PORT="$port" \
      SUBSERVERS=1 \
      CLIENTS_PER_SUBSERVER="$CLIENTS_PER_GROUP" \
      CONNECT_TIMEOUT="$CONNECT_TIMEOUT" \
      bash "$REPO_DIR/scripts/distributed_scripts/ggeur_headonly_download_qps/run_download_qps_clients.sh" \
      >"$RUN_DIR/clients_group${idx}.log" 2>&1 &
    client_pids+=("$!")
  done

  local status=0
  for pid in "${client_pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if ! wait "$server_pid"; then
    status=1
  fi

  {
    echo
    echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=$status"
  } | tee -a "$RUN_DIR/run_info.log"
  return "$status"
}

require_root
require_cmd ip
require_cmd tc
require_net_admin

cd "$REPO_DIR"
RUN_DIR="$OUT_ROOT/$RUN_ID"

if [ "$CLEANUP_ONLY" = "1" ]; then
  cleanup_netns
  exit 0
fi

setup_netns
if [ "$KEEP_NETNS" = "1" ]; then
  run_experiment
else
  trap cleanup_netns EXIT
  run_experiment
fi

echo "$RUN_DIR"
