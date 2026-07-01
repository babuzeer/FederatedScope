# HeadOnly Dispatch-Start QPS Test

This directory validates the first-stage HeadOnly acceptance metric:

```text
Dispatch-Start QPS
```

The primary path is:

```text
coordinator -> subserver processes -> logical clients
```

A successful dispatch-start event means:

```text
client requests model params
subserver sends model metadata
subserver sends the first parameter chunk
client ACKs immediately after receiving that first chunk
```

This is not full payload download QPS. Full payload mode is retained only as an
auxiliary bandwidth-related test.

## Default Protocol

The default ACK mode is:

```text
ACK_MODE=first-chunk
```

Supported modes:

```text
header       client ACKs after metadata only
first-chunk  client ACKs after metadata + first parameter chunk
full         client ACKs after the full MLP payload
```

The first-stage acceptance reports use `first-chunk`.

Default payload:

```text
512 * 65 + 65 = 33,345 params
33,345 * 4 bytes = 133,380 bytes, about 130 KiB
```

Default first chunk:

```text
FIRST_CHUNK_BYTES=4096
```

## Two-Machine Public-Network Run

Recommended topology:

```text
Machine A, server side:
  one coordinator process
  one Python subserver process per port
  ports 39301, 39302, ...

Machine B, client side:
  one Python client-group process per subserver
  each group can bind to a different local source IP
  all traffic connects to Machine A public IP/ports
```

Open or map the server public TCP ports first. For 8 subservers, expose:

```text
39301-39308/tcp
```

### Smoke Test

On Machine A:

```bash
cd /root/autodl-tmp/FederatedScope

RUN_ID=public_dispatch_smoke_1x10_server \
LISTEN_HOST=0.0.0.0 \
BASE_PORT=39301 \
SUBSERVERS=1 \
CLIENTS_PER_SUBSERVER=10 \
ACK_MODE=first-chunk \
bash scripts/distributed_scripts/ggeur_headonly_download_qps/run_download_qps_server.sh
```

For IPv6, use:

```bash
LISTEN_HOST='::'
```

On Machine B:

```bash
cd /root/autodl-tmp/FederatedScope

RUN_ID=public_dispatch_smoke_1x10_clients \
CONNECT_HOST=<machine_a_public_ip_or_host> \
BASE_PORT=39301 \
SUBSERVERS=1 \
CLIENTS_PER_SUBSERVER=10 \
ACK_MODE=first-chunk \
bash scripts/distributed_scripts/ggeur_headonly_download_qps/run_download_qps_clients.sh
```

For IPv6, pass the address without brackets:

```text
CONNECT_HOST=2001:db8::1234
```

## Public Port Mapping

If the public ports are the same as container/internal ports, only set
`BASE_PORT`.

If the platform maps internal ports to random public ports, pass one external
port per subserver on Machine B:

```bash
CONNECT_PORTS=51001,51002,51003,51004
```

The order maps to server-side ports in order:

```text
39301 -> first public port
39302 -> second public port
39303 -> third public port
39304 -> fourth public port
```

## Multiple Source IPs On Client Machine

If Machine B has multiple usable public or local source IPs, bind one client
group per source IP:

```bash
SOURCE_HOSTS=192.0.2.11,192.0.2.12,192.0.2.13,192.0.2.14
```

`SOURCE_HOSTS` must contain one IP per subserver for a single run.

If no source IP is specified, the OS chooses the outbound source address.

## Acceptance Matrix

Run this on Machine A:

```bash
cd /root/autodl-tmp/FederatedScope

RUN_PREFIX=public_dispatch_$(date +%Y%m%d_%H%M%S) \
LISTEN_HOST=0.0.0.0 \
BASE_PORT=39301 \
ACK_MODE=first-chunk \
bash scripts/distributed_scripts/ggeur_headonly_download_qps/run_public_dispatch_qps_plan_server.sh
```

Run this on Machine B with the same `RUN_PREFIX`:

```bash
cd /root/autodl-tmp/FederatedScope

RUN_PREFIX=<same_prefix_as_server> \
CONNECT_HOST=<machine_a_public_ip_or_host> \
BASE_PORT=39301 \
ACK_MODE=first-chunk \
bash scripts/distributed_scripts/ggeur_headonly_download_qps/run_public_dispatch_qps_plan_clients.sh
```

Default matrix:

```text
smoke_1x10
flat_1x500
h2_2x500
h4_4x500
h8_8x500
h4_4x2500
h8_8x1250
```

The last two cases are 10000-client runs.

Override the matrix with:

```bash
QPS_PLAN='smoke_1x10:1:10:120:120,h10_10x1000:10:1000:720:720'
```

Format:

```text
name:subservers:clients_per_subserver:ready_timeout:ack_timeout
```

## Result Files

Server-side run directory:

```text
exp/headonly_download_qps/<RUN_ID>/
```

Important files:

```text
download_qps_summary.json
server.log
run_info.log
```

Primary fields:

```text
expected_clients
first_chunk_ack_clients
success_ratio
dispatch_start_window_sec
dispatch_start_qps
latency_p50_sec
latency_p95_sec
latency_p99_sec
subserver_qps_min
subserver_qps_avg
subserver_qps_max
```

Summarize a matrix:

```bash
python3 scripts/summarize_headonly_qps_runs.py \
  exp/headonly_download_qps \
  --prefix <RUN_PREFIX> \
  --output exp/headonly_download_qps/<RUN_PREFIX>_summary.tsv
```

## Acceptance Criteria

For the first-stage 10000+ QPS target:

```text
expected_clients >= 10000
first_chunk_ack_clients >= 10000
success_ratio = 1.0
dispatch_start_qps >= 10000
latency_p95_sec <= 1.0
```

If `dispatch_start_qps >= 10000` but `latency_p95_sec > 1.0`, report it as a
tail-latency risk instead of a clean acceptance pass.
