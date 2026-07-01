# GGEUR HeadOnly 60-Client Real Distributed Training

This is the final 60-client OfficeHome HeadOnly entry:

```text
4 OfficeHome domains x 15 clients/domain = 60 clients
real FederatedScope distributed mode over gRPC
round 0: real feature extraction, statistics upload, covariance aggregation,
         GGEUR feature generation / augmentation
round 1+: HeadOnly MLP training on generated feature cache
cache-hot rerun: load per-client augmented HeadOnly cache first
```

It is not the synthetic QPS benchmark. It runs `federatedscope/main.py` with the
real GGEUR server/client workers.

## Network Requirement

FederatedScope distributed mode is bidirectional gRPC:

```text
clients -> server: join/statistics/model updates
server -> clients: assigned id/global covariances/model params/finish
```

So both sides need reachable public or mapped ports:

```text
server side: SERVER_HOST:SERVER_PORT
client side: CLIENT_HOST:CLIENT_PORT_BASE+1 ... CLIENT_PORT_BASE+60
```

If client processes bind locally to `0.0.0.0` but the server must call back
through a public IP or port mapping, use `CLIENT_HOST`/`CLIENT_HOSTS` as the
advertised address and keep `CLIENT_BIND_HOST=0.0.0.0`.

## Prepare Case

Run this on the client/local machine if that machine has the OfficeHome data:

```bash
cd /root/autodl-tmp/FederatedScope

RUN_ID=officehome_vit_headonly_60c_public \
DATA_ROOT=/root/autodl-tmp/datasets/OfficeHomeDataset_10072016 \
SERVER_HOST=<server_public_ip_or_dns> \
SERVER_BIND_HOST=0.0.0.0 \
SERVER_PORT=55051 \
CLIENT_HOST=<client_public_ip_or_dns_seen_by_server> \
CLIENT_BIND_HOST=0.0.0.0 \
CLIENT_PORT_BASE=56000 \
TOTAL_ROUNDS=100 \
GEN_NUM=20 \
FEATURE_CACHE_DIR=exp/headonly_system/cache/officehome_vit_headonly_60c \
HEADONLY_CACHE_VERSION=officehome_vit_headonly_60c_gen20 \
bash scripts/distributed_scripts/ggeur_headonly_60c/prepare_case.sh
```

The script writes the case directory:

```text
scripts/distributed_scripts/ggeur_multimachine/runs/<RUN_ID>/officehome_vit_fedavg_headonly
```

It also writes per-client manifests:

```text
exp/headonly_system/manifests/<RUN_ID>/client_000001/client_manifest.json
...
exp/headonly_system/manifests/<RUN_ID>/client_000060/client_manifest.json
```

The manifest split is fixed to:

```text
Art        clients 1-15
Clipart    clients 16-30
Product    clients 31-45
Real_World clients 46-60
```

with LDS enabled by default (`USE_LDS=1`, `LDS_ALPHA=0.1`).

## Multiple Client Public IPs

For multiple callback IPs on the client side, pass `CLIENT_HOSTS`.
Default assignment is block assignment:

```bash
CLIENT_HOSTS=ip1,ip2,ip3,ip4
CLIENT_HOST_ASSIGNMENT=block
```

For 60 clients and 4 IPs, this maps 15 consecutive clients to each IP.

If public callback ports differ from local bind ports, pass explicit advertised
ports:

```bash
CLIENT_ADVERTISE_PORTS=61001,61002,...,61060
CLIENT_BIND_PORT_BASE=56000
```

## Sync Case To Server

The server needs the generated case configs, but it does not need client image
data or manifests for training. Copy the generated case directory to the same
repo-relative path on the server:

```bash
scp -r scripts/distributed_scripts/ggeur_multimachine/runs/<RUN_ID> \
  <server_ssh>:/root/autodl-tmp/FederatedScope/scripts/distributed_scripts/ggeur_multimachine/runs/
```

Make sure the code version is synced on both machines.

## Start Training

On the server machine:

```bash
cd /root/autodl-tmp/FederatedScope

PYTHON_BIN=/root/miniconda3/envs/fs/bin/python \
bash scripts/distributed_scripts/ggeur_headonly_60c/launch_server.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/<RUN_ID>/officehome_vit_fedavg_headonly
```

On the client/local machine:

```bash
cd /root/autodl-tmp/FederatedScope

PYTHON_BIN=/root/miniconda3/envs/fs/bin/python \
CLIENT_START_GAP=1 \
bash scripts/distributed_scripts/ggeur_headonly_60c/launch_clients.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/<RUN_ID>/officehome_vit_fedavg_headonly
```

## Logs

All logs are under the case directory:

```text
logs/server.log
logs/client_1.log
...
logs/client_60.log
system/server_run_info.log
system/clients_run_info.log
pids/*.pid
```

Key log lines:

```text
Loaded augmented HeadOnly feature cache
HeadOnly cache-hot mode active
Feature extraction QPS
Local statistics payload bytes
Augmentation timing
Round <n> aggregation complete
Round <n> MLP Test Accuracy
Training finished
```

On cache-hot reruns, clients first try:

```text
FEATURE_CACHE_DIR/headonly_augmented/HEADONLY_CACHE_VERSION/office-home_client_000001.pt
...
```

They also fall back to the older flat cache layout:

```text
FEATURE_CACHE_DIR/client_000000.pt
FEATURE_CACHE_DIR/client_000001.pt
...
```

The default `FEATURE_CACHE_DIR`/`HEADONLY_CACHE_VERSION` are set to the existing
60-client gen20 cache name:

```text
exp/ggeur_headonly_real_cache/officehome_vitb16_60c_gen20_fcache_v1
```

If compatible metadata matches, round-0 extraction/statistics/augmentation is
skipped and training starts from cached generated features. If no cache exists
or metadata differs, the client regenerates data and saves a new cache.

## Status And Cleanup

Status:

```bash
bash scripts/distributed_scripts/ggeur_multimachine/status_case.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/<RUN_ID>/officehome_vit_fedavg_headonly
```

Stop:

```bash
bash scripts/distributed_scripts/ggeur_multimachine/stop_case.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/<RUN_ID>/officehome_vit_fedavg_headonly
```

Summarize:

```bash
python scripts/distributed_scripts/ggeur_multimachine/summarize_case.py \
  scripts/distributed_scripts/ggeur_multimachine/runs/<RUN_ID>/officehome_vit_fedavg_headonly
```
