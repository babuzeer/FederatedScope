# GGEUR Multi-machine Runner

This directory provides deployment scripts for real FederatedScope/GGEUR
distributed training on two machines:

- one server machine runs the FederatedScope server process
- one client machine simulates multiple FederatedScope client processes

The scripts generate YAML files consumed by `federatedscope/main.py`; they do
not replace the core training worker code.

## Matrix Coverage

The generator supports:

- datasets: `officehome`, `domainnet`
- models: `cnn`, `vit`, `mixer`
- methods: `fedavg`, `fedprox`, `fedopt`, `moon`, `fedproto`, `promptfl`
- optional strict head-only mode for MLP-head-only paths

Strict head-only mode skips `promptfl` by default because PromptFL is not an
MLP-head-only method.

## Generate One Case

```bash
cd /root/autodl-tmp/FederatedScope
/root/miniconda3/envs/fs/bin/python scripts/distributed_scripts/ggeur_multimachine/generate_configs.py \
  --run-id demo_vit_fedavg \
  --dataset officehome \
  --model vit \
  --method fedavg \
  --server-host 10.0.0.10 \
  --server-bind-host 0.0.0.0 \
  --client-host 10.0.0.11 \
  --clients 4 \
  --rounds 5
```

This writes configs under:

```text
scripts/distributed_scripts/ggeur_multimachine/runs/demo_vit_fedavg/
```

## Generate Full Matrix

```bash
/root/miniconda3/envs/fs/bin/python scripts/distributed_scripts/ggeur_multimachine/generate_configs.py \
  --run-id full_matrix \
  --dataset all \
  --model all \
  --method all \
  --server-host 10.0.0.10 \
  --server-bind-host 0.0.0.0 \
  --client-host 10.0.0.11 \
  --clients 4 \
  --rounds 5
```

This generates 36 cases.

## Generate Head-only Matrix

```bash
/root/miniconda3/envs/fs/bin/python scripts/distributed_scripts/ggeur_multimachine/generate_configs.py \
  --run-id headonly_matrix \
  --dataset all \
  --model all \
  --method all \
  --head-only \
  --server-host 10.0.0.10 \
  --server-bind-host 0.0.0.0 \
  --client-host 10.0.0.11 \
  --clients 4 \
  --rounds 5
```

This skips `promptfl` unless `--allow-promptfl-in-head-only` is explicitly set.

## Sync Case to the Client Machine

Run from the server/control repo:

```bash
bash scripts/distributed_scripts/ggeur_multimachine/sync_case.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/demo_vit_fedavg \
  client-machine-ssh-alias \
  /root/autodl-tmp/FederatedScope
```

The repository code, datasets, model weights, and Python environment should
already exist on both machines.

## Run One Case Remotely

```bash
REPO_DIR=/root/autodl-tmp/FederatedScope \
PYTHON_BIN=/root/miniconda3/envs/fs/bin/python \
bash scripts/distributed_scripts/ggeur_multimachine/run_remote_case.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/demo_vit_fedavg/officehome_vit_fedavg \
  server-machine-ssh-alias \
  client-machine-ssh-alias
```

The controller:

- starts the server via SSH
- waits for the server listen log
- starts all clients on the client machine
- waits for all tracked PIDs to exit
- writes `summary.json`
- cleans up on timeout or interruption

## Manual Launch

On the server machine:

```bash
PYTHON_BIN=/root/miniconda3/envs/fs/bin/python \
bash scripts/distributed_scripts/ggeur_multimachine/launch_server.sh <case_dir>
```

On the client machine:

```bash
PYTHON_BIN=/root/miniconda3/envs/fs/bin/python \
bash scripts/distributed_scripts/ggeur_multimachine/launch_clients.sh <case_dir>
```

## Robustness Mechanisms

- PID tracking per case
- explicit stop script with terminate then kill
- server listen readiness wait
- case timeout
- log-based completion summary
- return-code and error-tail reporting
- per-case port allocation when generating a matrix
- configurable `ggeur.distributed_stage_timeout`

## Current Scope

This is real FederatedScope distributed training over real gRPC communication
between machines. The strict head-only mode uses the existing GGEUR path with
frozen/cached/unloaded extractor and MLP-only standard training rounds.

The physical parameter-service/sub-server/root-server decomposition in the
head-only architecture is not yet split into separate long-running services in
this runner; the generated configs prepare the head-only training mode and
multi-machine launch surface first.
