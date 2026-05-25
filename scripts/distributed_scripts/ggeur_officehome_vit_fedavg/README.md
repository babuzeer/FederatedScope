# OfficeHome Distributed Sample

This directory contains the first real distributed sample task for the
current GGEUR codebase:

- Dataset: `OfficeHome`
- Clients: `4` total, `1` per domain
- Method: `GGEUR + FedAvg`
- Feature extractor: `CLIP ViT-B-16`

The configs are intentionally based on the small validation setup rather than
the full 60-client experiment, so the first distributed rollout is easier to
verify and debug.

## Files

- `officehome_vit_ggeur_fedavg_base.yaml`
  - Common reference settings for this sample task
- `officehome_vit_ggeur_fedavg_server.yaml`
  - Server config
- `officehome_vit_ggeur_fedavg_client_1.yaml` ... `client_4.yaml`
  - Per-client configs
- `start_server.sh`
  - Start the server process in background
- `start_client.sh`
  - Start one client process in background
- `run_local_managed.sh`
  - Start server and all 4 clients on one machine for the first real validation
- `stop.sh`
  - Stop tracked server/client processes

## Default assumptions

The configs default to single-host loopback addresses for the first rollout:

- server: `127.0.0.1:51051`
- client 1: `127.0.0.1:51052`
- client 2: `127.0.0.1:51053`
- client 3: `127.0.0.1:51054`
- client 4: `127.0.0.1:51055`

For multi-server deployment, change these fields in the client/server YAMLs:

- `distribute.server_host`
- `distribute.server_port`
- `distribute.client_host`
- `distribute.client_port`

## Data and model paths

Current defaults:

- dataset root: `/root/autodl-tmp/datasets/OfficeHomeDataset_10072016`
- CLIP local weights: `/root/autodl-tmp/models/open_clip_vitb16.bin`

If your server paths differ, update:

- `data.root`
- `ggeur.clip_model_path`

## Suggested rollout order

1. Start the server on the designated server machine:
   - `bash scripts/distributed_scripts/ggeur_officehome_vit_fedavg/start_server.sh`
2. Start each client on its designated client machine:
   - `bash scripts/distributed_scripts/ggeur_officehome_vit_fedavg/start_client.sh 1`
   - `bash scripts/distributed_scripts/ggeur_officehome_vit_fedavg/start_client.sh 2`
   - `bash scripts/distributed_scripts/ggeur_officehome_vit_fedavg/start_client.sh 3`
   - `bash scripts/distributed_scripts/ggeur_officehome_vit_fedavg/start_client.sh 4`
3. Stop processes with:
   - `bash scripts/distributed_scripts/ggeur_officehome_vit_fedavg/stop.sh`

## Single-machine first validation

If you want to validate the first real task on one server before multi-server
deployment, use:

- `bash scripts/distributed_scripts/ggeur_officehome_vit_fedavg/run_local_managed.sh`

This script will:

- start the server
- start all 4 clients with short delays
- write logs under `logs/`
- track PIDs under `pids/`
- stop tracked processes on `Ctrl+C`

## Scenario Validation

Before extending to more methods or multi-machine deployment, use the scenario
runner to validate basic distributed reactions:

```bash
bash scripts/distributed_scripts/ggeur_officehome_vit_fedavg/run_validation_scenario.sh normal
bash scripts/distributed_scripts/ggeur_officehome_vit_fedavg/run_validation_scenario.sh delayed_client
bash scripts/distributed_scripts/ggeur_officehome_vit_fedavg/run_validation_scenario.sh missing_client_timeout
bash scripts/distributed_scripts/ggeur_officehome_vit_fedavg/run_validation_scenario.sh port_conflict
bash scripts/distributed_scripts/ggeur_officehome_vit_fedavg/run_validation_scenario.sh client_wrong_server_port
bash scripts/distributed_scripts/ggeur_officehome_vit_fedavg/run_validation_scenario.sh client_crash_after_join
bash scripts/distributed_scripts/ggeur_officehome_vit_fedavg/run_validation_scenario.sh client_crash_during_round
bash scripts/distributed_scripts/ggeur_officehome_vit_fedavg/run_validation_scenario.sh repeat_start_stop
bash scripts/distributed_scripts/ggeur_officehome_vit_fedavg/run_validation_scenario.sh system_short
```

Each scenario writes logs under `logs/validation_*` and uses an isolated
validation PID directory, so it does not collide with normal sample runs.

`system_short` is the preferred quick infrastructure regression check. It runs
`port_conflict`, `client_wrong_server_port`, `client_crash_after_join`,
`client_crash_during_round`, and `repeat_start_stop` without waiting for a full
training job.

## Notes

- This sample uses `client_num: 4` and `sample_client_num: 4`.
- The server config uses `make_global_eval: False` so the first rollout stays
  focused on distributed training stability instead of server-side evaluation.
- The GGEUR data loader now skips building client training datasets when
  `federate.mode='distributed'` and `distribute.role='server'`. This keeps the
  server from owning client train splits while still allowing server-side test
  evaluation from `cfg.data.root` when needed.
- Each role writes to its own `outdir` to avoid local log/result collisions.
