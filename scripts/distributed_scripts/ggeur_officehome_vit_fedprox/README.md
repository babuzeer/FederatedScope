# OfficeHome GGEUR FedProx Distributed Sample

This directory is the second real distributed sample task for the current
GGEUR rollout.

- Dataset: `OfficeHome`
- Clients: `4` total, `1` per domain
- Method: `GGEUR + FedProx`
- Feature extractor: `CLIP ViT-B-16`

It intentionally mirrors the verified `GGEUR + FedAvg` sample and changes only
the method-specific FedProx settings, so failures can be attributed to the
method extension rather than data, paths, or distributed wiring.

## Files

- `officehome_vit_ggeur_fedprox_base.yaml`
  - Common reference settings for this sample task
- `officehome_vit_ggeur_fedprox_server.yaml`
  - Server config
- `officehome_vit_ggeur_fedprox_client_1.yaml` ... `client_4.yaml`
  - Per-client configs
- `start_server.sh`
  - Start the server process in background
- `start_client.sh`
  - Start one client process in background
- `run_local_managed.sh`
  - Start server and all 4 clients on one machine
- `stop.sh`
  - Stop tracked server/client processes

## Default assumptions

- server: `127.0.0.1:51151`
- client 1: `127.0.0.1:51152`
- client 2: `127.0.0.1:51153`
- client 3: `127.0.0.1:51154`
- client 4: `127.0.0.1:51155`

The port range intentionally differs from the FedAvg sample to avoid collisions
when old processes have not fully released sockets.

## Data and model paths

Current defaults:

- dataset root: `/root/autodl-tmp/datasets/OfficeHomeDataset_10072016`
- CLIP local weights: `/root/autodl-tmp/models/open_clip_vitb16.bin`

If your server paths differ, update:

- `data.root`
- `ggeur.clip_model_path`

## Single-machine validation

Use:

```bash
bash scripts/distributed_scripts/ggeur_officehome_vit_fedprox/run_local_managed.sh
```

This script starts server + 4 clients, writes logs under `logs/`, and tracks
PIDs under `pids/`.

## Validation Goal

This sample should verify that the distributed OfficeHome GGEUR path remains
stable when FedProx proximal regularization is enabled:

```yaml
fedprox:
  use: True
  mu: 0.1
```
