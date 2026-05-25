# OfficeHome FedAvg Distributed Baselines

This directory contains distributed baseline samples for OfficeHome using the
three model families currently in scope:

- `ViT-B16`
- `ConvNeXt`
- `MLP-Mixer`

Common constraints:

- method: `ggeur`
- no GGEUR data generation
- real dataset: `OfficeHome`
- distributed mode
- `4` clients, `1` per domain

The configs still use the GGEUR worker path because these three feature
extractor families are already integrated there, but Gaussian augmentation is
disabled:

- `num_generated_per_sample: 0`
- `num_generated_per_prototype: 0`
- `target_size_per_class: 0`

## Files

### ViT-B16

- `officehome_vitb16_fedavg_server.yaml`
- `officehome_vitb16_fedavg_client_1.yaml` ... `client_4.yaml`

### ConvNeXt

- `officehome_convnext_fedavg_server.yaml`
- `officehome_convnext_fedavg_client_1.yaml` ... `client_4.yaml`

Notes:

- Default config is offline-safe: `cnn_pretrained: False`
- If you already have local ConvNeXt weights, set `ggeur.cnn_checkpoint_path`

### Mixer

- `officehome_mixer_fedavg_server.yaml`
- `officehome_mixer_fedavg_client_1.yaml` ... `client_4.yaml`

Notes:

- Uses local `timm_checkpoint_path`
- Feature cache is disabled by default for first-pass distributed validation

## Validation priority

Recommended order:

1. `ViT-B16`
2. `ConvNeXt`
3. `Mixer`

This keeps the first distributed baseline closest to the existing ViT baseline
configs already used in the repo.
