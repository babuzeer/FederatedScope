# Third-Party Accuracy Matrix Rerun

Date: 2026-07-08

This note records the cross-domain heterogeneous accuracy rerun matrix prepared
for third-party testing.

## Scope

The active matrix for this test round contains 36 runs. PACS configs still
exist in the repository, but PACS is intentionally excluded for now.

- Datasets: `officehome`, `domainnet`
- Dataset/model groups:
  - `officehome/cnn`
  - `officehome/mixer`
  - `officehome/vit`
  - `domainnet/cnn`
  - `domainnet/mlp`
  - `domainnet/vit`
- Methods per group: `ggeur`, `fedavg`, `fedprox`, `fedproto`, `fedopt`, `moon`

Each run uses the existing YAML hyper-parameters in `scripts/example_configs`.
The third-party test entrypoint is config-driven:

- Matrix config: `scripts/thirdparty_accuracy_cases.yaml`
- Single-case runner: `scripts/run_thirdparty_accuracy_case.py`
- Background launcher: `scripts/start_thirdparty_accuracy_case.sh`
- Batch helper for producing current results: `scripts/run_thirdparty_accuracy_config_batch.sh`

The runner requires an explicit case unless `--allow-multiple` is passed. This
keeps the third-party test workflow as one method launch at a time.

## Server Paths

Expected dataset roots on the SeeTacloud server:

- OfficeHome: `/root/autodl-tmp/datasets/OfficeHomeDataset_10072016`
- DomainNet: `/root/autodl-tmp/datasets/DomainNet`
Use `scripts/run_thirdparty_accuracy_case.py --list-cases` to list supported
case names.

## Cache Policy

GGEUR training now uses generated feature cache by default:

- `ggeur.reuse_augmented_feature_cache: True`
- `ggeur.save_augmented_feature_cache: True`
- `ggeur.augmented_feature_cache_dir: ''`
- `ggeur.augmented_feature_cache_version: 'aug_fcache_v1'`

When the cache metadata matches the current dataset split, client count,
feature extractor, LDS settings, class count, and GGEUR generation parameters,
clients load the generated feature file first and skip repeated generation.
Changing `federate.client_num`, `num_generated_per_sample`,
`num_generated_per_prototype`, `target_size_per_class`, backbone, LDS settings,
or cache version creates a different cache namespace.

Set `ggeur.reuse_augmented_feature_cache False` to force regeneration while
still writing a new cache. Set `ggeur.save_augmented_feature_cache False` to
avoid writing generated feature caches.

## Per-Case Remote Commands

List cases:

```bash
cd /root/autodl-tmp/FederatedScope_thirdparty_20260708_1030
/root/miniconda3/envs/fs/bin/python scripts/run_thirdparty_accuracy_case.py --list-cases
```

Start one method in the background:

```bash
cd /root/autodl-tmp/FederatedScope_thirdparty_20260708_1030
RUN_ID=thirdparty_accuracy_manual \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/start_thirdparty_accuracy_case.sh officehome__cnn__ggeur
```

Run one method in the foreground:

```bash
cd /root/autodl-tmp/FederatedScope_thirdparty_20260708_1030
/root/miniconda3/envs/fs/bin/python scripts/run_thirdparty_accuracy_case.py \
  --case officehome__cnn__ggeur \
  --run-id thirdparty_accuracy_manual \
  --plot-group
```

After all six methods in the same dataset/model group finish under the same
`RUN_ID`, the `--plot-group` option updates the combined accuracy chart.

Run the current full OfficeHome + DomainNet matrix in the background:

```bash
cd /root/autodl-tmp/FederatedScope_thirdparty_20260708_1030
RUN_ID=thirdparty_accuracy_config_<timestamp> \
CUDA_VISIBLE_DEVICES=0 \
nohup bash scripts/run_thirdparty_accuracy_config_batch.sh . \
  > thirdparty_accuracy_config_nohup.log 2>&1 &
```

## Outputs

For run id `<run_id>`, outputs are written under:

```text
exp/ggeur_accuracy_reruns/<run_id>/
```

Important files:

- `manifest.json`: selected cases and config paths
- `state.json`: live per-case state for resume
- `summary.csv` and `summary.json`: final per-case status and last parsed accuracy
- `experiment_log.md`: start/end trace for each case
- `plots/*_all_methods_accuracy.png`: one combined chart per dataset/model group
- `launcher_logs/*.nohup.log`: background launcher stdout/stderr

## Legacy Batch Command

The previous full-matrix helper is still present for internal reruns, but it is
not the third-party test entrypoint.

```bash
cd /root/autodl-tmp/FederatedScope_thirdparty_<timestamp>
RUN_ID=thirdparty_accuracy_<timestamp> \
PYTHON_BIN=/root/miniconda3/envs/fs/bin/python \
CUDA_VISIBLE_DEVICES=0 \
THIRDPARTY_DATASETS="officehome domainnet" \
nohup bash scripts/run_thirdparty_accuracy_matrix.sh . \
  > thirdparty_accuracy_nohup.log 2>&1 &
```
