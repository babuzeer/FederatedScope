# HeadOnly cache generation on 4090

This bundle runs the full GGEUR HeadOnly path on the 4090 server:

- Round 0 extracts ViT-B/16 features, uploads statistics, generates augmented features.
- Generated HeadOnly feature caches are written to `exp/ggeur_headonly_real_cache`.
- The remaining rounds train the MLP head so the run is a complete training run.

Defaults:

- `CLIENT_NUM=120`
- `TOTAL_ROUNDS=20`
- `GEN_NUM=20`
- `PYTHON_BIN=/root/.local/share/mamba/envs/GGEUR/bin/python`
- `REPO_DIR=/root/autodl-tmp/FederatedScope`

`CLIENT_NUM` should be divisible by 4 for the OfficeHome LDS split.

Run on the 4090 server:

```bash
cd /root/autodl-tmp/FederatedScope
unzip -o headonly_cache_generation_4090_bundle.zip -d .
export PYTHON_BIN=/root/.local/share/mamba/envs/GGEUR/bin/python
CLIENT_NUM=120 TOTAL_ROUNDS=20 GEN_NUM=20 bash scripts/headonly_cache_generation/run_officehome_vit_cachegen_4090.sh
```

The script prints the run directory at the end. The main log is:

```bash
tail -f exp/headonly_cache_generation/runs/<run_id>/logs/server.log
```

The generated cache files are under:

```text
exp/ggeur_headonly_real_cache/officehome_vitb16_<CLIENT_NUM>c_gen20_fcache_v1/headonly_augmented/officehome_vitb16_<CLIENT_NUM>c_gen20_fcache_v1/
```

The script also writes:

- `system/run_info.log`
- `system/cache_manifest.txt`
- `metrics/metrics_summary.json`

For the 400-client random-sampling cache-generation run, use:

```bash
cd /root/autodl-tmp/FederatedScope
unset PYTHONPATH
PYTHONPATH=/root/autodl-tmp/FederatedScope /root/.local/share/mamba/envs/GGEUR/bin/python -m federatedscope.main --cfg scripts/headonly_cache_generation/officehome_vit_cachegen_400c_random25_gen20.yaml
```

That config uses 4 OfficeHome domains, 100 clients per domain, and 25 train
samples per client. Different clients from the same domain may sample the same
image; a single client's 25 samples are unique unless the domain has fewer than
25 train samples. The 400-client config keeps all cross-client prototypes for
GGEUR augmentation; the server shares one serialized prototype pool and each
client filters out its own entries locally.
