"""
Configuration for the GGEUR head-only hierarchical training PoC.

This module intentionally adds a switch and system-level settings without
changing the existing GGEUR server/client implementations.
"""

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_ggeur_headonly_cfg(cfg):
    if not hasattr(cfg, 'ggeur'):
        cfg.ggeur = CN()

    # Top-level switch requested by the head-only design document.
    cfg.ggeur.head_only_mode = False

    cfg.ggeur_headonly = CN()
    cfg.ggeur_headonly.use = False

    # Single-machine hierarchical FL settings.
    cfg.ggeur_headonly.client_total = 16
    cfg.ggeur_headonly.sample_clients_per_round = 8
    cfg.ggeur_headonly.num_sub_servers = 2
    cfg.ggeur_headonly.min_received_ratio = 0.8
    cfg.ggeur_headonly.round_timeout = 0.0
    cfg.ggeur_headonly.late_update_policy = 'discard'
    cfg.ggeur_headonly.process_workers = 0

    # Round-0 real data generation and feature-cache settings. When enabled,
    # the script loads the configured real dataset, extracts frozen-backbone
    # features, aggregates statistics, applies Gaussian augmentation, and
    # persists augmented feature caches before MLP-only FL starts.
    cfg.ggeur_headonly.run_data_generation = True
    cfg.ggeur_headonly.overwrite_feature_cache = True
    cfg.ggeur_headonly.feature_cache_dir = ''
    cfg.ggeur_headonly.feature_cache_version = 'fcache_v1'
    cfg.ggeur_headonly.reuse_feature_cache = True
    cfg.ggeur_headonly.augmentation_noise_scale = 1.0
    cfg.ggeur_headonly.covariance_epsilon = 1e-4
    cfg.ggeur_headonly.use_full_covariance = True
    cfg.ggeur_headonly.evaluate_test = True
    cfg.ggeur_headonly.eval_freq = 1
    cfg.ggeur_headonly.test_feature_cache = ''

    # Communication payload settings.
    cfg.ggeur_headonly.param_dtype = 'float32'
    cfg.ggeur_headonly.run_root = 'exp/ggeur_headonly_runs'
    cfg.ggeur_headonly.run_id = ''
    cfg.ggeur_headonly.output_json = ''

    cfg.register_cfg_check_fun(assert_ggeur_headonly_cfg)
    return cfg


def assert_ggeur_headonly_cfg(cfg):
    use_headonly = (
        hasattr(cfg, 'ggeur_headonly') and cfg.ggeur_headonly.use
    ) or (
        hasattr(cfg, 'ggeur') and
        getattr(cfg.ggeur, 'head_only_mode', False)
    )
    if not use_headonly:
        return

    incompatible = [
        'use_cnn_distillation',
        'use_feature_alignment',
        'use_separated_training',
        'use_end_to_end_finetune',
        'use_promptfl',
    ]
    enabled = [
        name for name in incompatible
        if hasattr(cfg.ggeur, name) and getattr(cfg.ggeur, name)
    ]
    if enabled:
        raise ValueError(
            'GGEUR head-only mode only supports the standard MLP head path. '
            f'Disable incompatible options: {enabled}'
        )

    if cfg.ggeur_headonly.num_sub_servers <= 0:
        raise ValueError('ggeur_headonly.num_sub_servers must be positive')
    if cfg.ggeur_headonly.client_total <= 0:
        raise ValueError('ggeur_headonly.client_total must be positive')
    if cfg.ggeur_headonly.sample_clients_per_round <= 0:
        raise ValueError(
            'ggeur_headonly.sample_clients_per_round must be positive'
        )
    if cfg.ggeur_headonly.sample_clients_per_round > cfg.ggeur_headonly.client_total:
        raise ValueError(
            'ggeur_headonly.sample_clients_per_round cannot exceed '
            'ggeur_headonly.client_total'
        )
    if not 0 < cfg.ggeur_headonly.min_received_ratio <= 1:
        raise ValueError('ggeur_headonly.min_received_ratio must be in (0, 1]')
    if cfg.ggeur_headonly.param_dtype not in ['float32', 'float16']:
        raise ValueError("ggeur_headonly.param_dtype must be 'float32' or 'float16'")


register_config('ggeur_headonly', extend_ggeur_headonly_cfg)
