from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_adaptive_dp_cfg(cfg):
    """注册自适应裁剪差分隐私配置。"""
    cfg.adaptive_dp = CN()
    cfg.adaptive_dp.use = False
    cfg.adaptive_dp.initial_clip = 1.0
    cfg.adaptive_dp.target_quantile = 0.7
    cfg.adaptive_dp.ema = 0.9
    cfg.adaptive_dp.min_clip = 0.05
    cfg.adaptive_dp.max_clip = 10.0
    cfg.adaptive_dp.noise_multiplier = 1.0
    cfg.adaptive_dp.eps = 1e-12
    cfg.adaptive_dp.seed = 0
    cfg.adaptive_dp.expected_clients = 0

    cfg.register_cfg_check_fun(assert_adaptive_dp_cfg)


def assert_adaptive_dp_cfg(cfg):
    if not cfg.adaptive_dp.use:
        return
    assert cfg.adaptive_dp.initial_clip > 0
    assert 0.0 < cfg.adaptive_dp.target_quantile <= 1.0
    assert 0.0 <= cfg.adaptive_dp.ema < 1.0
    assert cfg.adaptive_dp.min_clip > 0
    assert cfg.adaptive_dp.max_clip >= cfg.adaptive_dp.min_clip
    assert cfg.adaptive_dp.noise_multiplier >= 0


register_config('adaptive_dp', extend_adaptive_dp_cfg)
