"""客户端更新级自适应裁剪差分隐私工具。

机制：
    tilde_Delta = clip(Delta, C_t) + N(0, sigma^2 * C_t^2 * I)
    C_{t+1} = ema * C_t + (1 - ema) * quantile({||Delta_i||}, q)

目标：在保持噪声乘数 sigma 不变的前提下，通过自适应收紧裁剪界 C_t
来降低实际注入的噪声方差（约降至固定裁剪 DP 的 50%）。
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

TensorTree = Any


def _is_float_tensor(value: Any) -> bool:
    return torch.is_tensor(value) and torch.is_floating_point(value)


def _iter_float_tensors(payload: TensorTree):
    if _is_float_tensor(payload):
        yield payload
    elif isinstance(payload, Mapping):
        for value in payload.values():
            yield from _iter_float_tensors(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            yield from _iter_float_tensors(value)


def compute_update_norm(update: TensorTree) -> float:
    total_sq = 0.0
    has_tensor = False
    for tensor in _iter_float_tensors(update):
        t = tensor.detach().float().cpu()
        total_sq += float(torch.sum(t * t).item())
        has_tensor = True
    return float(total_sq ** 0.5) if has_tensor else 0.0


def clone_payload(payload: TensorTree) -> TensorTree:
    if torch.is_tensor(payload):
        return payload.detach().cpu().clone()
    if isinstance(payload, dict):
        return {key: clone_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [clone_payload(value) for value in payload]
    if isinstance(payload, tuple):
        return tuple(clone_payload(value) for value in payload)
    return copy.deepcopy(payload)


def scale_update(update: TensorTree, scale: float) -> TensorTree:
    if _is_float_tensor(update):
        return update.detach().cpu().float().clone() * float(scale)
    if torch.is_tensor(update):
        return update.detach().cpu().clone()
    if isinstance(update, dict):
        return {key: scale_update(value, scale) for key, value in update.items()}
    if isinstance(update, list):
        return [scale_update(value, scale) for value in update]
    if isinstance(update, tuple):
        return tuple(scale_update(value, scale) for value in update)
    return copy.deepcopy(update)


def add_gaussian_noise(update, clip_bound, noise_multiplier, generator=None):
    noise_std = float(noise_multiplier) * float(clip_bound)
    if _is_float_tensor(update):
        base = update.detach().cpu().float().clone()
        if noise_std <= 0:
            return base
        noise = torch.normal(
            mean=0.0, std=noise_std, size=base.shape,
            generator=generator, device=base.device)
        return base + noise
    if torch.is_tensor(update):
        return update.detach().cpu().clone()
    if isinstance(update, dict):
        return {k: add_gaussian_noise(v, clip_bound, noise_multiplier, generator)
                for k, v in update.items()}
    if isinstance(update, list):
        return [add_gaussian_noise(v, clip_bound, noise_multiplier, generator)
                for v in update]
    if isinstance(update, tuple):
        return tuple(add_gaussian_noise(v, clip_bound, noise_multiplier, generator)
                     for v in update)
    return copy.deepcopy(update)


def clip_update(update, clip_bound, eps=1e-12):
    raw_norm = compute_update_norm(update)
    clip_bound = float(clip_bound)
    scale = min(1.0, clip_bound / (raw_norm + float(eps))) if raw_norm > 0 else 1.0
    clipped = scale_update(update, scale)
    return clipped, {
        'raw_norm': float(raw_norm),
        'clip_bound': float(clip_bound),
        'clip_factor': float(scale),
        'clipped': bool(scale < 1.0),
    }


def sanitize_update(update, clip_bound, noise_multiplier, eps=1e-12, generator=None):
    clipped, stats = clip_update(update, clip_bound, eps=eps)
    noisy = add_gaussian_noise(clipped, clip_bound, noise_multiplier, generator)
    noise_std = float(noise_multiplier) * float(clip_bound)
    stats.update({
        'noise_multiplier': float(noise_multiplier),
        'noise_std': float(noise_std),
        'sanitized_norm': float(compute_update_norm(noisy)),
    })
    return noisy, stats


def update_clip_bound(old_clip, norms, target_quantile, ema, min_clip, max_clip):
    clean = np.asarray(
        [float(v) for v in norms if np.isfinite(float(v))], dtype=np.float64)
    if clean.size == 0:
        return float(old_clip)
    q = float(np.quantile(clean, float(target_quantile)))
    new_clip = float(ema) * float(old_clip) + (1.0 - float(ema)) * q
    return float(np.clip(new_clip, float(min_clip), float(max_clip)))


@dataclass
class AdaptiveDPConfig:
    use: bool = True
    initial_clip: float = 1.0
    target_quantile: float = 0.7
    ema: float = 0.9
    min_clip: float = 0.05
    max_clip: float = 10.0
    noise_multiplier: float = 1.0
    eps: float = 1e-12
    expected_clients: Optional[int] = None
    seed: Optional[int] = None

    @classmethod
    def from_cfg_node(cls, cfg_node):
        return cls(
            use=bool(getattr(cfg_node, 'use', False)),
            initial_clip=float(getattr(cfg_node, 'initial_clip', 1.0)),
            target_quantile=float(getattr(cfg_node, 'target_quantile', 0.7)),
            ema=float(getattr(cfg_node, 'ema', 0.9)),
            min_clip=float(getattr(cfg_node, 'min_clip', 0.05)),
            max_clip=float(getattr(cfg_node, 'max_clip', 10.0)),
            noise_multiplier=float(getattr(cfg_node, 'noise_multiplier', 1.0)),
            eps=float(getattr(cfg_node, 'eps', 1e-12)),
            expected_clients=getattr(cfg_node, 'expected_clients', None),
            seed=getattr(cfg_node, 'seed', None),
        )


@dataclass
class AdaptiveDPController:
    config: AdaptiveDPConfig
    current_clip: float = field(init=False)
    round_norms: MutableMapping[int, Dict[int, float]] = field(default_factory=dict)
    clip_history: Dict[int, float] = field(default_factory=dict)
    noise_std_history: Dict[int, float] = field(default_factory=dict)
    clipped_fraction_history: Dict[int, float] = field(default_factory=dict)
    _generator: Optional[torch.Generator] = field(default=None, init=False)

    def __post_init__(self):
        self.current_clip = float(self.config.initial_clip)
        if self.config.seed is not None:
            self._generator = torch.Generator(device='cpu')
            self._generator.manual_seed(int(self.config.seed))

    def expected_clients(self, fallback):
        if self.config.expected_clients is not None:
            return max(1, int(self.config.expected_clients))
        return max(1, int(fallback))

    def sanitize(self, update, round_idx, client_id, expected_clients):
        clip_used = float(self.current_clip)
        sanitized, stats = sanitize_update(
            update, clip_bound=clip_used,
            noise_multiplier=self.config.noise_multiplier,
            eps=self.config.eps, generator=self._generator)
        stats.update({
            'enabled': True,
            'mechanism': 'adaptive_client_update_dp',
            'round': int(round_idx),
            'client_id': int(client_id),
            'target_quantile': float(self.config.target_quantile),
            'ema': float(self.config.ema),
        })
        bucket = self.round_norms.setdefault(int(round_idx), {})
        bucket[int(client_id)] = float(stats['raw_norm'])
        exp_clients = self.expected_clients(expected_clients)
        if len(bucket) >= exp_clients and int(round_idx) not in self.clip_history:
            old_clip = self.current_clip
            norms = list(bucket.values())
            next_clip = update_clip_bound(
                old_clip, norms, self.config.target_quantile,
                self.config.ema, self.config.min_clip, self.config.max_clip)
            clipped_fraction = float(
                np.mean([n > old_clip for n in norms])) if norms else 0.0
            self.clip_history[int(round_idx)] = float(old_clip)
            self.noise_std_history[int(round_idx)] = float(stats['noise_std'])
            self.clipped_fraction_history[int(round_idx)] = clipped_fraction
            self.current_clip = float(next_clip)
            logger.info(
                '[AdaptiveDP] round=%s old_clip=%.6f next_clip=%.6f '
                'mean_norm=%.6f clipped_fraction=%.4f noise_std=%.6f',
                round_idx, old_clip, next_clip,
                float(np.mean(norms)), clipped_fraction,
                float(stats['noise_std']))
        return sanitized, stats

    def summarize_noise_variance(self):
        if self.noise_std_history:
            noise_stds = np.asarray(
                list(self.noise_std_history.values()), dtype=np.float64)
        else:
            noise_stds = np.asarray([], dtype=np.float64)
        if noise_stds.size == 0:
            return {
                'num_rounds': 0, 'mean_noise_std': 0.0,
                'mean_noise_variance': 0.0, 'min_noise_variance': 0.0,
                'max_noise_variance': 0.0, 'total_noise_variance': 0.0,
                'final_noise_variance': 0.0,
            }
        variances = np.square(noise_stds)
        return {
            'num_rounds': int(noise_stds.size),
            'mean_noise_std': float(noise_stds.mean()),
            'mean_noise_variance': float(variances.mean()),
            'min_noise_variance': float(variances.min()),
            'max_noise_variance': float(variances.max()),
            'total_noise_variance': float(variances.sum()),
            'final_noise_variance': float(variances[-1]),
        }


_GLOBAL_CONTROLLERS: Dict[str, AdaptiveDPController] = {}


def get_global_controller(config, key='default'):
    controller = _GLOBAL_CONTROLLERS.get(key)
    if controller is None:
        controller = AdaptiveDPController(config)
        _GLOBAL_CONTROLLERS[key] = controller
    return controller


def is_adaptive_dp_enabled(cfg):
    adaptive_dp_cfg = getattr(cfg, 'adaptive_dp', None)
    return adaptive_dp_cfg is not None and bool(getattr(adaptive_dp_cfg, 'use', False))


def get_controller_for_cfg(cfg, key='default'):
    if not is_adaptive_dp_enabled(cfg):
        return None
    config = AdaptiveDPConfig.from_cfg_node(cfg.adaptive_dp)
    return get_global_controller(config, key)
