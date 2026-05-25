"""
Timm-based feature extractor for GGEUR.

This provides a third feature extractor category besides:
  - 'clip' (ViT-based CLIP via open_clip)
  - 'cnn'  (torchvision/timm CNN backbones)

Use `cfg.ggeur.feature_extractor: 'timm'` and set `cfg.ggeur.timm_model`
to any model name supported by timm (e.g., GFNet / Mixer / etc.).

The extractor loads a pretrained image model, removes its classification head,
and returns a (B, D) embedding suitable for GGEUR statistics + Gaussian
feature generation, while training only the downstream MLP head.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm not installed. To use ggeur.feature_extractor='timm', "
                   "install with: pip install timm")


def _resolve_timm_model_name(model_name: str) -> str:
    """
    Resolve user-provided timm model name to an installed timm model id.

    This supports a few common short aliases (e.g. 'gfnet_tiny') and provides
    useful suggestions when the model is unknown.
    """
    name = str(model_name)
    available = set(timm.list_models())
    if name in available:
        return name

    # Common aliases across timm versions / papers
    alias_map = {
        # GFNet (common timm ids include the patch + resolution suffix)
        'gfnet_tiny': 'gfnet_tiny_patch4_224',
        'gfnet_small': 'gfnet_small_patch4_224',
        'gfnet_base': 'gfnet_base_patch4_224',
    }
    alias = alias_map.get(name)
    if alias and alias in available:
        return alias

    # Try common suffix patterns
    candidates = [
        f"{name}_patch4_224",
        f"{name}_patch16_224",
        f"{name}_224",
    ]
    for cand in candidates:
        if cand in available:
            return cand

    # Suggestions
    token = name.split('_', 1)[0]
    suggestions = [m for m in sorted(available) if token in m][:30]
    hint = ""
    if suggestions:
        hint = f" Suggestions (contains '{token}'): {', '.join(suggestions)}"
    raise RuntimeError(
        f"Unknown timm model ({name})."
        f"{hint} "
        f"Tip: run `python -c \"import timm; print([m for m in timm.list_models() if '{token}' in m])\"`."
    )


class TimmFeatureExtractor(nn.Module):
    """
    Generic timm feature extractor that outputs a pooled embedding.

    Args:
        model_name: timm model name, e.g. 'gfnet_tiny', 'mixer_b16_224', ...
        pretrained: whether to load pretrained weights
        freeze: whether to freeze backbone parameters (recommended)
        in_chans: number of input channels (default: 3)
        global_pool: pooling mode passed to timm create_model (default: 'avg')
    """

    def __init__(self,
                 model_name: str,
                 pretrained: bool = True,
                 freeze: bool = True,
                 checkpoint_path: str = '',
                 in_chans: int = 3,
                 global_pool: str = 'avg'):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for TimmFeatureExtractor. "
                              "Install with: pip install timm")

        self.model_name = str(model_name)
        self.pretrained = bool(pretrained)
        self.freeze = bool(freeze)
        self.checkpoint_path = str(checkpoint_path or '')
        self.in_chans = int(in_chans)
        self.global_pool = str(global_pool)

        resolved_name = _resolve_timm_model_name(self.model_name)

        # If a local checkpoint is provided, avoid any online download.
        # timm will load weights from checkpoint_path when provided.
        use_pretrained = self.pretrained
        ckpt = self.checkpoint_path.strip()
        if ckpt:
            if not Path(ckpt).exists():
                raise FileNotFoundError(f"timm_checkpoint_path not found: {ckpt}")
            use_pretrained = False

        # num_classes=0 removes the classification head and returns embeddings
        try:
            self.backbone = timm.create_model(
                resolved_name,
                pretrained=use_pretrained,
                num_classes=0,
                in_chans=self.in_chans,
                global_pool=self.global_pool,
                checkpoint_path=ckpt if ckpt else None,
            )
        except Exception as e:
            if ckpt:
                raise

            if use_pretrained:
                raise RuntimeError(
                    f"Failed to create timm model '{resolved_name}' with pretrained weights. "
                    f"This usually happens when the environment cannot reach HuggingFace to download weights.\n"
                    f"Fix options:\n"
                    f"  1) Download weights on a machine with internet and set cfg.ggeur.timm_checkpoint_path to the local file.\n"
                    f"  2) Set cfg.ggeur.timm_pretrained=False to run without pretrained weights.\n"
                    f"  3) If your network is slow, increase HuggingFace timeouts, e.g.:\n"
                    f"     HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=600\n"
                    f"Original error: {e}"
                ) from e

            raise

        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.feature_dim = self._infer_feature_dim()
        logger.info(f"TimmFeatureExtractor: {self.model_name}, feature_dim={self.feature_dim}, "
                    f"pretrained={self.pretrained}, frozen={self.freeze}")

    def _infer_feature_dim(self) -> int:
        declared_dim = getattr(self.backbone, 'num_features', None)

        # Use the actual forward output as the source of truth. Some timm
        # backbones keep num_features as the channel count even when forward
        # returns an unpooled spatial tensor that this wrapper flattens.
        default_cfg = getattr(self.backbone, 'default_cfg', {}) or {}
        input_size = default_cfg.get('input_size', (self.in_chans, 224, 224))
        if not (isinstance(input_size, (tuple, list)) and len(input_size) == 3):
            input_size = (self.in_chans, 224, 224)

        self.backbone.eval()
        with torch.no_grad():
            x = torch.zeros(1, int(input_size[0]), int(input_size[1]), int(input_size[2]))
            try:
                y = self.backbone(x)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to infer feature dim for timm model '{self.model_name}'. "
                    f"Please set cfg.ggeur.embedding_dim explicitly or choose a model with `num_features`. "
                    f"Original error: {e}"
                )
            if y.dim() > 2:
                y = y.view(y.size(0), -1)
            actual_dim = int(y.shape[1])

        if (isinstance(declared_dim, int) and declared_dim > 0
                and int(declared_dim) != actual_dim):
            logger.warning(
                f"TimmFeatureExtractor: model num_features={declared_dim} "
                f"but flattened forward dim={actual_dim}; using actual dim.")

        return actual_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            self.backbone.eval()
            with torch.no_grad():
                y = self.backbone(x)
        else:
            y = self.backbone(x)

        if y.dim() > 2:
            y = y.view(y.size(0), -1)
        return y

    def get_feature_dim(self) -> int:
        return int(self.feature_dim)
