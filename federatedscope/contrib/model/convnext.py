"""
ConvNeXt backbone sharing for memory-efficient federated learning.

Architecture:
  - SharedBackbone: single instance, frozen, on GPU, shared by ALL clients
  - ClassifierHead: one per client, only MLP (few hundred KB), trainable

Usage in standalone mode:
  - model.type = 'convnext_base_head'  -> each client gets ClassifierHead
  - StandaloneRunner._set_up creates ONE SharedBackbone and injects it into
    each trainer via ctx.shared_backbone
  - ConvNeXtTrainer._hook_on_batch_forward uses backbone + head

Memory:
  - Before: 10 clients × 350MB backbone = 3.5 GB
  - After:  1 shared backbone × 350MB + 10 × ~1.5MB head ≈ 365 MB
"""

import logging
import torch
import torch.nn as nn

from federatedscope.register import register_model

logger = logging.getLogger(__name__)

try:
    from torchvision.models import (
        convnext_tiny, convnext_small, convnext_base, convnext_large,
        ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights,
        ConvNeXt_Base_Weights, ConvNeXt_Large_Weights,
    )
    CONVNEXT_AVAILABLE = True
except ImportError:
    CONVNEXT_AVAILABLE = False
    logger.warning("ConvNeXt not available. Upgrade torchvision >= 0.13.")

# Feature dimensions for each variant
CONVNEXT_FEATURE_DIMS = {
    'convnext_tiny':  768,
    'convnext_small': 768,
    'convnext_base':  1024,
    'convnext_large': 1536,
}

CONVNEXT_WEIGHTS = {
    'convnext_tiny':  lambda: ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if CONVNEXT_AVAILABLE else None,
    'convnext_small': lambda: ConvNeXt_Small_Weights.IMAGENET1K_V1 if CONVNEXT_AVAILABLE else None,
    'convnext_base':  lambda: ConvNeXt_Base_Weights.IMAGENET1K_V1 if CONVNEXT_AVAILABLE else None,
    'convnext_large': lambda: ConvNeXt_Large_Weights.IMAGENET1K_V1 if CONVNEXT_AVAILABLE else None,
}

CONVNEXT_BUILDERS = {
    'convnext_tiny':  convnext_tiny if CONVNEXT_AVAILABLE else None,
    'convnext_small': convnext_small if CONVNEXT_AVAILABLE else None,
    'convnext_base':  convnext_base  if CONVNEXT_AVAILABLE else None,
    'convnext_large': convnext_large if CONVNEXT_AVAILABLE else None,
}


# ---------------------------------------------------------------------------
# SharedBackbone
# ---------------------------------------------------------------------------

class SharedBackbone(nn.Module):
    """
    Frozen ConvNeXt backbone shared by all FL clients in standalone mode.

    - Contains: features + avgpool (no classifier head)
    - All parameters: requires_grad=False
    - Singleton per (model_name, pretrained) via class-level registry
    - Device management: auto-moves to the device of the input tensor
    """

    _registry = {}  # {(model_name, pretrained) -> SharedBackbone instance}

    @classmethod
    def get_or_create(cls, model_name: str, pretrained: bool = True) -> 'SharedBackbone':
        """Return a singleton SharedBackbone for the given variant."""
        key = (model_name, pretrained)
        if key not in cls._registry:
            cls._registry[key] = cls(model_name, pretrained)
        return cls._registry[key]

    def __init__(self, model_name: str = 'convnext_base', pretrained: bool = True):
        super().__init__()
        if not CONVNEXT_AVAILABLE:
            raise ImportError("torchvision >= 0.13 required for ConvNeXt.")
        if model_name not in CONVNEXT_BUILDERS:
            raise ValueError(f"Unknown model_name: {model_name}. "
                             f"Choose from {list(CONVNEXT_BUILDERS.keys())}")

        self.model_name = model_name
        self.feature_dim = CONVNEXT_FEATURE_DIMS[model_name]

        weights = CONVNEXT_WEIGHTS[model_name]() if pretrained else None
        raw = CONVNEXT_BUILDERS[model_name](weights=weights)

        # Keep features + avgpool, drop the classifier head
        self.features = raw.features
        self.avgpool  = raw.avgpool

        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[SharedBackbone] Created {model_name} "
            f"(pretrained={pretrained}, params={n_params:,}). "
            f"Shared by all clients."
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features. Auto-moves to the same device as x.
        Output shape: (B, feature_dim)
        """
        # Auto device migration (cheap when device already matches)
        cur_device = next(self.parameters()).device
        if cur_device != x.device:
            self.to(x.device)

        feat = self.features(x)   # (B, C, H', W')
        feat = self.avgpool(feat)  # (B, C, 1, 1)
        feat = feat.flatten(1)    # (B, C)
        return feat

    def to_device(self, device):
        """Explicitly move backbone to a device (called during set_up)."""
        self.to(device)
        return self


# ---------------------------------------------------------------------------
# ClassifierHead  (the actual nn.Module each client holds as its "model")
# ---------------------------------------------------------------------------

class ClassifierHead(nn.Module):
    """
    Lightweight MLP classification head for ConvNeXt-based FL clients.

    Each client holds ONE ClassifierHead (~1-2 MB).
    The shared backbone is accessed via ctx.shared_backbone during training.

    Architecture: LayerNorm -> Linear(feature_dim, hidden_dim) -> ReLU
                  -> Dropout -> Linear(hidden_dim, num_classes)
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        num_classes:  int = 100,
        hidden_dim:   int = 256,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes  = num_classes
        self.hidden_dim   = hidden_dim

        if hidden_dim > 0:
            self.head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, num_classes),
            )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[ClassifierHead] feature_dim={feature_dim}, "
            f"hidden_dim={hidden_dim}, num_classes={num_classes}, "
            f"dropout={dropout}, params={n_params:,}"
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, feature_dim)  – pre-extracted by SharedBackbone
        Returns:
            logits: (B, num_classes)
        """
        return self.head(features)

    # -----------------------------------------------------------------------
    # Compatibility helpers used by FedMIA and eval hooks
    # -----------------------------------------------------------------------

    def get_embedding(self, features: torch.Tensor) -> torch.Tensor:
        """Pass-through: features are already embeddings."""
        return features


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_classifier_head(model_config, local_data=None) -> ClassifierHead:
    """Build a ClassifierHead from cfg.model."""
    model_type = model_config.type.lower()

    # Determine backbone variant to get feature_dim
    if 'tiny' in model_type:
        backbone_name = 'convnext_tiny'
    elif 'small' in model_type:
        backbone_name = 'convnext_small'
    elif 'large' in model_type:
        backbone_name = 'convnext_large'
    else:
        backbone_name = 'convnext_base'

    feature_dim = CONVNEXT_FEATURE_DIMS[backbone_name]
    num_classes = getattr(model_config, 'num_classes', 100)
    hidden_dim  = getattr(model_config, 'hidden_dim',  256)
    dropout     = getattr(model_config, 'dropout',     0.1)

    return ClassifierHead(
        feature_dim=feature_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def call_convnext_head_model(model_config, local_data=None):
    model_type = model_config.type.lower()
    if 'head' in model_type and model_type.startswith('convnext'):
        return build_classifier_head(model_config, local_data)
    return None


# Register the head-only model types
register_model('convnext_base_head',  call_convnext_head_model)
register_model('convnext_tiny_head',  call_convnext_head_model)
register_model('convnext_small_head', call_convnext_head_model)
register_model('convnext_large_head', call_convnext_head_model)
