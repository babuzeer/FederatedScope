"""
MLP head for offline-feature-based federated learning.

Input:  pre-extracted feature vectors (e.g. from ConvNeXt-Base, dim=1024)
Output: class logits

No backbone is loaded at all — backbone was used only in extract_features.py.

Usage in YAML:
    model:
      type: mlp_head
      feature_dim: 1024      # must match the backbone used in extraction
      num_classes: 100
      hidden_dim: 256        # 0 = single linear layer
      dropout: 0.1

    trainer:
      type: general          # standard GeneralTorchTrainer works fine

Architecture (hidden_dim > 0):
    LayerNorm(feature_dim)
    -> Linear(feature_dim, hidden_dim)
    -> ReLU
    -> Dropout(dropout)
    -> Linear(hidden_dim, num_classes)
"""

import logging

import torch
import torch.nn as nn

from federatedscope.register import register_model

logger = logging.getLogger(__name__)


class MLPHead(nn.Module):
    """
    Lightweight MLP that operates directly on pre-extracted feature vectors.

    Identical architecture to ClassifierHead in convnext.py so that
    pre-trained weights can be transferred if needed.
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
            # Single linear layer
            self.head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, num_classes),
            )

        n = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[MLPHead] feature_dim={feature_dim}, hidden_dim={hidden_dim}, "
            f"num_classes={num_classes}, dropout={dropout}, params={n:,}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, feature_dim)  — pre-extracted feature vectors
        Returns:
            logits: (B, num_classes)
        """
        return self.head(x)

    # -----------------------------------------------------------------------
    # Compatibility helpers (mirrors ClassifierHead API)
    # -----------------------------------------------------------------------

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return intermediate embedding (output of hidden layer)."""
        if self.hidden_dim > 0:
            # Run up to the penultimate linear and return activations
            embedding = self.head[:-1](x)   # through Dropout
        else:
            embedding = x
        return embedding


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

def build_mlp_head(model_config, local_data=None) -> MLPHead:
    feature_dim = getattr(model_config, 'feature_dim', 1024)
    num_classes  = getattr(model_config, 'num_classes',  100)
    hidden_dim   = getattr(model_config, 'hidden_dim',   256)
    dropout      = getattr(model_config, 'dropout',      0.1)
    return MLPHead(
        feature_dim=feature_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def call_mlp_head_model(model_config, local_data=None):
    if model_config.type.lower() == 'mlp_head':
        return build_mlp_head(model_config, local_data)
    return None


register_model('mlp_head', call_mlp_head_model)
