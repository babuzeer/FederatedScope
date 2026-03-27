"""
ConvNeXt Model for FedProto and other federated learning methods.

Provides ConvNeXt models as standalone models that can be used
with any federated learning method (FedProto, FedAvg, etc.)
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
    logger.warning("ConvNeXt models not available. Please upgrade torchvision.")


class ConvNeXtClassifier(nn.Module):
    """
    ConvNeXt model with classifier for federated learning.

    Supports:
    - Feature extraction via get_embedding()
    - Classification via forward()
    - Works with FedProto, FedAvg, and other FL methods
    """

    MODEL_CONFIGS = {
        'convnext_tiny': (768, ConvNeXt_Tiny_Weights if CONVNEXT_AVAILABLE else None),
        'convnext_small': (768, ConvNeXt_Small_Weights if CONVNEXT_AVAILABLE else None),
        'convnext_base': (1024, ConvNeXt_Base_Weights if CONVNEXT_AVAILABLE else None),
        'convnext_large': (1536, ConvNeXt_Large_Weights if CONVNEXT_AVAILABLE else None),
    }

    def __init__(self, model_name='convnext_base', num_classes=65, pretrained=True):
        """
        Initialize ConvNeXt classifier.

        Args:
            model_name: ConvNeXt variant (tiny, small, base, large)
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
        """
        super(ConvNeXtClassifier, self).__init__()

        if not CONVNEXT_AVAILABLE:
            raise ImportError("ConvNeXt models require torchvision >= 0.13")

        self.model_name = model_name.lower()
        self.num_classes = num_classes

        # Get feature dimension
        if self.model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. "
                           f"Available: {list(self.MODEL_CONFIGS.keys())}")

        self.feature_dim, weights_class = self.MODEL_CONFIGS[self.model_name]

        # Build backbone
        weights = weights_class.IMAGENET1K_V1 if pretrained and weights_class else None

        if self.model_name == 'convnext_tiny':
            self.backbone = convnext_tiny(weights=weights)
        elif self.model_name == 'convnext_small':
            self.backbone = convnext_small(weights=weights)
        elif self.model_name == 'convnext_base':
            self.backbone = convnext_base(weights=weights)
        elif self.model_name == 'convnext_large':
            self.backbone = convnext_large(weights=weights)

        # Remove original classifier, keep avgpool
        # ConvNeXt structure: features -> avgpool -> flatten -> classifier
        # We keep features + avgpool, replace classifier
        original_classifier = self.backbone.classifier
        self.backbone.classifier = nn.Identity()

        # Build new classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, num_classes)
        )

        logger.info(f"ConvNeXtClassifier: {model_name}, feature_dim={self.feature_dim}, "
                   f"num_classes={num_classes}, pretrained={pretrained}")

    def get_embedding(self, x):
        """
        Extract embeddings from images (for FedProto).

        Args:
            x: Input images, shape (B, 3, H, W)

        Returns:
            embeddings: Feature vectors, shape (B, feature_dim)
        """
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        return features

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input images, shape (B, 3, H, W)

        Returns:
            logits: Class logits, shape (B, num_classes)
        """
        features = self.get_embedding(x)
        logits = self.classifier(features)
        return logits


def build_convnext_model(model_config, local_data=None):
    """
    Build ConvNeXt model from config.

    Args:
        model_config: Configuration object with model settings
        local_data: Local data (unused, for compatibility)

    Returns:
        ConvNeXtClassifier instance
    """
    # Determine model variant from type
    model_type = model_config.type.lower()

    # Map model type to ConvNeXt variant
    if model_type in ['convnext_tiny', 'convnext-tiny']:
        model_name = 'convnext_tiny'
    elif model_type in ['convnext_small', 'convnext-small']:
        model_name = 'convnext_small'
    elif model_type in ['convnext_base', 'convnext-base', 'convnext']:
        model_name = 'convnext_base'
    elif model_type in ['convnext_large', 'convnext-large']:
        model_name = 'convnext_large'
    else:
        model_name = 'convnext_base'  # Default

    # Get num_classes
    num_classes = getattr(model_config, 'num_classes', 65)

    # Get pretrained setting (default True)
    pretrained = getattr(model_config, 'pretrained', True)

    model = ConvNeXtClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained
    )

    return model


def call_convnext_model(model_config, local_data=None):
    """Factory function for ConvNeXt models."""
    model_type = model_config.type.lower()
    if model_type.startswith('convnext'):
        return build_convnext_model(model_config, local_data)
    return None


# Register all ConvNeXt variants
register_model('convnext_tiny', call_convnext_model)
register_model('convnext_small', call_convnext_model)
register_model('convnext_base', call_convnext_model)
register_model('convnext_large', call_convnext_model)
register_model('convnext', call_convnext_model)  # Default to base
