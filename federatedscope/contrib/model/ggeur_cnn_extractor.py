"""
GGEUR_Clip CNN Feature Extractor

Provides CNN-based feature extractors to replace CLIP in GGEUR_Clip.
Supports ConvNeXt, ResNet, and other pretrained CNN backbones.

Key advantage: The same CNN is used for both feature extraction and final deployment,
eliminating the need for feature alignment.
"""

import logging
import os
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Check available backends
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm not installed. Some models may not be available. "
                   "Install with: pip install timm")

try:
    from torchvision import models as tv_models
    from torchvision.models import (
        convnext_tiny, convnext_small, convnext_base, convnext_large,
        ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights,
        ConvNeXt_Base_Weights, ConvNeXt_Large_Weights,
        resnet18, resnet34, resnet50, resnet101,
        ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights,
        efficientnet_b0, efficientnet_b4, efficientnet_b7,
        EfficientNet_B0_Weights, EfficientNet_B4_Weights, EfficientNet_B7_Weights
    )
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    logger.warning("torchvision models not available")


class CNNFeatureExtractor(nn.Module):
    """
    CNN-based feature extractor for GGEUR_Clip.

    Supports multiple backbone architectures:
    - ConvNeXt (Tiny, Small, Base, Large)
    - ResNet (18, 34, 50, 101)
    - EfficientNet (B0, B4, B7)

    Usage:
        extractor = CNNFeatureExtractor('convnext_base', pretrained=True)
        features = extractor(images)  # (B, feature_dim)
    """

    # Model configurations: (feature_dim, recommended_for_ggeur)
    MODEL_CONFIGS = {
        # ConvNeXt family (recommended)
        'convnext_tiny': (768, True),
        'convnext_small': (768, True),
        'convnext_base': (1024, True),  # Best balance
        'convnext_large': (1536, True),

        # ResNet family
        'resnet18': (512, False),
        'resnet34': (512, False),
        'resnet50': (2048, True),
        'resnet101': (2048, True),

        # EfficientNet family
        'efficientnet_b0': (1280, False),
        'efficientnet_b4': (1792, True),
        'efficientnet_b7': (2560, True),
    }

    def __init__(self,
                 model_name='convnext_base',
                 pretrained=True,
                 freeze=True,
                 checkpoint_path=''):
        """
        Initialize CNN feature extractor.

        Args:
            model_name: Name of the backbone model
            pretrained: Whether to use pretrained weights
            freeze: Whether to freeze the backbone (recommended for GGEUR_Clip Phase 1)
        """
        super(CNNFeatureExtractor, self).__init__()

        self.model_name = model_name.lower()
        self.pretrained = pretrained
        self.freeze = freeze
        self.checkpoint_path = str(checkpoint_path or '')

        # Get feature dimension
        if self.model_name in self.MODEL_CONFIGS:
            self.feature_dim = self.MODEL_CONFIGS[self.model_name][0]
        else:
            raise ValueError(f"Unsupported model: {model_name}. "
                           f"Available: {list(self.MODEL_CONFIGS.keys())}")

        # Build backbone
        self.backbone = self._build_backbone()

        self._load_checkpoint()

        # Freeze if requested
        if freeze:
            self._freeze_backbone()

        logger.info(f"CNNFeatureExtractor: {model_name}, feature_dim={self.feature_dim}, "
                   f"pretrained={pretrained}, frozen={freeze}, "
                   f"checkpoint_path={self.checkpoint_path or 'None'}")

    def _build_backbone(self):
        """Build the backbone network"""

        # ConvNeXt models
        if self.model_name == 'convnext_tiny':
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if self.pretrained else None
            model = convnext_tiny(weights=weights)
            model.classifier = nn.Identity()  # Remove classifier

        elif self.model_name == 'convnext_small':
            weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if self.pretrained else None
            model = convnext_small(weights=weights)
            model.classifier = nn.Identity()

        elif self.model_name == 'convnext_base':
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if self.pretrained else None
            model = convnext_base(weights=weights)
            model.classifier = nn.Identity()

        elif self.model_name == 'convnext_large':
            weights = ConvNeXt_Large_Weights.IMAGENET1K_V1 if self.pretrained else None
            model = convnext_large(weights=weights)
            model.classifier = nn.Identity()

        # ResNet models
        elif self.model_name == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if self.pretrained else None
            model = resnet18(weights=weights)
            model.fc = nn.Identity()

        elif self.model_name == 'resnet34':
            weights = ResNet34_Weights.IMAGENET1K_V1 if self.pretrained else None
            model = resnet34(weights=weights)
            model.fc = nn.Identity()

        elif self.model_name == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if self.pretrained else None
            model = resnet50(weights=weights)
            model.fc = nn.Identity()

        elif self.model_name == 'resnet101':
            weights = ResNet101_Weights.IMAGENET1K_V1 if self.pretrained else None
            model = resnet101(weights=weights)
            model.fc = nn.Identity()

        # EfficientNet models
        elif self.model_name == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if self.pretrained else None
            model = efficientnet_b0(weights=weights)
            model.classifier = nn.Identity()

        elif self.model_name == 'efficientnet_b4':
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if self.pretrained else None
            model = efficientnet_b4(weights=weights)
            model.classifier = nn.Identity()

        elif self.model_name == 'efficientnet_b7':
            weights = EfficientNet_B7_Weights.IMAGENET1K_V1 if self.pretrained else None
            model = efficientnet_b7(weights=weights)
            model.classifier = nn.Identity()

        # Try timm for other models
        elif TIMM_AVAILABLE:
            model = timm.create_model(self.model_name, pretrained=self.pretrained,
                                      num_classes=0)  # num_classes=0 removes classifier
        else:
            raise ValueError(f"Model {self.model_name} not found. "
                           f"Try installing timm: pip install timm")

        return model

    def _extract_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict):
            for key in ['state_dict', 'model', 'model_state_dict', 'net']:
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    return checkpoint[key]
        return checkpoint

    def _load_checkpoint(self):
        ckpt = self.checkpoint_path.strip()
        if not ckpt:
            return
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"cnn_checkpoint_path not found: {ckpt}")

        checkpoint = torch.load(ckpt, map_location='cpu')
        state_dict = self._extract_state_dict(checkpoint)
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported checkpoint format in {ckpt}")

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                key = key[len('module.'):]
            cleaned_state_dict[key] = value

        missing, unexpected = self.backbone.load_state_dict(
            cleaned_state_dict, strict=False)
        logger.info(
            f"CNNFeatureExtractor: Loaded checkpoint from {ckpt} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})")

    def _freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        logger.info(f"CNNFeatureExtractor: Backbone frozen")

    def unfreeze_backbone(self, unfreeze_layers=None):
        """
        Unfreeze backbone for fine-tuning.

        Args:
            unfreeze_layers: Number of layers to unfreeze from the end.
                           None means unfreeze all.
        """
        if unfreeze_layers is None:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
            self.freeze = False
            logger.info("CNNFeatureExtractor: All layers unfrozen")
        else:
            # Unfreeze last N layers (implementation depends on architecture)
            # For now, just unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
            self.freeze = False
            logger.info(f"CNNFeatureExtractor: Layers unfrozen")

    def forward(self, x):
        """
        Extract features from images.

        Args:
            x: Input images, shape (B, 3, H, W)

        Returns:
            features: Feature vectors, shape (B, feature_dim)
        """
        if self.freeze:
            self.backbone.eval()
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)

        # Ensure output is 2D (B, feature_dim)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)

        return features

    def get_feature_dim(self):
        """Get the output feature dimension"""
        return self.feature_dim


class GGEURCNNClassifier(nn.Module):
    """
    Complete GGEUR_Clip model with CNN backbone and MLP classifier.

    This is the final deployable model that combines:
    - CNN feature extractor (ConvNeXt, ResNet, etc.)
    - MLP classifier (trained on GGEUR_Clip-augmented features)

    Usage:
        model = GGEURCNNClassifier('convnext_base', num_classes=65)
        logits = model(images)
    """

    def __init__(self, backbone_name='convnext_base', num_classes=65,
                 hidden_dim=0, dropout=0.0, pretrained=True,
                 freeze_backbone=True, checkpoint_path=''):
        """
        Initialize GGEUR_Clip CNN classifier.

        Args:
            backbone_name: CNN backbone architecture
            num_classes: Number of output classes
            hidden_dim: MLP hidden dimension (0 for linear classifier)
            dropout: Dropout rate for MLP
            pretrained: Use pretrained backbone weights
            freeze_backbone: Freeze backbone during training
        """
        super(GGEURCNNClassifier, self).__init__()

        # Build feature extractor
        self.backbone = CNNFeatureExtractor(
            model_name=backbone_name,
            pretrained=pretrained,
            freeze=freeze_backbone,
            checkpoint_path=checkpoint_path,
        )

        feature_dim = self.backbone.get_feature_dim()

        # Build classifier (same architecture as GGEUR_Clip MLP)
        if hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = nn.Linear(feature_dim, num_classes)

        self.num_classes = num_classes
        self.feature_dim = feature_dim

        logger.info(f"GGEURCNNClassifier: {backbone_name} -> {feature_dim}d -> {num_classes} classes")

    def forward(self, x, return_features=False):
        """
        Forward pass.

        Args:
            x: Input images, shape (B, 3, H, W)
            return_features: If True, also return features

        Returns:
            logits: Class logits, shape (B, num_classes)
            features: (optional) Feature vectors, shape (B, feature_dim)
        """
        features = self.backbone(x)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits

    def get_features(self, x):
        """Extract features only"""
        return self.backbone(x)

    def load_classifier_weights(self, state_dict):
        """Load pretrained classifier weights (from GGEUR_Clip Phase 1)"""
        try:
            self.classifier.load_state_dict(state_dict)
            logger.info("GGEURCNNClassifier: Loaded classifier weights")
        except Exception as e:
            logger.error(f"Failed to load classifier weights: {e}")
            raise

    def unfreeze_backbone(self):
        """Unfreeze backbone for end-to-end fine-tuning"""
        self.backbone.unfreeze_backbone()


def get_cnn_feature_extractor(model_name='convnext_base', pretrained=True,
                               freeze=True, device='cuda',
                               checkpoint_path=''):
    """
    Factory function to create CNN feature extractor.

    Args:
        model_name: Name of the CNN backbone
        pretrained: Use pretrained weights
        freeze: Freeze backbone parameters
        device: Device to put model on

    Returns:
        extractor: CNNFeatureExtractor instance
    """
    extractor = CNNFeatureExtractor(
        model_name=model_name,
        pretrained=pretrained,
        freeze=freeze,
        checkpoint_path=checkpoint_path,
    )
    extractor = extractor.to(device)
    return extractor


def get_available_models():
    """Get list of available CNN models"""
    return list(CNNFeatureExtractor.MODEL_CONFIGS.keys())


def get_recommended_models():
    """Get list of recommended models for GGEUR_Clip"""
    return [name for name, (_, recommended) in CNNFeatureExtractor.MODEL_CONFIGS.items()
            if recommended]
