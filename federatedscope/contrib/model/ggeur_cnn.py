"""
GGEUR_Clip CNN Models for Knowledge Distillation and Feature Alignment

Provides CNN models that can be trained with:
1. Knowledge distillation from GGEUR_Clip's MLP teacher model
2. Feature alignment with CLIP features (from scratch training)

Supported architectures:
- ResNet18, ResNet34, ResNet50
- MobileNetV2
"""

import torch
import torch.nn as nn
try:
    from torchvision.models import resnet18, resnet34, resnet50, mobilenet_v2
    from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, MobileNet_V2_Weights
    USE_NEW_WEIGHTS_API = True
except ImportError:
    import torchvision.models as models
    USE_NEW_WEIGHTS_API = False

from federatedscope.register import register_model


class GGEUR_CNN_FeatureAlign(nn.Module):
    """
    CNN model with feature alignment capability for GGEUR_Clip.

    This model can be trained from scratch by aligning its features
    with CLIP features, enabling knowledge transfer without using
    pretrained weights.

    Architecture:
        Image → CNN Backbone → CNN Features → Projection → CLIP-aligned Features
                                    ↓
                              Classifier → Logits
    """

    def __init__(self, model_name='resnet18', num_classes=65,
                 clip_dim=512, pretrained=False):
        """
        Args:
            model_name: CNN backbone architecture
            num_classes: Number of output classes
            clip_dim: CLIP feature dimension (default 512 for ViT-B)
            pretrained: Whether to use ImageNet pretrained weights
        """
        super(GGEUR_CNN_FeatureAlign, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.clip_dim = clip_dim

        # Build backbone (WITHOUT pretrained by default for from-scratch training)
        if model_name == 'resnet18':
            if USE_NEW_WEIGHTS_API:
                weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = resnet18(weights=weights)
            else:
                self.backbone = resnet18(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features  # 512
            self.backbone.fc = nn.Identity()  # Remove original fc

        elif model_name == 'resnet34':
            if USE_NEW_WEIGHTS_API:
                weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = resnet34(weights=weights)
            else:
                self.backbone = resnet34(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features  # 512
            self.backbone.fc = nn.Identity()

        elif model_name == 'resnet50':
            if USE_NEW_WEIGHTS_API:
                weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = resnet50(weights=weights)
            else:
                self.backbone = resnet50(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features  # 2048
            self.backbone.fc = nn.Identity()

        elif model_name == 'mobilenet_v2':
            if USE_NEW_WEIGHTS_API:
                weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = mobilenet_v2(weights=weights)
            else:
                self.backbone = mobilenet_v2(pretrained=pretrained)
            self.feature_dim = self.backbone.classifier[1].in_features  # 1280
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported CNN model: {model_name}")

        # Projection layer: CNN features → CLIP feature space
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, clip_dim),
            nn.ReLU(),
            nn.Linear(clip_dim, clip_dim)
        )

        # Classifier: CNN features → class logits
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize projection and classifier weights"""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.classifier.weight, 0, 0.01)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x, return_features=False):
        """
        Forward pass.

        Args:
            x: Input images, shape (batch_size, 3, 224, 224)
            return_features: If True, also return projected features

        Returns:
            logits: Class logits, shape (batch_size, num_classes)
            proj_features: (optional) Projected features, shape (batch_size, clip_dim)
        """
        # Extract CNN features
        features = self.backbone(x)

        # Handle different backbone output shapes
        if features.dim() > 2:
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)

        # Classification
        logits = self.classifier(features)

        if return_features:
            # Project to CLIP space
            proj_features = self.projection(features)
            return logits, proj_features

        return logits

    def get_cnn_features(self, x):
        """Get raw CNN features (before projection)"""
        features = self.backbone(x)
        if features.dim() > 2:
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        return features

    def get_projected_features(self, x):
        """Get CLIP-aligned projected features"""
        features = self.get_cnn_features(x)
        return self.projection(features)


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss.

    Combines:
    1. Hard label loss (Cross-Entropy with ground truth)
    2. Soft label loss (KL-Divergence with teacher's soft predictions)

    total_loss = alpha * CE_loss + (1 - alpha) * T^2 * KL_loss
    """

    def __init__(self, temperature=4.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        """
        Compute distillation loss.

        Args:
            student_logits: CNN output logits
            teacher_logits: MLP teacher output logits
            labels: Ground truth labels

        Returns:
            total_loss: Combined loss
            ce_loss: Hard label loss (for logging)
            kl_loss: Soft label loss (for logging)
        """
        # Hard label loss
        ce_loss = self.ce_loss(student_logits, labels)

        # Soft label loss with temperature scaling
        soft_student = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        kl_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)

        # Combined loss
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss

        return total_loss, ce_loss, kl_loss


class FeatureAlignmentLoss(nn.Module):
    """
    Feature Alignment Loss for training CNN from scratch.

    Combines:
    1. Classification loss (Cross-Entropy with ground truth)
    2. Feature alignment loss (MSE between CNN projected features and CLIP features)

    total_loss = CE_loss + lambda * MSE_loss
    """

    def __init__(self, align_weight=1.0):
        """
        Args:
            align_weight: Weight for feature alignment loss (lambda)
        """
        super(FeatureAlignmentLoss, self).__init__()
        self.align_weight = align_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, logits, proj_features, clip_features, labels):
        """
        Compute feature alignment loss.

        Args:
            logits: CNN output logits (batch_size, num_classes)
            proj_features: CNN projected features (batch_size, clip_dim)
            clip_features: Target CLIP features (batch_size, clip_dim)
            labels: Ground truth labels

        Returns:
            total_loss: Combined loss
            ce_loss: Classification loss (for logging)
            align_loss: Feature alignment loss (for logging)
        """
        # Classification loss
        ce_loss = self.ce_loss(logits, labels)

        # Normalize features for better alignment
        proj_features_norm = nn.functional.normalize(proj_features, p=2, dim=1)
        clip_features_norm = nn.functional.normalize(clip_features, p=2, dim=1)

        # Feature alignment loss (MSE on normalized features)
        align_loss = self.mse_loss(proj_features_norm, clip_features_norm)

        # Combined loss
        total_loss = ce_loss + self.align_weight * align_loss

        return total_loss, ce_loss, align_loss


def build_ggeur_cnn_feature_align(model_config):
    """
    Build GGEUR_Clip CNN model with feature alignment from config.

    Args:
        model_config: Configuration object with model settings

    Returns:
        CNN model instance with feature alignment capability
    """
    # Get settings from config
    if hasattr(model_config, 'ggeur'):
        cnn_model = getattr(model_config.ggeur, 'cnn_model', 'resnet18')
        cnn_pretrained = getattr(model_config.ggeur, 'cnn_pretrained', False)  # Default False for from-scratch
        clip_dim = getattr(model_config.ggeur, 'embedding_dim', 512)
    else:
        cnn_model = 'resnet18'
        cnn_pretrained = False
        clip_dim = 512

    # Get num_classes
    if hasattr(model_config, 'model') and hasattr(model_config.model, 'num_classes'):
        num_classes = model_config.model.num_classes
    elif hasattr(model_config, 'num_classes'):
        num_classes = model_config.num_classes
    else:
        num_classes = 65

    model = GGEUR_CNN_FeatureAlign(
        model_name=cnn_model,
        num_classes=num_classes,
        clip_dim=clip_dim,
        pretrained=cnn_pretrained
    )

    return model


def call_ggeur_cnn_feature_align(model_config, local_data):
    """Factory function for GGEUR_Clip CNN with feature alignment."""
    if model_config.type.lower() == 'ggeur_cnn_feature_align':
        return build_ggeur_cnn_feature_align(model_config)
    return None


# Register the feature alignment model
register_model('ggeur_cnn_feature_align', call_ggeur_cnn_feature_align)


class GGEUR_CNN_Backbone(nn.Module):
    """
    CNN Backbone for Separated Training Mode.

    This model only contains the backbone network without a built-in classifier.
    The classifier is provided separately (pretrained on GGEUR_Clip augmented features).

    Architecture:
        Image → CNN Backbone → 512-dim Features → External Classifier → Logits
    """

    def __init__(self, model_name='resnet18', num_classes=65, pretrained=False):
        """
        Args:
            model_name: CNN backbone architecture
            num_classes: Number of output classes (for building internal classifier reference)
            pretrained: Whether to use ImageNet pretrained weights (usually False for from-scratch)
        """
        super(GGEUR_CNN_Backbone, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Build backbone WITHOUT pretrained weights (from scratch)
        if model_name == 'resnet18':
            if USE_NEW_WEIGHTS_API:
                weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = resnet18(weights=weights)
            else:
                self.backbone = resnet18(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features  # 512
            self.backbone.fc = nn.Identity()  # Remove original fc

        elif model_name == 'resnet34':
            if USE_NEW_WEIGHTS_API:
                weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = resnet34(weights=weights)
            else:
                self.backbone = resnet34(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features  # 512
            self.backbone.fc = nn.Identity()

        elif model_name == 'resnet50':
            if USE_NEW_WEIGHTS_API:
                weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = resnet50(weights=weights)
            else:
                self.backbone = resnet50(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features  # 2048
            self.backbone.fc = nn.Identity()

        elif model_name == 'mobilenet_v2':
            if USE_NEW_WEIGHTS_API:
                weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = mobilenet_v2(weights=weights)
            else:
                self.backbone = mobilenet_v2(pretrained=pretrained)
            self.feature_dim = self.backbone.classifier[1].in_features  # 1280
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported CNN model: {model_name}")

        # Feature projection layer: Map backbone features to 512-dim (CLIP/classifier compatible)
        # This is important when backbone feature_dim != 512 (e.g., ResNet50 has 2048)
        if self.feature_dim != 512:
            self.feature_proj = nn.Linear(self.feature_dim, 512)
        else:
            self.feature_proj = nn.Identity()

        # Initialize projection layer
        if isinstance(self.feature_proj, nn.Linear):
            nn.init.kaiming_normal_(self.feature_proj.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.feature_proj.bias, 0)

    def forward(self, x, classifier=None):
        """
        Forward pass.

        Args:
            x: Input images, shape (batch_size, 3, 224, 224)
            classifier: External classifier module (nn.Linear or nn.Sequential)
                        If None, only returns features

        Returns:
            If classifier is provided: logits (batch_size, num_classes)
            If classifier is None: features (batch_size, 512)
        """
        # Extract CNN features
        features = self.backbone(x)

        # Handle different backbone output shapes
        if features.dim() > 2:
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)

        # Project to 512-dim if needed
        features = self.feature_proj(features)

        if classifier is not None:
            # Apply external classifier
            logits = classifier(features)
            return logits
        else:
            return features

    def get_features(self, x):
        """Get 512-dim features compatible with pretrained classifier"""
        features = self.backbone(x)
        if features.dim() > 2:
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        return self.feature_proj(features)


def build_ggeur_cnn_backbone(model_config):
    """Build GGEUR_Clip CNN backbone for separated training."""
    if hasattr(model_config, 'ggeur'):
        cnn_model = getattr(model_config.ggeur, 'cnn_model', 'resnet18')
    else:
        cnn_model = 'resnet18'

    if hasattr(model_config, 'model') and hasattr(model_config.model, 'num_classes'):
        num_classes = model_config.model.num_classes
    else:
        num_classes = 65

    # Always train from scratch in separated training mode
    model = GGEUR_CNN_Backbone(
        model_name=cnn_model,
        num_classes=num_classes,
        pretrained=False
    )

    return model


def call_ggeur_cnn_backbone(model_config, local_data):
    """Factory function for GGEUR_Clip CNN backbone."""
    if model_config.type.lower() == 'ggeur_cnn_backbone':
        return build_ggeur_cnn_backbone(model_config)
    return None


# Register the backbone model
register_model('ggeur_cnn_backbone', call_ggeur_cnn_backbone)
