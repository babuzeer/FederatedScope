"""
GGEUR_Clip Configuration

This module defines configuration options for GGEUR_Clip method.
"""

import logging
from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

logger = logging.getLogger(__name__)


def extend_ggeur_cfg(cfg):
    """
    Extended configuration for GGEUR_Clip method.

    GGEUR_Clip uses CLIP features for federated learning on multi-domain
    datasets.
    """
    # ---------------------------------------------------------------------- #
    # GGEUR_Clip related options
    # ---------------------------------------------------------------------- #
    cfg.ggeur = CN()

    # Feature caching
    cfg.ggeur.use_feature_cache = True
    cfg.ggeur.feature_cache_dir = ''  # Empty means use default

    # CLIP model configuration
    cfg.ggeur.clip_model = 'ViT-B/32'
    cfg.ggeur.clip_pretrained = 'openai'
    cfg.ggeur.clip_model_path = ''  # Local path to CLIP weights

    # Feature extractor type: 'clip' or 'cnn'
    cfg.ggeur.feature_extractor = 'clip'

    # Feature dimensions
    cfg.ggeur.embedding_dim = 512  # CLIP ViT-B/32 embedding dimension

    # MLP classifier configuration
    cfg.ggeur.mlp_hidden_dim = 256
    cfg.ggeur.mlp_dropout = 0.5

    # GGEUR augmentation parameters
    cfg.ggeur.statistics_round = 0  # Round to collect statistics
    cfg.ggeur.target_size_per_class = 0  # 0 means use all
    cfg.ggeur.num_generated_per_sample = 5
    cfg.ggeur.num_generated_per_prototype = 10
    cfg.ggeur.use_cross_client_prototypes = True

    # Label Distribution Skew (LDS) mode
    cfg.ggeur.use_lds = False

    # CNN backbone configuration (for feature alignment or distillation)
    cfg.ggeur.cnn_backbone = 'convnext_base'
    cfg.ggeur.cnn_model = 'resnet18'
    cfg.ggeur.cnn_pretrained = True
    cfg.ggeur.freeze_backbone = True

    # CNN training configuration
    cfg.ggeur.cnn_lr = 0.01
    cfg.ggeur.cnn_local_epochs = 10
    cfg.ggeur.cnn_weight_decay = 5e-4
    cfg.ggeur.cnn_dropout = 0.5
    cfg.ggeur.cnn_use_augmentation = True
    cfg.ggeur.cnn_warmup_rounds = 0  # Warmup rounds before CNN training

    # Knowledge distillation mode (CNN learns from MLP teacher)
    cfg.ggeur.use_cnn_distillation = False
    cfg.ggeur.distill_temperature = 4.0
    cfg.ggeur.distill_alpha = 0.5  # Balance between CE and KL loss

    # Feature alignment mode (CNN from scratch with CLIP alignment)
    cfg.ggeur.use_feature_alignment = False
    cfg.ggeur.align_weight = 1.0
    cfg.ggeur.use_prototype_alignment = True
    cfg.ggeur.prototype_align_weight = 0.5

    # Separated training mode (Phase 1: classifier, Phase 2: CNN)
    cfg.ggeur.use_separated_training = False
    cfg.ggeur.classifier_pretrain_rounds = 20
    cfg.ggeur.freeze_classifier = True

    # End-to-end fine-tuning mode
    cfg.ggeur.use_end_to_end_finetune = False
    cfg.ggeur.finetune_start_round = 0


register_config("ggeur", extend_ggeur_cfg)
