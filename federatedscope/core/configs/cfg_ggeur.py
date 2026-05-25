"""
GGEUR_Clip (Gaussian Geometry-guided Feature Expansion with Unified Representation)
Configuration

Based on the paper's algorithm:
1. Extract CLIP features from local data
2. Compute local statistics (mean, covariance) per class
3. Server aggregates covariance matrices using parallel axis theorem
4. Clients perform Gaussian feature augmentation using global covariance
5. Train MLP classifier on augmented features via FedAvg
"""

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_ggeur_cfg(cfg):
    """
    Extend configuration with GGEUR_Clip-specific options.
    """
    cfg.ggeur = CN()

    # ========== Basic Settings ==========
    cfg.ggeur.use = False  # Whether to use GGEUR_Clip

    # ========== Feature Extractor Mode ==========
    # 'clip': Use CLIP (ViT-based, original method)
    # 'cnn': Use pretrained CNN (ConvNeXt, ResNet, EfficientNet, etc.)
    # 'timm': Use any timm vision backbone (e.g., GFNet / Mixer / etc.)
    cfg.ggeur.feature_extractor = 'clip'

    # ========== CNN Feature Extractor Settings ==========
    # CNN backbone for feature extraction
    # Supported models:
    #   - ConvNeXt: 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'
    #   - ResNet: 'resnet18', 'resnet34', 'resnet50', 'resnet101'
    #   - EfficientNet: 'efficientnet_b0', 'efficientnet_b4', 'efficientnet_b7'
    cfg.ggeur.cnn_backbone = 'convnext_base'
    # Whether to freeze CNN backbone during feature extraction
    cfg.ggeur.freeze_backbone = True

    # ========== timm Feature Extractor Settings ==========
    # Only used when cfg.ggeur.feature_extractor == 'timm'
    # Use a widely available timm model id by default (non-CNN, non-attention-ViT).
    # You can change to GFNet etc., e.g. 'gfnet_tiny_patch4_224' if your timm provides it.
    cfg.ggeur.timm_model = 'mixer_b16_224'
    cfg.ggeur.timm_pretrained = True
    # Optional local checkpoint to avoid downloading from HuggingFace at runtime.
    # If set, weights are loaded from this file and `timm_pretrained` is ignored.
    cfg.ggeur.timm_checkpoint_path = ''
    cfg.ggeur.timm_in_chans = 3
    cfg.ggeur.timm_global_pool = 'avg'

    # ========== CLIP Feature Extraction ==========
    cfg.ggeur.clip_model = 'ViT-B-16'  # CLIP backbone: ViT-B-16, ViT-B-32, etc.
    cfg.ggeur.clip_pretrained = 'openai'  # Pretrained weights source
    cfg.ggeur.clip_model_path = ''  # Local path to CLIP weights (if available)
    cfg.ggeur.embedding_dim = 512  # Feature dimension (512 for CLIP, 1024 for ConvNeXt-Base)

    # ========== Feature Caching ==========
    # Directory to cache extracted CLIP features (empty = auto, next to data.root)
    cfg.ggeur.feature_cache_dir = ''
    # Whether to use cached features if available
    cfg.ggeur.use_feature_cache = True
    # Unload feature extractor from GPU after extraction (saves VRAM in standalone mode)
    # Only applies when CNN distillation/alignment modes are disabled
    cfg.ggeur.unload_extractor_after_cache = True
    # Use fp16 (half precision) for feature extraction (saves VRAM, ~2x faster on Tensor Cores)
    cfg.ggeur.use_fp16_extraction = True
    # Batch size for feature extraction (larger = faster; reduce if OOM during extraction)
    cfg.ggeur.extract_batch_size = 64

    # ========== Feature Augmentation ==========
    # Number of samples to generate per original sample
    cfg.ggeur.num_generated_per_sample = 50
    # Number of samples to generate per prototype from other clients
    cfg.ggeur.num_generated_per_prototype = 50
    # Target number of samples per class after augmentation
    cfg.ggeur.target_size_per_class = 50

    # ========== MLP Classifier ==========
    cfg.ggeur.mlp_hidden_dim = 0  # Hidden dim (0 means no hidden layer, just linear)
    cfg.ggeur.mlp_dropout = 0.0  # Dropout rate

    # ========== Multi-domain Settings ==========
    # Whether to use cross-client prototypes for augmentation
    cfg.ggeur.use_cross_client_prototypes = True
    # Optional selected domains for DomainNet. Empty = auto-discover extracted domains.
    cfg.ggeur.domainnet_domains = []
    # If True, keep only classes present in every selected DomainNet domain.
    # If False, use the union of classes across selected domains.
    cfg.ggeur.domainnet_shared_classes_only = False

    # ========== Training Settings ==========
    cfg.ggeur.statistics_round = 0  # Round to collect statistics (usually 0)
    # Timeout in seconds for GGEUR-specific distributed phases. Set <= 0 to
    # disable the watchdog.
    cfg.ggeur.distributed_stage_timeout = 1800

    # ========== FedProto Integration Settings ==========
    # Whether to use FedProto-style prototype regularization during MLP training
    # If True: Add prototype distance loss to regularize embeddings
    cfg.ggeur.use_fedproto = False
    # Weight for prototype loss: total_loss = CE + proto_weight * proto_loss
    cfg.ggeur.proto_weight = 1.0
    # Distance metric for prototype loss: 'euclidean' or 'cosine'
    cfg.ggeur.proto_distance = 'cosine'
    # Temperature for cosine distance (only used when proto_distance='cosine')
    cfg.ggeur.proto_temperature = 0.1

    # ========== LDS (Label Distribution Skew) Settings ==========
    # Whether to use Dirichlet distribution for non-IID data split
    cfg.ggeur.use_lds = False
    # Dirichlet distribution alpha parameter (smaller = more non-IID)
    # alpha=0.1 creates highly skewed distribution (paper default)
    # alpha=0.5 creates moderately skewed distribution
    # alpha=1.0 creates nearly uniform distribution
    cfg.ggeur.lds_alpha = 0.1
    # Random seed for Dirichlet distribution (for reproducibility)
    cfg.ggeur.lds_seed = 42

    # ========== CNN Knowledge Distillation Settings ==========
    # Whether to enable CNN training with knowledge distillation from MLP teacher
    # If True: Train CNN using original images + MLP soft labels
    # If False: Standard GGEUR_Clip (MLP only on CLIP features)
    cfg.ggeur.use_cnn_distillation = False

    # CNN model architecture: 'resnet18', 'resnet34', 'resnet50', 'mobilenet_v2'
    cfg.ggeur.cnn_model = 'resnet18'
    # Whether to use pretrained CNN weights (ImageNet)
    cfg.ggeur.cnn_pretrained = True

    # Knowledge distillation parameters
    # Temperature for softmax (higher = softer labels)
    cfg.ggeur.distill_temperature = 4.0
    # Weight for distillation loss: total_loss = alpha * CE + (1-alpha) * KL
    cfg.ggeur.distill_alpha = 0.5

    # CNN training settings
    cfg.ggeur.cnn_lr = 0.01  # Learning rate for CNN (fine-tuning)
    cfg.ggeur.cnn_local_epochs = 10  # Local epochs for CNN training per round
    # Skip CNN distillation in first N rounds (wait for MLP to learn)
    cfg.ggeur.cnn_warmup_rounds = 0

    # CNN regularization settings (to prevent overfitting)
    cfg.ggeur.cnn_weight_decay = 5e-4  # Weight decay for CNN optimizer
    cfg.ggeur.cnn_dropout = 0.5  # Dropout rate for CNN (if applicable)
    cfg.ggeur.cnn_use_augmentation = True  # Whether to use data augmentation for CNN training

    # ========== CNN Feature Alignment Settings (From Scratch Training) ==========
    # Whether to use feature alignment mode (train CNN from scratch)
    # If True: Train CNN by aligning features with CLIP (no pretrained weights needed)
    # If False: Use knowledge distillation mode (requires use_cnn_distillation=True)
    cfg.ggeur.use_feature_alignment = False

    # Feature alignment loss weight (lambda in: L = CE + lambda * MSE)
    # Higher values encourage stronger feature alignment
    cfg.ggeur.align_weight = 1.0

    # Whether to also use global prototypes for alignment
    # If True: CNN features are encouraged to align with class prototypes
    cfg.ggeur.use_prototype_alignment = True

    # Prototype alignment weight
    cfg.ggeur.prototype_align_weight = 0.5

    # ========== Separated Training Settings ==========
    # Enable separated training mode:
    # Phase 1: Train classifier on GGEUR_Clip augmented features
    # Phase 2: Train CNN backbone with frozen classifier
    cfg.ggeur.use_separated_training = False

    # Number of rounds for classifier pre-training (Phase 1)
    # After these rounds, switch to CNN backbone training (Phase 2)
    cfg.ggeur.classifier_pretrain_rounds = 20

    # Whether to freeze classifier in Phase 2
    # If True: classifier parameters are not updated during CNN training
    cfg.ggeur.freeze_classifier = True

    # ========== End-to-End Fine-tuning Settings ==========
    # Enable end-to-end fine-tuning of CNN + classifier after initial training
    # This unfreezes the CNN backbone for better performance
    cfg.ggeur.use_end_to_end_finetune = False
    # Round to start fine-tuning (0 = from the beginning)
    cfg.ggeur.finetune_start_round = 30
    # Learning rate for fine-tuning (should be lower than initial training)
    cfg.ggeur.finetune_lr = 0.0001

    # ========== MOON Settings ==========
    # Model-Contrastive Federated Learning (MOON)
    # Ref: Li et al., "Model-Contrastive Federated Learning", CVPR 2021
    cfg.ggeur.use_moon = False
    # Contrastive loss weight (mu in the paper)
    cfg.ggeur.moon_mu = 5.0
    # Temperature for contrastive loss
    cfg.ggeur.moon_temperature = 0.5

    # ========== PromptFL Settings ==========
    # Enable federated soft prompt training (CoOp style) on augmented CLIP features.
    # Only supported when feature_extractor='clip'.
    # When enabled, ctx vectors (n_ctx × 512) are trained and federated instead of MLP.
    cfg.ggeur.use_promptfl = False
    # Number of learnable context tokens prepended to class name tokens
    cfg.ggeur.prompt_length = 16
    # Learning rate for prompt optimizer (Adam)
    cfg.ggeur.prompt_lr = 0.002
    # Local training epochs per round for prompt
    cfg.ggeur.prompt_local_epochs = 10
    # Optional cap on PromptFL local batches per round. 0 means no cap.
    cfg.ggeur.prompt_max_train_batches = 0
    # Temperature for cosine similarity logits
    cfg.ggeur.prompt_temperature = 0.07
    # Optional: manually specify class names (list of strings).
    # If empty, class names are inferred from the dataset (OfficeHome/PACS/etc.)
    cfg.ggeur.prompt_class_names = []
    # Text prompt template, {} is replaced by class name
    cfg.ggeur.prompt_template = 'a photo of a {}'
    # HuggingFace CLIP model directory or Hub ID for PromptFL
    # e.g. '/root/model/clip-vit-base-patch16' or 'openai/clip-vit-base-patch16'
    cfg.ggeur.hf_clip_model_id = 'openai/clip-vit-base-patch16'
    # FedProx proximal term weight for prompt (0.0 = disabled).
    # Penalizes ||ctx_local - ctx_global||^2 to reduce client drift.
    cfg.ggeur.prompt_proximal_mu = 0.0
    # Gaussian samples per prototype class in prompt_loader.
    # For classes missing locally, this many samples are generated around the
    # cross-client prototype mean using the global covariance matrix.
    cfg.ggeur.prompt_samples_per_proto = 20

    return cfg


register_config("ggeur", extend_ggeur_cfg)
