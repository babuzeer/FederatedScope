"""
GGEUR_Clip Client Implementation

Handles:
1. CLIP feature extraction from local images
2. Local statistics computation (mean, covariance per class)
3. Feature augmentation using global covariance from server
4. Standard FedAvg training on augmented features
5. (Optional) CNN training with knowledge distillation from MLP teacher
6. (Optional) CNN training with feature alignment (from scratch)
"""

import os
import logging
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from federatedscope.core.message import Message
from federatedscope.core.workers import Client
from federatedscope.register import register_worker

logger = logging.getLogger(__name__)


class AugmentedFeatureDataset(Dataset):
    """Dataset for augmented CLIP features"""

    def __init__(self, features, labels):
        if isinstance(features, np.ndarray):
            self.features = torch.from_numpy(features).float()
        else:
            self.features = features.float()

        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class AugmentedImageDataset(Dataset):
    """
    Wrapper dataset that applies strong data augmentation for CNN training.

    This is CRITICAL for preventing overfitting when training CNN from scratch.
    The augmentation helps CNN learn more generalizable features.
    """

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

        # 强数据增强变换
        from torchvision import transforms
        self.augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ])

        # 标准化（与原始数据集一致）
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 获取原始数据
        item = self.base_dataset[idx]
        if len(item) >= 2:
            image, label = item[0], item[1]
        else:
            return item

        # 如果image已经是tensor，转换为PIL进行增强
        if isinstance(image, torch.Tensor):
            # 反归一化（假设已经用ImageNet均值/标准差归一化）
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_denorm = image * std + mean
            image_denorm = torch.clamp(image_denorm, 0, 1)

            # 转为PIL
            pil_image = self.to_pil(image_denorm)

            # 应用增强
            augmented = self.augment_transform(pil_image)

            # 转回tensor并归一化
            image = self.normalize(self.to_tensor(augmented))

        return image, label


class GGEURClient(Client):
    """
    GGEUR_Clip Client that:
    1. Extracts features (CLIP or CNN) and computes local statistics
    2. Receives global covariance matrices from server
    3. Performs GGEUR_Clip feature augmentation
    4. Trains MLP classifier on augmented features
    5. (Optional) End-to-end fine-tuning with CNN backbone
    """

    def __init__(self, ID=-1, server_id=None, state=-1, config=None,
                 data=None, model=None, device='cpu', strategy=None,
                 is_unseen_client=False, *args, **kwargs):
        super(GGEURClient, self).__init__(ID, server_id, state, config,
                                          data, model, device, strategy,
                                          is_unseen_client, *args, **kwargs)

        if config is None:
            return

        self.ggeur_cfg = config.ggeur

        # ===== Feature Extractor Mode =====
        # 'clip': Use CLIP (ViT-based, original method)
        # 'cnn': Use pretrained CNN (ConvNeXt, ResNet, etc.)
        # 'timm': Use any timm vision backbone (e.g., GFNet / Mixer / etc.)
        self.feature_extractor_type = getattr(self.ggeur_cfg, 'feature_extractor', 'clip')

        # CLIP model (for 'clip' mode)
        self.clip_model = None
        self.clip_preprocess = None

        # CNN feature extractor (for 'cnn' mode)
        self.cnn_extractor = None

        # timm feature extractor (for 'timm' mode)
        self.timm_extractor = None

        # The actual embedding dimension used by the feature extractor.
        # (May differ from cfg.ggeur.embedding_dim if user changes backbone.)
        self.embedding_dim = getattr(self.ggeur_cfg, 'embedding_dim', 512)

        # Local features and labels
        self.local_features = {}  # {class_idx: features array}
        self.local_labels = {}

        # Local statistics
        self.local_means = {}  # {class_idx: mean vector}
        self.local_covs = {}  # {class_idx: covariance matrix}
        self.local_counts = {}  # {class_idx: sample count}

        # Global covariance from server
        self.global_cov_matrices = None
        self.other_prototypes = None  # Prototypes from other clients

        # Global prototypes for feature alignment
        self.global_prototypes = None  # {class_idx: mean vector}

        # Augmented data
        self.augmented_features = None
        self.augmented_labels = None
        self.augmented_loader = None

        # MLP classifier
        self.mlp_classifier = None

        # State tracking
        self.statistics_uploaded = False
        self.augmentation_done = False

        # ===== End-to-End Fine-tuning Mode (for CNN) =====
        self.use_end_to_end_finetune = getattr(self.ggeur_cfg, 'use_end_to_end_finetune', False)
        self.finetune_start_round = getattr(self.ggeur_cfg, 'finetune_start_round', 0)
        self.full_model = None  # Combined CNN + classifier for fine-tuning
        self.original_image_loader = None

        # ===== Legacy modes (for backward compatibility) =====
        self.use_cnn_distillation = getattr(self.ggeur_cfg, 'use_cnn_distillation', False)
        self.use_feature_alignment = getattr(self.ggeur_cfg, 'use_feature_alignment', False)
        self.cnn_model = None
        self.use_separated_training = getattr(self.ggeur_cfg, 'use_separated_training', False)
        self.training_phase = 'classifier'
        self.pretrained_classifier = None
        self.cnn_backbone = None

        # ===== MOON Mode =====
        self.use_moon = getattr(self.ggeur_cfg, 'use_moon', False)
        self.moon_prev_model = None    # Previous local model snapshot
        self.moon_global_model = None  # Global model snapshot (received this round)

        # ===== PromptFL Mode =====
        self.use_promptfl = getattr(self.ggeur_cfg, 'use_promptfl', False)
        self.prompt_learner = None
        self.text_encoder = None
        self.custom_clip = None        # CustomCLIP (HuggingFace-based) for PromptFL
        self.hf_clip_model = None      # HuggingFace CLIPModel (separate from open_clip)

    def _register_default_handlers(self):
        """Register message handlers"""
        super()._register_default_handlers()

        # Register handler for receiving global covariance matrices
        self.register_handlers('global_covariances',
                               self.callback_for_global_covariances)

    def _load_feature_extractor(self):
        """Load feature extractor (CLIP / CNN / timm based on config)"""
        if self.feature_extractor_type == 'cnn':
            self._load_cnn_extractor()
        elif self.feature_extractor_type == 'timm':
            self._load_timm_extractor()
        else:
            self._load_clip_model()

    def _load_cnn_extractor(self):
        """Load CNN feature extractor"""
        if self.cnn_extractor is not None:
            return

        try:
            from federatedscope.contrib.model.ggeur_cnn_extractor import CNNFeatureExtractor

            model_name = getattr(self.ggeur_cfg, 'cnn_backbone', 'convnext_base')
            pretrained = getattr(self.ggeur_cfg, 'cnn_pretrained', True)
            freeze = getattr(self.ggeur_cfg, 'freeze_backbone', True)

            self.cnn_extractor = CNNFeatureExtractor(
                model_name=model_name,
                pretrained=pretrained,
                freeze=freeze
            )
            self.cnn_extractor = self.cnn_extractor.to(self.device)

            self.embedding_dim = int(self.cnn_extractor.get_feature_dim())
            logger.info(f"Client {self.ID}: Loaded CNN extractor {model_name}, "
                       f"feature_dim={self.cnn_extractor.get_feature_dim()}")

        except Exception as e:
            logger.error(f"Client {self.ID}: Failed to load CNN extractor: {e}")
            raise

    def _load_timm_extractor(self):
        """Load timm feature extractor"""
        if self.timm_extractor is not None:
            return

        try:
            from federatedscope.contrib.model.ggeur_timm_extractor import TimmFeatureExtractor

            model_name = getattr(self.ggeur_cfg, 'timm_model', 'gfnet_tiny')
            pretrained = getattr(self.ggeur_cfg, 'timm_pretrained', True)
            checkpoint_path = getattr(self.ggeur_cfg, 'timm_checkpoint_path', '')
            freeze = getattr(self.ggeur_cfg, 'freeze_backbone', True)
            in_chans = getattr(self.ggeur_cfg, 'timm_in_chans', 3)
            global_pool = getattr(self.ggeur_cfg, 'timm_global_pool', 'avg')

            self.timm_extractor = TimmFeatureExtractor(
                model_name=model_name,
                pretrained=pretrained,
                freeze=freeze,
                checkpoint_path=checkpoint_path,
                in_chans=in_chans,
                global_pool=global_pool,
            )
            self.timm_extractor = self.timm_extractor.to(self.device)

            self.embedding_dim = int(self.timm_extractor.get_feature_dim())
            logger.info(f"Client {self.ID}: Loaded timm extractor {model_name}, "
                        f"feature_dim={self.embedding_dim}")

        except Exception as e:
            logger.error(f"Client {self.ID}: Failed to load timm extractor: {e}")
            raise

    def _load_clip_model(self):
        """Load CLIP model for feature extraction"""
        if self.clip_model is not None:
            return

        try:
            import open_clip

            model_name = self.ggeur_cfg.clip_model
            pretrained = self.ggeur_cfg.clip_pretrained
            local_path = self.ggeur_cfg.clip_model_path

            # Check for local model path first
            if local_path and os.path.exists(local_path):
                logger.info(f"Client {self.ID}: Loading CLIP from local path: {local_path}")
                self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=local_path
                )
            else:
                # Try to load from pretrained source (may require internet)
                logger.info(f"Client {self.ID}: Loading CLIP from pretrained: {pretrained}")
                try:
                    self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                        model_name, pretrained=pretrained
                    )
                except Exception as e:
                    logger.error(f"Failed to download CLIP model. Please set ggeur.clip_model_path to local weights file.")
                    logger.error(f"Error: {e}")
                    raise

            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            logger.info(f"Client {self.ID}: Loaded CLIP model {model_name}")

        except ImportError:
            logger.error("open_clip not installed. Please install: pip install open_clip_torch")
            raise

    def _get_feature_cache_path(self, domain=None):
        """Get the path for cached CLIP features"""
        # Check if caching is enabled
        if not getattr(self.ggeur_cfg, 'use_feature_cache', True):
            return None

        # Get cache directory from config or use default
        cache_dir = getattr(self.ggeur_cfg, 'feature_cache_dir', '')
        if not cache_dir:
            cache_dir = os.path.join(os.path.dirname(self._cfg.data.root), 'clip_feature_cache')

        os.makedirs(cache_dir, exist_ok=True)

        # Build cache filename based on dataset, domain, and model
        dataset_name = self._cfg.data.type.lower()

        if self.feature_extractor_type == 'cnn':
            model_name = getattr(self.ggeur_cfg, 'cnn_backbone', 'convnext_base')
            model_str = model_name.replace('/', '_').replace('-', '_')
            prefix = 'cnn'
        elif self.feature_extractor_type == 'timm':
            model_name = getattr(self.ggeur_cfg, 'timm_model', 'gfnet_tiny')
            model_str = str(model_name).replace('/', '_').replace('-', '_')
            prefix = 'timm'
        else:
            clip_model = self.ggeur_cfg.clip_model.replace('/', '_').replace('-', '_')
            pretrained = self.ggeur_cfg.clip_pretrained.replace('/', '_').replace('-', '_')
            model_str = f"{clip_model}_{pretrained}"
            prefix = 'clip'

        if domain:
            cache_filename = f"{dataset_name}_{domain}_{prefix}_{model_str}.npz"
        else:
            cache_filename = f"{dataset_name}_{prefix}_{model_str}.npz"

        return os.path.join(cache_dir, cache_filename)

    def _load_feature_cache(self, cache_path):
        """Load cached features from file"""
        if cache_path is None or not os.path.exists(cache_path):
            return {}

        try:
            data = np.load(cache_path, allow_pickle=True)
            # Cache format: {'paths': array of paths, 'features': array of features}
            if 'paths' in data and 'features' in data:
                paths = data['paths']
                features = data['features']
                cache = {str(p): f for p, f in zip(paths, features)}
                logger.info(f"Client {self.ID}: Loaded {len(cache)} cached features from {cache_path}")
                return cache
        except Exception as e:
            logger.warning(f"Client {self.ID}: Failed to load cache: {e}")

        return {}

    def _save_feature_cache(self, cache_path, feature_cache):
        """Save features to cache file"""
        if cache_path is None:
            return

        try:
            paths = list(feature_cache.keys())
            features = np.array([feature_cache[p] for p in paths])
            np.savez(cache_path, paths=np.array(paths), features=features)
            logger.info(f"Client {self.ID}: Saved {len(paths)} features to cache {cache_path}")
        except Exception as e:
            logger.warning(f"Client {self.ID}: Failed to save cache: {e}")

    def _extract_features(self):
        """
        Extract features from local data using either CLIP or CNN.
        Supports caching for both modes.
        """
        if self.feature_extractor_type == 'cnn':
            extractor_name = 'CNN'
        elif self.feature_extractor_type == 'timm':
            extractor_name = 'timm'
        else:
            extractor_name = 'CLIP'
        logger.info(f"Client {self.ID}: Extracting {extractor_name} features...")

        # Get train data
        train_data = self.trainer.ctx.data.get('train', None)
        if train_data is None:
            train_data = self.data.get('train', None)

        if train_data is None:
            logger.error(f"Client {self.ID}: No training data available")
            return

        # Get the underlying dataset (handle DataLoader -> Dataset -> possibly Subset)
        dataset = train_data.dataset if hasattr(train_data, 'dataset') else train_data

        # Handle Subset wrapper (when multiple clients share one domain)
        from torch.utils.data import Subset
        is_subset = isinstance(dataset, Subset)
        if is_subset:
            base_dataset = dataset.dataset
            subset_indices = dataset.indices
            domain = getattr(base_dataset, 'domain', None)
        else:
            base_dataset = dataset
            subset_indices = None
            domain = getattr(dataset, 'domain', None)

        cache_path = self._get_feature_cache_path(domain)

        # Load existing cache
        feature_cache = self._load_feature_cache(cache_path)
        cache_updated = False

        # Check if base dataset has image paths (PACS, Office-Home style)
        has_paths = hasattr(base_dataset, 'data') and len(base_dataset.data) > 0 and isinstance(base_dataset.data[0], str)

        self.local_features = {}
        self.local_labels = {}

        if has_paths:
            # Dataset with image paths - can use caching
            num_samples = len(subset_indices) if is_subset else len(base_dataset)
            logger.info(f"Client {self.ID}: Dataset has {num_samples} samples with paths")

            # Collect samples that need feature extraction
            paths_to_extract = []
            indices_to_extract = []  # Indices into the current dataset (Subset or base)
            base_indices_to_extract = []  # Indices into base_dataset for loading

            for local_idx in range(num_samples):
                # Get the index into the base dataset
                if is_subset:
                    base_idx = subset_indices[local_idx]
                else:
                    base_idx = local_idx

                img_path = base_dataset.data[base_idx]
                label = base_dataset.targets[base_idx]

                if img_path in feature_cache:
                    # Use cached feature
                    feat = feature_cache[img_path]
                    label = int(label)
                    if label not in self.local_features:
                        self.local_features[label] = []
                        self.local_labels[label] = []
                    self.local_features[label].append(feat)
                    self.local_labels[label].append(label)
                else:
                    # Need to extract
                    paths_to_extract.append(img_path)
                    indices_to_extract.append(local_idx)
                    base_indices_to_extract.append(base_idx)

            cached_count = num_samples - len(paths_to_extract)
            logger.info(f"Client {self.ID}: {cached_count} samples from cache, {len(paths_to_extract)} need extraction")

            # Extract features for non-cached samples
            if paths_to_extract:
                # Load feature extractor (CLIP or CNN)
                self._load_feature_extractor()

                # Create a mini dataloader for samples to extract
                batch_size = 32
                with torch.no_grad():
                    for i in range(0, len(base_indices_to_extract), batch_size):
                        batch_base_indices = base_indices_to_extract[i:i + batch_size]
                        batch_paths = paths_to_extract[i:i + batch_size]

                        # Load images from base dataset
                        images = []
                        labels = []
                        for base_idx in batch_base_indices:
                            img, lbl = base_dataset[base_idx]
                            images.append(img)
                            labels.append(lbl)

                        images = torch.stack(images).to(self.device)

                        # Extract features using appropriate extractor
                        if self.feature_extractor_type == 'cnn':
                            features = self.cnn_extractor(images)
                        elif self.feature_extractor_type == 'timm':
                            features = self.timm_extractor(images)
                        else:
                            features = self.clip_model.encode_image(images)
                        features = features.cpu().numpy()

                        for feat, label, path in zip(features, labels, batch_paths):
                            # Update cache
                            feature_cache[path] = feat
                            cache_updated = True

                            # Add to local features
                            label = int(label)
                            if label not in self.local_features:
                                self.local_features[label] = []
                                self.local_labels[label] = []
                            self.local_features[label].append(feat)
                            self.local_labels[label].append(label)

                # Save updated cache
                if cache_updated:
                    self._save_feature_cache(cache_path, feature_cache)

        else:
            # Fallback: Dataset without paths - cannot use caching
            logger.info(f"Client {self.ID}: Dataset does not have image paths, caching disabled")
            self._load_feature_extractor()

            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            with torch.no_grad():
                for batch in dataloader:
                    if len(batch) >= 2:
                        images, labels = batch[0], batch[1]
                    else:
                        continue

                    images = images.to(self.device)

                    # Skip invalid images
                    if images.shape[1] != 3:
                        continue

                    # Extract features using appropriate extractor
                    if self.feature_extractor_type == 'cnn':
                        features = self.cnn_extractor(images)
                    elif self.feature_extractor_type == 'timm':
                        features = self.timm_extractor(images)
                    else:
                        features = self.clip_model.encode_image(images)
                    features = features.cpu().numpy()
                    labels = labels.cpu().numpy()

                    for feat, label in zip(features, labels):
                        label = int(label)
                        if label not in self.local_features:
                            self.local_features[label] = []
                            self.local_labels[label] = []
                        self.local_features[label].append(feat)
                        self.local_labels[label].append(label)

        # Convert lists to numpy arrays
        for label in self.local_features:
            self.local_features[label] = np.array(self.local_features[label])

        total_samples = sum(len(v) for v in self.local_features.values())
        logger.info(f"Client {self.ID}: Extracted {total_samples} {extractor_name} features from {len(self.local_features)} classes")

    def _extract_clip_features(self):
        """Legacy method - now calls _extract_features()"""
        self._extract_features()

    def _compute_local_statistics(self):
        """Compute local mean and covariance for each class"""
        logger.info(f"Client {self.ID}: Computing local statistics...")

        self.local_means = {}
        self.local_covs = {}
        self.local_counts = {}

        for class_idx, features in self.local_features.items():
            if features.shape[0] == 0:
                continue

            n = features.shape[0]
            mean = np.mean(features, axis=0)

            # Compute covariance
            centered = features - mean
            cov = (1.0 / n) * np.dot(centered.T, centered)

            self.local_means[class_idx] = mean
            self.local_covs[class_idx] = cov
            self.local_counts[class_idx] = n

        logger.info(f"Client {self.ID}: Computed statistics for {len(self.local_means)} classes")

    def _upload_local_statistics(self):
        """Upload local statistics to server"""
        logger.info(f"Client {self.ID}: Uploading local statistics to server...")

        # Also send prototypes for cross-client augmentation
        prototypes = {}
        for class_idx, mean in self.local_means.items():
            prototypes[class_idx] = mean

        content = {
            'client_id': self.ID,
            'embedding_dim': int(self.embedding_dim),
            'means': self.local_means,
            'covs': self.local_covs,
            'counts': self.local_counts,
            'prototypes': prototypes
        }

        self.comm_manager.send(
            Message(
                msg_type='local_statistics',
                sender=self.ID,
                receiver=[self.server_id],
                state=self.state,
                content=content
            )
        )

        self.statistics_uploaded = True
        logger.info(f"Client {self.ID}: Statistics uploaded")

    def callback_for_global_covariances(self, message: Message):
        """Handle receiving global covariance matrices from server"""
        logger.info(f"Client {self.ID}: Received global covariances from server")

        content = message.content
        self.global_cov_matrices = content.get('cov_matrices', {})
        self.other_prototypes = content.get('other_prototypes', {}).get(self.ID, {})
        self.global_prototypes = content.get('global_prototypes', {})  # For feature alignment

        # Debug logging for received prototypes
        logger.info(f"Client {self.ID}: Received global_prototypes with {len(self.global_prototypes)} classes")
        if self.global_prototypes:
            sample_classes = list(self.global_prototypes.keys())[:3]
            for cls in sample_classes:
                proto = self.global_prototypes[cls]
                if hasattr(proto, 'shape'):
                    logger.info(f"  Class {cls}: shape={proto.shape}, mean={proto.mean():.4f}")

        # Perform augmentation
        self._perform_augmentation()

        # Build MLP classifier
        self._build_mlp_classifier()

        # Build PromptFL model if enabled
        if self.use_promptfl:
            self._build_prompt_model()

        # Build CNN based on mode
        if self.use_feature_alignment:
            # Feature alignment mode: train CNN from scratch
            self._load_clip_model()
            self._build_cnn_feature_align()
            self._setup_original_image_loader()
            logger.info(f"Client {self.ID}: Feature alignment ready - CLIP: {self.clip_model is not None}, "
                       f"CNN: {self.cnn_model is not None}, Prototypes: {len(self.global_prototypes)}")
        elif self.use_cnn_distillation:
            # Knowledge distillation mode
            self._load_clip_model()
            self._build_cnn_model()
            self._setup_original_image_loader()
            logger.info(f"Client {self.ID}: CNN distillation ready - CLIP: {self.clip_model is not None}, "
                       f"MLP: {self.mlp_classifier is not None}, CNN: {self.cnn_model is not None}")

        # Notify server that augmentation is complete
        logger.info(f"Client {self.ID}: Notifying server that augmentation is ready")
        self.comm_manager.send(
            Message(
                msg_type='augmentation_ready',
                sender=self.ID,
                receiver=[self.server_id],
                state=self.state,
                content=None
            )
        )

    def _nearest_pos_def(self, cov_matrix):
        """Ensure covariance matrix is positive definite"""
        # 对于大维度矩阵，使用简化方法
        dim = cov_matrix.shape[0]
        if dim > 512:
            # 对于高维矩阵，直接添加正则化而不做特征值分解
            # 这样更快且通常足够
            min_eig = np.min(np.real(np.linalg.eigvalsh(cov_matrix)))
            if min_eig < 1e-6:
                cov_matrix = cov_matrix + (1e-6 - min_eig) * np.eye(dim)
            return cov_matrix

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Scale small eigenvalues for better conditioning
        scale_factors = np.ones_like(eigenvalues)
        scale_factors[:10] = np.linspace(5, 1, 10)
        eigenvalues = eigenvalues * scale_factors

        # Clip negative eigenvalues
        eigenvalues[eigenvalues < 0] = 0

        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def _generate_samples(self, mean, cov_matrix, num_samples):
        """Generate samples from Gaussian distribution"""
        dim = cov_matrix.shape[0]

        # 对于高维矩阵，使用更快的采样方法
        if dim > 512:
            # 使用对角协方差近似（更快但损失一些精度）
            # 或者使用低秩近似
            try:
                # 尝试直接Cholesky分解
                jitter = 1e-5
                L = np.linalg.cholesky(cov_matrix + jitter * np.eye(dim))
                z = np.random.randn(num_samples, dim)
                samples = mean + z @ L.T
                return samples
            except np.linalg.LinAlgError:
                # 如果失败，使用对角近似
                var = np.diag(cov_matrix)
                var = np.maximum(var, 1e-6)
                std = np.sqrt(var)
                z = np.random.randn(num_samples, dim)
                samples = mean + z * std
                return samples

        cov_matrix = self._nearest_pos_def(cov_matrix)

        # Add jitter for numerical stability
        jitter = 1e-6

        while True:
            try:
                B = np.linalg.cholesky(cov_matrix + jitter * np.eye(dim))
                break
            except np.linalg.LinAlgError:
                jitter *= 10
                if jitter > 1:
                    B = np.eye(dim) * 0.1
                    break

        samples = np.random.multivariate_normal(mean, B @ B.T, num_samples)
        return samples

    def _perform_augmentation(self):
        """Perform GGEUR_Clip feature augmentation"""
        target_size = self.ggeur_cfg.target_size_per_class
        num_per_sample = self.ggeur_cfg.num_generated_per_sample
        num_per_prototype = self.ggeur_cfg.num_generated_per_prototype
        use_cross_client = self.ggeur_cfg.use_cross_client_prototypes

        # Check if augmentation is disabled (baseline mode)
        no_augmentation = (num_per_sample == 0 and num_per_prototype == 0) or not use_cross_client

        if no_augmentation:
            logger.info(f"Client {self.ID}: No augmentation mode - using original features only")
        else:
            logger.info(f"Client {self.ID}: Performing GGEUR_Clip augmentation...")

        all_features = []
        all_labels = []

        # Get all class indices
        all_classes = set(self.local_features.keys())
        if not no_augmentation and self.global_cov_matrices:
            all_classes.update(self.global_cov_matrices.keys())
        if not no_augmentation and self.other_prototypes:
            all_classes.update(self.other_prototypes.keys())

        total_classes = len(all_classes)

        # 获取特征维度用于日志
        feature_dim = self.embedding_dim
        if self.local_features:
            first_key = next(iter(self.local_features.keys()))
            if len(self.local_features[first_key]) > 0:
                feature_dim = self.local_features[first_key].shape[1]

        logger.info(f"Client {self.ID}: Processing {total_classes} classes, feature_dim={feature_dim}, "
                   f"num_per_sample={num_per_sample}, num_per_prototype={num_per_prototype}")

        for idx, class_idx in enumerate(all_classes):
            class_idx = int(class_idx)
            class_features = []

            # 显示进度
            if (idx + 1) % 10 == 0 or idx == 0:
                logger.info(f"Client {self.ID}: Augmenting class {idx+1}/{total_classes}")

            # 1. Original features from this client (always include)
            if class_idx in self.local_features:
                original = self.local_features[class_idx]
                class_features.append(original)

            # Skip augmentation if disabled
            if no_augmentation:
                if class_features:
                    combined = np.vstack(class_features)
                    all_features.append(combined)
                    all_labels.append(np.full(combined.shape[0], class_idx))
                continue

            # 2. Get global covariance matrix
            if class_idx in self.global_cov_matrices:
                cov_matrix = self.global_cov_matrices[class_idx]
            else:
                cov_matrix = np.eye(self.embedding_dim) * 0.01

            # 3. Expand original features using global covariance
            if num_per_sample > 0 and class_idx in self.local_features and self.local_features[class_idx].shape[0] > 0:
                for feat in self.local_features[class_idx]:
                    generated = self._generate_samples(feat, cov_matrix, num_per_sample)
                    class_features.append(generated)

            # 4. Generate from other clients' prototypes
            if use_cross_client and num_per_prototype > 0 and self.other_prototypes:
                if class_idx in self.other_prototypes:
                    for prototype in self.other_prototypes[class_idx]:
                        generated = self._generate_samples(prototype, cov_matrix, num_per_prototype)
                        class_features.append(generated)

            # Combine and sample to target size
            if class_features:
                combined = np.vstack(class_features)

                # target_size = 0 means use all samples
                if target_size > 0 and combined.shape[0] >= target_size:
                    indices = np.random.choice(combined.shape[0], target_size, replace=False)
                    selected = combined[indices]
                else:
                    selected = combined

                all_features.append(selected)
                all_labels.append(np.full(selected.shape[0], class_idx))

        logger.info(f"Client {self.ID}: Augmentation complete, building dataset...")

        if all_features:
            self.augmented_features = np.vstack(all_features)
            self.augmented_labels = np.concatenate(all_labels)

            # Create data loader
            dataset = AugmentedFeatureDataset(self.augmented_features, self.augmented_labels)
            self.augmented_loader = DataLoader(
                dataset,
                batch_size=self._cfg.dataloader.batch_size,
                shuffle=True
            )

            if no_augmentation:
                logger.info(f"Client {self.ID}: Original data - {self.augmented_features.shape[0]} samples, "
                            f"{len(np.unique(self.augmented_labels))} classes")
            else:
                logger.info(f"Client {self.ID}: Augmented data - {self.augmented_features.shape[0]} samples, "
                            f"{len(np.unique(self.augmented_labels))} classes")

        self.augmentation_done = True

    def _build_mlp_classifier(self):
        """Build MLP classifier for augmented features"""
        # IMPORTANT: Always use config's num_classes, not the unique labels in augmented data
        # In LDS mode, each client may only have a subset of classes, but the model
        # must support all classes for proper FedAvg aggregation
        num_classes = self._cfg.model.num_classes
        input_dim = self.embedding_dim
        hidden_dim = self.ggeur_cfg.mlp_hidden_dim

        if hidden_dim > 0:
            self.mlp_classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.ggeur_cfg.mlp_dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            # Simple linear classifier
            self.mlp_classifier = nn.Linear(input_dim, num_classes)

        self.mlp_classifier = self.mlp_classifier.to(self.device)
        logger.info(f"Client {self.ID}: Built MLP classifier with {num_classes} classes")

    def callback_funcs_for_model_para(self, message: Message):
        """
        Handle model parameters message.
        In Round 0: extract features and upload statistics
        In Round 1+: train on augmented data (and optionally CNN with distillation)

        For Separated Training Mode:
        - Phase 1 (classifier): Train classifier on GGEUR_Clip augmented features
        - Phase 2 (cnn_backbone): Train CNN backbone with frozen classifier
        """
        round_idx = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        # Update state
        self.state = round_idx

        # Statistics collection round
        if round_idx == self.ggeur_cfg.statistics_round and not self.statistics_uploaded:
            logger.info(f"Client {self.ID}: Round {round_idx} - Statistics collection phase")

            # Extract CLIP features
            self._extract_clip_features()

            # Compute local statistics
            self._compute_local_statistics()

            # Upload to server
            self._upload_local_statistics()

            return  # Don't do normal training in this round

        # Wait for augmentation to complete
        if not self.augmentation_done:
            logger.warning(f"Client {self.ID}: Augmentation not done, skipping training")
            # Send empty model update
            self.comm_manager.send(
                Message(
                    msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=(0, self.trainer.get_model_para())
                )
            )
            return

        # Handle Separated Training Mode
        if self.use_separated_training:
            self._handle_separated_training(message)
            return

        # Normal training round on augmented data
        logger.info(f"Client {self.ID}: Round {round_idx} - Training on augmented data")

        # Parse content - may contain both MLP and CNN parameters
        mlp_para = None
        cnn_para = None

        if content is not None:
            if isinstance(content, dict):
                if 'mlp' in content:
                    mlp_para = content.get('mlp')
                    cnn_para = content.get('cnn')
                else:
                    # Backward compatibility: content is just MLP parameters
                    mlp_para = content

        # Load global prompt ctx if PromptFL enabled
        if self.use_promptfl and content is not None and isinstance(content, dict):
            prompt_para = content.get('prompt')
            if prompt_para is not None and self.prompt_learner is not None:
                try:
                    ctx_tensor = prompt_para.get('ctx')
                    if ctx_tensor is not None:
                        self.prompt_learner.ctx.data = ctx_tensor.to(self.device)
                except Exception as e:
                    logger.debug(f"Client {self.ID}: Could not load prompt ctx: {e}")

        # Update MLP with global model parameters
        if mlp_para is not None and self.mlp_classifier is not None:
            try:
                self.mlp_classifier.load_state_dict(mlp_para, strict=False)
            except Exception as e:
                logger.debug(f"Client {self.ID}: Could not load MLP state dict: {e}")

        # MOON: snapshot global model after loading global params, before local training
        if self.use_moon and self.mlp_classifier is not None:
            self.moon_global_model = copy.deepcopy(self.mlp_classifier)
            self.moon_global_model.eval()
            for p in self.moon_global_model.parameters():
                p.requires_grad_(False)

        # Update CNN with global model parameters (for both distillation and feature alignment)
        if (self.use_cnn_distillation or self.use_feature_alignment) and cnn_para is not None and self.cnn_model is not None:
            try:
                self.cnn_model.load_state_dict(cnn_para, strict=False)
            except Exception as e:
                logger.debug(f"Client {self.ID}: Could not load CNN state dict: {e}")

        # Train MLP on augmented features
        mlp_sample_size, mlp_model_para, mlp_results = self._train_on_augmented_data()

        # MOON: save current local model as previous model for next round
        if self.use_moon and self.mlp_classifier is not None:
            self.moon_prev_model = copy.deepcopy(self.mlp_classifier)
            self.moon_prev_model.eval()
            for p in self.moon_prev_model.parameters():
                p.requires_grad_(False)

        # Train CNN based on mode
        cnn_warmup_rounds = getattr(self.ggeur_cfg, 'cnn_warmup_rounds', 0)

        if self.use_feature_alignment:
            # Feature alignment mode: train CNN from scratch
            should_train_cnn = round_idx > cnn_warmup_rounds

            if should_train_cnn:
                cnn_sample_size, cnn_model_para, cnn_results = self._train_cnn_with_feature_alignment()

                combined_para = {
                    'mlp': mlp_model_para,
                    'cnn': cnn_model_para
                }
                sample_size = mlp_sample_size
            else:
                logger.info(f"Client {self.ID}: CNN warmup - skipping training in round {round_idx}")
                combined_para = {
                    'mlp': mlp_model_para,
                    'cnn': copy.deepcopy(self.cnn_model.state_dict()) if self.cnn_model else None
                }
                sample_size = mlp_sample_size

        elif self.use_cnn_distillation:
            # Knowledge distillation mode
            should_train_cnn = round_idx > cnn_warmup_rounds

            if should_train_cnn:
                cnn_sample_size, cnn_model_para, cnn_results = self._train_cnn_with_distillation()

                combined_para = {
                    'mlp': mlp_model_para,
                    'cnn': cnn_model_para
                }
                sample_size = mlp_sample_size
            else:
                logger.info(f"Client {self.ID}: CNN warmup - skipping distillation in round {round_idx}")
                combined_para = {
                    'mlp': mlp_model_para,
                    'cnn': copy.deepcopy(self.cnn_model.state_dict()) if self.cnn_model else None
                }
                sample_size = mlp_sample_size
        else:
            # Standard mode: only MLP
            combined_para = mlp_model_para
            sample_size = mlp_sample_size

        # PromptFL: train soft prompts on augmented features and attach to combined_para
        if self.use_promptfl:
            _, prompt_para, _ = self._train_prompt_on_augmented_data()
            if isinstance(combined_para, dict):
                combined_para['prompt'] = prompt_para
            else:
                combined_para = {'mlp': combined_para, 'prompt': prompt_para}

        # Send model parameters
        self.comm_manager.send(
            Message(
                msg_type='model_para',
                sender=self.ID,
                receiver=[sender],
                state=self.state,
                timestamp=timestamp,
                content=(sample_size, combined_para)
            )
        )

    def _handle_separated_training(self, message: Message):
        """Handle training in separated training mode"""
        round_idx = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        # Parse phase from content
        if isinstance(content, dict) and 'phase' in content:
            phase = content['phase']
        else:
            phase = 'classifier'  # Default to classifier phase

        if phase == 'classifier':
            # Phase 1: Train classifier on GGEUR_Clip augmented features
            self.training_phase = 'classifier'
            logger.info(f"Client {self.ID}: Phase 1 (Classifier Training) - Round {round_idx}")

            # Update classifier with global parameters
            classifier_para = content.get('classifier') if isinstance(content, dict) else content
            if classifier_para is not None and self.mlp_classifier is not None:
                try:
                    self.mlp_classifier.load_state_dict(classifier_para, strict=False)
                except Exception as e:
                    logger.debug(f"Client {self.ID}: Could not load classifier: {e}")

            # Train classifier on augmented features
            sample_size, model_para, results = self._train_on_augmented_data()

            # Send classifier parameters
            self.comm_manager.send(
                Message(
                    msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=(sample_size, {'classifier': model_para})
                )
            )

        else:
            # Phase 2: Train CNN backbone with frozen classifier
            self.training_phase = 'cnn_backbone'
            logger.info(f"Client {self.ID}: Phase 2 (CNN Backbone Training) - Round {round_idx}")

            # Get pretrained classifier (frozen)
            classifier_para = content.get('classifier') if isinstance(content, dict) else None
            if classifier_para is not None:
                self._setup_pretrained_classifier(classifier_para)

            # Get CNN backbone parameters
            cnn_backbone_para = content.get('cnn_backbone') if isinstance(content, dict) else None

            # Build CNN backbone if not already built
            if self.cnn_backbone is None:
                self._build_cnn_backbone()

            # Load CNN backbone parameters
            if cnn_backbone_para is not None and self.cnn_backbone is not None:
                try:
                    self.cnn_backbone.load_state_dict(cnn_backbone_para, strict=False)
                except Exception as e:
                    logger.debug(f"Client {self.ID}: Could not load CNN backbone: {e}")

            # Setup image loader if not ready
            if self.original_image_loader is None:
                self._setup_original_image_loader()

            # Train CNN backbone with frozen classifier
            sample_size, backbone_para, results = self._train_cnn_backbone_separated()

            # Send only CNN backbone parameters
            self.comm_manager.send(
                Message(
                    msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=(sample_size, {'cnn_backbone': backbone_para})
                )
            )

    def _setup_pretrained_classifier(self, classifier_para):
        """Setup pretrained classifier for Phase 2"""
        if self.pretrained_classifier is not None:
            return  # Already setup

        # Build classifier with same architecture as MLP
        num_classes = self._cfg.model.num_classes
        input_dim = self.embedding_dim
        hidden_dim = self.ggeur_cfg.mlp_hidden_dim

        if hidden_dim > 0:
            self.pretrained_classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.ggeur_cfg.mlp_dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.pretrained_classifier = nn.Linear(input_dim, num_classes)

        self.pretrained_classifier = self.pretrained_classifier.to(self.device)

        # Load pretrained weights
        try:
            self.pretrained_classifier.load_state_dict(classifier_para, strict=False)
            logger.info(f"Client {self.ID}: Loaded pretrained classifier")
        except Exception as e:
            logger.warning(f"Client {self.ID}: Could not load pretrained classifier: {e}")

        # Freeze classifier
        for param in self.pretrained_classifier.parameters():
            param.requires_grad = False
        self.pretrained_classifier.eval()

        logger.info(f"Client {self.ID}: Pretrained classifier frozen for Phase 2")

    def _build_cnn_backbone(self):
        """Build CNN backbone for separated training Phase 2"""
        from federatedscope.contrib.model.ggeur_cnn import GGEUR_CNN_Backbone

        num_classes = self._cfg.model.num_classes
        cnn_model_name = getattr(self.ggeur_cfg, 'cnn_model', 'resnet18')

        self.cnn_backbone = GGEUR_CNN_Backbone(
            model_name=cnn_model_name,
            num_classes=num_classes,
            pretrained=False  # From scratch
        )
        self.cnn_backbone = self.cnn_backbone.to(self.device)

        logger.info(f"Client {self.ID}: Built CNN backbone ({cnn_model_name}) for Phase 2 - FROM SCRATCH")

    def _train_cnn_backbone_separated(self):
        """
        Train CNN backbone with frozen pretrained classifier.

        Loss function: CE(classifier(backbone(x)), y) + λ * MSE(backbone_feat, CLIP_feat)

        The alignment loss is CRITICAL because:
        - The classifier was trained on CLIP features
        - CNN backbone must produce CLIP-like features for classifier to work
        - Without alignment, CNN features are in a completely different space
        """
        if self.cnn_backbone is None or self.pretrained_classifier is None:
            logger.warning(f"Client {self.ID}: CNN backbone or classifier not ready")
            return 0, {}, {}

        if self.original_image_loader is None:
            logger.warning(f"Client {self.ID}: Image loader not ready")
            return 0, {}, {}

        # Load CLIP model for feature alignment (CRITICAL!)
        if self.clip_model is None:
            logger.info(f"Client {self.ID}: Loading CLIP model for feature alignment...")
            self._load_clip_model()

        if self.clip_model is None:
            logger.warning(f"Client {self.ID}: CLIP model not available, training without alignment")

        # Training settings
        cnn_lr = getattr(self.ggeur_cfg, 'cnn_lr', 0.01)
        cnn_epochs = getattr(self.ggeur_cfg, 'cnn_local_epochs', 10)
        align_weight = getattr(self.ggeur_cfg, 'align_weight', 1.0)  # Feature alignment weight
        cnn_weight_decay = getattr(self.ggeur_cfg, 'cnn_weight_decay', 5e-4)  # 正则化
        cnn_dropout = getattr(self.ggeur_cfg, 'cnn_dropout', 0.5)  # Dropout比例

        # 诊断信息
        num_samples = len(self.original_image_loader.dataset) if hasattr(self.original_image_loader, 'dataset') else 0
        logger.info(f"Client {self.ID}: Phase 2 训练诊断 - CLIP加载:{self.clip_model is not None}, "
                   f"align_weight={align_weight}, lr={cnn_lr}, epochs={cnn_epochs}, samples={num_samples}")

        self.cnn_backbone.train()
        self.pretrained_classifier.eval()  # Classifier stays frozen
        if self.clip_model is not None:
            self.clip_model.eval()

        optimizer = torch.optim.SGD(
            self.cnn_backbone.parameters(),
            lr=cnn_lr,
            momentum=0.9,
            weight_decay=cnn_weight_decay  # 增强正则化
        )

        ce_criterion = nn.CrossEntropyLoss()
        mse_criterion = nn.MSELoss()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_align_loss = 0.0
        total_correct = 0
        total_samples = 0

        logger.info(f"Client {self.ID}: CNN training with CLIP feature alignment, align_weight={align_weight}")

        for epoch in range(cnn_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch in self.original_image_loader:
                if len(batch) >= 2:
                    images, labels = batch[0], batch[1]
                else:
                    continue

                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                # Forward: backbone -> features
                cnn_features = self.cnn_backbone(images)  # 512-dim features

                # Get CLIP features as alignment target (原始特征，不归一化!)
                if self.clip_model is not None:
                    with torch.no_grad():
                        clip_features = self.clip_model.encode_image(images).float()
                        # 不再归一化! 保持原始尺度以匹配分类器期望的输入

                # Classification loss (through frozen classifier)
                logits = self.pretrained_classifier(cnn_features)
                ce_loss = ce_criterion(logits, labels)

                # Feature alignment loss - 使用原始特征对齐 (不归一化)
                # 这样CNN学到的特征尺度也会匹配CLIP
                if self.clip_model is not None:
                    # 方法1: 直接MSE (可能尺度太大)
                    # align_loss = mse_criterion(cnn_features, clip_features)

                    # 方法2: 余弦相似度 + 尺度匹配
                    # 余弦相似度损失 (方向对齐)
                    cnn_norm = F.normalize(cnn_features, p=2, dim=1)
                    clip_norm = F.normalize(clip_features, p=2, dim=1)
                    cosine_loss = 1 - (cnn_norm * clip_norm).sum(dim=1).mean()

                    # 尺度匹配损失 (让CNN特征的范数接近CLIP特征的范数)
                    cnn_magnitude = torch.norm(cnn_features, p=2, dim=1)
                    clip_magnitude = torch.norm(clip_features, p=2, dim=1)
                    magnitude_loss = mse_criterion(cnn_magnitude, clip_magnitude) / (clip_magnitude.mean() ** 2 + 1e-6)

                    # 组合: 方向 + 尺度
                    align_loss = cosine_loss + 0.1 * magnitude_loss

                    loss = ce_loss + align_weight * align_loss
                    total_align_loss += align_loss.item() * images.size(0)

                    # 第一个batch打印诊断信息
                    if epoch == 0 and epoch_samples == 0:
                        logger.info(f"Client {self.ID}: 诊断 - CNN范数={cnn_magnitude.mean().item():.2f}, "
                                   f"CLIP范数={clip_magnitude.mean().item():.2f}, "
                                   f"cosine_loss={cosine_loss.item():.4f}, "
                                   f"magnitude_loss={magnitude_loss.item():.4f}")
                else:
                    loss = ce_loss
                    align_loss = torch.tensor(0.0)

                # Backward: only updates backbone
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cnn_backbone.parameters(), max_norm=1.0)
                optimizer.step()

                # Statistics
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_ce_loss += ce_loss.item() * batch_size
                _, predicted = torch.max(logits, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += batch_size

                epoch_loss += loss.item() * batch_size
                epoch_correct += (predicted == labels).sum().item()
                epoch_samples += batch_size

            if epoch_samples > 0:
                epoch_acc = epoch_correct / epoch_samples
                # 计算平均特征范数用于诊断
                logger.info(f"Client {self.ID}: CNN Epoch {epoch+1}/{cnn_epochs}, "
                           f"Loss={epoch_loss/epoch_samples:.4f}, Acc={epoch_acc:.4f}")

        # Compute averages
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_ce_loss = total_ce_loss / total_samples
            avg_align_loss = total_align_loss / total_samples
            accuracy = total_correct / total_samples
        else:
            avg_loss = avg_ce_loss = avg_align_loss = accuracy = 0

        logger.info(f"Client {self.ID}: CNN Training - loss={avg_loss:.4f}, "
                   f"ce={avg_ce_loss:.4f}, align={avg_align_loss:.4f}, acc={accuracy:.4f}")

        # Get backbone parameters
        backbone_para = copy.deepcopy(self.cnn_backbone.state_dict())

        results = {
            'cnn_train_loss': avg_loss,
            'cnn_ce_loss': avg_ce_loss,
            'cnn_align_loss': avg_align_loss,
            'cnn_train_acc': accuracy,
            'cnn_train_total': total_samples
        }

        return total_samples, backbone_para, results

    def _train_on_augmented_data(self):
        """Train MLP classifier on augmented features with optional FedProto/FedProx regularization"""
        if self.augmented_loader is None or self.mlp_classifier is None:
            return 0, {}, {}

        self.mlp_classifier.train()
        optimizer = torch.optim.Adam(self.mlp_classifier.parameters(),
                                     lr=self._cfg.train.optimizer.lr)
        criterion = nn.CrossEntropyLoss()

        # FedProx settings (proximal term to the received global model)
        use_fedprox = getattr(self._cfg.fedprox, 'use', False) if hasattr(self._cfg, 'fedprox') else False
        fedprox_mu = getattr(self._cfg.fedprox, 'mu', 0.0) if hasattr(self._cfg, 'fedprox') else 0.0
        global_params_ref = None
        if use_fedprox and fedprox_mu > 0:
            # Snapshot current model params as the global reference before local updates
            global_params_ref = {
                k: v.detach().clone()
                for k, v in self.mlp_classifier.state_dict().items()
            }

        # MOON settings
        use_moon = getattr(self.ggeur_cfg, 'use_moon', False)
        moon_mu = getattr(self.ggeur_cfg, 'moon_mu', 5.0)
        moon_temperature = getattr(self.ggeur_cfg, 'moon_temperature', 0.5)
        if use_moon:
            logger.info(f"Client {self.ID}: MOON enabled - mu={moon_mu}, temperature={moon_temperature}")

        # FedProto settings
        use_fedproto = getattr(self.ggeur_cfg, 'use_fedproto', False)
        proto_weight = getattr(self.ggeur_cfg, 'proto_weight', 1.0)
        proto_distance = getattr(self.ggeur_cfg, 'proto_distance', 'cosine')
        proto_temperature = getattr(self.ggeur_cfg, 'proto_temperature', 0.1)

        # Debug logging for FedProto
        logger.info(f"Client {self.ID}: FedProto settings - use_fedproto={use_fedproto}, "
                   f"proto_weight={proto_weight}, proto_distance={proto_distance}, "
                   f"global_prototypes_count={len(self.global_prototypes) if self.global_prototypes else 0}")
        if use_fedprox and fedprox_mu > 0:
            logger.info(f"Client {self.ID}: FedProx enabled - mu={fedprox_mu}")

        # Prepare global prototypes tensor if using FedProto
        proto_tensor = None
        if use_fedproto and self.global_prototypes:
            num_classes = self._cfg.model.num_classes
            embedding_dim = self.embedding_dim
            proto_tensor = torch.zeros(num_classes, embedding_dim).to(self.device)
            for class_idx, proto in self.global_prototypes.items():
                class_idx = int(class_idx)
                if class_idx < num_classes:
                    if isinstance(proto, np.ndarray):
                        proto_tensor[class_idx] = torch.from_numpy(proto).float()
                    else:
                        proto_tensor[class_idx] = proto.float()
            logger.info(f"Client {self.ID}: FedProto enabled with {len(self.global_prototypes)} prototypes")

        total_loss = 0.0
        total_ce_loss = 0.0
        total_proto_loss = 0.0
        total_prox_loss = 0.0
        total_moon_loss = 0.0
        total_correct = 0
        total_samples = 0

        local_epochs = self._cfg.train.local_update_steps

        for epoch in range(local_epochs):
            for features, labels in self.augmented_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.mlp_classifier(features)
                ce_loss = criterion(outputs, labels)

                # Compute FedProto prototype loss
                proto_loss = torch.tensor(0.0, device=self.device)
                if use_fedproto and proto_tensor is not None:
                    # NOTE: If we compute prototype loss directly on `features`,
                    # it will NOT affect training because `features` are inputs
                    # (no gradients). Therefore we compute FedProto loss on
                    # trainable quantities:
                    #  - for linear classifier: align class weight vectors with prototypes
                    #  - for MLP: align a trainable representation with transformed prototypes

                    # Build a mask for available prototypes (some classes may be missing under LDS)
                    proto_norms = torch.norm(proto_tensor, p=2, dim=1)
                    proto_mask = proto_norms > 0

                    if isinstance(self.mlp_classifier, nn.Linear):
                        # Align classifier weights with corresponding prototypes
                        if proto_mask.any():
                            weight = self.mlp_classifier.weight  # (C, D)
                            weight_sel = weight[proto_mask]
                            proto_sel = proto_tensor[proto_mask]

                            if proto_distance == 'cosine':
                                weight_norm = nn.functional.normalize(weight_sel, p=2, dim=1)
                                proto_norm = nn.functional.normalize(proto_sel, p=2, dim=1)
                                similarity = (weight_norm * proto_norm).sum(dim=1)
                                proto_loss = (1 - similarity).mean()
                            else:
                                proto_loss = nn.functional.mse_loss(weight_sel, proto_sel)
                    else:
                        # If the classifier is a small MLP, align the *trainable* representation.
                        # We use the first Linear (+ ReLU if present) as a representation mapper.
                        mapper = None
                        mapper_act = None
                        if isinstance(self.mlp_classifier, nn.Sequential) and len(self.mlp_classifier) >= 1:
                            if isinstance(self.mlp_classifier[0], nn.Linear):
                                mapper = self.mlp_classifier[0]
                            if len(self.mlp_classifier) >= 2 and isinstance(self.mlp_classifier[1], nn.ReLU):
                                mapper_act = self.mlp_classifier[1]

                        if mapper is not None and proto_mask.any():
                            rep = mapper(features)
                            proto_rep = mapper(proto_tensor)
                            if mapper_act is not None:
                                rep = mapper_act(rep)
                                proto_rep = mapper_act(proto_rep)

                            target_protos = proto_rep[labels]
                            if proto_distance == 'cosine':
                                rep_norm = nn.functional.normalize(rep, p=2, dim=1)
                                target_norm = nn.functional.normalize(target_protos, p=2, dim=1)
                                similarity = (rep_norm * target_norm).sum(dim=1)
                                proto_loss = (1 - similarity).mean()
                            else:
                                proto_loss = nn.functional.mse_loss(rep, target_protos)

                # Compute FedProx proximal loss on model parameters
                prox_loss = torch.tensor(0.0, device=self.device)
                if use_fedprox and fedprox_mu > 0 and global_params_ref is not None:
                    for name, param in self.mlp_classifier.named_parameters():
                        if not param.requires_grad:
                            continue
                        ref = global_params_ref.get(name, None)
                        if ref is None:
                            continue
                        if isinstance(ref, torch.Tensor) and ref.device != param.device:
                            ref = ref.to(param.device)
                        prox_loss = prox_loss + torch.sum((param - ref) ** 2)

                # Compute MOON contrastive loss
                # L_con = -log( exp(sim(z,z_global)/τ) / (exp(sim(z,z_global)/τ) + exp(sim(z,z_prev)/τ)) )
                moon_loss = torch.tensor(0.0, device=self.device)
                if use_moon and self.moon_global_model is not None:
                    with torch.no_grad():
                        z_global = self.moon_global_model(features)
                        if self.moon_prev_model is not None:
                            z_prev = self.moon_prev_model(features)
                        else:
                            # First round: no previous local model, use global as prev (loss = 0)
                            z_prev = z_global.clone()

                    z = outputs  # current model logits as representation
                    z_norm = F.normalize(z, dim=1)
                    z_global_norm = F.normalize(z_global, dim=1)
                    z_prev_norm = F.normalize(z_prev, dim=1)

                    sim_global = (z_norm * z_global_norm).sum(dim=1, keepdim=True) / moon_temperature
                    sim_prev = (z_norm * z_prev_norm).sum(dim=1, keepdim=True) / moon_temperature

                    logits_con = torch.cat([sim_global, sim_prev], dim=1)
                    labels_con = torch.zeros(features.size(0), dtype=torch.long, device=self.device)
                    moon_loss = criterion(logits_con, labels_con)

                # Total loss
                loss = ce_loss + proto_weight * proto_loss
                if use_fedprox and fedprox_mu > 0:
                    loss = loss + (fedprox_mu / 2.0) * prox_loss
                if use_moon:
                    loss = loss + moon_mu * moon_loss
                loss.backward()
                optimizer.step()

                batch_size = features.size(0)
                total_loss += loss.item() * batch_size
                total_ce_loss += ce_loss.item() * batch_size
                total_proto_loss += proto_loss.item() * batch_size
                total_prox_loss += prox_loss.item() * batch_size
                total_moon_loss += moon_loss.item() * batch_size
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_ce_loss = total_ce_loss / total_samples if total_samples > 0 else 0
        avg_proto_loss = total_proto_loss / total_samples if total_samples > 0 else 0
        avg_prox_loss = total_prox_loss / total_samples if total_samples > 0 else 0
        avg_moon_loss = total_moon_loss / total_samples if total_samples > 0 else 0
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        if use_fedproto:
            logger.info(f"Client {self.ID}: Train loss={avg_loss:.4f} (CE={avg_ce_loss:.4f}, "
                       f"proto={avg_proto_loss:.4f}), accuracy={accuracy:.4f}")
        elif use_fedprox and fedprox_mu > 0:
            logger.info(f"Client {self.ID}: Train loss={avg_loss:.4f} (CE={avg_ce_loss:.4f}, "
                       f"prox={avg_prox_loss:.4f}), accuracy={accuracy:.4f}")
        elif use_moon:
            logger.info(f"Client {self.ID}: Train loss={avg_loss:.4f} (CE={avg_ce_loss:.4f}, "
                       f"moon={avg_moon_loss:.4f}), accuracy={accuracy:.4f}")
        else:
            logger.info(f"Client {self.ID}: Train loss={avg_loss:.4f}, accuracy={accuracy:.4f}")

        # Get model parameters
        model_para = copy.deepcopy(self.mlp_classifier.state_dict())

        results = {
            'train_loss': avg_loss,
            'train_ce_loss': avg_ce_loss,
            'train_proto_loss': avg_proto_loss,
            'train_prox_loss': avg_prox_loss,
            'train_moon_loss': avg_moon_loss,
            'train_acc': accuracy,
            'train_total': total_samples
        }

        return total_samples, model_para, results

    # ==================== CNN Knowledge Distillation Methods ====================

    def _build_cnn_model(self):
        """Build CNN model for knowledge distillation"""
        from federatedscope.contrib.model.ggeur_cnn import GGEUR_CNN_FeatureAlign

        num_classes = self._cfg.model.num_classes
        cnn_model_name = getattr(self.ggeur_cfg, 'cnn_model', 'resnet18')
        cnn_pretrained = getattr(self.ggeur_cfg, 'cnn_pretrained', True)
        clip_dim = getattr(self.ggeur_cfg, 'embedding_dim', 512)

        # Use the new FeatureAlign model for consistency
        self.cnn_model = GGEUR_CNN_FeatureAlign(
            model_name=cnn_model_name,
            num_classes=num_classes,
            clip_dim=clip_dim,
            pretrained=cnn_pretrained
        )
        self.cnn_model = self.cnn_model.to(self.device)

        logger.info(f"Client {self.ID}: Built CNN model ({cnn_model_name}) with {num_classes} classes, pretrained={cnn_pretrained}")

    def _build_cnn_feature_align(self):
        """Build CNN model for feature alignment (from scratch training)"""
        from federatedscope.contrib.model.ggeur_cnn import GGEUR_CNN_FeatureAlign

        num_classes = self._cfg.model.num_classes
        cnn_model_name = getattr(self.ggeur_cfg, 'cnn_model', 'resnet18')
        clip_dim = getattr(self.ggeur_cfg, 'embedding_dim', 512)

        # Feature alignment mode: always start from scratch (no pretrained weights)
        self.cnn_model = GGEUR_CNN_FeatureAlign(
            model_name=cnn_model_name,
            num_classes=num_classes,
            clip_dim=clip_dim,
            pretrained=False  # From scratch!
        )
        self.cnn_model = self.cnn_model.to(self.device)

        logger.info(f"Client {self.ID}: Built CNN ({cnn_model_name}) for feature alignment - FROM SCRATCH, {num_classes} classes")

    def _setup_original_image_loader(self):
        """Setup DataLoader for original images (for CNN training)"""
        # Get train data
        train_data = self.trainer.ctx.data.get('train', None)
        if train_data is None:
            train_data = self.data.get('train', None)

        if train_data is None:
            logger.error(f"Client {self.ID}: No training data available for CNN")
            return

        # Get the underlying dataset
        dataset = train_data.dataset if hasattr(train_data, 'dataset') else train_data

        # 为CNN训练创建带数据增强的包装数据集
        use_augmentation = getattr(self.ggeur_cfg, 'cnn_use_augmentation', True)
        if use_augmentation:
            dataset = AugmentedImageDataset(dataset)
            logger.info(f"Client {self.ID}: CNN training with data augmentation enabled")

        # Create DataLoader for original images
        batch_size = self._cfg.dataloader.batch_size
        self.original_image_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 to avoid multiprocessing issues
            drop_last=False
        )

        logger.info(f"Client {self.ID}: Setup original image loader with {len(dataset)} samples")

    def _train_cnn_with_distillation(self):
        """
        Train CNN with knowledge distillation from MLP teacher.

        The loss function is:
        L = alpha * CE(CNN(x), y) + (1-alpha) * T^2 * KL(softmax(CNN(x)/T), softmax(MLP(CLIP(x))/T))

        Returns:
            sample_size: Number of samples trained
            cnn_para: CNN model parameters
            results: Training results dict
        """
        if self.cnn_model is None or self.original_image_loader is None:
            logger.warning(f"Client {self.ID}: CNN or image loader not ready")
            return 0, {}, {}

        # Ensure CLIP model is loaded (may not be loaded if features were cached)
        if self.clip_model is None:
            logger.info(f"Client {self.ID}: Loading CLIP model for CNN distillation...")
            self._load_clip_model()

        if self.mlp_classifier is None or self.clip_model is None:
            logger.warning(f"Client {self.ID}: MLP teacher or CLIP not ready - "
                          f"MLP: {self.mlp_classifier is not None}, CLIP: {self.clip_model is not None}")
            return 0, {}, {}

        # Training settings
        temperature = getattr(self.ggeur_cfg, 'distill_temperature', 4.0)
        alpha = getattr(self.ggeur_cfg, 'distill_alpha', 0.5)
        cnn_lr = getattr(self.ggeur_cfg, 'cnn_lr', 0.01)
        cnn_epochs = getattr(self.ggeur_cfg, 'cnn_local_epochs', 5)

        self.cnn_model.train()
        self.mlp_classifier.eval()  # Teacher is frozen
        self.clip_model.eval()

        # Use differential learning rates: lower for backbone, higher for classifier
        backbone_params = []
        classifier_params = []
        for name, param in self.cnn_model.named_parameters():
            if 'fc' in name or 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = torch.optim.SGD([
            {'params': backbone_params, 'lr': cnn_lr * 0.1},  # Lower LR for backbone
            {'params': classifier_params, 'lr': cnn_lr}       # Higher LR for classifier
        ], momentum=0.9, weight_decay=1e-4)

        ce_criterion = nn.CrossEntropyLoss()
        kl_criterion = nn.KLDivLoss(reduction='batchmean')

        total_loss = 0.0
        total_ce_loss = 0.0
        total_kl_loss = 0.0
        total_correct = 0
        total_samples = 0
        teacher_correct = 0  # Track teacher accuracy for debugging

        # Debug: check data loader
        num_batches = len(self.original_image_loader)
        logger.info(f"Client {self.ID}: CNN training with {num_batches} batches, "
                   f"backbone_lr={cnn_lr*0.1}, classifier_lr={cnn_lr}, epochs={cnn_epochs}")

        for epoch in range(cnn_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            epoch_teacher_correct = 0

            for batch in self.original_image_loader:
                if len(batch) >= 2:
                    images, labels = batch[0], batch[1]
                else:
                    continue

                images = images.to(self.device)
                labels = labels.to(self.device)

                # Get teacher's soft labels
                with torch.no_grad():
                    # Extract CLIP features
                    clip_features = self.clip_model.encode_image(images)
                    # Get MLP teacher's logits
                    teacher_logits = self.mlp_classifier(clip_features.float())
                    # Track teacher accuracy for debugging
                    _, teacher_pred = torch.max(teacher_logits, 1)
                    epoch_teacher_correct += (teacher_pred == labels).sum().item()

                # Get student's (CNN) logits
                optimizer.zero_grad()
                student_logits = self.cnn_model(images)

                # Hard label loss (Cross-Entropy)
                ce_loss = ce_criterion(student_logits, labels)

                # Soft label loss (KL-Divergence with temperature)
                soft_student = F.log_softmax(student_logits / temperature, dim=1)
                soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
                kl_loss = kl_criterion(soft_student, soft_teacher) * (temperature ** 2)

                # Combined loss
                loss = alpha * ce_loss + (1 - alpha) * kl_loss

                # Backward and optimize
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.cnn_model.parameters(), max_norm=1.0)
                optimizer.step()

                # Statistics
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_ce_loss += ce_loss.item() * batch_size
                total_kl_loss += kl_loss.item() * batch_size

                _, predicted = torch.max(student_logits, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += batch_size
                teacher_correct += (teacher_pred == labels).sum().item()

                epoch_loss += loss.item() * batch_size
                epoch_correct += (predicted == labels).sum().item()
                epoch_samples += batch_size

            if epoch_samples > 0:
                epoch_acc = epoch_correct / epoch_samples
                epoch_teacher_acc = epoch_teacher_correct / epoch_samples
                logger.info(f"Client {self.ID}: CNN Epoch {epoch+1}/{cnn_epochs}, "
                            f"Loss={epoch_loss/epoch_samples:.4f}, "
                            f"CNN_Acc={epoch_acc:.4f}, Teacher_Acc={epoch_teacher_acc:.4f}")

        # Compute averages
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_ce_loss = total_ce_loss / total_samples
            avg_kl_loss = total_kl_loss / total_samples
            accuracy = total_correct / total_samples
            teacher_accuracy = teacher_correct / total_samples
        else:
            avg_loss = avg_ce_loss = avg_kl_loss = accuracy = teacher_accuracy = 0

        logger.info(f"Client {self.ID}: CNN Train - loss={avg_loss:.4f}, "
                   f"ce_loss={avg_ce_loss:.4f}, kl_loss={avg_kl_loss:.4f}, "
                   f"cnn_acc={accuracy:.4f}, teacher_acc={teacher_accuracy:.4f}")

        # Get CNN model parameters
        cnn_para = copy.deepcopy(self.cnn_model.state_dict())

        results = {
            'cnn_train_loss': avg_loss,
            'cnn_ce_loss': avg_ce_loss,
            'cnn_kl_loss': avg_kl_loss,
            'cnn_train_acc': accuracy,
            'cnn_teacher_acc': teacher_accuracy,
            'cnn_train_total': total_samples
        }

        return total_samples, cnn_para, results

    # ==================== CNN Feature Alignment Methods ====================

    def _train_cnn_with_feature_alignment(self):
        """
        Train CNN from scratch using feature alignment with CLIP.

        The loss function is:
        L = CE(CNN(x), y) + λ * MSE(proj(CNN_feat(x)), CLIP_feat(x))

        Optionally, also align with global prototypes:
        L += λ_proto * MSE(proj(CNN_feat(x)), prototype[y])

        Returns:
            sample_size: Number of samples trained
            cnn_para: CNN model parameters
            results: Training results dict
        """
        if self.cnn_model is None or self.original_image_loader is None:
            logger.warning(f"Client {self.ID}: CNN or image loader not ready")
            return 0, {}, {}

        # Ensure CLIP model is loaded
        if self.clip_model is None:
            logger.info(f"Client {self.ID}: Loading CLIP model for feature alignment...")
            self._load_clip_model()

        if self.clip_model is None:
            logger.warning(f"Client {self.ID}: CLIP model not available")
            return 0, {}, {}

        # Training settings
        cnn_lr = getattr(self.ggeur_cfg, 'cnn_lr', 0.01)
        cnn_epochs = getattr(self.ggeur_cfg, 'cnn_local_epochs', 10)
        align_weight = getattr(self.ggeur_cfg, 'align_weight', 1.0)
        use_prototype_align = getattr(self.ggeur_cfg, 'use_prototype_alignment', True)
        prototype_weight = getattr(self.ggeur_cfg, 'prototype_align_weight', 0.5)

        self.cnn_model.train()
        self.clip_model.eval()

        # Optimizer
        optimizer = torch.optim.SGD(
            self.cnn_model.parameters(),
            lr=cnn_lr,
            momentum=0.9,
            weight_decay=1e-4
        )

        # Loss functions
        ce_criterion = nn.CrossEntropyLoss()
        mse_criterion = nn.MSELoss()

        # Convert global prototypes to tensor
        prototype_tensor = None
        if use_prototype_align and self.global_prototypes:
            num_classes = self._cfg.model.num_classes
            clip_dim = getattr(self.ggeur_cfg, 'embedding_dim', 512)
            prototype_tensor = torch.zeros(num_classes, clip_dim).to(self.device)
            for class_idx, proto in self.global_prototypes.items():
                class_idx = int(class_idx)
                if class_idx < num_classes:
                    if isinstance(proto, np.ndarray):
                        prototype_tensor[class_idx] = torch.from_numpy(proto).float()
                    else:
                        prototype_tensor[class_idx] = proto.float()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_align_loss = 0.0
        total_proto_loss = 0.0
        total_correct = 0
        total_samples = 0

        num_batches = len(self.original_image_loader)
        logger.info(f"Client {self.ID}: Feature alignment training with {num_batches} batches, "
                   f"lr={cnn_lr}, epochs={cnn_epochs}, align_weight={align_weight}")

        for epoch in range(cnn_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch in self.original_image_loader:
                if len(batch) >= 2:
                    images, labels = batch[0], batch[1]
                else:
                    continue

                images = images.to(self.device)
                labels = labels.to(self.device)

                # Get CLIP features (target for alignment)
                with torch.no_grad():
                    clip_features = self.clip_model.encode_image(images).float()
                    clip_features_norm = F.normalize(clip_features, p=2, dim=1)

                # Forward pass through CNN
                optimizer.zero_grad()
                logits, proj_features = self.cnn_model(images, return_features=True)
                proj_features_norm = F.normalize(proj_features, p=2, dim=1)

                # Classification loss
                ce_loss = ce_criterion(logits, labels)

                # Feature alignment loss (align CNN features with CLIP features)
                align_loss = mse_criterion(proj_features_norm, clip_features_norm)

                # Prototype alignment loss (optional)
                proto_loss = torch.tensor(0.0).to(self.device)
                if use_prototype_align and prototype_tensor is not None:
                    # Get prototypes for each sample's class
                    target_prototypes = prototype_tensor[labels]
                    target_prototypes_norm = F.normalize(target_prototypes, p=2, dim=1)
                    proto_loss = mse_criterion(proj_features_norm, target_prototypes_norm)

                # Combined loss
                loss = ce_loss + align_weight * align_loss
                if use_prototype_align:
                    loss = loss + prototype_weight * proto_loss

                # Backward and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cnn_model.parameters(), max_norm=1.0)
                optimizer.step()

                # Statistics
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_ce_loss += ce_loss.item() * batch_size
                total_align_loss += align_loss.item() * batch_size
                total_proto_loss += proto_loss.item() * batch_size

                _, predicted = torch.max(logits, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += batch_size

                epoch_loss += loss.item() * batch_size
                epoch_correct += (predicted == labels).sum().item()
                epoch_samples += batch_size

            if epoch_samples > 0:
                epoch_acc = epoch_correct / epoch_samples
                logger.info(f"Client {self.ID}: CNN Epoch {epoch+1}/{cnn_epochs}, "
                            f"Loss={epoch_loss/epoch_samples:.4f}, Acc={epoch_acc:.4f}")

        # Compute averages
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_ce_loss = total_ce_loss / total_samples
            avg_align_loss = total_align_loss / total_samples
            avg_proto_loss = total_proto_loss / total_samples
            accuracy = total_correct / total_samples
        else:
            avg_loss = avg_ce_loss = avg_align_loss = avg_proto_loss = accuracy = 0

        logger.info(f"Client {self.ID}: CNN Feature Align - loss={avg_loss:.4f}, "
                   f"ce={avg_ce_loss:.4f}, align={avg_align_loss:.4f}, "
                   f"proto={avg_proto_loss:.4f}, acc={accuracy:.4f}")

        # Get CNN model parameters
        cnn_para = copy.deepcopy(self.cnn_model.state_dict())

        results = {
            'cnn_train_loss': avg_loss,
            'cnn_ce_loss': avg_ce_loss,
            'cnn_align_loss': avg_align_loss,
            'cnn_proto_loss': avg_proto_loss,
            'cnn_train_acc': accuracy,
            'cnn_train_total': total_samples
        }

        return total_samples, cnn_para, results

    # ==================== PromptFL Methods ====================

    def _get_class_names(self):
        """Get class names from dataset or config, falling back to generic names."""
        # 1. Check config override
        cfg_names = getattr(self.ggeur_cfg, 'prompt_class_names', [])
        if cfg_names:
            return list(cfg_names)

        # 2. Try to infer from dataset type
        data_type = self._cfg.data.type.lower()
        try:
            if 'office' in data_type and 'home' in data_type:
                from federatedscope.cv.dataset.office_home import OfficeHome
                return list(OfficeHome.CLASSES)
            elif 'pacs' in data_type:
                from federatedscope.cv.dataset.pacs import PACS
                return list(PACS.CLASSES)
            elif 'office' in data_type and 'caltech' in data_type:
                from federatedscope.cv.dataset.office_caltech import OfficeCaltech10
                return list(OfficeCaltech10.CLASSES)
        except Exception as e:
            logger.warning(f"Client {self.ID}: Could not load class names from dataset: {e}")

        # 3. Fallback: generic names
        num_classes = self._cfg.model.num_classes
        logger.warning(f"Client {self.ID}: Using generic class names for {num_classes} classes")
        return [f"class {i}" for i in range(num_classes)]

    def _build_prompt_model(self):
        """Build CustomCLIP (HuggingFace-based) for PromptFL."""
        from federatedscope.contrib.model.ggeur_prompt import CustomCLIP
        from transformers import CLIPModel, CLIPProcessor

        n_ctx = getattr(self.ggeur_cfg, 'prompt_length', 16)
        class_names = self._get_class_names()
        model_path = getattr(self.ggeur_cfg, 'clip_model_path', '')
        clip_pretrained = getattr(self.ggeur_cfg, 'clip_pretrained', 'openai')

        # Determine HuggingFace model identifier
        # Support local path or HuggingFace hub id
        if model_path and os.path.isdir(model_path):
            hf_model_id = model_path
        else:
            # Map open_clip model names to HuggingFace ids
            clip_model_name = getattr(self.ggeur_cfg, 'clip_model', 'ViT-B-16')
            hf_model_id = getattr(self.ggeur_cfg, 'hf_clip_model_id', 'openai/clip-vit-base-patch16')

        try:
            logger.info(f"Client {self.ID}: Loading HuggingFace CLIP from {hf_model_id}")
            self.hf_clip_model = CLIPModel.from_pretrained(hf_model_id)
            processor = CLIPProcessor.from_pretrained(hf_model_id)
        except Exception as e:
            logger.error(f"Client {self.ID}: Failed to load HF CLIP: {e}")
            return

        template = getattr(self.ggeur_cfg, 'prompt_template', 'a photo of a {}')

        self.custom_clip = CustomCLIP(
            clip_model=self.hf_clip_model,
            processor=processor,
            classnames=class_names,
            n_ctx=n_ctx,
            template=template,
            device=self.device,
        )
        # Keep backward-compat references
        self.prompt_learner = self.custom_clip.prompt_learner
        self.text_encoder = self.custom_clip.text_encoder

        logger.info(f"Client {self.ID}: Built CustomCLIP with {n_ctx} ctx tokens, "
                    f"{len(class_names)} classes")

    def _train_prompt_on_augmented_data(self):
        """
        Train soft prompt ctx vectors on augmented CLIP features.

        Uses CustomCLIP's PromptLearner + TextEncoder (HuggingFace-based).
        Image features are pre-extracted; only ctx vectors are updated.

        Returns:
            sample_size, prompt_para dict, results dict
        """
        if self.custom_clip is None or self.augmented_loader is None:
            return 0, {}, {}

        lr = getattr(self.ggeur_cfg, 'prompt_lr', 0.002)
        local_epochs = getattr(self.ggeur_cfg, 'prompt_local_epochs', 10)

        self.custom_clip.prompt_learner.train()

        optimizer = torch.optim.Adam([self.custom_clip.prompt_learner.ctx], lr=lr)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(local_epochs):
            for features, labels in self.augmented_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Normalize pre-extracted image features
                img_feats = F.normalize(features, p=2, dim=1)

                # Get text features via CustomCLIP's prompt + text encoder
                prompt_embeds, attn_mask = self.custom_clip.prompt_learner()
                text_feats = self.custom_clip.text_encoder(
                    prompt_embeds, attn_mask, self.custom_clip.clip_model
                )
                text_feats = F.normalize(text_feats, p=2, dim=1)

                # Scale by learned logit_scale (same as CustomCLIP.forward)
                logit_scale = self.custom_clip.clip_model.logit_scale.exp()
                logits = logit_scale * img_feats @ text_feats.T

                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = features.size(0)
                total_loss += loss.item() * batch_size
                _, predicted = torch.max(logits, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        logger.info(f"Client {self.ID}: Prompt training - loss={avg_loss:.4f}, acc={accuracy:.4f}, "
                    f"samples={total_samples}")

        prompt_para = {'ctx': copy.deepcopy(self.custom_clip.prompt_learner.ctx.data.cpu())}
        results = {'prompt_loss': avg_loss, 'prompt_acc': accuracy, 'prompt_total': total_samples}

        return total_samples, prompt_para, results


def call_ggeur_worker(method):
    """Factory function for GGEUR_Clip worker"""
    if method.lower() == 'ggeur':
        from federatedscope.contrib.worker.ggeur_server import GGEURServer
        return {
            'client': GGEURClient,
            'server': GGEURServer
        }
    return None


register_worker('ggeur', call_ggeur_worker)
