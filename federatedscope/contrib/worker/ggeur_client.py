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
import gc
import base64
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
            transforms.ColorJitter(brightness=0.4,
                                   contrast=0.4,
                                   saturation=0.4,
                                   hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ])

        # 标准化（与原始数据集一致）
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

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
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        super(GGEURClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, is_unseen_client, *args, **kwargs)

        if config is None:
            return

        self.ggeur_cfg = config.ggeur

        # ===== Feature Extractor Mode =====
        # 'clip': Use CLIP (ViT-based, original method)
        # 'cnn': Use pretrained CNN (ConvNeXt, ResNet, etc.)
        self.feature_extractor_type = getattr(self.ggeur_cfg,
                                              'feature_extractor', 'clip')

        # CLIP model (for 'clip' mode)
        self.clip_model = None
        self.clip_preprocess = None

        # CNN feature extractor (for 'cnn' mode)
        self.cnn_extractor = None

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
        self.use_end_to_end_finetune = getattr(self.ggeur_cfg,
                                               'use_end_to_end_finetune',
                                               False)
        self.finetune_start_round = getattr(self.ggeur_cfg,
                                            'finetune_start_round', 0)
        self.full_model = None  # Combined CNN + classifier for fine-tuning
        self.original_image_loader = None

        # ===== Legacy modes (for backward compatibility) =====
        self.use_cnn_distillation = getattr(self.ggeur_cfg,
                                            'use_cnn_distillation', False)
        self.use_feature_alignment = getattr(self.ggeur_cfg,
                                             'use_feature_alignment', False)
        self.cnn_model = None
        self.use_separated_training = getattr(self.ggeur_cfg,
                                              'use_separated_training', False)
        self.training_phase = 'classifier'
        self.pretrained_classifier = None
        self.cnn_backbone = None

    def _register_default_handlers(self):
        """Register message handlers"""
        super()._register_default_handlers()

        # Register handler for receiving global covariance matrices
        self.register_handlers('global_covariances',
                               self.callback_for_global_covariances)

    def _load_feature_extractor(self):
        """Load feature extractor (CLIP or CNN based on config)"""
        if self.feature_extractor_type == 'cnn':
            self._load_cnn_extractor()
        else:
            self._load_clip_model()

    def _load_cnn_extractor(self):
        """Load CNN feature extractor"""
        if self.cnn_extractor is not None:
            return

        try:
            from federatedscope.contrib.model.ggeur_cnn_extractor import CNNFeatureExtractor

            model_name = getattr(self.ggeur_cfg, 'cnn_backbone',
                                 'convnext_base')
            pretrained = getattr(self.ggeur_cfg, 'cnn_pretrained', True)
            freeze = getattr(self.ggeur_cfg, 'freeze_backbone', True)

            self.cnn_extractor = CNNFeatureExtractor(model_name=model_name,
                                                     pretrained=pretrained,
                                                     freeze=freeze)
            self.cnn_extractor = self.cnn_extractor.to(self.device)

            logger.info(
                f"Client {self.ID}: Loaded CNN extractor {model_name}, "
                f"feature_dim={self.cnn_extractor.get_feature_dim()}")

        except Exception as e:
            logger.error(
                f"Client {self.ID}: Failed to load CNN extractor: {e}")
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
                logger.info(
                    f"Client {self.ID}: Loading CLIP from local path: {local_path}"
                )
                self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=local_path)
            else:
                # Try to load from pretrained source (may require internet)
                logger.info(
                    f"Client {self.ID}: Loading CLIP from pretrained: {pretrained}"
                )
                try:
                    self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                        model_name, pretrained=pretrained)
                except Exception as e:
                    logger.error(
                        f"Failed to download CLIP model. Please set ggeur.clip_model_path to local weights file."
                    )
                    logger.error(f"Error: {e}")
                    raise

            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            logger.info(f"Client {self.ID}: Loaded CLIP model {model_name}")

        except ImportError:
            logger.error(
                "open_clip not installed. Please install: pip install open_clip_torch"
            )
            raise

    def _get_feature_cache_path(self, domain=None):
        """Get the path for cached CLIP features"""
        # Check if caching is enabled
        if not getattr(self.ggeur_cfg, 'use_feature_cache', True):
            return None

        # Get cache directory from config or use default
        cache_dir = getattr(self.ggeur_cfg, 'feature_cache_dir', '')
        if not cache_dir:
            cache_dir = os.path.join(os.path.dirname(self._cfg.data.root),
                                     'clip_feature_cache')

        # Cache isolation for distributed mode
        if (hasattr(self._cfg, 'federate')
                and self._cfg.federate.mode == 'distributed'):
            cache_dir = os.path.join(cache_dir, f'client_{self.ID}')

        os.makedirs(cache_dir, exist_ok=True)

        # Build cache filename based on dataset, domain, and model
        dataset_name = self._cfg.data.type.lower()

        if self.feature_extractor_type == 'cnn':
            model_name = getattr(self.ggeur_cfg, 'cnn_backbone',
                                 'convnext_base')
            model_str = model_name.replace('/', '_').replace('-', '_')
            prefix = 'cnn'
        else:
            clip_model = self.ggeur_cfg.clip_model.replace('/', '_').replace(
                '-', '_')
            pretrained = self.ggeur_cfg.clip_pretrained.replace('/',
                                                                '_').replace(
                                                                    '-', '_')
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
                logger.info(
                    f"Client {self.ID}: Loaded {len(cache)} cached features from {cache_path}"
                )
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
            logger.info(
                f"Client {self.ID}: Saved {len(paths)} features to cache {cache_path}"
            )
        except Exception as e:
            logger.warning(f"Client {self.ID}: Failed to save cache: {e}")

    def _extract_features(self):
        """
        Extract features from local data using either CLIP or CNN.
        Supports caching for both modes.
        """
        extractor_name = 'CNN' if self.feature_extractor_type == 'cnn' else 'CLIP'
        logger.info(
            f"Client {self.ID}: Extracting {extractor_name} features...")

        # Get train data
        train_data = self.trainer.ctx.data.get('train', None)
        if train_data is None:
            train_data = self.data.get('train', None)

        if train_data is None:
            logger.error(f"Client {self.ID}: No training data available")
            return

        # Get the underlying dataset (handle DataLoader -> Dataset -> possibly Subset)
        dataset = train_data.dataset if hasattr(train_data,
                                                'dataset') else train_data

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
        has_paths = hasattr(base_dataset, 'data') and len(
            base_dataset.data) > 0 and isinstance(base_dataset.data[0], str)

        self.local_features = {}
        self.local_labels = {}

        if has_paths:
            # Dataset with image paths - can use caching
            num_samples = len(subset_indices) if is_subset else len(
                base_dataset)
            logger.info(
                f"Client {self.ID}: Dataset has {num_samples} samples with paths"
            )

            # Collect samples that need feature extraction
            paths_to_extract = []
            indices_to_extract = [
            ]  # Indices into the current dataset (Subset or base)
            base_indices_to_extract = [
            ]  # Indices into base_dataset for loading

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
            logger.info(
                f"Client {self.ID}: {cached_count} samples from cache, {len(paths_to_extract)} need extraction"
            )

            # Extract features for non-cached samples
            if paths_to_extract:
                # Load feature extractor (CLIP or CNN)
                self._load_feature_extractor()

                # Create a mini dataloader for samples to extract
                batch_size = 32
                with torch.no_grad():
                    for i in range(0, len(base_indices_to_extract),
                                   batch_size):
                        batch_base_indices = base_indices_to_extract[
                            i:i + batch_size]
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
                        else:
                            features = self.clip_model.encode_image(images)
                        features = features.cpu().numpy()

                        for feat, label, path in zip(features, labels,
                                                     batch_paths):
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
            logger.info(
                f"Client {self.ID}: Dataset does not have image paths, caching disabled"
            )
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
        expected_dim = self.ggeur_cfg.embedding_dim
        for label in self.local_features:
            self.local_features[label] = np.array(self.local_features[label])

            # Validate feature shape after conversion
            feature_shape = self.local_features[label].shape

            # Ensure 2D shape (n_samples, embedding_dim)
            if len(feature_shape) != 2:
                raise ValueError(
                    f"Features for class {label} should be 2D, got shape {feature_shape}"
                )

            if feature_shape[1] != expected_dim:
                raise ValueError(
                    f"Features dimension mismatch: expected {expected_dim}, got {feature_shape[1]}"
                )

        total_samples = sum(len(v) for v in self.local_features.values())
        logger.info(
            f"Client {self.ID}: Extracted {total_samples} {extractor_name} features from {len(self.local_features)} classes"
        )

    def _extract_clip_features(self):
        """Legacy method - now calls _extract_features()"""
        self._extract_features()

    def _compute_local_statistics(self):
        """Compute local mean and covariance for each class"""
        logger.info(f"Client {self.ID}: Computing local statistics...")

        self.local_means = {}
        self.local_covs = {}
        self.local_counts = {}

        # Get expected embedding dimension from config
        expected_dim = self.ggeur_cfg.embedding_dim

        for class_idx, features in self.local_features.items():
            if features.shape[0] == 0:
                continue

            n = features.shape[0]

            # Validate features shape
            if len(features.shape) != 2:
                raise ValueError(
                    f"Features for class {class_idx} should be 2D array (n_samples, embedding_dim), got shape {features.shape}"
                )

            if features.shape[1] != expected_dim:
                raise ValueError(
                    f"Features for class {class_idx} dimension mismatch: expected {expected_dim}, got {features.shape[1]}"
                )

            mean = np.mean(features, axis=0)

            # Validate mean shape
            assert mean.shape == (
                expected_dim,
            ), f"Mean shape mismatch: {mean.shape} vs ({expected_dim},)"

            # Compute covariance
            centered = features - mean
            cov = np.dot(centered.T, centered) / np.float32(n)

            # Validate cov shape
            assert cov.shape == (
                expected_dim, expected_dim
            ), f"Cov shape mismatch: {cov.shape} vs ({expected_dim}, {expected_dim})"

            self.local_means[class_idx] = mean
            self.local_covs[class_idx] = cov
            self.local_counts[class_idx] = n

        logger.info(
            f"Client {self.ID}: Computed statistics for {len(self.local_means)} classes"
        )

    def _upload_local_statistics(self):
        """Upload local statistics to server"""
        logger.info(
            f"Client {self.ID}: Uploading local statistics to server...")

        # Convert numpy arrays to lists for safe gRPC transmission
        # This ensures consistent serialization across all clients
        means_serialized = {}
        covs_serialized = {}
        prototypes_serialized = {}

        for class_idx in self.local_means.keys():
            # Ensure class_idx is int
            class_idx = int(class_idx)

            # Serialize mean (512,) -> list
            mean = self.local_means[class_idx]
            assert isinstance(
                mean, np.ndarray), f"mean should be ndarray, got {type(mean)}"
            assert mean.shape == (self.ggeur_cfg.embedding_dim,), \
                f"mean shape should be ({self.ggeur_cfg.embedding_dim},), got {mean.shape}"
            means_serialized[class_idx] = mean.tolist()

            # Serialize cov (512, 512) -> base64 encoded binary
            # This avoids creating ~262K Python float objects per matrix
            cov = self.local_covs[class_idx]
            assert isinstance(
                cov, np.ndarray), f"cov should be ndarray, got {type(cov)}"
            assert cov.shape == (self.ggeur_cfg.embedding_dim, self.ggeur_cfg.embedding_dim), \
                f"cov shape should be ({self.ggeur_cfg.embedding_dim}, {self.ggeur_cfg.embedding_dim}), got {cov.shape}"
            cov_f32 = cov.astype(
                np.float32) if cov.dtype != np.float32 else cov
            covs_serialized[class_idx] = {
                '_b64': base64.b64encode(cov_f32.tobytes()).decode('ascii'),
                '_shape': list(cov_f32.shape),
                '_dtype': 'float32',
            }

            # Serialize prototype (same as mean)
            prototypes_serialized[class_idx] = mean.tolist()

        content = {
            'client_id': self.ID,
            'means': means_serialized,
            'covs': covs_serialized,
            'counts': self.local_counts,  # counts are already ints
            'prototypes': prototypes_serialized
        }

        logger.info(
            f"Client {self.ID}: Serialized statistics for {len(means_serialized)} classes"
        )

        self.comm_manager.send(
            Message(msg_type='local_statistics',
                    sender=self.ID,
                    receiver=[self.server_id],
                    state=self.state,
                    content=content))

        self.statistics_uploaded = True
        logger.info(f"Client {self.ID}: Statistics uploaded")

    def callback_for_global_covariances(self, message: Message):
        """Handle receiving global covariance matrices from server"""
        logger.info(
            f"Client {self.ID}: Received global covariances from server")

        content = message.content
        raw_cov = content.get('cov_matrices', {})
        self.global_cov_matrices = {}
        for k, v in raw_cov.items():
            if isinstance(v, dict) and '_b64' in v:
                buf = base64.b64decode(v['_b64'])
                arr = np.frombuffer(buf, dtype=np.dtype(v['_dtype']))
                self.global_cov_matrices[k] = arr.reshape(v['_shape']).copy()
            elif isinstance(v, list):
                self.global_cov_matrices[k] = np.array(v)
            else:
                self.global_cov_matrices[k] = v
        raw_other = content.get('other_prototypes', {}).get(self.ID, {})
        self.other_prototypes = {
            k: [np.array(p) if isinstance(p, list) else p
                for p in v] if isinstance(v, list) else v
            for k, v in raw_other.items()
        }
        raw_global = content.get('global_prototypes',
                                 {})  # For feature alignment
        self.global_prototypes = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in raw_global.items()
        }

        # Perform augmentation
        self._perform_augmentation()

        # Build MLP classifier
        self._build_mlp_classifier()

        # Build CNN based on mode
        if self.use_feature_alignment:
            # Feature alignment mode: train CNN from scratch
            self._load_clip_model()
            self._build_cnn_feature_align()
            self._setup_original_image_loader()
            logger.info(
                f"Client {self.ID}: Feature alignment ready - CLIP: {self.clip_model is not None}, "
                f"CNN: {self.cnn_model is not None}, Prototypes: {len(self.global_prototypes)}"
            )
        elif self.use_cnn_distillation:
            # Knowledge distillation mode
            self._load_clip_model()
            self._build_cnn_model()
            self._setup_original_image_loader()
            logger.info(
                f"Client {self.ID}: CNN distillation ready - CLIP: {self.clip_model is not None}, "
                f"MLP: {self.mlp_classifier is not None}, CNN: {self.cnn_model is not None}"
            )

        # Notify server that augmentation is complete
        logger.info(
            f"Client {self.ID}: Notifying server that augmentation is ready")
        self.comm_manager.send(
            Message(msg_type='augmentation_ready',
                    sender=self.ID,
                    receiver=[self.server_id],
                    state=self.state,
                    content=None))

        # Release large data structures no longer needed after augmentation
        self._cleanup_after_augmentation()

    def _cleanup_after_augmentation(self):
        """Release large data structures no longer needed after augmentation.

        Unloads CLIP model after augmentation since feature extraction is done.
        """
        # Unload CLIP model if not needed for distillation/alignment
        if not self.use_cnn_distillation and not self.use_feature_alignment:
            if self.clip_model is not None:
                del self.clip_model
                self.clip_model = None
                self.clip_preprocess = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.debug(
                    f"Client {self.ID}: Unloaded CLIP model (~600 MB freed)")

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

    def _precompute_cholesky(self, cov_matrix):
        """Precompute Cholesky factor for efficient repeated sampling from the same covariance.

        Instead of calling _nearest_pos_def + cholesky per sample (N+3 times per class),
        this computes it once per class, reducing ~14 GB total allocations to ~200 MB.

        Returns float32 to avoid upcasting all generated samples to float64
        (which would double memory usage for all intermediate augmentation arrays).
        """
        dim = cov_matrix.shape[0]

        if dim > 512:
            try:
                jitter = 1e-5
                L = np.linalg.cholesky(cov_matrix + jitter * np.eye(dim))
                return L.astype(np.float32)
            except np.linalg.LinAlgError:
                var = np.diag(cov_matrix)
                var = np.maximum(var, 1e-6)
                return np.diag(np.sqrt(var)).astype(np.float32)

        cov_matrix = self._nearest_pos_def(cov_matrix)
        jitter = 1e-6

        while True:
            try:
                L = np.linalg.cholesky(cov_matrix + jitter * np.eye(dim))
                return L.astype(np.float32)
            except np.linalg.LinAlgError:
                jitter *= 10
                if jitter > 1:
                    return (np.eye(dim) * 0.1).astype(np.float32)

    def _generate_samples_fast(self, mean, cholesky_L, num_samples):
        """Generate samples using precomputed Cholesky factor (avoids redundant decompositions)"""
        dim = cholesky_L.shape[0]
        z = np.random.randn(num_samples, dim).astype(np.float32)
        return mean + z @ cholesky_L.T

    def _perform_augmentation(self):
        """Perform GGEUR_Clip feature augmentation"""
        target_size = self.ggeur_cfg.target_size_per_class
        num_per_sample = self.ggeur_cfg.num_generated_per_sample
        num_per_prototype = self.ggeur_cfg.num_generated_per_prototype
        use_cross_client = self.ggeur_cfg.use_cross_client_prototypes

        # Check if augmentation is disabled (baseline mode)
        no_augmentation = (num_per_sample == 0
                           and num_per_prototype == 0) or not use_cross_client

        if no_augmentation:
            logger.info(
                f"Client {self.ID}: No augmentation mode - using original features only"
            )
        else:
            logger.info(
                f"Client {self.ID}: Performing GGEUR_Clip augmentation...")

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
        feature_dim = self.ggeur_cfg.embedding_dim
        if self.local_features:
            first_key = next(iter(self.local_features.keys()))
            if len(self.local_features[first_key]) > 0:
                feature_dim = self.local_features[first_key].shape[1]

        logger.info(
            f"Client {self.ID}: Processing {total_classes} classes, feature_dim={feature_dim}, "
            f"num_per_sample={num_per_sample}, num_per_prototype={num_per_prototype}"
        )

        for idx, class_idx in enumerate(all_classes):
            class_idx = int(class_idx)
            class_features = []

            # 显示进度
            if (idx + 1) % 10 == 0 or idx == 0:
                logger.info(
                    f"Client {self.ID}: Augmenting class {idx+1}/{total_classes}"
                )

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

            # 2. Get global covariance matrix and precompute Cholesky once per class
            if class_idx in self.global_cov_matrices:
                cov_matrix = self.global_cov_matrices[class_idx]
            else:
                cov_matrix = np.eye(self.ggeur_cfg.embedding_dim) * 0.01

            cholesky_L = self._precompute_cholesky(cov_matrix)

            # 3. Expand original features using global covariance
            if num_per_sample > 0 and class_idx in self.local_features and self.local_features[
                    class_idx].shape[0] > 0:
                for feat in self.local_features[class_idx]:
                    generated = self._generate_samples_fast(
                        feat, cholesky_L, num_per_sample)
                    class_features.append(generated)

            # 4. Generate from other clients' prototypes
            if use_cross_client and num_per_prototype > 0 and self.other_prototypes:
                if class_idx in self.other_prototypes:
                    for prototype in self.other_prototypes[class_idx]:
                        generated = self._generate_samples_fast(
                            prototype, cholesky_L, num_per_prototype)
                        class_features.append(generated)

            del cholesky_L

            # Combine and sample to target size
            if class_features:
                combined = np.vstack(class_features)

                # target_size = 0 means use all samples
                if target_size > 0 and combined.shape[0] >= target_size:
                    indices = np.random.choice(combined.shape[0],
                                               target_size,
                                               replace=False)
                    selected = combined[indices]
                else:
                    selected = combined

                all_features.append(selected)
                all_labels.append(np.full(selected.shape[0], class_idx))

        logger.info(
            f"Client {self.ID}: Augmentation complete, building dataset...")

        if all_features:
            self.augmented_features = np.vstack(all_features)
            self.augmented_labels = np.concatenate(all_labels)

            # Create data loader
            dataset = AugmentedFeatureDataset(self.augmented_features,
                                              self.augmented_labels)
            self.augmented_loader = DataLoader(
                dataset,
                batch_size=self._cfg.dataloader.batch_size,
                shuffle=True)

            if no_augmentation:
                logger.info(
                    f"Client {self.ID}: Original data - {self.augmented_features.shape[0]} samples, "
                    f"{len(np.unique(self.augmented_labels))} classes")
            else:
                logger.info(
                    f"Client {self.ID}: Augmented data - {self.augmented_features.shape[0]} samples, "
                    f"{len(np.unique(self.augmented_labels))} classes")

        self.augmentation_done = True

    def _build_mlp_classifier(self):
        """Build MLP classifier for augmented features"""
        # IMPORTANT: Always use config's num_classes, not the unique labels in augmented data
        # In LDS mode, each client may only have a subset of classes, but the model
        # must support all classes for proper FedAvg aggregation
        num_classes = self._cfg.model.out_channels
        input_dim = self.ggeur_cfg.embedding_dim
        hidden_dim = self.ggeur_cfg.mlp_hidden_dim

        if hidden_dim > 0:
            self.mlp_classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(self.ggeur_cfg.mlp_dropout),
                nn.Linear(hidden_dim, num_classes))
        else:
            # Simple linear classifier
            self.mlp_classifier = nn.Linear(input_dim, num_classes)

        self.mlp_classifier = self.mlp_classifier.to(self.device)
        logger.info(
            f"Client {self.ID}: Built MLP classifier with {num_classes} classes"
        )

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
            logger.info(
                f"Client {self.ID}: Round {round_idx} - Statistics collection phase"
            )

            # Extract CLIP features
            self._extract_clip_features()

            # Compute local statistics
            self._compute_local_statistics()

            # Upload to server
            self._upload_local_statistics()

            return  # Don't do normal training in this round

        # Wait for augmentation to complete
        if not self.augmentation_done:
            logger.warning(
                f"Client {self.ID}: Augmentation not done, skipping training")
            # Send empty model update
            self.comm_manager.send(
                Message(msg_type='model_para',
                        sender=self.ID,
                        receiver=[sender],
                        state=self.state,
                        timestamp=timestamp,
                        content=(0, self.trainer.get_model_para())))
            return

        # Handle Separated Training Mode
        if self.use_separated_training:
            self._handle_separated_training(message)
            return

        # Normal training round on augmented data
        logger.info(
            f"Client {self.ID}: Round {round_idx} - Training on augmented data"
        )

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

        # Update MLP with global model parameters
        if mlp_para is not None and self.mlp_classifier is not None:
            try:
                self.mlp_classifier.load_state_dict(mlp_para)
            except Exception as e:
                logger.debug(
                    f"Client {self.ID}: Could not load MLP state dict: {e}")

        # Update CNN with global model parameters (for both distillation and feature alignment)
        if (self.use_cnn_distillation or self.use_feature_alignment
            ) and cnn_para is not None and self.cnn_model is not None:
            try:
                self.cnn_model.load_state_dict(cnn_para)
            except Exception as e:
                logger.debug(
                    f"Client {self.ID}: Could not load CNN state dict: {e}")

        # Train MLP on augmented features
        mlp_sample_size, mlp_model_para, mlp_results = self._train_on_augmented_data(
        )

        # Train CNN based on mode
        cnn_warmup_rounds = getattr(self.ggeur_cfg, 'cnn_warmup_rounds', 0)

        if self.use_feature_alignment:
            # Feature alignment mode: train CNN from scratch
            should_train_cnn = round_idx > cnn_warmup_rounds

            if should_train_cnn:
                cnn_sample_size, cnn_model_para, cnn_results = self._train_cnn_with_feature_alignment(
                )

                combined_para = {'mlp': mlp_model_para, 'cnn': cnn_model_para}
                sample_size = mlp_sample_size
            else:
                logger.info(
                    f"Client {self.ID}: CNN warmup - skipping training in round {round_idx}"
                )
                combined_para = {
                    'mlp': mlp_model_para,
                    'cnn': copy.deepcopy(self.cnn_model.state_dict())
                    if self.cnn_model else None
                }
                sample_size = mlp_sample_size

        elif self.use_cnn_distillation:
            # Knowledge distillation mode
            should_train_cnn = round_idx > cnn_warmup_rounds

            if should_train_cnn:
                cnn_sample_size, cnn_model_para, cnn_results = self._train_cnn_with_distillation(
                )

                combined_para = {'mlp': mlp_model_para, 'cnn': cnn_model_para}
                sample_size = mlp_sample_size
            else:
                logger.info(
                    f"Client {self.ID}: CNN warmup - skipping distillation in round {round_idx}"
                )
                combined_para = {
                    'mlp': mlp_model_para,
                    'cnn': copy.deepcopy(self.cnn_model.state_dict())
                    if self.cnn_model else None
                }
                sample_size = mlp_sample_size
        else:
            # Standard mode: only MLP
            combined_para = mlp_model_para
            sample_size = mlp_sample_size

        # Send model parameters
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=(sample_size, combined_para)))

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
            logger.info(
                f"Client {self.ID}: Phase 1 (Classifier Training) - Round {round_idx}"
            )

            # Update classifier with global parameters
            classifier_para = content.get('classifier') if isinstance(
                content, dict) else content
            if classifier_para is not None and self.mlp_classifier is not None:
                try:
                    self.mlp_classifier.load_state_dict(classifier_para)
                except Exception as e:
                    logger.debug(
                        f"Client {self.ID}: Could not load classifier: {e}")

            # Train classifier on augmented features
            sample_size, model_para, results = self._train_on_augmented_data()

            # Send classifier parameters
            self.comm_manager.send(
                Message(msg_type='model_para',
                        sender=self.ID,
                        receiver=[sender],
                        state=self.state,
                        timestamp=timestamp,
                        content=(sample_size, {
                            'classifier': model_para
                        })))

        else:
            # Phase 2: Train CNN backbone with frozen classifier
            self.training_phase = 'cnn_backbone'
            logger.info(
                f"Client {self.ID}: Phase 2 (CNN Backbone Training) - Round {round_idx}"
            )

            # Get pretrained classifier (frozen)
            classifier_para = content.get('classifier') if isinstance(
                content, dict) else None
            if classifier_para is not None:
                self._setup_pretrained_classifier(classifier_para)

            # Get CNN backbone parameters
            cnn_backbone_para = content.get('cnn_backbone') if isinstance(
                content, dict) else None

            # Build CNN backbone if not already built
            if self.cnn_backbone is None:
                self._build_cnn_backbone()

            # Load CNN backbone parameters
            if cnn_backbone_para is not None and self.cnn_backbone is not None:
                try:
                    self.cnn_backbone.load_state_dict(cnn_backbone_para)
                except Exception as e:
                    logger.debug(
                        f"Client {self.ID}: Could not load CNN backbone: {e}")

            # Setup image loader if not ready
            if self.original_image_loader is None:
                self._setup_original_image_loader()

            # Train CNN backbone with frozen classifier
            sample_size, backbone_para, results = self._train_cnn_backbone_separated(
            )

            # Send only CNN backbone parameters
            self.comm_manager.send(
                Message(msg_type='model_para',
                        sender=self.ID,
                        receiver=[sender],
                        state=self.state,
                        timestamp=timestamp,
                        content=(sample_size, {
                            'cnn_backbone': backbone_para
                        })))

    def _setup_pretrained_classifier(self, classifier_para):
        """Setup pretrained classifier for Phase 2"""
        if self.pretrained_classifier is not None:
            return  # Already setup

        # Build classifier with same architecture as MLP
        num_classes = self._cfg.model.out_channels
        input_dim = self.ggeur_cfg.embedding_dim
        hidden_dim = self.ggeur_cfg.mlp_hidden_dim

        if hidden_dim > 0:
            self.pretrained_classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(self.ggeur_cfg.mlp_dropout),
                nn.Linear(hidden_dim, num_classes))
        else:
            self.pretrained_classifier = nn.Linear(input_dim, num_classes)

        self.pretrained_classifier = self.pretrained_classifier.to(self.device)

        # Load pretrained weights
        try:
            self.pretrained_classifier.load_state_dict(classifier_para)
            logger.info(f"Client {self.ID}: Loaded pretrained classifier")
        except Exception as e:
            logger.warning(
                f"Client {self.ID}: Could not load pretrained classifier: {e}")

        # Freeze classifier
        for param in self.pretrained_classifier.parameters():
            param.requires_grad = False
        self.pretrained_classifier.eval()

        logger.info(
            f"Client {self.ID}: Pretrained classifier frozen for Phase 2")

    def _build_cnn_backbone(self):
        """Build CNN backbone for separated training Phase 2"""
        from federatedscope.contrib.model.ggeur_cnn import GGEUR_CNN_Backbone

        num_classes = self._cfg.model.out_channels
        cnn_model_name = getattr(self.ggeur_cfg, 'cnn_model', 'resnet18')

        self.cnn_backbone = GGEUR_CNN_Backbone(
            model_name=cnn_model_name,
            num_classes=num_classes,
            pretrained=False  # From scratch
        )
        self.cnn_backbone = self.cnn_backbone.to(self.device)

        logger.info(
            f"Client {self.ID}: Built CNN backbone ({cnn_model_name}) for Phase 2 - FROM SCRATCH"
        )

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
            logger.warning(
                f"Client {self.ID}: CNN backbone or classifier not ready")
            return 0, {}, {}

        if self.original_image_loader is None:
            logger.warning(f"Client {self.ID}: Image loader not ready")
            return 0, {}, {}

        # Load CLIP model for feature alignment (CRITICAL!)
        if self.clip_model is None:
            logger.info(
                f"Client {self.ID}: Loading CLIP model for feature alignment..."
            )
            self._load_clip_model()

        if self.clip_model is None:
            logger.warning(
                f"Client {self.ID}: CLIP model not available, training without alignment"
            )

        # Training settings
        cnn_lr = getattr(self.ggeur_cfg, 'cnn_lr', 0.01)
        cnn_epochs = getattr(self.ggeur_cfg, 'cnn_local_epochs', 10)
        align_weight = getattr(self.ggeur_cfg, 'align_weight',
                               1.0)  # Feature alignment weight
        cnn_weight_decay = getattr(self.ggeur_cfg, 'cnn_weight_decay',
                                   5e-4)  # 正则化
        cnn_dropout = getattr(self.ggeur_cfg, 'cnn_dropout', 0.5)  # Dropout比例

        # 诊断信息
        num_samples = len(self.original_image_loader.dataset) if hasattr(
            self.original_image_loader, 'dataset') else 0
        logger.info(
            f"Client {self.ID}: Phase 2 训练诊断 - CLIP加载:{self.clip_model is not None}, "
            f"align_weight={align_weight}, lr={cnn_lr}, epochs={cnn_epochs}, samples={num_samples}"
        )

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

        logger.info(
            f"Client {self.ID}: CNN training with CLIP feature alignment, align_weight={align_weight}"
        )

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
                        clip_features = self.clip_model.encode_image(
                            images).float()
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
                    magnitude_loss = mse_criterion(
                        cnn_magnitude,
                        clip_magnitude) / (clip_magnitude.mean()**2 + 1e-6)

                    # 组合: 方向 + 尺度
                    align_loss = cosine_loss + 0.1 * magnitude_loss

                    loss = ce_loss + align_weight * align_loss
                    total_align_loss += align_loss.item() * images.size(0)

                    # 第一个batch打印诊断信息
                    if epoch == 0 and epoch_samples == 0:
                        logger.info(
                            f"Client {self.ID}: 诊断 - CNN范数={cnn_magnitude.mean().item():.2f}, "
                            f"CLIP范数={clip_magnitude.mean().item():.2f}, "
                            f"cosine_loss={cosine_loss.item():.4f}, "
                            f"magnitude_loss={magnitude_loss.item():.4f}")
                else:
                    loss = ce_loss
                    align_loss = torch.tensor(0.0)

                # Backward: only updates backbone
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cnn_backbone.parameters(),
                                               max_norm=1.0)
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
                logger.info(
                    f"Client {self.ID}: CNN Epoch {epoch+1}/{cnn_epochs}, "
                    f"Loss={epoch_loss/epoch_samples:.4f}, Acc={epoch_acc:.4f}"
                )

        # Compute averages
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_ce_loss = total_ce_loss / total_samples
            avg_align_loss = total_align_loss / total_samples
            accuracy = total_correct / total_samples
        else:
            avg_loss = avg_ce_loss = avg_align_loss = accuracy = 0

        logger.info(
            f"Client {self.ID}: CNN Training - loss={avg_loss:.4f}, "
            f"ce={avg_ce_loss:.4f}, align={avg_align_loss:.4f}, acc={accuracy:.4f}"
        )

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
        """Train MLP classifier on augmented features"""
        if self.augmented_loader is None or self.mlp_classifier is None:
            return 0, {}, {}

        self.mlp_classifier.train()
        optimizer = torch.optim.Adam(self.mlp_classifier.parameters(),
                                     lr=self._cfg.train.optimizer.lr)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        local_epochs = self._cfg.train.local_update_steps

        for epoch in range(local_epochs):
            for features, labels in self.augmented_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.mlp_classifier(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += features.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        logger.info(
            f"Client {self.ID}: Train loss={avg_loss:.4f}, accuracy={accuracy:.4f}"
        )

        # Get model parameters
        model_para = copy.deepcopy(self.mlp_classifier.state_dict())

        results = {
            'train_loss': avg_loss,
            'train_acc': accuracy,
            'train_total': total_samples
        }

        return total_samples, model_para, results

    # ==================== CNN Knowledge Distillation Methods ====================

    def _build_cnn_model(self):
        """Build CNN model for knowledge distillation"""
        from federatedscope.contrib.model.ggeur_cnn import GGEUR_CNN_FeatureAlign

        num_classes = self._cfg.model.out_channels
        cnn_model_name = getattr(self.ggeur_cfg, 'cnn_model', 'resnet18')
        cnn_pretrained = getattr(self.ggeur_cfg, 'cnn_pretrained', True)
        clip_dim = getattr(self.ggeur_cfg, 'embedding_dim', 512)

        # Use the new FeatureAlign model for consistency
        self.cnn_model = GGEUR_CNN_FeatureAlign(model_name=cnn_model_name,
                                                num_classes=num_classes,
                                                clip_dim=clip_dim,
                                                pretrained=cnn_pretrained)
        self.cnn_model = self.cnn_model.to(self.device)

        logger.info(
            f"Client {self.ID}: Built CNN model ({cnn_model_name}) with {num_classes} classes, pretrained={cnn_pretrained}"
        )

    def _build_cnn_feature_align(self):
        """Build CNN model for feature alignment (from scratch training)"""
        from federatedscope.contrib.model.ggeur_cnn import GGEUR_CNN_FeatureAlign

        num_classes = self._cfg.model.out_channels
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

        logger.info(
            f"Client {self.ID}: Built CNN ({cnn_model_name}) for feature alignment - FROM SCRATCH, {num_classes} classes"
        )

    def _setup_original_image_loader(self):
        """Setup DataLoader for original images (for CNN training)"""
        # Get train data
        train_data = self.trainer.ctx.data.get('train', None)
        if train_data is None:
            train_data = self.data.get('train', None)

        if train_data is None:
            logger.error(
                f"Client {self.ID}: No training data available for CNN")
            return

        # Get the underlying dataset
        dataset = train_data.dataset if hasattr(train_data,
                                                'dataset') else train_data

        # 为CNN训练创建带数据增强的包装数据集
        use_augmentation = getattr(self.ggeur_cfg, 'cnn_use_augmentation',
                                   True)
        if use_augmentation:
            dataset = AugmentedImageDataset(dataset)
            logger.info(
                f"Client {self.ID}: CNN training with data augmentation enabled"
            )

        # Create DataLoader for original images
        batch_size = self._cfg.dataloader.batch_size
        self.original_image_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 to avoid multiprocessing issues
            drop_last=False)

        logger.info(
            f"Client {self.ID}: Setup original image loader with {len(dataset)} samples"
        )

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
            logger.info(
                f"Client {self.ID}: Loading CLIP model for CNN distillation..."
            )
            self._load_clip_model()

        if self.mlp_classifier is None or self.clip_model is None:
            logger.warning(
                f"Client {self.ID}: MLP teacher or CLIP not ready - "
                f"MLP: {self.mlp_classifier is not None}, CLIP: {self.clip_model is not None}"
            )
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

        optimizer = torch.optim.SGD(
            [
                {
                    'params': backbone_params,
                    'lr': cnn_lr * 0.1
                },  # Lower LR for backbone
                {
                    'params': classifier_params,
                    'lr': cnn_lr
                }  # Higher LR for classifier
            ],
            momentum=0.9,
            weight_decay=1e-4)

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
        logger.info(
            f"Client {self.ID}: CNN training with {num_batches} batches, "
            f"backbone_lr={cnn_lr*0.1}, classifier_lr={cnn_lr}, epochs={cnn_epochs}"
        )

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
                    epoch_teacher_correct += (
                        teacher_pred == labels).sum().item()

                # Get student's (CNN) logits
                optimizer.zero_grad()
                student_logits = self.cnn_model(images)

                # Hard label loss (Cross-Entropy)
                ce_loss = ce_criterion(student_logits, labels)

                # Soft label loss (KL-Divergence with temperature)
                soft_student = F.log_softmax(student_logits / temperature,
                                             dim=1)
                soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
                kl_loss = kl_criterion(soft_student,
                                       soft_teacher) * (temperature**2)

                # Combined loss
                loss = alpha * ce_loss + (1 - alpha) * kl_loss

                # Backward and optimize
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.cnn_model.parameters(),
                                               max_norm=1.0)
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
                logger.info(
                    f"Client {self.ID}: CNN Epoch {epoch+1}/{cnn_epochs}, "
                    f"Loss={epoch_loss/epoch_samples:.4f}, "
                    f"CNN_Acc={epoch_acc:.4f}, Teacher_Acc={epoch_teacher_acc:.4f}"
                )

        # Compute averages
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_ce_loss = total_ce_loss / total_samples
            avg_kl_loss = total_kl_loss / total_samples
            accuracy = total_correct / total_samples
            teacher_accuracy = teacher_correct / total_samples
        else:
            avg_loss = avg_ce_loss = avg_kl_loss = accuracy = teacher_accuracy = 0

        logger.info(
            f"Client {self.ID}: CNN Train - loss={avg_loss:.4f}, "
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
            logger.info(
                f"Client {self.ID}: Loading CLIP model for feature alignment..."
            )
            self._load_clip_model()

        if self.clip_model is None:
            logger.warning(f"Client {self.ID}: CLIP model not available")
            return 0, {}, {}

        # Training settings
        cnn_lr = getattr(self.ggeur_cfg, 'cnn_lr', 0.01)
        cnn_epochs = getattr(self.ggeur_cfg, 'cnn_local_epochs', 10)
        align_weight = getattr(self.ggeur_cfg, 'align_weight', 1.0)
        use_prototype_align = getattr(self.ggeur_cfg,
                                      'use_prototype_alignment', True)
        prototype_weight = getattr(self.ggeur_cfg, 'prototype_align_weight',
                                   0.5)

        self.cnn_model.train()
        self.clip_model.eval()

        # Optimizer
        optimizer = torch.optim.SGD(self.cnn_model.parameters(),
                                    lr=cnn_lr,
                                    momentum=0.9,
                                    weight_decay=1e-4)

        # Loss functions
        ce_criterion = nn.CrossEntropyLoss()
        mse_criterion = nn.MSELoss()

        # Convert global prototypes to tensor
        prototype_tensor = None
        if use_prototype_align and self.global_prototypes:
            num_classes = self._cfg.model.out_channels
            clip_dim = getattr(self.ggeur_cfg, 'embedding_dim', 512)
            prototype_tensor = torch.zeros(num_classes,
                                           clip_dim).to(self.device)
            for class_idx, proto in self.global_prototypes.items():
                class_idx = int(class_idx)
                if class_idx < num_classes:
                    if isinstance(proto, np.ndarray):
                        prototype_tensor[class_idx] = torch.from_numpy(
                            proto).float()
                    else:
                        prototype_tensor[class_idx] = proto.float()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_align_loss = 0.0
        total_proto_loss = 0.0
        total_correct = 0
        total_samples = 0

        num_batches = len(self.original_image_loader)
        logger.info(
            f"Client {self.ID}: Feature alignment training with {num_batches} batches, "
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
                    clip_features = self.clip_model.encode_image(
                        images).float()
                    clip_features_norm = F.normalize(clip_features, p=2, dim=1)

                # Forward pass through CNN
                optimizer.zero_grad()
                logits, proj_features = self.cnn_model(images,
                                                       return_features=True)
                proj_features_norm = F.normalize(proj_features, p=2, dim=1)

                # Classification loss
                ce_loss = ce_criterion(logits, labels)

                # Feature alignment loss (align CNN features with CLIP features)
                align_loss = mse_criterion(proj_features_norm,
                                           clip_features_norm)

                # Prototype alignment loss (optional)
                proto_loss = torch.tensor(0.0).to(self.device)
                if use_prototype_align and prototype_tensor is not None:
                    # Get prototypes for each sample's class
                    target_prototypes = prototype_tensor[labels]
                    target_prototypes_norm = F.normalize(target_prototypes,
                                                         p=2,
                                                         dim=1)
                    proto_loss = mse_criterion(proj_features_norm,
                                               target_prototypes_norm)

                # Combined loss
                loss = ce_loss + align_weight * align_loss
                if use_prototype_align:
                    loss = loss + prototype_weight * proto_loss

                # Backward and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cnn_model.parameters(),
                                               max_norm=1.0)
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
                logger.info(
                    f"Client {self.ID}: CNN Epoch {epoch+1}/{cnn_epochs}, "
                    f"Loss={epoch_loss/epoch_samples:.4f}, Acc={epoch_acc:.4f}"
                )

        # Compute averages
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_ce_loss = total_ce_loss / total_samples
            avg_align_loss = total_align_loss / total_samples
            avg_proto_loss = total_proto_loss / total_samples
            accuracy = total_correct / total_samples
        else:
            avg_loss = avg_ce_loss = avg_align_loss = avg_proto_loss = accuracy = 0

        logger.info(
            f"Client {self.ID}: CNN Feature Align - loss={avg_loss:.4f}, "
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


def call_ggeur_worker(method):
    """Factory function for GGEUR_Clip worker"""
    if method.lower() == 'ggeur':
        from federatedscope.contrib.worker.ggeur_server import GGEURServer
        return {'client': GGEURClient, 'server': GGEURServer}
    return None


register_worker('ggeur', call_ggeur_worker)
