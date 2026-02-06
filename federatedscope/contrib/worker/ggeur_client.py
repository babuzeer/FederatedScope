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

# Shared (process-wide) BERT cache to avoid loading a huge model per client in
# standalone simulation.
_SHARED_BERT_EXTRACTORS = {}  # key -> (tokenizer, model)


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
        # 'bert': Use pretrained BERT for text feature extraction
        self.feature_extractor_type = getattr(self.ggeur_cfg, 'feature_extractor', 'clip')

        # CLIP model (for 'clip' mode)
        self.clip_model = None
        self.clip_preprocess = None

        # CNN feature extractor (for 'cnn' mode)
        self.cnn_extractor = None

        # BERT feature extractor (for 'bert' mode)
        self.bert_model = None
        self.bert_tokenizer = None

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

        # ===== FedProto (Prototype Regularization on Representations) =====
        # This is an optional comparison method built on top of the GGEUR pipeline
        # (keeps the same evaluation logic; adds an extra prototype loss during
        # local training and prototype exchange across clients/rounds).
        self.use_fedproto = bool(getattr(self.ggeur_cfg, 'use_fedproto', False))
        self.fedproto_proto_weight = float(getattr(self.ggeur_cfg, 'fedproto_proto_weight', 1.0))
        self.fedproto_distance_metric = str(getattr(self.ggeur_cfg, 'fedproto_distance_metric', 'mse')).lower()
        self.fedproto_normalize = bool(getattr(self.ggeur_cfg, 'fedproto_normalize', False))
        self.fedproto_global_prototypes = {}  # {class_idx: torch.Tensor}
        self.fedproto_local_prototypes = {}  # {class_idx: torch.Tensor}
        self.fedproto_local_counts = {}  # {class_idx: int}

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

    def _register_default_handlers(self):
        """Register message handlers"""
        super()._register_default_handlers()

        # Register handler for receiving global covariance matrices
        self.register_handlers('global_covariances',
                               self.callback_for_global_covariances)

    def _load_feature_extractor(self):
        """Load feature extractor (CLIP or CNN based on config)"""
        if self.feature_extractor_type == 'bert':
            self._load_bert_model()
        elif self.feature_extractor_type == 'cnn':
            self._load_cnn_extractor()
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

            logger.info(f"Client {self.ID}: Loaded CNN extractor {model_name}, "
                       f"feature_dim={self.cnn_extractor.get_feature_dim()}")

        except Exception as e:
            logger.error(f"Client {self.ID}: Failed to load CNN extractor: {e}")
            raise

    def _load_bert_model(self):
        """Load pretrained BERT model/tokenizer for text feature extraction"""
        if self.bert_model is not None and self.bert_tokenizer is not None:
            return

        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as e:
            logger.error("transformers not installed. Please install: pip install transformers")
            raise e

        model_path = getattr(self.ggeur_cfg, 'bert_model_path', '') or ''
        tokenizer_path = getattr(self.ggeur_cfg, 'bert_tokenizer_path', '') or model_path
        local_only = getattr(self.ggeur_cfg, 'bert_local_files_only', True)
        use_pretrained = getattr(self.ggeur_cfg, 'bert_use_pretrained_weights', True)

        if not model_path:
            raise ValueError("When ggeur.feature_extractor='bert', please set ggeur.bert_model_path to a local model dir")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BERT model path not found: {model_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"BERT tokenizer path not found: {tokenizer_path}")

        # Reuse a shared extractor to avoid loading a large model per client
        cache_key = (
            os.path.abspath(model_path),
            os.path.abspath(tokenizer_path),
            str(self.device),
            bool(local_only),
            bool(use_pretrained),
        )
        if cache_key in _SHARED_BERT_EXTRACTORS:
            self.bert_tokenizer, self.bert_model = _SHARED_BERT_EXTRACTORS[cache_key]
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=local_only)
            if use_pretrained:
                model = AutoModel.from_pretrained(model_path, local_files_only=local_only)
            else:
                cfg = AutoConfig.from_pretrained(model_path, local_files_only=local_only)
                model = AutoModel.from_config(cfg)
            model = model.to(self.device)
            model.eval()
            _SHARED_BERT_EXTRACTORS[cache_key] = (tokenizer, model)
            self.bert_tokenizer, self.bert_model = tokenizer, model

        hidden_size = getattr(getattr(self.bert_model, 'config', None), 'hidden_size', None)
        mode_str = "pretrained" if use_pretrained else "random_init"
        logger.info(f"Client {self.ID}: Loaded BERT extractor ({mode_str}) from {model_path} (hidden_size={hidden_size})")

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
            if self.feature_extractor_type == 'bert':
                cache_dir = os.path.join(os.path.dirname(self._cfg.data.root), 'text_feature_cache')
            else:
                cache_dir = os.path.join(os.path.dirname(self._cfg.data.root), 'clip_feature_cache')

        os.makedirs(cache_dir, exist_ok=True)

        # Build cache filename based on dataset, domain, and model
        dataset_name = self._cfg.data.type.lower()

        if self.feature_extractor_type == 'bert':
            model_path = getattr(self.ggeur_cfg, 'bert_model_path', 'bert')
            model_name = os.path.basename(str(model_path).rstrip('/\\')) or 'bert'
            max_len = getattr(self.ggeur_cfg, 'bert_max_length', 128)
            pooling = getattr(self.ggeur_cfg, 'bert_pooling', 'cls')
            use_pretrained = getattr(self.ggeur_cfg, 'bert_use_pretrained_weights', True)
            mode_str = "pre" if use_pretrained else "rand"
            if use_pretrained:
                model_str = f"{model_name}_maxlen{max_len}_{pooling}_{mode_str}"
            else:
                seed = getattr(self._cfg, 'seed', 0)
                model_str = f"{model_name}_maxlen{max_len}_{pooling}_{mode_str}_seed{seed}"
            prefix = 'bert'
        elif self.feature_extractor_type == 'cnn':
            model_name = getattr(self.ggeur_cfg, 'cnn_backbone', 'convnext_base')
            model_str = model_name.replace('/', '_').replace('-', '_')
            prefix = 'cnn'
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

    def _extract_bert_features(self, base_dataset, subset_indices, cache_path):
        """
        Extract sentence-level BERT embeddings from a text dataset.

        Args:
            base_dataset: underlying Dataset (may be CPSDTextDataset)
            subset_indices: optional list of indices (for Subset)
            cache_path: npz cache path for {id -> embedding}
        """
        self._load_bert_model()

        # Load existing cache (id -> embedding)
        feature_cache = self._load_feature_cache(cache_path)
        cache_updated = False

        # Resolve indices
        if subset_indices is None:
            indices = list(range(len(base_dataset)))
        else:
            indices = list(subset_indices)

        # Collect missing ids for batch extraction
        base_indices_to_extract = []
        ids_to_extract = []

        self.local_features = {}
        self.local_labels = {}

        for base_idx in indices:
            # Stable sample id for cache lookup
            if hasattr(base_dataset, 'get_id'):
                sample_id = str(base_dataset.get_id(base_idx))
            else:
                sample_id = f"{getattr(base_dataset, 'domain', 'text')}:{base_idx}"

            # Label (prefer `.targets`)
            if hasattr(base_dataset, 'targets'):
                label = int(base_dataset.targets[base_idx])
                text = None
            else:
                text, label = base_dataset[base_idx]
                label = int(label)

            if sample_id in feature_cache:
                feat = feature_cache[sample_id]
                if label not in self.local_features:
                    self.local_features[label] = []
                    self.local_labels[label] = []
                self.local_features[label].append(feat)
                self.local_labels[label].append(label)
            else:
                base_indices_to_extract.append(base_idx)
                ids_to_extract.append(sample_id)

        # Batch extract missing samples
        max_len = int(getattr(self.ggeur_cfg, 'bert_max_length', 128))
        pooling = str(getattr(self.ggeur_cfg, 'bert_pooling', 'cls')).lower()
        batch_size = int(getattr(self.ggeur_cfg, 'bert_batch_size', 32))
        if batch_size <= 0:
            batch_size = 32

        import numpy as np
        import torch

        with torch.no_grad():
            for i in range(0, len(base_indices_to_extract), batch_size):
                batch_base_idx = base_indices_to_extract[i:i + batch_size]
                batch_ids = ids_to_extract[i:i + batch_size]

                texts = []
                labels = []
                for base_idx in batch_base_idx:
                    text, label = base_dataset[base_idx]
                    texts.append(str(text))
                    labels.append(int(label))

                encoded = self.bert_tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors='pt'
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                outputs = self.bert_model(**encoded)
                hidden = outputs.last_hidden_state  # (B, T, H)

                if pooling == 'mean':
                    attn = encoded.get('attention_mask', None)
                    if attn is None:
                        emb = hidden.mean(dim=1)
                    else:
                        mask = attn.unsqueeze(-1).float()
                        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                else:
                    # default: [CLS]
                    emb = hidden[:, 0, :]

                emb_np = emb.detach().cpu().numpy().astype(np.float32)

                for vec, label, sample_id in zip(emb_np, labels, batch_ids):
                    feature_cache[sample_id] = vec
                    cache_updated = True
                    if label not in self.local_features:
                        self.local_features[label] = []
                        self.local_labels[label] = []
                    self.local_features[label].append(vec)
                    self.local_labels[label].append(label)

        # Save updated cache
        if cache_updated:
            self._save_feature_cache(cache_path, feature_cache)

        # Convert lists to numpy arrays
        for label in list(self.local_features.keys()):
            self.local_features[label] = np.array(self.local_features[label])

    def _extract_features(self):
        """
        Extract features from local data using either CLIP or CNN.
        Supports caching for both modes.
        """
        if self.feature_extractor_type == 'bert':
            extractor_name = 'BERT'
        elif self.feature_extractor_type == 'cnn':
            extractor_name = 'CNN'
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

        # Text (BERT) feature extraction path
        if self.feature_extractor_type == 'bert':
            self._extract_bert_features(base_dataset, subset_indices, cache_path)
            total_samples = sum(len(v) for v in self.local_features.values())
            logger.info(f"Client {self.ID}: Extracted {total_samples} BERT features from {len(self.local_features)} classes")
            return

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
        feature_dim = self.ggeur_cfg.embedding_dim
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
                cov_matrix = np.eye(self.ggeur_cfg.embedding_dim) * 0.01

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
        """Build classifier for augmented features (MLP for vision, RNN/LSTM for text)"""
        # IMPORTANT: Always use config's num_classes, not the unique labels in augmented data
        # In LDS mode, each client may only have a subset of classes, but the model
        # must support all classes for proper FedAvg aggregation
        num_classes = self._cfg.model.num_classes
        model_type = str(getattr(self._cfg.model, 'type', 'ggeur_mlp')).lower()

        # Text mode: train RNN/LSTM on frozen-feature embeddings
        if model_type in {'ggeur_rnn', 'ggeur_lstm'}:
            from federatedscope.contrib.model.ggeur_text_rnn import GGEURTextRNNClassifier

            input_dim = int(getattr(self.ggeur_cfg, 'embedding_dim', 768))
            hidden_dim = int(getattr(self._cfg.model, 'hidden', 256))
            num_layers = int(getattr(self._cfg.model, 'layer', 1))
            dropout = float(getattr(self._cfg.model, 'dropout', 0.0))
            rnn_type = 'rnn' if model_type == 'ggeur_rnn' else 'lstm'

            self.mlp_classifier = GGEURTextRNNClassifier(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_layers=num_layers,
                dropout=dropout,
                rnn_type=rnn_type,
            )
            self.mlp_classifier = self.mlp_classifier.to(self.device)
            logger.info(f"Client {self.ID}: Built {model_type} classifier with {num_classes} classes")
            return

        # Vision mode: train MLP on embeddings
        input_dim = self.ggeur_cfg.embedding_dim
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
                # FedProto: receive global prototypes from server (if provided)
                if self.use_fedproto and 'fedproto_global_prototypes' in content:
                    self._set_fedproto_global_prototypes(content.get('fedproto_global_prototypes'))

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
                logger.debug(f"Client {self.ID}: Could not load MLP state dict: {e}")

        # Update CNN with global model parameters (for both distillation and feature alignment)
        if (self.use_cnn_distillation or self.use_feature_alignment) and cnn_para is not None and self.cnn_model is not None:
            try:
                self.cnn_model.load_state_dict(cnn_para)
            except Exception as e:
                logger.debug(f"Client {self.ID}: Could not load CNN state dict: {e}")

        # Train MLP on augmented features
        mlp_sample_size, mlp_model_para, mlp_results = self._train_on_augmented_data()

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

        # FedProto: attach local prototypes for server aggregation
        if self.use_fedproto:
            if isinstance(combined_para, dict):
                if 'mlp' not in combined_para:
                    combined_para = {'mlp': combined_para}
            else:
                combined_para = {'mlp': combined_para}

            combined_para['fedproto_local_prototypes'] = copy.deepcopy(self.fedproto_local_prototypes)
            combined_para['fedproto_local_counts'] = copy.deepcopy(self.fedproto_local_counts)

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
                    self.mlp_classifier.load_state_dict(classifier_para)
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
                    self.cnn_backbone.load_state_dict(cnn_backbone_para)
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
        input_dim = self.ggeur_cfg.embedding_dim
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
            self.pretrained_classifier.load_state_dict(classifier_para)
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

    def _set_fedproto_global_prototypes(self, prototypes):
        """Set FedProto global prototypes received from server (move to device)."""
        if not self.use_fedproto:
            return

        if not prototypes:
            self.fedproto_global_prototypes = {}
            return

        proto_dict = {}
        if isinstance(prototypes, dict):
            for class_idx, proto in prototypes.items():
                try:
                    class_idx = int(class_idx)
                except Exception:
                    continue

                if proto is None:
                    continue

                if isinstance(proto, torch.Tensor):
                    tensor = proto.detach().to(self.device).float()
                else:
                    try:
                        tensor = torch.tensor(proto, device=self.device, dtype=torch.float32)
                    except Exception:
                        continue

                proto_dict[class_idx] = tensor

        self.fedproto_global_prototypes = proto_dict
        logger.info(
            f"Client {self.ID}: FedProto global prototypes updated ({len(self.fedproto_global_prototypes)} classes)"
        )

    def _compute_fedproto_local_prototypes(self):
        """Compute local prototypes on the current augmented dataset (CPU tensors)."""
        if not self.use_fedproto or self.augmented_loader is None or self.mlp_classifier is None:
            return {}, {}

        was_training = self.mlp_classifier.training
        self.mlp_classifier.eval()

        proto_sums = {}
        proto_counts = {}

        with torch.no_grad():
            for features, labels in self.augmented_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                try:
                    _, embeddings = self.mlp_classifier(features, return_features=True)
                except TypeError:
                    # Model does not support returning features (disable proto upload)
                    if was_training:
                        self.mlp_classifier.train()
                    return {}, {}

                embeddings = embeddings.detach()
                for cls in labels.unique():
                    cls_int = int(cls.item())
                    mask = labels == cls
                    cnt = int(mask.sum().item())
                    if cnt <= 0:
                        continue

                    emb_sum = embeddings[mask].sum(dim=0).cpu()
                    if cls_int not in proto_sums:
                        proto_sums[cls_int] = emb_sum
                        proto_counts[cls_int] = cnt
                    else:
                        proto_sums[cls_int] += emb_sum
                        proto_counts[cls_int] += cnt

        local_prototypes = {}
        for cls_int, sum_vec in proto_sums.items():
            cnt = int(proto_counts.get(cls_int, 0))
            if cnt > 0:
                local_prototypes[cls_int] = (sum_vec / float(cnt)).float()

        if was_training:
            self.mlp_classifier.train()

        return local_prototypes, proto_counts

    def _train_on_augmented_data(self):
        """Train MLP classifier on augmented features"""
        if self.augmented_loader is None or self.mlp_classifier is None:
            return 0, {}, {}

        self.mlp_classifier.train()
        optimizer = torch.optim.Adam(self.mlp_classifier.parameters(),
                                     lr=self._cfg.train.optimizer.lr)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_proto_loss = 0.0
        total_prox_loss = 0.0
        total_correct = 0
        total_samples = 0

        local_epochs = self._cfg.train.local_update_steps

        # FedProx (proximal regularization) settings
        fedprox_cfg = getattr(self._cfg, 'fedprox', None)
        use_fedprox = bool(getattr(fedprox_cfg, 'use', False)) if fedprox_cfg is not None else False
        fedprox_mu = float(getattr(fedprox_cfg, 'mu', 0.0)) if fedprox_cfg is not None else 0.0
        if fedprox_mu <= 0:
            use_fedprox = False

        global_param_snapshot = None
        if use_fedprox:
            # Snapshot the received global parameters (before local updates)
            global_param_snapshot = [
                p.detach().clone() for p in self.mlp_classifier.parameters() if p.requires_grad
            ]

        # Reset local prototypes each round (avoid stale values)
        self.fedproto_local_prototypes = {}
        self.fedproto_local_counts = {}

        for epoch in range(local_epochs):
            for features, labels in self.augmented_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                embeddings = None
                if self.use_fedproto:
                    try:
                        outputs, embeddings = self.mlp_classifier(features, return_features=True)
                    except TypeError:
                        outputs = self.mlp_classifier(features)
                        embeddings = None
                else:
                    outputs = self.mlp_classifier(features)

                ce_loss = criterion(outputs, labels)
                loss = ce_loss

                proto_loss = None
                if self.use_fedproto and embeddings is not None and self.fedproto_global_prototypes:
                    # Build prototype targets for the batch labels
                    proto_targets = torch.zeros_like(embeddings)
                    valid_mask = torch.zeros(labels.shape[0], device=self.device, dtype=torch.bool)

                    for class_idx, proto in self.fedproto_global_prototypes.items():
                        mask = labels == int(class_idx)
                        if mask.any():
                            proto_targets[mask] = proto
                            valid_mask |= mask

                    if valid_mask.any():
                        emb_sel = embeddings[valid_mask]
                        proto_sel = proto_targets[valid_mask]

                        if self.fedproto_normalize:
                            emb_sel = F.normalize(emb_sel, p=2, dim=1)
                            proto_sel = F.normalize(proto_sel, p=2, dim=1)

                        metric = str(self.fedproto_distance_metric).lower()
                        if metric in {'cos', 'cosine'}:
                            proto_loss = 1.0 - F.cosine_similarity(emb_sel, proto_sel, dim=1).mean()
                        else:
                            # Default: squared L2 distance (mean over samples)
                            proto_loss = (emb_sel - proto_sel).pow(2).sum(dim=1).mean()

                        loss = ce_loss + self.fedproto_proto_weight * proto_loss

                prox_reg = None
                if use_fedprox and global_param_snapshot is not None:
                    prox_term = torch.zeros((), device=self.device)
                    idx = 0
                    for p in self.mlp_classifier.parameters():
                        if not p.requires_grad:
                            continue
                        prox_term = prox_term + (p - global_param_snapshot[idx]).pow(2).sum()
                        idx += 1
                    prox_reg = 0.5 * fedprox_mu * prox_term
                    loss = loss + prox_reg

                loss.backward()
                optimizer.step()

                batch_size = int(features.size(0))
                total_loss += loss.item() * batch_size
                total_ce_loss += ce_loss.item() * batch_size
                if proto_loss is not None:
                    total_proto_loss += proto_loss.item() * batch_size
                if prox_reg is not None:
                    total_prox_loss += prox_reg.item() * batch_size
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_ce_loss = total_ce_loss / total_samples if total_samples > 0 else 0
        avg_proto_loss = total_proto_loss / total_samples if total_samples > 0 else 0
        avg_prox_loss = total_prox_loss / total_samples if total_samples > 0 else 0
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        if self.use_fedproto or use_fedprox:
            logger.info(
                f"Client {self.ID}: Train loss={avg_loss:.4f} (ce={avg_ce_loss:.4f}, "
                f"proto={avg_proto_loss:.4f}, prox={avg_prox_loss:.4f}), "
                f"accuracy={accuracy:.4f}"
            )
        else:
            logger.info(f"Client {self.ID}: Train loss={avg_loss:.4f}, accuracy={accuracy:.4f}")

        # Compute and cache local prototypes for upload (FedProto)
        if self.use_fedproto:
            local_prototypes, local_counts = self._compute_fedproto_local_prototypes()
            self.fedproto_local_prototypes = local_prototypes
            self.fedproto_local_counts = local_counts

        # Get model parameters
        model_para = copy.deepcopy(self.mlp_classifier.state_dict())

        results = {
            'train_loss': avg_loss,
            'train_ce_loss': avg_ce_loss,
            'train_acc': accuracy,
            'train_total': total_samples
        }
        if self.use_fedproto:
            results['train_proto_loss'] = avg_proto_loss
        if use_fedprox:
            results['train_prox_loss'] = avg_prox_loss

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
