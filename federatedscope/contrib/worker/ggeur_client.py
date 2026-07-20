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
import time
import logging
import copy
import base64
import io
import zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from federatedscope.attack.auxiliary.a3fl_utils import \
    parse_attacker_ids, should_a3fl_attack
from federatedscope.core.message import Message, b64serializer
from federatedscope.core.auxiliaries.utils import param2tensor
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

    # Shared CLIP model for PromptFL text encoding - loaded once, reused by all clients
    _shared_prompt_clip = None        # open_clip model (CPU when idle, GPU during forward)
    _shared_prompt_tokenizer = None
    _shared_prompt_clip_name = None   # track which model is loaded

    def __init__(self, ID=-1, server_id=None, state=-1, config=None,
                 data=None, model=None, device='cpu', strategy=None,
                 is_unseen_client=False, *args, **kwargs):
        super(GGEURClient, self).__init__(ID, server_id, state, config,
                                          data, model, device, strategy,
                                          is_unseen_client, *args, **kwargs)

        if config is None:
            return

        self.ggeur_cfg = config.ggeur
        self.head_only_mode = getattr(self.ggeur_cfg, 'head_only_mode', False)
        if self.head_only_mode:
            logger.info(
                f"Client {self.ID}: GGEUR HeadOnly system mode active. "
                "Round 0 uses the real backbone for feature statistics and "
                "augmentation; later rounds train the MLP head on augmented "
                "feature cache.")

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
        self._cov_factor_cache = {}

        # Global prototypes for feature alignment
        self.global_prototypes = None  # {class_idx: mean vector}

        # Augmented data
        self.augmented_features = None
        self.augmented_labels = None
        self.augmented_loader = None
        self.base_augmented_features = None
        self.base_augmented_labels = None

        # Real local features saved before augmentation (for PromptFL training)
        self.real_local_features = {}
        self.prompt_loader = None

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
        self.global_prompt_ctx = None  # Latest global PromptFL context

        # ===== A3FL Mode =====
        attack_method = str(getattr(config.attack, 'attack_method', '')).lower()
        self.a3fl_enabled = attack_method == 'a3fl'
        self.a3fl_cfg = getattr(config.attack, 'a3fl', None)
        self.a3fl_attacker_ids = set(parse_attacker_ids(config.attack.attacker_id))
        self.a3fl_is_attacker = self.a3fl_enabled and self.ID in self.a3fl_attacker_ids
        self.a3fl_trigger = None
        self.a3fl_mask = None
        self.a3fl_latest_meta = {'active': False, 'client_id': int(self.ID)}

        # ===== CERBERUS Mode =====
        # Ported to the GGEUR feature-head path from doc/attack/user.py:
        # poisoned CE + clean-anchor distance + optional peer-model cosine.
        self.cerberus_enabled = attack_method == 'cerberus'
        self.cerberus_cfg = getattr(config.attack, 'cerberus', None)
        self.cerberus_attacker_ids = set(parse_attacker_ids(config.attack.attacker_id))
        self.cerberus_is_attacker = (
            self.cerberus_enabled and self.ID in self.cerberus_attacker_ids)
        self.cerberus_trigger = None
        self.cerberus_mask = None
        self.cerberus_peer_models = {}
        self.cerberus_latest_meta = {
            'active': False,
            'client_id': int(self.ID)
        }

        # ===== SABRE Mode =====
        # Full-image additive trigger on the GGEUR feature-head path.
        self.sabre_enabled = attack_method == 'sabre'
        self.sabre_cfg = getattr(config.attack, 'sabre', None)
        self.sabre_attacker_ids = set(parse_attacker_ids(config.attack.attacker_id))
        self.sabre_is_attacker = (
            self.sabre_enabled and self.ID in self.sabre_attacker_ids)
        self.sabre_trigger = None
        self.sabre_mask = None
        self.sabre_latest_meta = {
            'active': False,
            'client_id': int(self.ID)
        }

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

    def _unload_feature_extractor(self):
        """Unload feature extractor from GPU to free VRAM after feature caching."""
        unloaded = False
        if self.clip_model is not None:
            self.clip_model = self.clip_model.cpu()
            del self.clip_model
            self.clip_model = None
            unloaded = True
        if self.cnn_extractor is not None:
            self.cnn_extractor = self.cnn_extractor.cpu()
            del self.cnn_extractor
            self.cnn_extractor = None
            unloaded = True
        if self.timm_extractor is not None:
            self.timm_extractor = self.timm_extractor.cpu()
            del self.timm_extractor
            self.timm_extractor = None
            unloaded = True
        if unloaded:
            torch.cuda.empty_cache()
            logger.info(f"Client {self.ID}: Feature extractor unloaded from GPU")

    def _extractor_forward(self, images, allow_input_grad=False):
        if self.feature_extractor_type == 'cnn':
            if allow_input_grad and self.cnn_extractor is not None and \
                    getattr(self.cnn_extractor, 'freeze', False):
                features = self.cnn_extractor.backbone(images)
                if features.dim() > 2:
                    features = features.view(features.size(0), -1)
                return features
            return self.cnn_extractor(images)
        if self.feature_extractor_type == 'timm':
            if allow_input_grad and self.timm_extractor is not None and \
                    getattr(self.timm_extractor, 'freeze', False):
                features = self.timm_extractor.backbone(images)
                if features.dim() > 2:
                    features = features.view(features.size(0), -1)
                return features
            return self.timm_extractor(images)
        return self.clip_model.encode_image(images)

    def _get_train_dataset_base(self):
        train_data = self.trainer.ctx.data.get('train', None)
        if train_data is None:
            train_data = self.data.get('train', None)
        if train_data is None:
            return None, None

        dataset = train_data.dataset if hasattr(train_data, 'dataset') else train_data
        from torch.utils.data import Subset
        if isinstance(dataset, Subset):
            return dataset.dataset, list(dataset.indices)
        return dataset, list(range(len(dataset)))

    def _ensure_a3fl_trigger(self, sample_image):
        if self.a3fl_trigger is not None and self.a3fl_mask is not None:
            return
        if sample_image.dim() != 3:
            raise ValueError('A3FL requires image tensor with shape [C, H, W].')

        _, height, width = sample_image.shape
        trigger_size = max(1, int(getattr(self.a3fl_cfg, 'trigger_size', 5)))
        trigger_offset = max(0, int(getattr(self.a3fl_cfg, 'trigger_offset', 2)))
        patch_h = min(trigger_size, height)
        patch_w = min(trigger_size, width)
        start_h = min(trigger_offset, max(0, height - patch_h))
        start_w = min(trigger_offset, max(0, width - patch_w))

        self.a3fl_trigger = torch.full(
            (1, sample_image.shape[0], height, width),
            float(getattr(self.a3fl_cfg, 'trigger_init', 0.5)),
            device=self.device)
        self.a3fl_mask = torch.zeros_like(self.a3fl_trigger)
        self.a3fl_mask[:, :, start_h:start_h + patch_h,
                       start_w:start_w + patch_w] = 1.0

    def _apply_a3fl_trigger(self, images):
        if self.a3fl_trigger is None or self.a3fl_mask is None:
            return images
        return self.a3fl_trigger * self.a3fl_mask + images * (1.0 -
                                                              self.a3fl_mask)

    def _a3fl_to_visual_tensor(self, images):
        images = images.detach().cpu().float()
        if images.dim() == 3:
            images = images.unsqueeze(0)

        if self.feature_extractor_type == 'clip':
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                dtype=images.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                               dtype=images.dtype).view(1, 3, 1, 1)
            images = images * std + mean
        elif images.shape[1] == 3:
            mean = torch.tensor([0.485, 0.456, 0.406],
                                dtype=images.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225],
                               dtype=images.dtype).view(1, 3, 1, 1)
            if images.min() < 0.0 or images.max() > 1.0:
                images = images * std + mean

        return torch.clamp(images, 0.0, 1.0)

    def _save_a3fl_trigger_visuals(self, round_idx, clean_images,
                                   poisoned_images):
        if not bool(getattr(self.a3fl_cfg, 'save_trigger_samples', False)):
            return
        if clean_images is None or poisoned_images is None or \
                clean_images.numel() == 0 or poisoned_images.numel() == 0:
            return

        from torchvision.utils import save_image

        max_samples = max(
            1, int(getattr(self.a3fl_cfg, 'save_trigger_max_samples', 4)))
        clean_images = clean_images[:max_samples]
        poisoned_images = poisoned_images[:max_samples]

        clean_vis = self._a3fl_to_visual_tensor(clean_images)
        poisoned_vis = self._a3fl_to_visual_tensor(poisoned_images)
        delta_vis = torch.clamp(
            (poisoned_vis - clean_vis).abs() * 4.0, 0.0, 1.0)
        trigger_vis = self._a3fl_to_visual_tensor(
            self.a3fl_trigger * self.a3fl_mask)
        mask_vis = self.a3fl_mask.detach().cpu().float()
        if mask_vis.dim() == 4 and mask_vis.shape[1] > 1:
            mask_vis = mask_vis[:, :1, :, :]

        round_dir = os.path.join(self._cfg.outdir, 'a3fl_samples',
                                 f'client_{self.ID}', f'round_{int(round_idx)}')
        os.makedirs(round_dir, exist_ok=True)

        save_image(clean_vis,
                   os.path.join(round_dir, 'clean_grid.png'),
                   nrow=min(max_samples, clean_vis.shape[0]))
        save_image(poisoned_vis,
                   os.path.join(round_dir, 'poisoned_grid.png'),
                   nrow=min(max_samples, poisoned_vis.shape[0]))
        save_image(delta_vis,
                   os.path.join(round_dir, 'delta_grid.png'),
                   nrow=min(max_samples, delta_vis.shape[0]))
        save_image(trigger_vis,
                   os.path.join(round_dir, 'trigger.png'))
        save_image(mask_vis,
                   os.path.join(round_dir, 'mask.png'))
        logger.info(
            f"Client {self.ID}: Saved A3FL trigger visualizations to {round_dir}")

    def _evaluate_a3fl_target_rate(self,
                                   base_dataset,
                                   subset_indices,
                                   max_batches=None):
        if self.mlp_classifier is None or not subset_indices:
            return 0.0, 0

        batch_size = max(1, int(getattr(self.ggeur_cfg, 'extract_batch_size', 64)))
        target_label = int(self._cfg.attack.target_label_ind)
        max_batches = max_batches or len(subset_indices)

        self.mlp_classifier.eval()
        self._load_feature_extractor()
        if self.feature_extractor_type == 'clip' and self.clip_model is not None:
            self.clip_model.eval()
        if self.feature_extractor_type == 'cnn' and self.cnn_extractor is not None:
            self.cnn_extractor.eval()
        if self.feature_extractor_type == 'timm' and self.timm_extractor is not None:
            self.timm_extractor.eval()

        total = 0
        target_hits = 0
        processed_batches = 0
        with torch.no_grad():
            for batch_start in range(0, len(subset_indices), batch_size):
                batch_indices = subset_indices[batch_start:batch_start + batch_size]
                if not batch_indices:
                    continue
                images = []
                for base_idx in batch_indices:
                    image, _ = base_dataset[base_idx]
                    images.append(image)
                images = torch.stack(images).to(self.device)
                poisoned_images = self._apply_a3fl_trigger(images)
                features = self._extractor_forward(
                    poisoned_images, allow_input_grad=True).float()
                logits = self.mlp_classifier(features)
                preds = torch.argmax(logits, dim=1)
                target_hits += preds.eq(target_label).sum().item()
                total += preds.shape[0]
                processed_batches += 1
                if processed_batches >= max_batches:
                    break

        return (target_hits / total if total > 0 else 0.0), total

    def _compute_a3fl_debug_metrics(self, images, poisoned_images):
        if self.mlp_classifier is None or images is None or poisoned_images is None:
            return {}

        target_label = int(self._cfg.attack.target_label_ind)
        with torch.no_grad():
            clean_features = self._extractor_forward(images).float()
            poison_features = self._extractor_forward(poisoned_images).float()
            clean_logits = self.mlp_classifier(clean_features)
            poison_logits = self.mlp_classifier(poison_features)

            clean_preds = torch.argmax(clean_logits, dim=1)
            poison_preds = torch.argmax(poison_logits, dim=1)

            clean_target_logits = clean_logits[:, target_label]
            poison_target_logits = poison_logits[:, target_label]

            feature_shift = torch.norm(
                poison_features - clean_features, dim=1).mean().item()
            clean_target_rate = clean_preds.eq(target_label).float().mean().item()
            poison_target_rate = poison_preds.eq(target_label).float().mean().item()
            target_logit_gain = (
                poison_target_logits - clean_target_logits).mean().item()
            clean_target_logit = clean_target_logits.mean().item()
            poison_target_logit = poison_target_logits.mean().item()

        return {
            'feature_shift_l2': float(feature_shift),
            'clean_target_rate': float(clean_target_rate),
            'poison_target_rate': float(poison_target_rate),
            'target_logit_gain': float(target_logit_gain),
            'clean_target_logit': float(clean_target_logit),
            'poison_target_logit': float(poison_target_logit),
        }

    def _strengthen_a3fl_mlp_update(self, global_state_dict, local_state_dict):
        if global_state_dict is None or local_state_dict is None:
            return local_state_dict

        update_scale = float(getattr(self.a3fl_cfg, 'update_scale', 1.0))
        target_row_scale = float(
            getattr(self.a3fl_cfg, 'target_row_scale', 1.0))
        if update_scale == 1.0 and target_row_scale == 1.0:
            return local_state_dict

        target_label = int(self._cfg.attack.target_label_ind)
        num_classes = int(self._cfg.model.num_classes)
        strengthened_state = copy.deepcopy(local_state_dict)
        total_delta_norm = 0.0
        target_delta_norm = 0.0

        for key, local_tensor in strengthened_state.items():
            global_tensor = global_state_dict.get(key, None)
            if global_tensor is None or not torch.is_tensor(local_tensor):
                continue

            global_tensor = global_tensor.to(local_tensor.device)
            delta = local_tensor - global_tensor

            if update_scale != 1.0:
                delta = delta * update_scale

            if target_row_scale != 1.0:
                if delta.dim() >= 2 and delta.shape[0] == num_classes:
                    delta[target_label] = delta[target_label] * target_row_scale
                elif delta.dim() == 1 and delta.shape[0] == num_classes:
                    delta[target_label] = delta[target_label] * target_row_scale

            strengthened_state[key] = global_tensor + delta
            total_delta_norm += delta.norm().item()

            if delta.dim() >= 2 and delta.shape[0] == num_classes:
                target_delta_norm += delta[target_label].norm().item()
            elif delta.dim() == 1 and delta.shape[0] == num_classes:
                target_delta_norm += delta[target_label].abs().item()

        self.a3fl_latest_meta.update({
            'update_scale': float(update_scale),
            'target_row_scale': float(target_row_scale),
            'strengthened_total_delta_norm': float(total_delta_norm),
            'strengthened_target_delta_norm': float(target_delta_norm),
        })
        logger.info(
            f"Client {self.ID}: A3FL update strengthen - "
            f"update_scale={update_scale:.2f}, "
            f"target_row_scale={target_row_scale:.2f}, "
            f"total_delta_norm={total_delta_norm:.4f}, "
            f"target_delta_norm={target_delta_norm:.4f}")
        return strengthened_state

    def _run_a3fl_trigger_search(self):
        if not self.a3fl_enabled or not self.a3fl_is_attacker or \
                self.mlp_classifier is None:
            return False

        base_dataset, subset_indices = self._get_train_dataset_base()
        if base_dataset is None or not subset_indices:
            logger.warning(f"Client {self.ID}: No dataset available for A3FL trigger search")
            return False

        self._load_feature_extractor()
        first_image, _ = base_dataset[subset_indices[0]]
        self._ensure_a3fl_trigger(first_image.to(self.device))

        batch_size = max(1, int(getattr(self.ggeur_cfg, 'extract_batch_size', 64)))
        outer_epochs = max(1, int(getattr(self.a3fl_cfg, 'trigger_outer_epochs', 5)))
        batch_limit = max(1, int(getattr(self.a3fl_cfg, 'trigger_search_batches', 1)))
        trigger_lr = float(getattr(self.a3fl_cfg, 'trigger_lr', 0.01))
        clip_min = float(getattr(self.a3fl_cfg, 'trigger_clip_min', -2.0))
        clip_max = float(getattr(self.a3fl_cfg, 'trigger_clip_max', 2.0))
        target_label = int(self._cfg.attack.target_label_ind)
        criterion = nn.CrossEntropyLoss()

        self.mlp_classifier.eval()
        if self.feature_extractor_type == 'clip' and self.clip_model is not None:
            self.clip_model.eval()
        if self.feature_extractor_type == 'cnn' and self.cnn_extractor is not None:
            self.cnn_extractor.eval()
        if self.feature_extractor_type == 'timm' and self.timm_extractor is not None:
            self.timm_extractor.eval()

        trigger = self.a3fl_trigger.detach().clone()
        processed_batches = 0
        max_batches = outer_epochs * batch_limit
        for _ in range(outer_epochs):
            for batch_start in range(0, len(subset_indices), batch_size):
                batch_indices = subset_indices[batch_start:batch_start + batch_size]
                if not batch_indices:
                    continue
                images = []
                for base_idx in batch_indices:
                    image, _ = base_dataset[base_idx]
                    images.append(image)
                images = torch.stack(images).to(self.device)
                labels = torch.full((images.shape[0], ),
                                    target_label,
                                    dtype=torch.long,
                                    device=self.device)

                trigger.requires_grad_()
                poisoned_images = trigger * self.a3fl_mask + images * (
                    1.0 - self.a3fl_mask)
                features = self._extractor_forward(
                    poisoned_images, allow_input_grad=True).float()
                logits = self.mlp_classifier(features)
                loss = criterion(logits, labels)
                grad = torch.autograd.grad(loss, trigger)[0]
                trigger = trigger.detach() - trigger_lr * grad.sign()
                trigger = torch.clamp(trigger, clip_min, clip_max)
                processed_batches += 1
                if processed_batches >= max_batches:
                    break
            if processed_batches >= max_batches:
                break

        self.a3fl_trigger = trigger.detach()
        target_rate, target_total = self._evaluate_a3fl_target_rate(
            base_dataset, subset_indices, max_batches=batch_limit)
        debug_sample_count = min(len(subset_indices), batch_size)
        debug_metrics = {}
        if debug_sample_count > 0:
            debug_images = []
            for base_idx in subset_indices[:debug_sample_count]:
                image, _ = base_dataset[base_idx]
                debug_images.append(image)
            debug_images = torch.stack(debug_images).to(self.device)
            debug_poisoned_images = self._apply_a3fl_trigger(debug_images)
            debug_metrics = self._compute_a3fl_debug_metrics(
                debug_images, debug_poisoned_images)
            self.a3fl_latest_meta.update({
                'trigger_target_rate': float(target_rate),
                'trigger_eval_samples': int(target_total),
                'trigger_feature_shift_l2':
                    debug_metrics['feature_shift_l2'],
                'trigger_clean_target_rate':
                    debug_metrics['clean_target_rate'],
                'trigger_poison_target_rate':
                    debug_metrics['poison_target_rate'],
                'trigger_target_logit_gain':
                    debug_metrics['target_logit_gain'],
                'trigger_clean_target_logit':
                    debug_metrics['clean_target_logit'],
                'trigger_poison_target_logit':
                    debug_metrics['poison_target_logit'],
            })
        logger.info(
            f"Client {self.ID}: A3FL trigger search finished using {processed_batches} batches, "
            f"target_hit_rate={target_rate:.4f} on {target_total} samples")
        if debug_metrics:
            logger.info(
                f"Client {self.ID}: A3FL trigger debug - "
                f"clean_target_rate={debug_metrics['clean_target_rate']:.4f}, "
                f"poison_target_rate={debug_metrics['poison_target_rate']:.4f}, "
                f"target_logit_gain={debug_metrics['target_logit_gain']:.4f}, "
                f"feature_shift_l2={debug_metrics['feature_shift_l2']:.4f}")
        return processed_batches > 0

    def _restore_base_augmented_dataset(self):
        if self.base_augmented_features is None or self.base_augmented_labels is None:
            return
        self.augmented_features = self.base_augmented_features.copy()
        self.augmented_labels = self.base_augmented_labels.copy()
        dataset = AugmentedFeatureDataset(self.augmented_features,
                                          self.augmented_labels)
        self.augmented_loader = DataLoader(dataset,
                                           batch_size=self._cfg.dataloader.batch_size,
                                           shuffle=True)

    def _inject_a3fl_poison_features(self, round_idx):
        if not self.a3fl_enabled:
            return

        self._restore_base_augmented_dataset()
        active = self.a3fl_is_attacker and should_a3fl_attack(
            self._cfg, round_idx, self.ID, self._cfg.federate.sample_client_num)
        self.a3fl_latest_meta = {
            'active': bool(active),
            'client_id': int(self.ID),
            'round': int(round_idx),
            'target_label': int(self._cfg.attack.target_label_ind),
        }
        if not active:
            return

        if not self._run_a3fl_trigger_search():
            return

        base_dataset, subset_indices = self._get_train_dataset_base()
        if base_dataset is None or not subset_indices:
            return

        poison_count = max(1, int(len(subset_indices) * float(self._cfg.attack.poison_ratio)))
        poison_count = min(poison_count, len(subset_indices))
        rng = np.random.RandomState(int(self._cfg.seed) + int(round_idx) + int(self.ID) * 997)
        selected_indices = rng.choice(subset_indices, size=poison_count, replace=False).tolist()

        self._load_feature_extractor()
        poison_features = []
        sample_clean_images = []
        sample_poisoned_images = []
        debug_clean_feature_chunks = []
        debug_poison_feature_chunks = []
        batch_size = max(1, int(getattr(self.ggeur_cfg, 'extract_batch_size', 64)))
        sample_budget = max(
            1, int(getattr(self.a3fl_cfg, 'save_trigger_max_samples', 4)))
        with torch.no_grad():
            for start in range(0, len(selected_indices), batch_size):
                batch_indices = selected_indices[start:start + batch_size]
                images = []
                for base_idx in batch_indices:
                    image, _ = base_dataset[base_idx]
                    images.append(image)
                images = torch.stack(images).to(self.device)
                poisoned_images = self._apply_a3fl_trigger(images)
                saved_samples = sum(
                    tensor.shape[0] for tensor in sample_clean_images)
                if saved_samples < sample_budget:
                    remain = sample_budget - saved_samples
                    sample_clean_images.append(images[:remain].detach().cpu())
                    sample_poisoned_images.append(
                        poisoned_images[:remain].detach().cpu())
                    debug_clean_feature_chunks.append(
                        self._extractor_forward(images[:remain]).float().cpu())
                    debug_poison_feature_chunks.append(
                        self._extractor_forward(
                            poisoned_images[:remain]).float().cpu())
                features = self._extractor_forward(poisoned_images).float()
                poison_features.append(features.cpu().numpy())

        if not poison_features:
            return

        poison_features = np.vstack(poison_features)
        poison_repeat = max(
            1, int(getattr(self.a3fl_cfg, 'poison_feature_repeat', 1)))
        if poison_repeat > 1:
            poison_features = np.repeat(poison_features,
                                        poison_repeat,
                                        axis=0)
        poison_labels = np.full(poison_features.shape[0],
                                int(self._cfg.attack.target_label_ind),
                                dtype=np.int64)

        self.augmented_features = np.vstack(
            [self.augmented_features, poison_features]).astype(np.float32)
        self.augmented_labels = np.concatenate(
            [self.augmented_labels, poison_labels]).astype(np.int64)
        dataset = AugmentedFeatureDataset(self.augmented_features,
                                          self.augmented_labels)
        self.augmented_loader = DataLoader(dataset,
                                           batch_size=self._cfg.dataloader.batch_size,
                                           shuffle=True)

        self.a3fl_latest_meta.update({
            'trigger': self.a3fl_trigger.detach().cpu(),
            'mask': self.a3fl_mask.detach().cpu(),
            'poisoned_samples': int(poison_features.shape[0]),
            'poison_feature_repeat': int(poison_repeat),
        })
        if debug_clean_feature_chunks and debug_poison_feature_chunks:
            debug_clean_features = torch.cat(debug_clean_feature_chunks, dim=0)
            debug_poison_features = torch.cat(debug_poison_feature_chunks, dim=0)
            inject_feature_shift = torch.norm(
                debug_poison_features - debug_clean_features,
                dim=1).mean().item()
            self.a3fl_latest_meta.update({
                'inject_feature_shift_l2': float(inject_feature_shift),
            })
            logger.info(
                f"Client {self.ID}: A3FL inject debug - "
                f"poisoned_samples={poison_features.shape[0]}, "
                f"inject_feature_shift_l2={inject_feature_shift:.4f}")
        if sample_clean_images and sample_poisoned_images:
            self._save_a3fl_trigger_visuals(
                round_idx, torch.cat(sample_clean_images, dim=0),
                torch.cat(sample_poisoned_images, dim=0))
        logger.info(
            f"Client {self.ID}: Injected {poison_features.shape[0]} A3FL poisoned feature samples "
            f"in round {round_idx} (repeat={poison_repeat})")

    def _get_cerberus_start_round(self):
        start_round = int(getattr(self.cerberus_cfg, 'start_round', -1))
        if start_round >= 0:
            return start_round
        return int(getattr(self._cfg.attack, 'inject_round', 0))

    def _is_cerberus_active_round(self, round_idx):
        if not self.cerberus_enabled:
            return False
        start_round = self._get_cerberus_start_round()
        if int(round_idx) < start_round:
            return False
        poison_epochs = int(getattr(self.cerberus_cfg, 'poison_epochs', 0))
        if poison_epochs <= 0:
            return True
        return int(round_idx) < start_round + poison_epochs

    def _should_cerberus_attack(self, round_idx):
        return (
            self.cerberus_is_attacker and
            self._is_cerberus_active_round(round_idx)
        )

    def _get_cerberus_pattern(self, height, width):
        pattern = getattr(self.cerberus_cfg, 'poison_pattern', None)
        if pattern:
            coords = []
            for pos in pattern:
                if len(pos) < 2:
                    continue
                row = min(max(int(pos[0]), 0), height - 1)
                col = min(max(int(pos[1]), 0), width - 1)
                coords.append((row, col))
            if coords:
                return coords

        pattern_size = max(1, int(getattr(self.cerberus_cfg, 'pattern_size', 4)))
        pattern_offset = max(0, int(getattr(self.cerberus_cfg, 'pattern_offset', 0)))
        rows = range(pattern_offset, min(height, pattern_offset + pattern_size))
        cols = range(pattern_offset, min(width, pattern_offset + pattern_size))
        return [(row, col) for row in rows for col in cols]

    def _ensure_cerberus_trigger(self, sample_image):
        if self.cerberus_trigger is not None and self.cerberus_mask is not None:
            return
        if sample_image.dim() != 3:
            raise ValueError('CERBERUS requires image tensor with shape [C, H, W].')

        channels, height, width = sample_image.shape
        trigger_init = float(getattr(self.cerberus_cfg, 'trigger_init', 0.5))
        self.cerberus_trigger = torch.full(
            (1, channels, height, width),
            trigger_init,
            device=self.device)
        self.cerberus_mask = torch.zeros_like(self.cerberus_trigger)
        for row, col in self._get_cerberus_pattern(height, width):
            self.cerberus_mask[:, :, row, col] = 1.0

    def _apply_cerberus_trigger(self, images):
        if self.cerberus_trigger is None or self.cerberus_mask is None:
            return images
        return self.cerberus_trigger * self.cerberus_mask + images * (
            1.0 - self.cerberus_mask)

    def _get_sabre_start_round(self):
        start_round = int(getattr(self.sabre_cfg, 'start_round', -1))
        if start_round >= 0:
            return start_round
        return int(getattr(self._cfg.attack, 'inject_round', 0))

    def _is_sabre_active_round(self, round_idx):
        if not self.sabre_enabled:
            return False
        start_round = self._get_sabre_start_round()
        if int(round_idx) < start_round:
            return False
        poison_epochs = int(getattr(self.sabre_cfg, 'poison_epochs', 0))
        if poison_epochs <= 0:
            return True
        return int(round_idx) < start_round + poison_epochs

    def _should_sabre_attack(self, round_idx):
        return (
            self.sabre_is_attacker and
            self._is_sabre_active_round(round_idx)
        )

    def _ensure_sabre_trigger(self, sample_image):
        if self.sabre_trigger is not None and self.sabre_mask is not None:
            return
        if sample_image.dim() != 3:
            raise ValueError('SABRE requires image tensor with shape [C, H, W].')

        channels, height, width = sample_image.shape
        shape = (1, channels, height, width)
        init_mode = str(getattr(
            self.sabre_cfg, 'trigger_init_mode', 'uniform')).lower()
        trigger_init = float(getattr(self.sabre_cfg, 'trigger_init', 0.0))
        random_scale = float(getattr(
            self.sabre_cfg, 'trigger_random_scale', 0.01))
        seed = int(getattr(self.sabre_cfg, 'trigger_seed', 0)) + \
            int(self.ID) * 1009

        if init_mode in ('uniform', 'random_uniform'):
            generator = torch.Generator(device='cpu')
            generator.manual_seed(seed)
            trigger = torch.empty(shape).uniform_(
                -random_scale, random_scale, generator=generator)
        elif init_mode in ('normal', 'gaussian', 'random_normal'):
            generator = torch.Generator(device='cpu')
            generator.manual_seed(seed)
            trigger = torch.randn(shape, generator=generator) * random_scale
        else:
            trigger = torch.full(shape, trigger_init)

        self.sabre_trigger = trigger.to(self.device)
        self.sabre_mask = torch.ones_like(self.sabre_trigger)

    def _apply_sabre_trigger(self, images, trigger=None, mask=None):
        trigger = self.sabre_trigger if trigger is None else trigger
        mask = self.sabre_mask if mask is None else mask
        if trigger is None or mask is None:
            return images
        if images.dim() == 3:
            images = images.unsqueeze(0)
        clip_min = float(getattr(self.sabre_cfg, 'image_clip_min', -3.0))
        clip_max = float(getattr(self.sabre_cfg, 'image_clip_max', 3.0))
        delta = trigger.to(images.device) * mask.to(images.device)
        return torch.clamp(images + delta, clip_min, clip_max)

    def _move_prompt_buffers(self, device):
        self.prompt_learner.token_prefix = self.prompt_learner.token_prefix.to(device)
        self.prompt_learner.token_suffix = self.prompt_learner.token_suffix.to(device)
        self.prompt_learner.tokenized_prompts = \
            self.prompt_learner.tokenized_prompts.to(device)

    def _optimize_cerberus_trigger(self, base_dataset, candidate_indices,
                                   round_idx):
        if self.mlp_classifier is None or self.cerberus_trigger is None or \
                self.cerberus_mask is None or base_dataset is None or \
                not candidate_indices:
            return

        target_label = int(self._cfg.attack.target_label_ind)
        steps = max(0, int(getattr(
            self.cerberus_cfg, 'trigger_search_steps', 0)))
        if steps <= 0:
            return

        batch_size = max(1, int(getattr(
            self.cerberus_cfg, 'trigger_search_batch_size', 8)))
        max_batches = max(1, int(getattr(
            self.cerberus_cfg, 'trigger_search_batches', 2)))
        lr = float(getattr(self.cerberus_cfg, 'trigger_search_lr', 0.05))
        clip_min = float(getattr(
            self.cerberus_cfg, 'trigger_search_clip_min', -2.5))
        clip_max = float(getattr(
            self.cerberus_cfg, 'trigger_search_clip_max', 2.5))
        proj_norm = float(getattr(
            self.cerberus_cfg, 'trigger_search_proj_norm', 12.0))
        target_margin = float(getattr(
            self.cerberus_cfg, 'trigger_search_target_margin', 1.0))
        gain_weight = float(getattr(
            self.cerberus_cfg, 'trigger_search_gain_weight', 0.5))
        gain_margin = float(getattr(
            self.cerberus_cfg, 'trigger_search_gain_margin', 0.5))
        l2_weight = float(getattr(
            self.cerberus_cfg, 'trigger_search_l2_weight', 1e-4))

        search_indices = []
        for base_idx in candidate_indices:
            try:
                _, label = base_dataset[base_idx]
            except Exception:
                continue
            label_value = int(label.item()) if torch.is_tensor(label) else int(label)
            if label_value != target_label:
                search_indices.append(base_idx)
            if len(search_indices) >= batch_size * max_batches:
                break
        if not search_indices:
            search_indices = list(candidate_indices[:batch_size * max_batches])
        if not search_indices:
            return

        self._load_feature_extractor()
        if self.feature_extractor_type == 'clip' and self.clip_model is not None:
            self.clip_model.eval()
        if self.feature_extractor_type == 'cnn' and self.cnn_extractor is not None:
            self.cnn_extractor.eval()
        if self.feature_extractor_type == 'timm' and self.timm_extractor is not None:
            self.timm_extractor.eval()

        extractor_modules = [
            module for module in
            (self.clip_model, self.cnn_extractor, self.timm_extractor)
            if module is not None
        ]
        saved_requires_grad = []
        for module in extractor_modules:
            for param in module.parameters():
                saved_requires_grad.append((param, param.requires_grad))
                param.requires_grad_(False)
        for param in self.mlp_classifier.parameters():
            saved_requires_grad.append((param, param.requires_grad))
            param.requires_grad_(False)

        trigger_base = self.cerberus_trigger.detach().clone()
        trigger = trigger_base.clone().requires_grad_(True)
        mask = self.cerberus_mask.detach()
        optimizer = torch.optim.Adam([trigger], lr=lr)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_ce = 0.0
        total_margin = 0.0
        total_gain = 0.0
        total_batches = 0
        try:
            for step in range(steps):
                offset = (step * batch_size) % len(search_indices)
                if offset + batch_size <= len(search_indices):
                    batch_indices = search_indices[offset:offset + batch_size]
                else:
                    batch_indices = search_indices[offset:] + \
                        search_indices[:batch_size - (len(search_indices) - offset)]

                images = []
                labels = []
                for base_idx in batch_indices:
                    image, label = base_dataset[base_idx]
                    images.append(image)
                    labels.append(
                        int(label.item()) if torch.is_tensor(label)
                        else int(label))
                images = torch.stack(images).to(self.device)
                labels = torch.as_tensor(labels,
                                         dtype=torch.long,
                                         device=self.device)
                target_labels = torch.full_like(labels, target_label)

                optimizer.zero_grad()
                poisoned_images = trigger * mask + images * (1.0 - mask)
                poison_features = self._extractor_forward(
                    poisoned_images, allow_input_grad=True).float()
                poison_logits = self.mlp_classifier(poison_features)
                poison_ce = criterion(poison_logits, target_labels)

                target_logits = poison_logits[:, target_label]
                other_logits = poison_logits.clone()
                if 0 <= target_label < other_logits.size(1):
                    other_logits[:, target_label] = -1e9
                max_other_logits = other_logits.max(dim=1).values
                target_margin_loss = F.relu(
                    max_other_logits - target_logits + target_margin).mean()

                with torch.no_grad():
                    clean_features = self._extractor_forward(images).float()
                    clean_logits = self.mlp_classifier(clean_features)
                    clean_target_logits = clean_logits[:, target_label]
                gain_loss = F.relu(
                    gain_margin - (target_logits - clean_target_logits)).mean()
                l2_loss = torch.norm((trigger - trigger_base) * mask, p=2)
                loss = poison_ce + target_margin_loss + \
                    gain_weight * gain_loss + l2_weight * l2_loss
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    trigger.mul_(mask).add_(trigger_base * (1.0 - mask))
                    trigger.clamp_(clip_min, clip_max)
                    if proj_norm > 0:
                        delta = (trigger - trigger_base) * mask
                        delta_norm = torch.norm(delta, p=2)
                        if delta_norm > proj_norm:
                            delta = delta * (proj_norm / (delta_norm + 1e-12))
                            trigger.copy_(trigger_base + delta)
                            trigger.mul_(mask).add_(
                                trigger_base * (1.0 - mask))

                total_loss += loss.item()
                total_ce += poison_ce.item()
                total_margin += target_margin_loss.item()
                total_gain += gain_loss.item()
                total_batches += 1

            self.cerberus_trigger = trigger.detach()
        finally:
            for param, requires_grad in saved_requires_grad:
                param.requires_grad_(requires_grad)

        if total_batches <= 0:
            return

        eval_images = []
        eval_labels = []
        for base_idx in search_indices[:batch_size]:
            image, label = base_dataset[base_idx]
            eval_images.append(image)
            eval_labels.append(
                int(label.item()) if torch.is_tensor(label) else int(label))
        eval_images = torch.stack(eval_images).to(self.device)
        eval_labels = torch.as_tensor(eval_labels,
                                      dtype=torch.long,
                                      device=self.device)
        with torch.no_grad():
            clean_features = self._extractor_forward(eval_images).float()
            clean_logits = self.mlp_classifier(clean_features)
            poisoned_images = self._apply_cerberus_trigger(eval_images)
            poison_features = self._extractor_forward(poisoned_images).float()
            poison_logits = self.mlp_classifier(poison_features)
            clean_preds = torch.argmax(clean_logits, dim=1)
            poison_preds = torch.argmax(poison_logits, dim=1)
            clean_target_rate = clean_preds.eq(target_label).float().mean()
            poison_target_rate = poison_preds.eq(target_label).float().mean()
            target_gain = (
                poison_logits[:, target_label] -
                clean_logits[:, target_label]).mean()
            eval_target_labels = torch.full_like(eval_labels, target_label)
            eval_ce = criterion(poison_logits, eval_target_labels)

        trigger_delta_norm = torch.norm(
            (self.cerberus_trigger - trigger_base) * mask, p=2).item()
        self.cerberus_latest_meta.update({
            'trigger_search_steps': int(steps),
            'trigger_search_batches': int(max_batches),
            'trigger_search_loss': float(total_loss / total_batches),
            'trigger_search_ce': float(total_ce / total_batches),
            'trigger_search_margin': float(total_margin / total_batches),
            'trigger_search_gain_loss': float(total_gain / total_batches),
            'trigger_search_eval_ce': float(eval_ce.item()),
            'trigger_search_clean_target_rate': float(
                clean_target_rate.item()),
            'trigger_search_poison_target_rate': float(
                poison_target_rate.item()),
            'trigger_search_target_logit_gain': float(target_gain.item()),
            'trigger_delta_norm': float(trigger_delta_norm),
        })
        logger.info(
            f"Client {self.ID}: CERBERUS trigger search round {round_idx} "
            f"steps={steps}, batches={max_batches}, "
            f"loss={total_loss / total_batches:.4f}, "
            f"CE={total_ce / total_batches:.4f}, "
            f"poison_target_rate={poison_target_rate.item():.4f}, "
            f"clean_target_rate={clean_target_rate.item():.4f}, "
            f"target_logit_gain={target_gain.item():.4f}, "
            f"delta_norm={trigger_delta_norm:.4f}")

    def _build_cerberus_poison_feature_pool(self, round_idx):
        if self.mlp_classifier is None:
            return None, None

        target_label = int(self._cfg.attack.target_label_ind)
        base_dataset, subset_indices = self._get_train_dataset_base()
        poison_ratio = float(getattr(self._cfg.attack, 'poison_ratio', 0.1))

        if base_dataset is None or not subset_indices:
            if self.augmented_features is None or len(self.augmented_features) == 0:
                logger.warning(
                    f"Client {self.ID}: CERBERUS could not find local data "
                    "for poisoned feature construction")
                return None, None
            poison_count = max(1, int(len(self.augmented_features) * poison_ratio))
            poison_count = min(poison_count, len(self.augmented_features))
            rng = np.random.RandomState(
                int(self._cfg.seed) + int(round_idx) + int(self.ID) * 1009)
            selected = rng.choice(len(self.augmented_features),
                                  size=poison_count,
                                  replace=False)
            poison_features = torch.from_numpy(
                self.augmented_features[selected]).float()
            poison_labels = torch.full((poison_count, ),
                                       target_label,
                                       dtype=torch.long)
            return poison_features, poison_labels

        poison_count = max(1, int(len(subset_indices) * poison_ratio))
        poison_count = min(poison_count, len(subset_indices))
        rng = np.random.RandomState(
            int(self._cfg.seed) + int(round_idx) + int(self.ID) * 1009)
        selected_indices = rng.choice(subset_indices,
                                      size=poison_count,
                                      replace=False).tolist()

        self._load_feature_extractor()
        first_image, _ = base_dataset[selected_indices[0]]
        self._ensure_cerberus_trigger(first_image.to(self.device))
        self._optimize_cerberus_trigger(base_dataset, selected_indices,
                                        round_idx)
        self.cerberus_latest_meta.update({
            'trigger': self.cerberus_trigger.detach().cpu(),
            'mask': self.cerberus_mask.detach().cpu(),
        })

        if self.feature_extractor_type == 'clip' and self.clip_model is not None:
            self.clip_model.eval()
        if self.feature_extractor_type == 'cnn' and self.cnn_extractor is not None:
            self.cnn_extractor.eval()
        if self.feature_extractor_type == 'timm' and self.timm_extractor is not None:
            self.timm_extractor.eval()

        poison_features = []
        batch_size = max(1, int(getattr(self.ggeur_cfg, 'extract_batch_size', 64)))
        with torch.no_grad():
            for start in range(0, len(selected_indices), batch_size):
                batch_indices = selected_indices[start:start + batch_size]
                images = []
                for base_idx in batch_indices:
                    image, _ = base_dataset[base_idx]
                    images.append(image)
                images = torch.stack(images).to(self.device)
                poisoned_images = self._apply_cerberus_trigger(images)
                features = self._extractor_forward(poisoned_images).float()
                poison_features.append(features.detach().cpu())

        if not poison_features:
            return None, None

        poison_features = torch.cat(poison_features, dim=0)
        poison_labels = torch.full((poison_features.shape[0], ),
                                   target_label,
                                   dtype=torch.long)
        self.cerberus_latest_meta.update({
            'poisoned_samples': int(poison_features.shape[0]),
            'target_label': int(target_label),
        })
        return poison_features, poison_labels

    def _train_cerberus_clean_anchor(self):
        anchor_model = copy.deepcopy(self.mlp_classifier)
        anchor_model.train()

        clean_lr = float(getattr(
            self.cerberus_cfg,
            'clean_anchor_lr',
            getattr(self.cerberus_cfg, 'shadow_lr', self._cfg.train.optimizer.lr)))
        clean_epochs = int(getattr(
            self.cerberus_cfg,
            'clean_anchor_epochs',
            getattr(self.cerberus_cfg, 'shadow_epochs', 1)))
        clean_epochs = max(1, clean_epochs)

        optimizer = torch.optim.Adam(anchor_model.parameters(), lr=clean_lr)
        criterion = nn.CrossEntropyLoss()
        for _ in range(clean_epochs):
            for features, labels in self.augmented_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = anchor_model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        anchor_state = {
            name: param.detach().clone()
            for name, param in anchor_model.named_parameters()
            if param.requires_grad
        }
        return anchor_state

    def _cerberus_anchor_distance(self, anchor_state):
        distance = torch.tensor(0.0, device=self.device)
        for name, param in self.mlp_classifier.named_parameters():
            if not param.requires_grad or name not in anchor_state:
                continue
            anchor_param = anchor_state[name].to(param.device)
            distance = distance + torch.norm(param - anchor_param, p=2) ** 2
        return distance

    def _cerberus_peer_cosine(self):
        if not self.cerberus_peer_models:
            return torch.tensor(0.0, device=self.device)

        peer_terms = []
        trainable_params = [
            (name, param)
            for name, param in self.mlp_classifier.named_parameters()
            if param.requires_grad
        ]
        for _, peer_model in self.cerberus_peer_models.items():
            layer_terms = []
            for name, param in trainable_params:
                if name not in peer_model:
                    continue
                peer_param = peer_model[name]
                if not torch.is_tensor(peer_param):
                    try:
                        peer_param = param2tensor(peer_param)
                    except Exception:
                        continue
                peer_param = peer_param.to(param.device).view(-1)
                param_flat = param.view(-1)
                eps = 1e-8
                cosine = F.cosine_similarity(param_flat + eps,
                                             peer_param + eps,
                                             dim=0)
                layer_terms.append(torch.abs(cosine))
            if layer_terms:
                peer_terms.append(torch.stack(layer_terms).mean())

        if not peer_terms:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(peer_terms).mean()

    def _sample_cerberus_poison_batch(self, poison_features, poison_labels,
                                      batch_size):
        pool_size = poison_features.shape[0]
        if pool_size == 0:
            return None, None

        poison_per_batch = int(getattr(self.cerberus_cfg,
                                       'poisoning_per_batch', 0))
        if poison_per_batch <= 0:
            poison_ratio = float(getattr(self._cfg.attack, 'poison_ratio', 0.1))
            poison_per_batch = max(1, int(batch_size * poison_ratio))
        poison_per_batch = min(max(1, poison_per_batch), batch_size)

        replace = pool_size < poison_per_batch
        indices = np.random.choice(pool_size,
                                   size=poison_per_batch,
                                   replace=replace)
        poison_x = poison_features[indices].to(self.device)
        poison_y = poison_labels[indices].to(self.device)
        return poison_x, poison_y

    def _extract_cerberus_peer_models(self, content):
        self.cerberus_peer_models = {}
        if not isinstance(content, dict):
            return

        peer_models = {}
        cerberus_payload = content.get('cerberus', None)
        if isinstance(cerberus_payload, dict):
            peer_models = cerberus_payload.get('peer_models', peer_models)
        if isinstance(peer_models, dict):
            self.cerberus_peer_models = {
                peer_id: peer_state
                for peer_id, peer_state in peer_models.items()
                if int(peer_id) != int(self.ID) and isinstance(peer_state, dict)
            }

        shared_trigger = content.get('cerberus_shared_trigger', None)
        if shared_trigger is None and isinstance(cerberus_payload, dict):
            shared_trigger = cerberus_payload.get('shared_trigger', None)
        if isinstance(shared_trigger, dict):
            trigger = shared_trigger.get('trigger', None)
            mask = shared_trigger.get('mask', None)
            if trigger is not None and mask is not None:
                try:
                    trigger = param2tensor(trigger)
                    mask = param2tensor(mask)
                except Exception:
                    pass
                if isinstance(trigger, torch.Tensor) and isinstance(mask, torch.Tensor):
                    self.cerberus_trigger = trigger.to(self.device).float()
                    self.cerberus_mask = mask.to(self.device).float()
                    source_client = shared_trigger.get(
                        'source_client_id', 'unknown')
                    source_round = shared_trigger.get(
                        'source_round', 'unknown')
                    logger.info(
                        f"Client {self.ID}: Loaded shared CERBERUS trigger "
                        f"from client {source_client}, round {source_round}")

    def _extract_shared_attack_trigger(self, content, attack_name):
        if not isinstance(content, dict):
            return

        payload = content.get(attack_name, None)
        shared_trigger = content.get(f'{attack_name}_shared_trigger', None)
        if shared_trigger is None and isinstance(payload, dict):
            shared_trigger = payload.get('shared_trigger', None)
        if not isinstance(shared_trigger, dict):
            return

        trigger = shared_trigger.get('trigger', None)
        mask = shared_trigger.get('mask', None)
        if trigger is None or mask is None:
            return
        try:
            trigger = param2tensor(trigger)
            mask = param2tensor(mask)
        except Exception as exc:
            logger.debug(
                f"Client {self.ID}: Could not restore shared "
                f"{attack_name.upper()} trigger: {exc}")
            return
        if not isinstance(trigger, torch.Tensor) or \
                not isinstance(mask, torch.Tensor):
            return
        if tuple(trigger.shape) != tuple(mask.shape):
            logger.debug(
                f"Client {self.ID}: Ignored shared {attack_name.upper()} "
                f"trigger with mismatched trigger/mask shapes "
                f"{tuple(trigger.shape)} vs {tuple(mask.shape)}")
            return

        setattr(self, f'{attack_name}_trigger', trigger.to(self.device).float())
        setattr(self, f'{attack_name}_mask', mask.to(self.device).float())
        source_client = shared_trigger.get('source_client_id', 'unknown')
        source_round = shared_trigger.get('source_round', 'unknown')
        logger.info(
            f"Client {self.ID}: Loaded shared {attack_name.upper()} trigger "
            f"from client {source_client}, round {source_round}")

    def _train_cerberus_on_augmented_data(self, round_idx):
        if self.augmented_loader is None or self.mlp_classifier is None:
            return 0, {}, {}

        self.cerberus_latest_meta = {
            'active': True,
            'client_id': int(self.ID),
            'round': int(round_idx),
            'target_label': int(self._cfg.attack.target_label_ind),
        }

        poison_features, poison_labels = self._build_cerberus_poison_feature_pool(
            round_idx)
        if poison_features is None or poison_labels is None:
            logger.warning(
                f"Client {self.ID}: CERBERUS poison pool is empty; "
                "falling back to clean augmented training")
            self.cerberus_latest_meta['active'] = False
            return self._train_on_augmented_data()

        anchor_state = self._train_cerberus_clean_anchor()

        poison_lr = float(getattr(self.cerberus_cfg,
                                  'poison_lr',
                                  self._cfg.train.optimizer.lr))
        optimizer_name = str(getattr(self.cerberus_cfg,
                                     'poison_optimizer',
                                     'adam')).lower()
        if optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.mlp_classifier.parameters(),
                                        lr=poison_lr)
        else:
            optimizer = torch.optim.Adam(self.mlp_classifier.parameters(),
                                         lr=poison_lr)

        criterion = nn.CrossEntropyLoss()
        alpha_loss = float(getattr(self.cerberus_cfg, 'alpha_loss', 0.01))
        beta_loss = float(getattr(self.cerberus_cfg, 'beta_loss', 0.01))
        clean_ce_weight = float(getattr(
            self.cerberus_cfg, 'clean_ce_weight', 1.0))
        poison_ce_weight = float(getattr(
            self.cerberus_cfg, 'poison_ce_weight', 1.0))
        clean_target_suppression_weight = float(getattr(
            self.cerberus_cfg, 'clean_target_suppression_weight', 0.0))
        clean_target_margin = float(getattr(
            self.cerberus_cfg, 'clean_target_margin', 0.5))
        target_label = int(self._cfg.attack.target_label_ind)
        internal_epochs = max(
            1, int(getattr(self.cerberus_cfg, 'internal_poison_epochs', 1)))
        preserve_clean_batches = bool(getattr(
            self.cerberus_cfg, 'preserve_clean_batches', False))

        self.mlp_classifier.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_clean_ce_loss = 0.0
        total_poison_ce_loss = 0.0
        total_clean_target_suppression_loss = 0.0
        total_anchor_loss = 0.0
        total_peer_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_clean_correct = 0
        total_clean_samples = 0
        total_poison_correct = 0
        total_poison_samples = 0
        total_clean_target_predictions = 0

        for _ in range(internal_epochs):
            for clean_features, clean_labels in self.augmented_loader:
                clean_features = clean_features.to(self.device)
                clean_labels = clean_labels.to(self.device)
                poison_x, poison_y = self._sample_cerberus_poison_batch(
                    poison_features, poison_labels, clean_features.size(0))
                if poison_x is None:
                    continue

                optimizer.zero_grad()
                if preserve_clean_batches:
                    clean_outputs = self.mlp_classifier(clean_features)
                    poison_outputs = self.mlp_classifier(poison_x)
                    clean_ce_loss = criterion(clean_outputs, clean_labels)
                    poison_ce_loss = criterion(poison_outputs, poison_y)
                    ce_loss = (
                        clean_ce_weight * clean_ce_loss +
                        poison_ce_weight * poison_ce_loss)
                    outputs = torch.cat([poison_outputs, clean_outputs], dim=0)
                    labels = torch.cat([poison_y, clean_labels], dim=0)
                    clean_pred = torch.argmax(clean_outputs, dim=1)
                    poison_pred = torch.argmax(poison_outputs, dim=1)
                    clean_eval_labels = clean_labels
                    clean_outputs_for_suppression = clean_outputs
                else:
                    features = clean_features.clone()
                    labels = clean_labels.clone()
                    poison_num = poison_x.size(0)
                    features[:poison_num] = poison_x
                    labels[:poison_num] = poison_y
                    outputs = self.mlp_classifier(features)
                    poison_ce_loss = criterion(
                        outputs[:poison_num], labels[:poison_num])
                    if poison_num < outputs.size(0):
                        clean_ce_loss = criterion(
                            outputs[poison_num:], labels[poison_num:])
                    else:
                        clean_ce_loss = torch.tensor(
                            0.0, device=self.device)
                    ce_loss = (
                        clean_ce_weight * clean_ce_loss +
                        poison_ce_weight * poison_ce_loss)
                    poison_pred = torch.argmax(outputs[:poison_num], dim=1)
                    clean_pred = torch.argmax(outputs[poison_num:], dim=1)
                    clean_eval_labels = labels[poison_num:]
                    clean_outputs_for_suppression = outputs[poison_num:]
                clean_target_suppression_loss = torch.tensor(
                    0.0, device=self.device)
                if clean_target_suppression_weight > 0.0 and \
                        0 <= target_label < clean_outputs_for_suppression.size(1):
                    non_target_mask = clean_eval_labels != target_label
                    if non_target_mask.any():
                        non_target_outputs = clean_outputs_for_suppression[
                            non_target_mask]
                        non_target_labels = clean_eval_labels[non_target_mask]
                        target_logits = non_target_outputs[:, target_label]
                        true_logits = non_target_outputs.gather(
                            1, non_target_labels.view(-1, 1)).squeeze(1)
                        clean_target_suppression_loss = F.relu(
                            target_logits - true_logits +
                            clean_target_margin).mean()
                anchor_loss = self._cerberus_anchor_distance(anchor_state)
                peer_loss = self._cerberus_peer_cosine()
                loss = ce_loss + alpha_loss * anchor_loss + \
                    beta_loss * peer_loss + \
                    clean_target_suppression_weight * \
                    clean_target_suppression_loss
                loss.backward()
                optimizer.step()

                batch_size = outputs.size(0)
                total_loss += loss.item() * batch_size
                total_ce_loss += ce_loss.item() * batch_size
                total_clean_ce_loss += clean_ce_loss.item() * batch_size
                total_poison_ce_loss += poison_ce_loss.item() * batch_size
                total_clean_target_suppression_loss += \
                    clean_target_suppression_loss.item() * batch_size
                total_anchor_loss += anchor_loss.item() * batch_size
                total_peer_loss += peer_loss.item() * batch_size
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += batch_size
                total_poison_correct += (
                    poison_pred == poison_y).sum().item()
                total_poison_samples += poison_y.numel()
                if clean_pred.numel() > 0:
                    total_clean_correct += (
                        clean_pred == clean_eval_labels
                    ).sum().item()
                    total_clean_target_predictions += clean_pred.eq(
                        target_label).sum().item()
                    total_clean_samples += clean_pred.numel()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_ce_loss = total_ce_loss / total_samples if total_samples > 0 else 0
        avg_clean_ce_loss = (
            total_clean_ce_loss / total_samples if total_samples > 0 else 0)
        avg_poison_ce_loss = (
            total_poison_ce_loss / total_samples if total_samples > 0 else 0)
        avg_clean_target_suppression_loss = (
            total_clean_target_suppression_loss / total_samples
            if total_samples > 0 else 0)
        avg_anchor_loss = (
            total_anchor_loss / total_samples if total_samples > 0 else 0)
        avg_peer_loss = (
            total_peer_loss / total_samples if total_samples > 0 else 0)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        clean_accuracy = (
            total_clean_correct / total_clean_samples
            if total_clean_samples > 0 else 0)
        poison_accuracy = (
            total_poison_correct / total_poison_samples
            if total_poison_samples > 0 else 0)
        clean_target_rate = (
            total_clean_target_predictions / total_clean_samples
            if total_clean_samples > 0 else 0)

        logger.info(
            f"Client {self.ID}: CERBERUS train loss={avg_loss:.4f} "
            f"(CE={avg_ce_loss:.4f}, clean_CE={avg_clean_ce_loss:.4f}, "
            f"poison_CE={avg_poison_ce_loss:.4f}, "
            f"clean_target_supp={avg_clean_target_suppression_loss:.4f}, "
            f"anchor={avg_anchor_loss:.4f}, "
            f"peer={avg_peer_loss:.4f}), accuracy={accuracy:.4f}, "
            f"clean_acc={clean_accuracy:.4f}, "
            f"clean_target_rate={clean_target_rate:.4f}, "
            f"poison_acc={poison_accuracy:.4f}")

        model_para = copy.deepcopy(self.mlp_classifier.state_dict())
        if bool(getattr(self.cerberus_cfg,
                        'constrain_update_to_anchor', False)):
            gamma = float(getattr(
                self.cerberus_cfg, 'anchor_residual_gamma', 0.5))
            gamma = min(1.0, max(0.0, gamma))
            constrained_para = copy.deepcopy(model_para)
            for name, value in model_para.items():
                if name not in anchor_state or not isinstance(value, torch.Tensor):
                    continue
                anchor_value = anchor_state[name].to(value.device)
                if tuple(anchor_value.shape) != tuple(value.shape):
                    continue
                constrained_para[name] = anchor_value + gamma * (
                    value - anchor_value)
            model_para = constrained_para
            logger.info(
                f"Client {self.ID}: CERBERUS constrained update to "
                f"clean anchor with gamma={gamma:.4f}")
        self.cerberus_latest_meta.update({
            'alpha_loss': float(alpha_loss),
            'beta_loss': float(beta_loss),
            'clean_ce_weight': float(clean_ce_weight),
            'poison_ce_weight': float(poison_ce_weight),
            'clean_target_suppression_weight': float(
                clean_target_suppression_weight),
            'clean_target_margin': float(clean_target_margin),
            'anchor_loss': float(avg_anchor_loss),
            'peer_loss': float(avg_peer_loss),
            'train_loss': float(avg_loss),
            'train_clean_ce_loss': float(avg_clean_ce_loss),
            'train_poison_ce_loss': float(avg_poison_ce_loss),
            'train_clean_target_suppression_loss': float(
                avg_clean_target_suppression_loss),
            'train_acc': float(accuracy),
            'train_clean_acc': float(clean_accuracy),
            'train_clean_target_rate': float(clean_target_rate),
            'train_poison_acc': float(poison_accuracy),
        })
        if bool(getattr(self.cerberus_cfg, 'share_model_meta', False)):
            self.cerberus_latest_meta['model'] = copy.deepcopy(model_para)

        results = {
            'train_loss': avg_loss,
            'train_ce_loss': avg_ce_loss,
            'train_clean_ce_loss': avg_clean_ce_loss,
            'train_poison_ce_loss': avg_poison_ce_loss,
            'train_clean_target_suppression_loss':
            avg_clean_target_suppression_loss,
            'train_cerberus_anchor_loss': avg_anchor_loss,
            'train_cerberus_peer_loss': avg_peer_loss,
            'train_acc': accuracy,
            'train_clean_acc': clean_accuracy,
            'train_clean_target_rate': clean_target_rate,
            'train_poison_acc': poison_accuracy,
            'train_total': total_samples
        }

        # FedAvg sample_size should reflect the clean local data scale, not
        # poisoned replay volume or internal CERBERUS epochs.
        clean_weight_samples = total_clean_samples
        if internal_epochs > 0:
            clean_weight_samples = int(round(
                float(total_clean_samples) / float(internal_epochs)))
        clean_weight_samples = max(1, clean_weight_samples)
        self.cerberus_latest_meta['aggregation_sample_size'] = int(
            clean_weight_samples)
        results['aggregation_sample_size'] = int(clean_weight_samples)
        return clean_weight_samples, model_para, results

    def _optimize_sabre_trigger(self, base_dataset, candidate_indices,
                                round_idx):
        if self.mlp_classifier is None or self.sabre_trigger is None or \
                self.sabre_mask is None or base_dataset is None or \
                not candidate_indices:
            return

        target_label = int(self._cfg.attack.target_label_ind)
        steps = max(0, int(getattr(
            self.sabre_cfg, 'trigger_search_steps', 0)))
        if steps <= 0:
            return

        batch_size = max(1, int(getattr(
            self.sabre_cfg, 'trigger_search_batch_size', 8)))
        max_batches = max(1, int(getattr(
            self.sabre_cfg, 'trigger_search_batches', 2)))
        lr = float(getattr(self.sabre_cfg, 'trigger_search_lr', 0.01))
        clip_min = float(getattr(
            self.sabre_cfg, 'trigger_search_clip_min', -0.05))
        clip_max = float(getattr(
            self.sabre_cfg, 'trigger_search_clip_max', 0.05))
        proj_norm = float(getattr(
            self.sabre_cfg, 'trigger_search_proj_norm', 4.0))
        target_margin = float(getattr(
            self.sabre_cfg, 'trigger_search_target_margin', 1.0))
        gain_weight = float(getattr(
            self.sabre_cfg, 'trigger_search_gain_weight', 0.5))
        gain_margin = float(getattr(
            self.sabre_cfg, 'trigger_search_gain_margin', 0.5))
        l2_weight = float(getattr(
            self.sabre_cfg, 'trigger_search_l2_weight', 1e-4))

        search_indices = []
        for base_idx in candidate_indices:
            try:
                _, label = base_dataset[base_idx]
            except Exception:
                continue
            label_value = int(label.item()) if torch.is_tensor(label) else int(label)
            if label_value != target_label:
                search_indices.append(base_idx)
            if len(search_indices) >= batch_size * max_batches:
                break
        if not search_indices:
            search_indices = list(candidate_indices[:batch_size * max_batches])
        if not search_indices:
            return

        self._load_feature_extractor()
        if self.feature_extractor_type == 'clip' and self.clip_model is not None:
            self.clip_model.eval()
        if self.feature_extractor_type == 'cnn' and self.cnn_extractor is not None:
            self.cnn_extractor.eval()
        if self.feature_extractor_type == 'timm' and self.timm_extractor is not None:
            self.timm_extractor.eval()

        extractor_modules = [
            module for module in
            (self.clip_model, self.cnn_extractor, self.timm_extractor)
            if module is not None
        ]
        saved_requires_grad = []
        for module in extractor_modules:
            for param in module.parameters():
                saved_requires_grad.append((param, param.requires_grad))
                param.requires_grad_(False)
        for param in self.mlp_classifier.parameters():
            saved_requires_grad.append((param, param.requires_grad))
            param.requires_grad_(False)

        trigger_base = self.sabre_trigger.detach().clone()
        trigger = trigger_base.clone().requires_grad_(True)
        mask = self.sabre_mask.detach()
        optimizer = torch.optim.Adam([trigger], lr=lr)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_ce = 0.0
        total_margin = 0.0
        total_gain = 0.0
        total_batches = 0
        try:
            for step in range(steps):
                offset = (step * batch_size) % len(search_indices)
                if offset + batch_size <= len(search_indices):
                    batch_indices = search_indices[offset:offset + batch_size]
                else:
                    batch_indices = search_indices[offset:] + \
                        search_indices[:batch_size - (len(search_indices) - offset)]

                images = []
                labels = []
                for base_idx in batch_indices:
                    image, label = base_dataset[base_idx]
                    images.append(image)
                    labels.append(
                        int(label.item()) if torch.is_tensor(label)
                        else int(label))
                images = torch.stack(images).to(self.device)
                labels = torch.as_tensor(labels,
                                         dtype=torch.long,
                                         device=self.device)
                target_labels = torch.full_like(labels, target_label)

                optimizer.zero_grad()
                poisoned_images = self._apply_sabre_trigger(
                    images, trigger=trigger, mask=mask)
                poison_features = self._extractor_forward(
                    poisoned_images, allow_input_grad=True).float()
                poison_logits = self.mlp_classifier(poison_features)
                poison_ce = criterion(poison_logits, target_labels)

                target_logits = poison_logits[:, target_label]
                other_logits = poison_logits.clone()
                if 0 <= target_label < other_logits.size(1):
                    other_logits[:, target_label] = -1e9
                max_other_logits = other_logits.max(dim=1).values
                target_margin_loss = F.relu(
                    max_other_logits - target_logits + target_margin).mean()

                with torch.no_grad():
                    clean_features = self._extractor_forward(images).float()
                    clean_logits = self.mlp_classifier(clean_features)
                    clean_target_logits = clean_logits[:, target_label]
                gain_loss = F.relu(
                    gain_margin - (target_logits - clean_target_logits)).mean()
                l2_loss = torch.norm(trigger * mask, p=2)
                loss = poison_ce + target_margin_loss + \
                    gain_weight * gain_loss + l2_weight * l2_loss
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    trigger.mul_(mask).add_(trigger_base * (1.0 - mask))
                    trigger.clamp_(clip_min, clip_max)
                    if proj_norm > 0:
                        delta = trigger * mask
                        delta_norm = torch.norm(delta, p=2)
                        if delta_norm > proj_norm:
                            delta = delta * (proj_norm / (delta_norm + 1e-12))
                            trigger.copy_(delta)
                            trigger.mul_(mask).add_(
                                trigger_base * (1.0 - mask))

                total_loss += loss.item()
                total_ce += poison_ce.item()
                total_margin += target_margin_loss.item()
                total_gain += gain_loss.item()
                total_batches += 1

            self.sabre_trigger = trigger.detach()
        finally:
            for param, requires_grad in saved_requires_grad:
                param.requires_grad_(requires_grad)

        if total_batches <= 0:
            return

        eval_images = []
        eval_labels = []
        for base_idx in search_indices[:batch_size]:
            image, label = base_dataset[base_idx]
            eval_images.append(image)
            eval_labels.append(
                int(label.item()) if torch.is_tensor(label) else int(label))
        eval_images = torch.stack(eval_images).to(self.device)
        eval_labels = torch.as_tensor(eval_labels,
                                      dtype=torch.long,
                                      device=self.device)
        with torch.no_grad():
            clean_features = self._extractor_forward(eval_images).float()
            clean_logits = self.mlp_classifier(clean_features)
            poisoned_images = self._apply_sabre_trigger(eval_images)
            poison_features = self._extractor_forward(poisoned_images).float()
            poison_logits = self.mlp_classifier(poison_features)
            clean_preds = torch.argmax(clean_logits, dim=1)
            poison_preds = torch.argmax(poison_logits, dim=1)
            clean_target_rate = clean_preds.eq(target_label).float().mean()
            poison_target_rate = poison_preds.eq(target_label).float().mean()
            target_gain = (
                poison_logits[:, target_label] -
                clean_logits[:, target_label]).mean()
            eval_target_labels = torch.full_like(eval_labels, target_label)
            eval_ce = criterion(poison_logits, eval_target_labels)

        trigger_norm = torch.norm(self.sabre_trigger * self.sabre_mask,
                                  p=2).item()
        self.sabre_latest_meta.update({
            'trigger_search_steps': int(steps),
            'trigger_search_batches': int(max_batches),
            'trigger_search_loss': float(total_loss / total_batches),
            'trigger_search_ce': float(total_ce / total_batches),
            'trigger_search_margin': float(total_margin / total_batches),
            'trigger_search_gain_loss': float(total_gain / total_batches),
            'trigger_search_eval_ce': float(eval_ce.item()),
            'trigger_search_clean_target_rate': float(
                clean_target_rate.item()),
            'trigger_search_poison_target_rate': float(
                poison_target_rate.item()),
            'trigger_search_target_logit_gain': float(target_gain.item()),
            'trigger_norm': float(trigger_norm),
        })
        logger.info(
            f"Client {self.ID}: SABRE trigger search round {round_idx} "
            f"steps={steps}, batches={max_batches}, "
            f"loss={total_loss / total_batches:.4f}, "
            f"CE={total_ce / total_batches:.4f}, "
            f"poison_target_rate={poison_target_rate.item():.4f}, "
            f"clean_target_rate={clean_target_rate.item():.4f}, "
            f"target_logit_gain={target_gain.item():.4f}, "
            f"trigger_norm={trigger_norm:.4f}")

    def _build_sabre_poison_feature_pool(self, round_idx):
        if self.mlp_classifier is None:
            return None, None

        target_label = int(self._cfg.attack.target_label_ind)
        base_dataset, subset_indices = self._get_train_dataset_base()
        poison_ratio = float(getattr(self._cfg.attack, 'poison_ratio', 0.1))

        if base_dataset is None or not subset_indices:
            if self.augmented_features is None or len(self.augmented_features) == 0:
                logger.warning(
                    f"Client {self.ID}: SABRE could not find local data "
                    "for poisoned feature construction")
                return None, None
            poison_count = max(1, int(len(self.augmented_features) * poison_ratio))
            poison_count = min(poison_count, len(self.augmented_features))
            rng = np.random.RandomState(
                int(self._cfg.seed) + int(round_idx) + int(self.ID) * 1009)
            selected = rng.choice(len(self.augmented_features),
                                  size=poison_count,
                                  replace=False)
            poison_features = torch.from_numpy(
                self.augmented_features[selected]).float()
            poison_labels = torch.full((poison_count, ),
                                       target_label,
                                       dtype=torch.long)
            return poison_features, poison_labels

        poison_count = max(1, int(len(subset_indices) * poison_ratio))
        max_poison = int(getattr(self.sabre_cfg, 'max_poison_samples', 0))
        if max_poison > 0:
            poison_count = min(poison_count, max_poison)
        poison_count = min(poison_count, len(subset_indices))
        rng = np.random.RandomState(
            int(self._cfg.seed) + int(round_idx) + int(self.ID) * 1009)
        selected_indices = rng.choice(subset_indices,
                                      size=poison_count,
                                      replace=False).tolist()

        self._load_feature_extractor()
        first_image, _ = base_dataset[selected_indices[0]]
        self._ensure_sabre_trigger(first_image.to(self.device))
        self._optimize_sabre_trigger(base_dataset, selected_indices, round_idx)
        self.sabre_latest_meta.update({
            'trigger': self.sabre_trigger.detach().cpu(),
            'mask': self.sabre_mask.detach().cpu(),
            'trigger_mode': 'additive_full_image',
        })

        if self.feature_extractor_type == 'clip' and self.clip_model is not None:
            self.clip_model.eval()
        if self.feature_extractor_type == 'cnn' and self.cnn_extractor is not None:
            self.cnn_extractor.eval()
        if self.feature_extractor_type == 'timm' and self.timm_extractor is not None:
            self.timm_extractor.eval()

        poison_features = []
        batch_size = max(1, int(getattr(self.ggeur_cfg, 'extract_batch_size', 64)))
        with torch.no_grad():
            for start in range(0, len(selected_indices), batch_size):
                batch_indices = selected_indices[start:start + batch_size]
                images = []
                for base_idx in batch_indices:
                    image, _ = base_dataset[base_idx]
                    images.append(image)
                images = torch.stack(images).to(self.device)
                poisoned_images = self._apply_sabre_trigger(images)
                features = self._extractor_forward(poisoned_images).float()
                poison_features.append(features.detach().cpu())

        if not poison_features:
            return None, None

        poison_features = torch.cat(poison_features, dim=0)
        poison_labels = torch.full((poison_features.shape[0], ),
                                   target_label,
                                   dtype=torch.long)
        repeat = max(1, int(getattr(self.sabre_cfg, 'poison_feature_repeat', 1)))
        if repeat > 1:
            poison_features = poison_features.repeat((repeat, 1))
            poison_labels = poison_labels.repeat(repeat)

        self.sabre_latest_meta.update({
            'poisoned_samples': int(poison_features.shape[0]),
            'target_label': int(target_label),
            'trigger_norm': float(torch.norm(
                self.sabre_trigger * self.sabre_mask, p=2).item()),
            'trigger_linf': float(torch.max(torch.abs(
                self.sabre_trigger * self.sabre_mask)).item()),
        })
        return poison_features, poison_labels

    def _train_sabre_clean_anchor(self):
        anchor_model = copy.deepcopy(self.mlp_classifier)
        anchor_model.train()

        clean_lr = float(getattr(
            self.sabre_cfg, 'clean_anchor_lr',
            getattr(self.sabre_cfg, 'poison_lr',
                    self._cfg.train.optimizer.lr)))
        clean_epochs = max(
            1, int(getattr(self.sabre_cfg, 'clean_anchor_epochs', 1)))

        optimizer = torch.optim.Adam(anchor_model.parameters(), lr=clean_lr)
        criterion = nn.CrossEntropyLoss()
        for _ in range(clean_epochs):
            for features, labels in self.augmented_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = anchor_model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        anchor_state = {
            name: param.detach().clone()
            for name, param in anchor_model.named_parameters()
            if param.requires_grad
        }
        return anchor_state

    def _sabre_anchor_distance(self, anchor_state):
        distance = torch.tensor(0.0, device=self.device)
        for name, param in self.mlp_classifier.named_parameters():
            if not param.requires_grad or name not in anchor_state:
                continue
            anchor_param = anchor_state[name].to(param.device)
            distance = distance + torch.norm(param - anchor_param, p=2) ** 2
        return distance

    def _sample_sabre_poison_batch(self, poison_features, poison_labels,
                                   batch_size):
        pool_size = poison_features.shape[0]
        if pool_size == 0:
            return None, None

        poison_per_batch = int(getattr(self.sabre_cfg,
                                       'poisoning_per_batch', 0))
        if poison_per_batch <= 0:
            poison_ratio = float(getattr(self._cfg.attack, 'poison_ratio', 0.1))
            poison_per_batch = max(1, int(batch_size * poison_ratio))
        poison_per_batch = min(max(1, poison_per_batch), batch_size)

        replace = pool_size < poison_per_batch
        indices = np.random.choice(pool_size,
                                   size=poison_per_batch,
                                   replace=replace)
        poison_x = poison_features[indices].to(self.device)
        poison_y = poison_labels[indices].to(self.device)
        return poison_x, poison_y

    def _train_sabre_on_augmented_data(self, round_idx):
        if self.augmented_loader is None or self.mlp_classifier is None:
            return 0, {}, {}

        self.sabre_latest_meta = {
            'active': True,
            'client_id': int(self.ID),
            'round': int(round_idx),
            'target_label': int(self._cfg.attack.target_label_ind),
        }

        poison_features, poison_labels = self._build_sabre_poison_feature_pool(
            round_idx)
        if poison_features is None or poison_labels is None:
            logger.warning(
                f"Client {self.ID}: SABRE poison pool is empty; "
                "falling back to clean augmented training")
            self.sabre_latest_meta['active'] = False
            return self._train_on_augmented_data()

        anchor_state = self._train_sabre_clean_anchor()
        poison_lr = float(getattr(self.sabre_cfg,
                                  'poison_lr',
                                  self._cfg.train.optimizer.lr))
        optimizer_name = str(getattr(self.sabre_cfg,
                                     'poison_optimizer',
                                     'adam')).lower()
        if optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.mlp_classifier.parameters(),
                                        lr=poison_lr)
        else:
            optimizer = torch.optim.Adam(self.mlp_classifier.parameters(),
                                         lr=poison_lr)

        criterion = nn.CrossEntropyLoss()
        anchor_loss_weight = float(getattr(self.sabre_cfg,
                                           'anchor_loss_weight', 0.01))
        clean_ce_weight = float(getattr(
            self.sabre_cfg, 'clean_ce_weight', 1.0))
        poison_ce_weight = float(getattr(
            self.sabre_cfg, 'poison_ce_weight', 1.0))
        clean_target_suppression_weight = float(getattr(
            self.sabre_cfg, 'clean_target_suppression_weight', 0.0))
        clean_target_margin = float(getattr(
            self.sabre_cfg, 'clean_target_margin', 0.5))
        target_label = int(self._cfg.attack.target_label_ind)
        internal_epochs = max(
            1, int(getattr(self.sabre_cfg, 'internal_poison_epochs', 1)))
        preserve_clean_batches = bool(getattr(
            self.sabre_cfg, 'preserve_clean_batches', True))

        self.mlp_classifier.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_clean_ce_loss = 0.0
        total_poison_ce_loss = 0.0
        total_clean_target_suppression_loss = 0.0
        total_anchor_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_clean_correct = 0
        total_clean_samples = 0
        total_poison_correct = 0
        total_poison_samples = 0
        total_clean_target_predictions = 0

        for _ in range(internal_epochs):
            for clean_features, clean_labels in self.augmented_loader:
                clean_features = clean_features.to(self.device)
                clean_labels = clean_labels.to(self.device)
                poison_x, poison_y = self._sample_sabre_poison_batch(
                    poison_features, poison_labels, clean_features.size(0))
                if poison_x is None:
                    continue

                optimizer.zero_grad()
                if preserve_clean_batches:
                    clean_outputs = self.mlp_classifier(clean_features)
                    poison_outputs = self.mlp_classifier(poison_x)
                    clean_ce_loss = criterion(clean_outputs, clean_labels)
                    poison_ce_loss = criterion(poison_outputs, poison_y)
                    ce_loss = (
                        clean_ce_weight * clean_ce_loss +
                        poison_ce_weight * poison_ce_loss)
                    outputs = torch.cat([poison_outputs, clean_outputs], dim=0)
                    labels = torch.cat([poison_y, clean_labels], dim=0)
                    clean_pred = torch.argmax(clean_outputs, dim=1)
                    poison_pred = torch.argmax(poison_outputs, dim=1)
                    clean_eval_labels = clean_labels
                    clean_outputs_for_suppression = clean_outputs
                else:
                    features = clean_features.clone()
                    labels = clean_labels.clone()
                    poison_num = poison_x.size(0)
                    features[:poison_num] = poison_x
                    labels[:poison_num] = poison_y
                    outputs = self.mlp_classifier(features)
                    poison_ce_loss = criterion(
                        outputs[:poison_num], labels[:poison_num])
                    if poison_num < outputs.size(0):
                        clean_ce_loss = criterion(
                            outputs[poison_num:], labels[poison_num:])
                    else:
                        clean_ce_loss = torch.tensor(
                            0.0, device=self.device)
                    ce_loss = (
                        clean_ce_weight * clean_ce_loss +
                        poison_ce_weight * poison_ce_loss)
                    poison_pred = torch.argmax(outputs[:poison_num], dim=1)
                    clean_pred = torch.argmax(outputs[poison_num:], dim=1)
                    clean_eval_labels = labels[poison_num:]
                    clean_outputs_for_suppression = outputs[poison_num:]

                clean_target_suppression_loss = torch.tensor(
                    0.0, device=self.device)
                if clean_target_suppression_weight > 0.0 and \
                        clean_outputs_for_suppression.numel() > 0 and \
                        0 <= target_label < clean_outputs_for_suppression.size(1):
                    non_target_mask = clean_eval_labels != target_label
                    if non_target_mask.any():
                        non_target_outputs = clean_outputs_for_suppression[
                            non_target_mask]
                        non_target_labels = clean_eval_labels[non_target_mask]
                        target_logits = non_target_outputs[:, target_label]
                        true_logits = non_target_outputs.gather(
                            1, non_target_labels.view(-1, 1)).squeeze(1)
                        clean_target_suppression_loss = F.relu(
                            target_logits - true_logits +
                            clean_target_margin).mean()

                anchor_loss = self._sabre_anchor_distance(anchor_state)
                loss = ce_loss + anchor_loss_weight * anchor_loss + \
                    clean_target_suppression_weight * \
                    clean_target_suppression_loss
                loss.backward()
                optimizer.step()

                batch_size = outputs.size(0)
                total_loss += loss.item() * batch_size
                total_ce_loss += ce_loss.item() * batch_size
                total_clean_ce_loss += clean_ce_loss.item() * batch_size
                total_poison_ce_loss += poison_ce_loss.item() * batch_size
                total_clean_target_suppression_loss += \
                    clean_target_suppression_loss.item() * batch_size
                total_anchor_loss += anchor_loss.item() * batch_size
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += batch_size
                total_poison_correct += (
                    poison_pred == poison_y).sum().item()
                total_poison_samples += poison_y.numel()
                if clean_pred.numel() > 0:
                    total_clean_correct += (
                        clean_pred == clean_eval_labels
                    ).sum().item()
                    total_clean_target_predictions += clean_pred.eq(
                        target_label).sum().item()
                    total_clean_samples += clean_pred.numel()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_ce_loss = total_ce_loss / total_samples if total_samples > 0 else 0
        avg_clean_ce_loss = (
            total_clean_ce_loss / total_samples if total_samples > 0 else 0)
        avg_poison_ce_loss = (
            total_poison_ce_loss / total_samples if total_samples > 0 else 0)
        avg_clean_target_suppression_loss = (
            total_clean_target_suppression_loss / total_samples
            if total_samples > 0 else 0)
        avg_anchor_loss = (
            total_anchor_loss / total_samples if total_samples > 0 else 0)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        clean_accuracy = (
            total_clean_correct / total_clean_samples
            if total_clean_samples > 0 else 0)
        poison_accuracy = (
            total_poison_correct / total_poison_samples
            if total_poison_samples > 0 else 0)
        clean_target_rate = (
            total_clean_target_predictions / total_clean_samples
            if total_clean_samples > 0 else 0)

        logger.info(
            f"Client {self.ID}: SABRE train loss={avg_loss:.4f} "
            f"(CE={avg_ce_loss:.4f}, clean_CE={avg_clean_ce_loss:.4f}, "
            f"poison_CE={avg_poison_ce_loss:.4f}, "
            f"clean_target_supp={avg_clean_target_suppression_loss:.4f}, "
            f"anchor={avg_anchor_loss:.4f}), accuracy={accuracy:.4f}, "
            f"clean_acc={clean_accuracy:.4f}, "
            f"clean_target_rate={clean_target_rate:.4f}, "
            f"poison_acc={poison_accuracy:.4f}")

        model_para = copy.deepcopy(self.mlp_classifier.state_dict())
        if bool(getattr(self.sabre_cfg,
                        'constrain_update_to_anchor', False)):
            gamma = float(getattr(
                self.sabre_cfg, 'anchor_residual_gamma', 0.5))
            gamma = min(1.0, max(0.0, gamma))
            constrained_para = copy.deepcopy(model_para)
            for name, value in model_para.items():
                if name not in anchor_state or not isinstance(value, torch.Tensor):
                    continue
                anchor_value = anchor_state[name].to(value.device)
                if tuple(anchor_value.shape) != tuple(value.shape):
                    continue
                constrained_para[name] = anchor_value + gamma * (
                    value - anchor_value)
            model_para = constrained_para
            logger.info(
                f"Client {self.ID}: SABRE constrained update to "
                f"clean anchor with gamma={gamma:.4f}")

        self.sabre_latest_meta.update({
            'anchor_loss_weight': float(anchor_loss_weight),
            'clean_ce_weight': float(clean_ce_weight),
            'poison_ce_weight': float(poison_ce_weight),
            'clean_target_suppression_weight': float(
                clean_target_suppression_weight),
            'clean_target_margin': float(clean_target_margin),
            'anchor_loss': float(avg_anchor_loss),
            'train_loss': float(avg_loss),
            'train_clean_ce_loss': float(avg_clean_ce_loss),
            'train_poison_ce_loss': float(avg_poison_ce_loss),
            'train_clean_target_suppression_loss': float(
                avg_clean_target_suppression_loss),
            'train_acc': float(accuracy),
            'train_clean_acc': float(clean_accuracy),
            'train_clean_target_rate': float(clean_target_rate),
            'train_poison_acc': float(poison_accuracy),
        })

        clean_weight_samples = total_clean_samples
        if internal_epochs > 0:
            clean_weight_samples = int(round(
                float(total_clean_samples) / float(internal_epochs)))
        clean_weight_samples = max(1, clean_weight_samples)
        self.sabre_latest_meta['aggregation_sample_size'] = int(
            clean_weight_samples)
        results = {
            'train_loss': avg_loss,
            'train_ce_loss': avg_ce_loss,
            'train_clean_ce_loss': avg_clean_ce_loss,
            'train_poison_ce_loss': avg_poison_ce_loss,
            'train_clean_target_suppression_loss':
            avg_clean_target_suppression_loss,
            'train_sabre_anchor_loss': avg_anchor_loss,
            'train_acc': accuracy,
            'train_clean_acc': clean_accuracy,
            'train_clean_target_rate': clean_target_rate,
            'train_poison_acc': poison_accuracy,
            'train_total': total_samples,
            'aggregation_sample_size': int(clean_weight_samples),
        }
        return clean_weight_samples, model_para, results

    def _load_cnn_extractor(self):
        """Load CNN feature extractor"""
        if self.cnn_extractor is not None:
            return

        try:
            from federatedscope.contrib.model.ggeur_cnn_extractor import CNNFeatureExtractor

            model_name = getattr(self.ggeur_cfg, 'cnn_backbone', 'convnext_base')
            pretrained = getattr(self.ggeur_cfg, 'cnn_pretrained', True)
            checkpoint_path = getattr(self.ggeur_cfg, 'cnn_checkpoint_path', '')
            freeze = getattr(self.ggeur_cfg, 'freeze_backbone', True)

            self.cnn_extractor = CNNFeatureExtractor(
                model_name=model_name,
                pretrained=pretrained,
                freeze=freeze,
                checkpoint_path=checkpoint_path,
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
            visual_dim = getattr(getattr(self.clip_model, 'visual', None),
                                 'output_dim', None)
            if visual_dim is None and hasattr(self.clip_model,
                                              'text_projection'):
                visual_dim = int(self.clip_model.text_projection.shape[1])
            if visual_dim is not None:
                self.embedding_dim = int(visual_dim)
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

        dim_str = f"d{int(self.embedding_dim)}"
        if domain:
            cache_filename = (
                f"{dataset_name}_{domain}_{prefix}_{model_str}_{dim_str}.npz")
        else:
            cache_filename = f"{dataset_name}_{prefix}_{model_str}_{dim_str}.npz"

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

    def _is_valid_feature_vector(self, feat):
        """Check whether a cached/extracted feature matches current dim."""
        if feat is None:
            return False

        feat_arr = np.asarray(feat)
        if feat_arr.ndim != 1:
            return False

        expected_dim = int(getattr(self, 'embedding_dim',
                                   getattr(self.ggeur_cfg,
                                           'embedding_dim', 0)))
        if expected_dim > 0 and feat_arr.shape[0] != expected_dim:
            return False

        return True

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

        # Load before cache lookup so embedding_dim reflects the real extractor
        # output, not the config default from a previous backbone.
        self._load_feature_extractor()

        cache_path = self._get_feature_cache_path(domain)

        # Load existing cache
        feature_cache = self._load_feature_cache(cache_path)
        if feature_cache:
            valid_cache = {
                path: feat
                for path, feat in feature_cache.items()
                if self._is_valid_feature_vector(feat)
            }
            dropped = len(feature_cache) - len(valid_cache)
            if dropped:
                logger.warning(
                    f"Client {self.ID}: Dropped {dropped} cached features "
                    f"with wrong dim; expected {self.embedding_dim}")
            feature_cache = valid_cache
        cache_updated = False

        # Check if base dataset has image paths (PACS, Office-Home style)
        has_paths = hasattr(base_dataset, 'data') and len(base_dataset.data) > 0 and isinstance(base_dataset.data[0], str)

        self.local_features = {}
        self.local_labels = {}

        # QPS tracking
        _qps_samples = 0
        _qps_time = 0.0

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
                    if not self._is_valid_feature_vector(feat):
                        logger.warning(
                            f"Client {self.ID}: Skip invalid cached feature "
                            f"for {img_path} with shape "
                            f"{np.asarray(feat).shape}, expected dim "
                            f"{self.embedding_dim}")
                        paths_to_extract.append(img_path)
                        indices_to_extract.append(local_idx)
                        base_indices_to_extract.append(base_idx)
                        continue
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
                # Create a mini dataloader for samples to extract
                batch_size = getattr(self.ggeur_cfg, 'extract_batch_size', 64)
                use_fp16 = getattr(self.ggeur_cfg, 'use_fp16_extraction', True) and torch.cuda.is_available()
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
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
                        _t0 = time.time()
                        if self.feature_extractor_type == 'cnn':
                            features = self.cnn_extractor(images)
                        elif self.feature_extractor_type == 'timm':
                            features = self.timm_extractor(images)
                        else:
                            features = self.clip_model.encode_image(images)
                        _qps_samples += len(images)
                        _qps_time += time.time() - _t0
                        features = features.cpu().numpy()

                        for feat, label, path in zip(features, labels, batch_paths):
                            if not self._is_valid_feature_vector(feat):
                                logger.warning(
                                    f"Client {self.ID}: Skip extracted "
                                    f"feature for {path} with invalid shape "
                                    f"{np.asarray(feat).shape}, expected dim "
                                    f"{self.embedding_dim}")
                                continue
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

            use_fp16 = getattr(self.ggeur_cfg, 'use_fp16_extraction', True) and torch.cuda.is_available()
            extract_batch_size = getattr(self.ggeur_cfg, 'extract_batch_size', 64)
            dataloader = DataLoader(dataset, batch_size=extract_batch_size, shuffle=False)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
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
                    _t0 = time.time()
                    if self.feature_extractor_type == 'cnn':
                        features = self.cnn_extractor(images)
                    elif self.feature_extractor_type == 'timm':
                        features = self.timm_extractor(images)
                    else:
                        features = self.clip_model.encode_image(images)
                    _qps_samples += len(images)
                    _qps_time += time.time() - _t0
                    features = features.cpu().numpy()
                    labels = labels.cpu().numpy()

                    for feat, label in zip(features, labels):
                        if not self._is_valid_feature_vector(feat):
                            logger.warning(
                                f"Client {self.ID}: Skip extracted feature "
                                f"with invalid shape {np.asarray(feat).shape}, "
                                f"expected dim {self.embedding_dim}")
                            continue
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
        if _qps_time > 0:
            qps = _qps_samples / _qps_time
            logger.info(f"Client {self.ID}: Feature extraction QPS={qps:.1f} img/s "
                        f"({_qps_samples} samples in {_qps_time:.2f}s)")
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
            if len(features.shape) != 2:
                logger.warning(
                    f"Client {self.ID}: Skip class {class_idx} with invalid "
                    f"feature shape {features.shape}; expected 2D features.")
                continue
            if int(features.shape[1]) != int(self.embedding_dim):
                logger.warning(
                    f"Client {self.ID}: Skip class {class_idx} with feature "
                    f"dim {features.shape[1]}; expected {self.embedding_dim}.")
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

    def _validate_local_statistics(self):
        """Validate statistics dimensions before sending them to the server."""
        expected_dim = int(self.embedding_dim)
        invalid = []

        for class_idx, mean in self.local_means.items():
            mean_arr = np.asarray(mean)
            cov_arr = np.asarray(self.local_covs.get(class_idx))
            if mean_arr.shape != (expected_dim, ):
                invalid.append(
                    f"class {class_idx}: mean shape {mean_arr.shape}")
                continue
            if cov_arr.shape != (expected_dim, expected_dim):
                invalid.append(f"class {class_idx}: cov shape {cov_arr.shape}")

        if invalid:
            detail = '; '.join(invalid[:5])
            raise ValueError(
                f"Client {self.ID}: Invalid local statistics for "
                f"embedding_dim={expected_dim}. {detail}")

    def _upload_local_statistics(self):
        """Upload local statistics to server"""
        logger.info(f"Client {self.ID}: Uploading local statistics to server...")
        self._validate_local_statistics()

        # Also send prototypes for cross-client augmentation
        prototypes = {}
        for class_idx, mean in self.local_means.items():
            prototypes[class_idx] = mean

        content = {
            'client_id': self.ID,
            'embedding_dim': int(self.embedding_dim),
            'means': self._serialize_array_payload(self.local_means),
            'covs': self._serialize_array_payload(self.local_covs),
            'counts': self.local_counts,
            'prototypes': self._serialize_array_payload(prototypes)
        }
        payload_bytes = self._sizeof_content(content)
        logger.info(
            f"Client {self.ID}: Local statistics payload bytes={payload_bytes} "
            f"(classes={len(self.local_means)}, embedding_dim={self.embedding_dim})")

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

    @staticmethod
    def _serialize_array_payload(payload):
        """Encode array/tensor leaves to compact strings for gRPC transfer."""
        if isinstance(payload, dict):
            return {
                key: GGEURClient._serialize_array_payload(value)
                for key, value in payload.items()
            }
        if isinstance(payload, list):
            return [
                GGEURClient._serialize_array_payload(value)
                for value in payload
            ]
        if isinstance(payload, tuple):
            return [
                GGEURClient._serialize_array_payload(value)
                for value in payload
            ]
        if isinstance(payload, (np.ndarray, torch.Tensor)):
            return GGEURClient._serialize_ndarray_payload(payload)
        return payload

    @staticmethod
    def _serialize_ndarray_payload(payload):
        """Serialize ndarray/tensor as compressed base64 to keep gRPC payloads small."""
        if isinstance(payload, torch.Tensor):
            payload = payload.detach().cpu().numpy()
        array = np.asarray(payload, dtype=np.float16)
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        compressed = zlib.compress(buffer.getvalue(), level=3)
        return 'znp:' + base64.b64encode(compressed).decode('ascii')

    @staticmethod
    def _sizeof_content(content) -> int:
        """Best-effort serialized payload size estimate for system logs."""
        if content is None:
            return 0
        if isinstance(content, str):
            return len(content.encode('utf-8'))
        if isinstance(content, bytes):
            return len(content)
        if isinstance(content, np.ndarray):
            return content.nbytes
        if isinstance(content, torch.Tensor):
            return content.numel() * content.element_size()
        if isinstance(content, dict):
            return sum(GGEURClient._sizeof_content(v)
                       for v in content.values())
        if isinstance(content, (list, tuple)):
            return sum(GGEURClient._sizeof_content(v) for v in content)
        return 0

    def callback_for_global_covariances(self, message: Message):
        """Handle receiving global covariance matrices from server"""
        logger.info(f"Client {self.ID}: Received global covariances from server")

        content = message.content
        self.global_cov_matrices = self._normalize_covariance_mapping(
            content.get('cov_matrices', {}))
        other_prototypes = content.get('other_prototypes', {})
        self.other_prototypes = self._normalize_prototype_mapping(
            self._get_class_value(other_prototypes, self.ID) or {})
        self.global_prototypes = self._normalize_prototype_mapping(
            content.get('global_prototypes', {}))  # For feature alignment

        # Debug logging for received prototypes
        logger.info(f"Client {self.ID}: Received global_prototypes with {len(self.global_prototypes)} classes")
        if self.global_prototypes:
            sample_classes = list(self.global_prototypes.keys())[:3]
            for cls in sample_classes:
                proto = self.global_prototypes[cls]
                if hasattr(proto, 'shape'):
                    logger.info(f"  Class {cls}: shape={proto.shape}, mean={proto.mean():.4f}")

        # Save real local features before augmentation clears them (for PromptFL)
        if self.use_promptfl:
            self.real_local_features = {k: v.copy() for k, v in self.local_features.items()}

        # Perform augmentation
        self._perform_augmentation()

        # Build MLP classifier
        self._build_mlp_classifier()

        # Build PromptFL model if enabled
        if self.use_promptfl:
            self._build_prompt_model()
            self._build_prompt_loader()

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
                content='ready'
            )
        )

    @staticmethod
    def _get_class_value(mapping, class_idx):
        """Read int-keyed payloads after serializers convert keys to strings."""
        if not isinstance(mapping, dict):
            return None
        if class_idx in mapping:
            return mapping[class_idx]
        str_key = str(class_idx)
        if str_key in mapping:
            return mapping[str_key]
        try:
            int_key = int(class_idx)
        except (TypeError, ValueError):
            return None
        return mapping.get(int_key)

    def _normalize_covariance_mapping(self, cov_matrices):
        """Convert received covariance payloads to {int: np.ndarray}."""
        normalized = {}
        if not isinstance(cov_matrices, dict):
            return normalized

        expected_dim = int(self.embedding_dim)
        expected_shape = (expected_dim, expected_dim)
        for class_idx, cov in cov_matrices.items():
            try:
                key = int(class_idx)
            except (TypeError, ValueError):
                logger.warning(
                    f"Client {self.ID}: Skip covariance with invalid class "
                    f"key {class_idx}")
                continue

            cov_arr = self._payload_to_ndarray(cov)
            if cov_arr is None:
                logger.warning(
                    f"Client {self.ID}: Skip covariance for class {key}; "
                    f"payload could not be decoded")
                continue
            cov_arr = np.asarray(cov_arr, dtype=np.float32)
            if cov_arr.shape != expected_shape:
                logger.warning(
                    f"Client {self.ID}: Skip covariance for class {key} "
                    f"with shape {cov_arr.shape}, expected {expected_shape}")
                continue
            normalized[key] = cov_arr

        return normalized

    def _normalize_prototype_mapping(self, prototypes):
        """Convert received prototype payloads to ndarray values keyed by int."""
        normalized = {}
        if not isinstance(prototypes, dict):
            return normalized

        expected_dim = int(self.embedding_dim)
        expected_shape = (expected_dim, )
        for class_idx, value in prototypes.items():
            try:
                key = int(class_idx)
            except (TypeError, ValueError):
                logger.warning(
                    f"Client {self.ID}: Skip prototype with invalid class "
                    f"key {class_idx}")
                continue

            if (isinstance(value, list) and value and isinstance(
                    value[0], (list, tuple, np.ndarray, str, bytes))):
                proto_list = []
                for proto in value:
                    proto_arr = self._payload_to_ndarray(proto)
                    if proto_arr is None:
                        logger.warning(
                            f"Client {self.ID}: Skip prototype for class "
                            f"{key}; payload could not be decoded")
                        continue
                    proto_arr = np.asarray(proto_arr, dtype=np.float32)
                    if proto_arr.shape == expected_shape:
                        proto_list.append(proto_arr)
                    else:
                        logger.warning(
                            f"Client {self.ID}: Skip prototype for class "
                            f"{key} with shape {proto_arr.shape}, expected "
                            f"{expected_shape}")
                normalized[key] = proto_list
            else:
                proto_arr = self._payload_to_ndarray(value)
                if proto_arr is None:
                    logger.warning(
                        f"Client {self.ID}: Skip prototype for class {key}; "
                        f"payload could not be decoded")
                    continue
                proto_arr = np.asarray(proto_arr, dtype=np.float32)
                if proto_arr.shape == expected_shape:
                    normalized[key] = proto_arr
                else:
                    logger.warning(
                        f"Client {self.ID}: Skip prototype for class {key} "
                        f"with shape {proto_arr.shape}, expected "
                        f"{expected_shape}")

        return normalized

    @staticmethod
    def _payload_to_ndarray(payload):
        """Decode compact gRPC payloads back to ndarray-compatible values."""
        if isinstance(payload, bytes):
            payload = payload.decode('utf-8')
        if isinstance(payload, str):
            if payload.startswith('znp:'):
                try:
                    raw = zlib.decompress(base64.b64decode(payload[4:]))
                    return np.load(io.BytesIO(raw), allow_pickle=False)
                except Exception:
                    return None
            try:
                payload = param2tensor(payload)
            except Exception:
                return None
        if isinstance(payload, torch.Tensor):
            return payload.detach().cpu().numpy()
        return payload

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
        """Generate samples from Gaussian distribution."""
        mean = np.asarray(mean, dtype=np.float32)
        cov_matrix = np.asarray(cov_matrix, dtype=np.float32)

        expected_dim = int(self.embedding_dim)
        if mean.shape != (expected_dim, ):
            raise ValueError(
                f"Client {self.ID}: Cannot generate samples with mean shape "
                f"{mean.shape}, expected ({expected_dim},).")
        if cov_matrix.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Client {self.ID}: Cannot generate samples with covariance "
                f"shape {cov_matrix.shape}, expected "
                f"({expected_dim}, {expected_dim}).")

        dim = cov_matrix.shape[0]
        cache_key = id(cov_matrix)
        cached_factor = self._cov_factor_cache.get(cache_key)

        if cached_factor is None:
            cov_matrix = self._nearest_pos_def(cov_matrix)

            jitter = 1e-6
            factor = None
            factor_type = 'cholesky'
            while True:
                try:
                    factor = np.linalg.cholesky(
                        cov_matrix + jitter * np.eye(dim, dtype=np.float32))
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10
                    if jitter > 1:
                        var = np.diag(cov_matrix)
                        var = np.maximum(var, 1e-6)
                        factor = np.sqrt(var)
                        factor_type = 'diag'
                        break

            cached_factor = (factor_type, factor.astype(np.float32))
            self._cov_factor_cache[cache_key] = cached_factor

        factor_type, factor = cached_factor
        z = np.random.randn(num_samples, dim).astype(np.float32)
        if factor_type == 'diag':
            return mean + z * factor
        return mean + z @ factor.T

    def _perform_augmentation(self):
        """Perform GGEUR_Clip feature augmentation"""
        augmentation_start = time.time()
        if self._try_load_augmented_feature_cache():
            self.augmentation_done = True
            self.local_features = {}
            self.local_labels = {}
            return

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
            augmentation_elapsed = time.time() - augmentation_start
            aug_qps = (
                self.augmented_features.shape[0] / augmentation_elapsed
                if augmentation_elapsed > 0 else 0.0
            )
            logger.info(
                f"Client {self.ID}: Augmentation timing - "
                f"samples={self.augmented_features.shape[0]}, "
                f"time={augmentation_elapsed:.4f}s, "
                f"qps={aug_qps:.2f} samples/s")
            self._save_augmented_feature_cache()
            self.base_augmented_features = self.augmented_features.copy()
            self.base_augmented_labels = self.augmented_labels.copy()

        self.augmentation_done = True

        # Free raw features from memory - augmented_features/loader are all we need now
        self.local_features = {}
        self.local_labels = {}

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

    def _augmented_cache_metadata(self):
        splits = getattr(self._cfg.data, 'splits', [])
        try:
            splits = list(splits)
        except Exception:
            splits = []
        return {
            'source': 'real_dataset',
            'mode': 'ggeur_headonly_augmented_features',
            'client_id': int(self.ID),
            'client_num': int(self._cfg.federate.client_num),
            'dataset': str(self._cfg.data.type),
            'data_root': str(self._cfg.data.root),
            'splits': splits,
            'seed': int(getattr(self._cfg, 'seed', 0)),
            'feature_extractor': str(self.feature_extractor_type),
            'embedding_dim': int(self.embedding_dim),
            'num_classes': int(self._cfg.model.num_classes),
            'num_generated_per_sample':
                int(self.ggeur_cfg.num_generated_per_sample),
            'num_generated_per_prototype':
                int(self.ggeur_cfg.num_generated_per_prototype),
            'target_size_per_class':
                int(self.ggeur_cfg.target_size_per_class),
            'headonly_cache_version':
                str(getattr(self.ggeur_cfg, 'headonly_cache_version',
                            'fcache_v1')),
        }

    def _get_augmented_feature_cache_path(self):
        if not getattr(self.ggeur_cfg, 'head_only_mode', False):
            return None
        cache_dir = getattr(self.ggeur_cfg, 'feature_cache_dir', '')
        if not cache_dir:
            return None
        version = str(getattr(self.ggeur_cfg, 'headonly_cache_version',
                              'fcache_v1')).replace('/', '_')
        dataset = str(self._cfg.data.type).replace('/', '_')
        path = os.path.join(
            cache_dir,
            'headonly_augmented',
            version,
            f'{dataset}_client_{int(self.ID):06d}.pt',
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def _try_load_augmented_feature_cache(self):
        path = self._get_augmented_feature_cache_path()
        if path is None or not os.path.exists(path):
            return False
        try:
            cached = torch.load(path, map_location='cpu')
            metadata = cached.get('metadata', {})
            expected = self._augmented_cache_metadata()
            if metadata != expected:
                logger.info(
                    f"Client {self.ID}: Ignore augmented cache metadata "
                    f"mismatch at {path}")
                return False
            features = cached['features']
            labels = cached['labels']
            self.augmented_features = (
                features.numpy() if isinstance(features, torch.Tensor)
                else np.asarray(features)
            )
            self.augmented_labels = (
                labels.numpy() if isinstance(labels, torch.Tensor)
                else np.asarray(labels)
            )
            self.base_augmented_features = self.augmented_features.copy()
            self.base_augmented_labels = self.augmented_labels.copy()
            dataset = AugmentedFeatureDataset(
                self.augmented_features, self.augmented_labels)
            self.augmented_loader = DataLoader(
                dataset,
                batch_size=self._cfg.dataloader.batch_size,
                shuffle=True,
            )
            logger.info(
                f"Client {self.ID}: Loaded augmented HeadOnly feature cache "
                f"from {path} ({len(self.augmented_labels)} samples)")
            return True
        except Exception as error:
            logger.warning(
                f"Client {self.ID}: Failed to load augmented cache {path}: "
                f"{error}")
            return False

    def _save_augmented_feature_cache(self):
        path = self._get_augmented_feature_cache_path()
        if path is None or self.augmented_features is None:
            return
        try:
            torch.save({
                'features': torch.as_tensor(self.augmented_features).float(),
                'labels': torch.as_tensor(self.augmented_labels).long(),
                'metadata': self._augmented_cache_metadata(),
            }, path)
            logger.info(
                f"Client {self.ID}: Saved augmented HeadOnly feature cache "
                f"to {path} ({len(self.augmented_labels)} samples)")
        except Exception as error:
            logger.warning(
                f"Client {self.ID}: Failed to save augmented cache {path}: "
                f"{error}")

    def _try_start_from_augmented_cache(self):
        """Load generated HeadOnly samples and skip round-0 generation."""
        if not getattr(self.ggeur_cfg,
                       'headonly_skip_round0_if_augmented_cache_exists',
                       False):
            return False
        if not getattr(self.ggeur_cfg, 'head_only_mode', False):
            return False
        if not self._try_load_augmented_feature_cache():
            return False

        self._build_mlp_classifier()
        self.augmentation_done = True
        self.statistics_uploaded = True
        self.local_features = {}
        self.local_labels = {}
        logger.info(
            f"Client {self.ID}: HeadOnly cache-hot mode active; skip "
            "round-0 feature extraction/statistics/augmentation and train "
            "on cached generated samples.")
        return True

    def _should_skip_statistics_phase(self):
        """Whether baseline mode can skip server-side statistics exchange."""
        return (
            int(getattr(self.ggeur_cfg, 'num_generated_per_sample', 0)) == 0
            and int(getattr(self.ggeur_cfg, 'num_generated_per_prototype', 0)) == 0
            and int(getattr(self.ggeur_cfg, 'target_size_per_class', 0)) == 0
            and not getattr(self.ggeur_cfg, 'use_fedproto', False)
            and not getattr(self.ggeur_cfg, 'use_cnn_distillation', False)
            and not getattr(self.ggeur_cfg, 'use_feature_alignment', False)
            and not getattr(self.ggeur_cfg, 'use_promptfl', False)
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
            logger.info(f"Client {self.ID}: Round {round_idx} - Statistics collection phase")

            if self._try_start_from_augmented_cache():
                self.comm_manager.send(
                    Message(
                        msg_type='augmentation_ready',
                        sender=self.ID,
                        receiver=[self.server_id],
                        state=self.state,
                        content='ready'
                    )
                )
                return

            # Extract CLIP features
            self._extract_clip_features()

            if self._should_skip_statistics_phase():
                logger.info(f"Client {self.ID}: Baseline mode detected, skip "
                            f"statistics upload and prepare local features "
                            f"for FedAvg directly")
                self._perform_augmentation()
                self._build_mlp_classifier()
                self.statistics_uploaded = True

                if getattr(self.ggeur_cfg, 'unload_extractor_after_cache', True):
                    self._unload_feature_extractor()

                self.comm_manager.send(
                    Message(
                        msg_type='augmentation_ready',
                        sender=self.ID,
                        receiver=[self.server_id],
                        state=self.state,
                        content='ready'
                    )
                )
                return

            # Compute local statistics
            self._compute_local_statistics()

            # Upload to server
            self._upload_local_statistics()

            # Unload extractor to free VRAM (features are cached to disk)
            if getattr(self.ggeur_cfg, 'unload_extractor_after_cache', True):
                if not (self.use_cnn_distillation or self.use_feature_alignment):
                    self._unload_feature_extractor()

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
                        # Save global ctx for FedProx proximal term
                        self.global_prompt_ctx = ctx_tensor.to(self.device).detach().clone()
                except Exception as e:
                    logger.debug(f"Client {self.ID}: Could not load prompt ctx: {e}")

        if self.a3fl_enabled:
            self._extract_shared_attack_trigger(content, 'a3fl')

        if self.cerberus_enabled:
            self._extract_cerberus_peer_models(content)

        if self.sabre_enabled:
            self._extract_shared_attack_trigger(content, 'sabre')

        # Update MLP with global model parameters
        if mlp_para is not None and self.mlp_classifier is not None:
            try:
                self.mlp_classifier.load_state_dict(mlp_para)
            except Exception as e:
                logger.debug(f"Client {self.ID}: Could not load MLP state dict: {e}")
        a3fl_global_mlp_state = None
        if self.mlp_classifier is not None:
            a3fl_global_mlp_state = copy.deepcopy(
                self.mlp_classifier.state_dict())

        # MOON: snapshot global model after loading global params, before local training
        if self.use_moon and self.mlp_classifier is not None:
            self.moon_global_model = copy.deepcopy(self.mlp_classifier)
            self.moon_global_model.eval()
            for p in self.moon_global_model.parameters():
                p.requires_grad_(False)

        # Update CNN with global model parameters (for both distillation and feature alignment)
        if (self.use_cnn_distillation or self.use_feature_alignment) and cnn_para is not None and self.cnn_model is not None:
            try:
                self.cnn_model.load_state_dict(cnn_para)
            except Exception as e:
                logger.debug(f"Client {self.ID}: Could not load CNN state dict: {e}")

        # Apply A3FL poisoning on the GGEUR augmented feature training set.
        self._inject_a3fl_poison_features(round_idx)

        # Train MLP on augmented features
        if self._should_cerberus_attack(round_idx):
            logger.info(
                f"Client {self.ID}: Round {round_idx} - CERBERUS attack training")
            mlp_sample_size, mlp_model_para, mlp_results = \
                self._train_cerberus_on_augmented_data(round_idx)
            self.sabre_latest_meta = {
                'active': False,
                'client_id': int(self.ID),
                'round': int(round_idx),
            }
        elif self._should_sabre_attack(round_idx):
            logger.info(
                f"Client {self.ID}: Round {round_idx} - SABRE attack training")
            mlp_sample_size, mlp_model_para, mlp_results = \
                self._train_sabre_on_augmented_data(round_idx)
            self.cerberus_latest_meta = {
                'active': False,
                'client_id': int(self.ID),
                'round': int(round_idx),
            }
        else:
            self.cerberus_latest_meta = {
                'active': False,
                'client_id': int(self.ID),
                'round': int(round_idx),
            }
            self.sabre_latest_meta = {
                'active': False,
                'client_id': int(self.ID),
                'round': int(round_idx),
            }
            mlp_sample_size, mlp_model_para, mlp_results = \
                self._train_on_augmented_data()
        if self.a3fl_enabled and self.a3fl_latest_meta.get('active', False):
            mlp_model_para = self._strengthen_a3fl_mlp_update(
                a3fl_global_mlp_state, mlp_model_para)

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

        if self.a3fl_enabled:
            if isinstance(combined_para, dict) and 'mlp' in combined_para:
                combined_para['a3fl'] = copy.deepcopy(self.a3fl_latest_meta)
            else:
                combined_para = {
                    'mlp': combined_para,
                    'a3fl': copy.deepcopy(self.a3fl_latest_meta)
                }

        if self.cerberus_enabled:
            if isinstance(combined_para, dict) and 'mlp' in combined_para:
                combined_para['cerberus'] = copy.deepcopy(
                    self.cerberus_latest_meta)
            else:
                combined_para = {
                    'mlp': combined_para,
                    'cerberus': copy.deepcopy(self.cerberus_latest_meta)
                }

        if self.sabre_enabled:
            if isinstance(combined_para, dict) and 'mlp' in combined_para:
                combined_para['sabre'] = copy.deepcopy(
                    self.sabre_latest_meta)
            else:
                combined_para = {
                    'mlp': combined_para,
                    'sabre': copy.deepcopy(self.sabre_latest_meta)
                }

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
        """Build PromptFL components. CLIP backbone is shared across all clients
        (class-level singleton) to avoid loading N copies into GPU memory."""
        import open_clip
        from federatedscope.contrib.model.ggeur_prompt import PromptLearner, TextEncoder

        n_ctx = getattr(self.ggeur_cfg, 'prompt_length', 16)
        class_names = self._get_class_names()
        clip_model_name = getattr(self.ggeur_cfg, 'clip_model', 'ViT-B-16')
        model_path = getattr(self.ggeur_cfg, 'clip_model_path', '')
        clip_pretrained = getattr(self.ggeur_cfg, 'clip_pretrained', 'openai')
        template = getattr(self.ggeur_cfg, 'prompt_template', 'a photo of a {}')

        # Load shared CLIP once (CPU), reused by all clients
        if GGEURClient._shared_prompt_clip is None or GGEURClient._shared_prompt_clip_name != clip_model_name:
            try:
                if model_path and os.path.isfile(model_path):
                    logger.info(f"Client {self.ID}: Loading shared prompt CLIP from {model_path}")
                    clip_model, _, _ = open_clip.create_model_and_transforms(
                        clip_model_name, pretrained=model_path
                    )
                else:
                    clip_model, _, _ = open_clip.create_model_and_transforms(
                        clip_model_name, pretrained=clip_pretrained
                    )
                for p in clip_model.parameters():
                    p.requires_grad = False
                clip_model.eval()
                GGEURClient._shared_prompt_clip = clip_model.cpu()
                GGEURClient._shared_prompt_tokenizer = open_clip.get_tokenizer(clip_model_name)
                GGEURClient._shared_prompt_clip_name = clip_model_name
                logger.info(f"Shared prompt CLIP loaded (CPU), will be moved to GPU only during forward pass")
            except Exception as e:
                logger.error(f"Client {self.ID}: Failed to load shared CLIP: {e}")
                return

        # Each client only stores its own ctx (16x512 = 32KB) and token buffers
        # PromptLearner is built on CPU using the shared clip_model
        try:
            self.prompt_learner = PromptLearner(
                clip_model=GGEURClient._shared_prompt_clip,
                tokenizer=GGEURClient._shared_prompt_tokenizer,
                classnames=class_names,
                n_ctx=n_ctx,
                template=template,
                device=torch.device('cpu'),  # buffers on CPU, moved to GPU during forward
            )
            self.text_encoder = TextEncoder(device=self.device)
            # Move only the small trainable ctx to GPU; buffers stay CPU until needed
            self.prompt_learner.ctx.data = self.prompt_learner.ctx.data.to(self.device)
        except Exception as e:
            logger.error(f"Client {self.ID}: Failed to build PromptLearner: {e}")
            return

        logger.info(f"Client {self.ID}: PromptLearner ready ({n_ctx} ctx tokens, "
                    f"{len(class_names)} classes, shared CLIP backbone)")

    def _build_prompt_loader(self):
        """
        Build a prompt-specific data loader using:
        1. Real local CLIP features (not Gaussian samples) for classes this client has
        2. Gaussian samples around OTHER clients' prototype means for missing classes
           (anchored to real feature statistics, provides cross-domain coverage)
        """
        all_features = []
        all_labels = []

        requested_n_per_proto = getattr(self.ggeur_cfg, 'prompt_samples_per_proto', 20)
        local_classes = set()
        local_class_sizes = []
        n_local_samples = 0

        for class_idx, feats in self.real_local_features.items():
            if feats.shape[0] <= 0:
                continue
            local_classes.add(int(class_idx))
            local_class_sizes.append(int(feats.shape[0]))

        avg_local_per_class = int(round(np.mean(local_class_sizes))) if local_class_sizes else 1
        effective_n_per_proto = max(1, min(int(requested_n_per_proto), avg_local_per_class))
        target_local_per_class = max(1, effective_n_per_proto)

        # 1. Real local features (upsample lightly for class balance)
        for class_idx, feats in self.real_local_features.items():
            if feats.shape[0] <= 0:
                continue

            balanced_feats = feats
            if feats.shape[0] < target_local_per_class:
                repeat_idx = np.random.choice(feats.shape[0],
                                              target_local_per_class - feats.shape[0],
                                              replace=True)
                balanced_feats = np.concatenate([feats, feats[repeat_idx]], axis=0)

            all_features.append(balanced_feats)
            all_labels.append(np.full(balanced_feats.shape[0], int(class_idx)))
            n_local_samples += int(balanced_feats.shape[0])

        # 2. Gaussian samples around other clients' prototype means for missing classes
        n_proto_samples = 0
        if self.other_prototypes:
            for class_idx, prototypes in self.other_prototypes.items():
                if int(class_idx) not in local_classes and len(prototypes) > 0:
                    proto_mean = np.mean(np.stack(prototypes), axis=0)
                    cov = self.global_cov_matrices.get(
                        class_idx,
                        self.global_cov_matrices.get(int(class_idx), None)
                    )
                    if cov is None:
                        cov = np.eye(self.embedding_dim) * 0.01
                    generated = self._generate_samples(proto_mean, cov, effective_n_per_proto)
                    all_features.append(generated)
                    all_labels.append(np.full(effective_n_per_proto, int(class_idx)))
                    n_proto_samples += int(effective_n_per_proto)

        if not all_features:
            logger.warning(f"Client {self.ID}: No data for prompt_loader, falling back to augmented_loader")
            self.prompt_loader = self.augmented_loader
            return

        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)

        dataset = AugmentedFeatureDataset(features, labels)
        self.prompt_loader = DataLoader(
            dataset,
            batch_size=self._cfg.dataloader.batch_size,
            shuffle=True
        )

        logger.info(f"Client {self.ID}: prompt_loader built - "
                    f"{features.shape[0]} samples, {len(np.unique(labels))} classes "
                    f"({n_local_samples} balanced local, {n_proto_samples} proto-augmented, "
                    f"requested_proto={requested_n_per_proto}, effective_proto={effective_n_per_proto})")

    def _train_prompt_on_augmented_data(self,
                                        prompt_loader=None,
                                        lr=None,
                                        local_epochs=None,
                                        prompt_mu=None,
                                        log_prefix='Prompt'):
        """
        Train soft prompt ctx vectors on augmented CLIP features.
        The shared CLIP backbone is moved to GPU only for this client's turn,
        then moved back to CPU to free VRAM for the next client.
        """
        prompt_loader = self.prompt_loader if prompt_loader is None else prompt_loader
        if self.prompt_learner is None or prompt_loader is None:
            return 0, {}, {}

        if lr is None:
            lr = getattr(self.ggeur_cfg, 'prompt_lr', 0.002)
        if local_epochs is None:
            local_epochs = getattr(self.ggeur_cfg, 'prompt_local_epochs', 1)
        max_train_batches = int(
            getattr(self.ggeur_cfg, 'prompt_max_train_batches', 0))

        # Move shared CLIP to GPU for this client's forward pass
        clip_model = GGEURClient._shared_prompt_clip.to(self.device)
        clip_model.eval()

        # Move prompt buffers to GPU
        self._move_prompt_buffers(self.device)
        self.prompt_learner.train()

        optimizer = torch.optim.Adam([self.prompt_learner.ctx], lr=lr)

        # Use configured temperature instead of CLIP's logit_scale (~100).
        # logit_scale ≈ 100 amplifies gradients 100x, causing client drift.
        temperature = getattr(self.ggeur_cfg, 'prompt_temperature', 0.07)
        logit_scale = 1.0 / temperature

        # FedProx proximal term: keeps local ctx close to global ctx
        if prompt_mu is None:
            prompt_mu = getattr(self.ggeur_cfg, 'prompt_proximal_mu', 0.0)
        global_ctx = self.global_prompt_ctx  # may be None in round 0

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch_idx, (features, labels) in enumerate(prompt_loader):
                if max_train_batches > 0 and batch_idx >= max_train_batches:
                    break

                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                prompt_embeds, eot_pos = self.prompt_learner()
                text_feats = self.text_encoder(prompt_embeds, eot_pos, clip_model)
                text_feats_norm = F.normalize(text_feats, p=2, dim=1)

                img_feats = F.normalize(features, p=2, dim=1)
                logits = logit_scale * (img_feats @ text_feats_norm.T)
                loss = F.cross_entropy(logits, labels)

                if prompt_mu > 0.0 and global_ctx is not None:
                    proximal_loss = (prompt_mu / 2.0) * ((self.prompt_learner.ctx - global_ctx) ** 2).sum()
                    loss = loss + proximal_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_([self.prompt_learner.ctx], max_norm=1.0)
                optimizer.step()

                with torch.no_grad():
                    _, predicted = torch.max(logits, 1)
                    epoch_correct += (predicted == labels).sum().item()
                    epoch_samples += features.size(0)
                    epoch_loss += loss.item() * features.size(0)

            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples

        # Move shared CLIP back to CPU and free GPU memory
        GGEURClient._shared_prompt_clip = clip_model.cpu()
        # Move prompt buffers back to CPU
        self._move_prompt_buffers(torch.device('cpu'))
        torch.cuda.empty_cache()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        logger.info(f"Client {self.ID}: {log_prefix} training - "
                    f"loss={avg_loss:.4f}, acc={accuracy:.4f}, "
                    f"samples={total_samples}")

        prompt_para = {
            'ctx': copy.deepcopy(self.prompt_learner.ctx.data.cpu()),
            'sample_size': int(total_samples)
        }
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
