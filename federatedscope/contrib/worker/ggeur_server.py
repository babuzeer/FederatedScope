"""
GGEUR_Clip Server Implementation

Handles:
1. Collecting local statistics from all clients
2. Aggregating covariance matrices using parallel axis theorem
3. Broadcasting global covariance matrices to clients
4. Standard FedAvg aggregation for MLP training
5. Evaluating on test sets from all domains
6. (Optional) CNN model aggregation and evaluation for knowledge distillation
7. (Optional) CNN model aggregation and evaluation for feature alignment
"""

import os
import gc
import logging
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from federatedscope.core.message import Message
from federatedscope.core.workers import Server

logger = logging.getLogger(__name__)


class GGEURServer(Server):
    """
    GGEUR_Clip Server that:
    1. Collects local statistics (means, covariances, counts) from clients
    2. Aggregates covariance matrices using parallel axis theorem
    3. Broadcasts global covariances to all clients
    4. Performs standard FedAvg aggregation on MLP parameters
    5. Evaluates on test sets from all domains after each round
    6. (Optional) Aggregates and evaluates CNN model for knowledge distillation
    7. (Optional) Aggregates and evaluates CNN model for feature alignment
    """
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 unseen_clients_id=None,
                 **kwargs):
        super(GGEURServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy,
                             unseen_clients_id, **kwargs)

        if config is None:
            return

        self.ggeur_cfg = config.ggeur

        # Statistics collection buffers
        self.local_statistics_buffer = {}  # {client_id: statistics}
        self.statistics_collected = False

        # Aggregated covariance matrices
        self.global_cov_matrices = {}  # {class_idx: cov_matrix}

        # All client prototypes for cross-client augmentation
        self.all_prototypes = {}  # {client_id: {class_idx: prototype}}

        # Global prototypes (aggregated means) for feature alignment
        self.global_prototypes = {}  # {class_idx: mean_vector}

        # MLP model for aggregation
        self.global_mlp = None

        # Track which clients have completed augmentation
        self.augmentation_ready_clients = set()

        # Test data for evaluation
        self.test_features = {}  # {domain_name: features array}
        self.test_labels = {}  # {domain_name: labels array}
        self.test_data_loaded = False

        # CLIP model for test feature extraction
        self.clip_model = None
        self.clip_preprocess = None

        # Tracking best model
        self.best_avg_accuracy = 0.0
        self.best_model_state = None
        self.test_accuracies_history = {}  # {domain: [acc_per_round]}

        # ===== CNN Mode =====
        self.use_cnn_distillation = getattr(self.ggeur_cfg,
                                            'use_cnn_distillation', False)
        self.use_feature_alignment = getattr(self.ggeur_cfg,
                                             'use_feature_alignment', False)
        self.global_cnn = None  # Global CNN model
        self.cnn_test_accuracies_history = {}  # {domain: [acc_per_round]}
        self.best_cnn_avg_accuracy = 0.0
        self.best_cnn_model_state = None
        # Test images for CNN evaluation (original images, not CLIP features)
        self.test_images_loaded = False
        self.test_image_loaders = {}  # {domain_name: DataLoader}

        # ===== Feature Extractor Mode =====
        # 'clip': Use CLIP (ViT-based, original method)
        # 'cnn': Use pretrained CNN (ConvNeXt, ResNet, etc.)
        self.feature_extractor_type = getattr(self.ggeur_cfg,
                                              'feature_extractor', 'clip')
        self.cnn_extractor = None  # CNN feature extractor for evaluation

        # ===== Separated Training Mode =====
        self.use_separated_training = getattr(self.ggeur_cfg,
                                              'use_separated_training', False)
        self.classifier_pretrain_rounds = getattr(
            self.ggeur_cfg, 'classifier_pretrain_rounds', 20)
        self.freeze_classifier = getattr(self.ggeur_cfg, 'freeze_classifier',
                                         True)
        # Phase tracking: 'classifier' or 'cnn_backbone'
        self.training_phase = 'classifier'
        # Store pretrained classifier for Phase 2
        self.pretrained_classifier = None

    def _register_default_handlers(self):
        """Register message handlers"""
        super()._register_default_handlers()

        # Register handler for local statistics
        self.register_handlers('local_statistics',
                               self.callback_for_local_statistics)

        # Register handler for augmentation ready signal
        self.register_handlers('augmentation_ready',
                               self.callback_for_augmentation_ready)

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        """
        Override broadcast_model_para to handle GGEUR's special startup flow.

        In GGEUR, the server doesn't broadcast initial model parameters at
        Round 0. Instead, clients first collect local statistics, and the
        server creates the global MLP only after receiving all statistics.
        """
        if self.state == 0:
            # Round 0: Send a signal to clients to start collecting statistics
            # Send a non-empty dict with a signal (Message system doesn't
            # support None or empty dict)
            logger.info(
                "Server: Round 0 - sending signal to collect statistics")
            for client_id in range(1, self._client_num + 1):
                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[client_id],
                            state=self.state,
                            content={'signal': 'collect_statistics'}))
        else:
            # For subsequent rounds, use the custom training round method
            self._start_training_round()

    def _build_global_mlp(self, num_classes):
        """Build global MLP classifier"""
        input_dim = self.ggeur_cfg.embedding_dim
        hidden_dim = self.ggeur_cfg.mlp_hidden_dim

        if hidden_dim > 0:
            self.global_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(self.ggeur_cfg.mlp_dropout),
                nn.Linear(hidden_dim, num_classes))
        else:
            self.global_mlp = nn.Linear(input_dim, num_classes)

        self.global_mlp = self.global_mlp.to(self.device)
        logger.info(
            f"Server: Built global MLP classifier with {num_classes} classes")

    def _build_global_cnn(self, num_classes):
        """Build global CNN model for knowledge distillation or feature alignment"""
        from federatedscope.contrib.model.ggeur_cnn import GGEUR_CNN_FeatureAlign

        cnn_model_name = getattr(self.ggeur_cfg, 'cnn_model', 'resnet18')
        clip_dim = getattr(self.ggeur_cfg, 'embedding_dim', 512)

        # For feature alignment: no pretrained weights (from scratch)
        # For distillation: use pretrained weights
        if self.use_feature_alignment:
            cnn_pretrained = False
            mode_str = "feature alignment (from scratch)"
        else:
            cnn_pretrained = getattr(self.ggeur_cfg, 'cnn_pretrained', True)
            mode_str = f"distillation (pretrained={cnn_pretrained})"

        self.global_cnn = GGEUR_CNN_FeatureAlign(model_name=cnn_model_name,
                                                 num_classes=num_classes,
                                                 clip_dim=clip_dim,
                                                 pretrained=cnn_pretrained)
        self.global_cnn = self.global_cnn.to(self.device)
        logger.info(
            f"Server: Built global CNN ({cnn_model_name}) for {mode_str} with {num_classes} classes"
        )

    def _load_clip_model(self):
        """Load CLIP model for test feature extraction"""
        if self.clip_model is not None:
            return

        try:
            import open_clip

            model_name = self.ggeur_cfg.clip_model
            pretrained = self.ggeur_cfg.clip_pretrained
            local_path = self.ggeur_cfg.clip_model_path

            if local_path and os.path.exists(local_path):
                logger.info(
                    f"Server: Loading CLIP from local path: {local_path}")
                self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=local_path)
            else:
                logger.info(
                    f"Server: Loading CLIP from pretrained: {pretrained}")
                self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained)

            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            logger.info(f"Server: Loaded CLIP model {model_name}")

        except ImportError:
            logger.error(
                "open_clip not installed. Please install: pip install open_clip_torch"
            )
            raise

    def _load_cnn_extractor(self):
        """Load CNN feature extractor for test feature extraction"""
        if self.cnn_extractor is not None:
            return

        try:
            from federatedscope.contrib.model.ggeur_cnn_extractor import CNNFeatureExtractor

            model_name = getattr(self.ggeur_cfg, 'cnn_backbone',
                                 'convnext_base')
            pretrained = getattr(self.ggeur_cfg, 'cnn_pretrained', True)
            freeze = True  # Always freeze for feature extraction

            self.cnn_extractor = CNNFeatureExtractor(model_name=model_name,
                                                     pretrained=pretrained,
                                                     freeze=freeze)
            self.cnn_extractor = self.cnn_extractor.to(self.device)

            logger.info(f"Server: Loaded CNN extractor {model_name}, "
                        f"feature_dim={self.cnn_extractor.get_feature_dim()}")

        except Exception as e:
            logger.error(f"Server: Failed to load CNN extractor: {e}")
            raise

    def _load_feature_extractor(self):
        """Load the appropriate feature extractor (CLIP or CNN)"""
        if self.feature_extractor_type == 'cnn':
            self._load_cnn_extractor()
        else:
            self._load_clip_model()

    def _get_test_cache_path(self, domain):
        """Get cache path for test features"""
        cache_dir = getattr(self.ggeur_cfg, 'feature_cache_dir', '')
        if not cache_dir:
            cache_dir = os.path.join(os.path.dirname(self._cfg.data.root),
                                     'clip_feature_cache')

        # Cache isolation for distributed mode
        if (hasattr(self._cfg, 'federate')
                and self._cfg.federate.mode == 'distributed'):
            cache_dir = os.path.join(cache_dir, 'server')

        os.makedirs(cache_dir, exist_ok=True)

        # Get split parameters to include in cache filename
        data_type = self._cfg.data.type.lower()
        if hasattr(self._cfg.data, 'splits'):
            splits = tuple(self._cfg.data.splits)
        else:
            if 'office' in data_type and 'home' in data_type:
                splits = (0.7, 0.0, 0.3)
            else:
                splits = (0.8, 0.1, 0.1)
        seed = self._cfg.seed if hasattr(self._cfg, 'seed') else 123

        dataset_name = data_type

        # Build model string based on feature extractor type
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

        # Include split params in filename to ensure cache invalidation when params change
        split_str = f"split{int(splits[0]*100)}_{int(splits[1]*100)}_{int(100-splits[0]*100-splits[1]*100)}_seed{seed}"
        cache_filename = f"{dataset_name}_{domain}_test_{prefix}_{model_str}_{split_str}.npz"

        return os.path.join(cache_dir, cache_filename)

    def _load_test_data_and_features(self):
        """Load test data from all domains and extract CLIP features"""
        if self.test_data_loaded:
            return

        logger.info("Server: Loading test data from all domains...")

        data_type = self._cfg.data.type.lower()
        data_root = self._cfg.data.root

        # Get the same split ratios and seed as client data loading
        if hasattr(self._cfg.data, 'splits'):
            splits = tuple(self._cfg.data.splits)
        else:
            # Default splits based on dataset type
            if 'office' in data_type and 'home' in data_type:
                splits = (0.7, 0.0, 0.3)  # Same as ggeur_data.py
            else:
                splits = (0.8, 0.1, 0.1)

        train_ratio, val_ratio = splits[0], splits[1]
        seed = self._cfg.seed if hasattr(self._cfg, 'seed') else 123

        logger.info(
            f"Server: Using splits={splits}, seed={seed} (same as client data)"
        )

        # Determine domains based on dataset type
        if 'pacs' in data_type:
            domains = ['photo', 'art_painting', 'cartoon', 'sketch']
            from federatedscope.cv.dataset.pacs import PACS
            dataset_class = PACS
        elif 'office' in data_type and 'home' in data_type:
            domains = ['Art', 'Clipart', 'Product', 'Real_World']
            from federatedscope.cv.dataset.office_home import OfficeHome
            dataset_class = OfficeHome
        elif 'office' in data_type and 'caltech' in data_type:
            domains = ['amazon', 'caltech', 'dslr', 'webcam']
            from federatedscope.cv.dataset.office_caltech import OfficeCaltech10
            dataset_class = OfficeCaltech10
        else:
            logger.warning(
                f"Server: Unknown dataset type {data_type}, skipping test evaluation"
            )
            return

        # Load test data for each domain
        for domain in domains:
            cache_path = self._get_test_cache_path(domain)

            # Try to load from cache first
            if os.path.exists(cache_path):
                try:
                    data = np.load(cache_path)
                    self.test_features[domain] = data['features']
                    self.test_labels[domain] = data['labels']
                    logger.info(
                        f"Server: Loaded {len(self.test_labels[domain])} cached test features for {domain}"
                    )
                    continue
                except Exception as e:
                    logger.warning(
                        f"Server: Failed to load cache for {domain}: {e}")

            # Load test dataset with SAME parameters as client data
            try:
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])

                test_dataset = dataset_class(root=data_root,
                                             domain=domain,
                                             split='test',
                                             transform=transform,
                                             train_ratio=train_ratio,
                                             val_ratio=val_ratio,
                                             seed=seed)

                if len(test_dataset) == 0:
                    logger.warning(f"Server: No test data for domain {domain}")
                    continue

                # Log the split info for verification
                logger.info(
                    f"Server: {domain} test set has {len(test_dataset)} samples"
                )

                # Extract features using appropriate extractor (CLIP or CNN)
                self._load_feature_extractor()
                features_list = []
                labels_list = []

                dataloader = DataLoader(test_dataset,
                                        batch_size=32,
                                        shuffle=False)

                extractor_name = 'CNN' if self.feature_extractor_type == 'cnn' else 'CLIP'

                with torch.no_grad():
                    for images, labels in dataloader:
                        images = images.to(self.device)
                        if self.feature_extractor_type == 'cnn':
                            features = self.cnn_extractor(images)
                        else:
                            features = self.clip_model.encode_image(images)
                        features_list.append(features.cpu().numpy())
                        labels_list.append(labels.numpy())

                self.test_features[domain] = np.vstack(features_list)
                self.test_labels[domain] = np.concatenate(labels_list)

                # Save to cache
                np.savez(cache_path,
                         features=self.test_features[domain],
                         labels=self.test_labels[domain])

                logger.info(
                    f"Server: Extracted and cached {len(self.test_labels[domain])} test features for {domain}"
                )

            except Exception as e:
                logger.warning(
                    f"Server: Failed to load test data for {domain}: {e}")
                import traceback
                traceback.print_exc()

        self.test_data_loaded = True
        logger.info(
            f"Server: Loaded test data for {len(self.test_features)} domains")

        # Unload CLIP model after test features are extracted/cached - no longer needed
        if self.clip_model is not None:
            del self.clip_model
            self.clip_model = None
            self.clip_preprocess = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug(
                "Server: Unloaded CLIP model after test feature extraction (~600 MB freed)"
            )

    def _evaluate_on_test_sets(self):
        """Evaluate global model on all test sets"""
        if self.global_mlp is None:
            return {}

        # Load test data if not already loaded
        if not self.test_data_loaded:
            self._load_test_data_and_features()

        if not self.test_features:
            return {}

        self.global_mlp.eval()
        results = {}

        with torch.no_grad():
            for domain, features in self.test_features.items():
                labels = self.test_labels[domain]

                features_tensor = torch.from_numpy(features).float().to(
                    self.device)
                labels_tensor = torch.from_numpy(labels).long().to(self.device)

                outputs = self.global_mlp(features_tensor)
                _, predicted = torch.max(outputs, 1)

                correct = (predicted == labels_tensor).sum().item()
                total = labels_tensor.size(0)
                accuracy = correct / total if total > 0 else 0

                results[domain] = accuracy

                # Track history
                if domain not in self.test_accuracies_history:
                    self.test_accuracies_history[domain] = []
                self.test_accuracies_history[domain].append(accuracy)

        # Compute average
        if results:
            avg_accuracy = sum(results.values()) / len(results)
            results['average'] = avg_accuracy

            if 'average' not in self.test_accuracies_history:
                self.test_accuracies_history['average'] = []
            self.test_accuracies_history['average'].append(avg_accuracy)

            # Track best model
            if avg_accuracy > self.best_avg_accuracy:
                self.best_avg_accuracy = avg_accuracy
                self.best_model_state = copy.deepcopy(
                    self.global_mlp.state_dict())

        return results

    def callback_for_local_statistics(self, message: Message):
        """Handle receiving local statistics from a client"""
        client_id = message.sender
        content = message.content

        logger.info(
            f"Server: Received local statistics from client {client_id}")

        # Expected embedding dimension from config
        expected_dim = self.ggeur_cfg.embedding_dim

        means = content['means']
        covs = content['covs']
        counts = content['counts']

        # Convert lists back to numpy arrays
        # Client sends: list -> Server converts: numpy array
        validated_means = {}
        validated_covs = {}

        for class_idx in means.keys():
            class_idx = int(class_idx)  # Ensure int type

            # Convert mean list -> numpy array
            mean_list = means[class_idx]
            mean = np.array(mean_list, dtype=np.float32)

            # Convert cov nested list -> numpy array
            cov_list = covs[class_idx]
            cov = np.array(cov_list, dtype=np.float32)

            # Validate shapes
            if mean.shape != (expected_dim, ):
                raise ValueError(
                    f"Client {client_id}, Class {class_idx}: mean shape mismatch - expected ({expected_dim},), got {mean.shape}"
                )

            if cov.shape != (expected_dim, expected_dim):
                raise ValueError(
                    f"Client {client_id}, Class {class_idx}: cov shape mismatch - expected ({expected_dim}, {expected_dim}), got {cov.shape}"
                )

            validated_means[class_idx] = mean
            validated_covs[class_idx] = cov

        # Store statistics with validated numpy arrays
        self.local_statistics_buffer[client_id] = {
            'means': validated_means,
            'covs': validated_covs,
            'counts': counts,
            'prototypes': content['prototypes']
        }

        # Store prototypes for cross-client sharing
        self.all_prototypes[client_id] = content['prototypes']

        logger.info(
            f"Server: Validated statistics from client {client_id} ({len(validated_means)} classes)"
        )

        # Check if all clients have uploaded statistics
        if len(self.local_statistics_buffer) >= self._client_num:
            logger.info(
                f"Server: Received statistics from all {self._client_num} clients"
            )

            # Aggregate covariance matrices
            self._aggregate_covariances()

            # Compute global prototypes (aggregated means for each class)
            self._compute_global_prototypes()

            # Prepare other client prototypes for each client
            other_prototypes = self._prepare_other_prototypes()

            # Build global MLP
            # Use out_channels as the number of classes
            num_classes = self._cfg.model.out_channels
            if num_classes > 0:
                self._build_global_mlp(num_classes)

                # Build global CNN if using CNN mode
                if self.use_cnn_distillation or self.use_feature_alignment:
                    self._build_global_cnn(num_classes)

            # Broadcast global covariances to all clients
            self._broadcast_global_covariances(other_prototypes)

            self.statistics_collected = True

    def callback_for_augmentation_ready(self, message: Message):
        """Handle client signaling augmentation is complete"""
        client_id = message.sender
        self.augmentation_ready_clients.add(client_id)

        logger.info(
            f"Server: Client {client_id} augmentation ready ({len(self.augmentation_ready_clients)}/{self._client_num})"
        )

        # When all clients are ready, start training
        if len(self.augmentation_ready_clients) >= self._client_num:
            logger.info(
                "Server: All clients ready, starting FedAvg training...")
            self.state = 1  # Move to round 1
            self._start_training_round()

    def _start_training_round(self):
        """Start a new training round by broadcasting model"""
        logger.info(f"Server: Starting training round {self.state}")

        # Check for phase transition in separated training mode
        if self.use_separated_training:
            if self.state == self.classifier_pretrain_rounds + 1 and self.training_phase == 'classifier':
                # Transition from Phase 1 to Phase 2
                self._transition_to_cnn_phase()

        # Prepare model parameters based on current mode and phase
        if self.use_separated_training:
            model_para = self._prepare_separated_training_params()
        elif self.use_cnn_distillation or self.use_feature_alignment:
            # Send both MLP and CNN parameters (for distillation or feature alignment)
            model_para = {
                'mlp': copy.deepcopy(self.global_mlp.state_dict())
                if self.global_mlp else None,
                'cnn': copy.deepcopy(self.global_cnn.state_dict())
                if self.global_cnn else None
            }
        else:
            # Standard mode: only MLP
            if self.global_mlp is not None:
                model_para = copy.deepcopy(self.global_mlp.state_dict())
            else:
                model_para = None

        # Broadcast to all clients
        for client_id in range(1, self._client_num + 1):
            self.comm_manager.send(
                Message(msg_type='model_para',
                        sender=self.ID,
                        receiver=[client_id],
                        state=self.state,
                        content=model_para))

    def _prepare_separated_training_params(self):
        """Prepare model parameters for separated training mode"""
        if self.training_phase == 'classifier':
            # Phase 1: Only send classifier (MLP) parameters
            logger.info(
                f"Server: Phase 1 (Classifier Training) - Round {self.state}/{self.classifier_pretrain_rounds}"
            )
            return {
                'phase': 'classifier',
                'classifier': copy.deepcopy(self.global_mlp.state_dict())
                if self.global_mlp else None
            }
        else:
            # Phase 2: Send frozen classifier + CNN backbone
            logger.info(
                f"Server: Phase 2 (CNN Backbone Training) - Round {self.state}"
            )
            return {
                'phase': 'cnn_backbone',
                'classifier': copy.deepcopy(self.pretrained_classifier)
                if self.pretrained_classifier else None,
                'cnn_backbone': copy.deepcopy(self.global_cnn.state_dict())
                if self.global_cnn else None,
                'freeze_classifier': self.freeze_classifier
            }

    def _transition_to_cnn_phase(self):
        """Transition from classifier training to CNN backbone training"""
        logger.info("=" * 60)
        logger.info("Server: Transitioning to Phase 2 - CNN Backbone Training")
        logger.info("=" * 60)

        # Save the pretrained classifier
        if self.global_mlp is not None:
            self.pretrained_classifier = copy.deepcopy(
                self.global_mlp.state_dict())
            logger.info(
                f"Server: Saved pretrained classifier (Best MLP accuracy: {self.best_avg_accuracy:.4f})"
            )
            logger.info(
                f"Server: Classifier state_dict keys: {list(self.pretrained_classifier.keys())}"
            )
        else:
            logger.error(
                "Server: global_mlp is None! Cannot save pretrained classifier!"
            )
            return

        # Build CNN backbone (without classifier - will use pretrained one)
        self._build_cnn_backbone_only()

        # Update phase
        self.training_phase = 'cnn_backbone'
        logger.info(
            f"Server: Phase transition complete. Now in '{self.training_phase}' phase"
        )

    def _build_cnn_backbone_only(self):
        """Build CNN backbone for separated training (Phase 2)"""
        from federatedscope.contrib.model.ggeur_cnn import GGEUR_CNN_Backbone

        num_classes = self._cfg.model.out_channels
        cnn_model_name = getattr(self.ggeur_cfg, 'cnn_model', 'resnet18')

        # Always train from scratch in separated training mode
        self.global_cnn = GGEUR_CNN_Backbone(
            model_name=cnn_model_name,
            num_classes=num_classes,
            pretrained=False  # From scratch
        )
        self.global_cnn = self.global_cnn.to(self.device)
        logger.info(
            f"Server: Built CNN backbone ({cnn_model_name}) for Phase 2 - FROM SCRATCH"
        )

    def _build_classifier_from_state_dict(self, state_dict):
        """Build classifier model from saved state dict for evaluation"""
        num_classes = self._cfg.model.out_channels
        input_dim = self.ggeur_cfg.embedding_dim
        hidden_dim = self.ggeur_cfg.mlp_hidden_dim

        if hidden_dim > 0:
            classifier = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Dropout(self.ggeur_cfg.mlp_dropout),
                                       nn.Linear(hidden_dim, num_classes))
        else:
            classifier = nn.Linear(input_dim, num_classes)

        classifier = classifier.to(self.device)

        try:
            classifier.load_state_dict(state_dict)
            logger.debug(
                f"Server: Classifier loaded successfully, keys: {list(state_dict.keys())}"
            )
        except Exception as e:
            logger.error(f"Server: Failed to load classifier state dict: {e}")
            logger.error(
                f"Server: Expected keys: {list(classifier.state_dict().keys())}"
            )
            logger.error(f"Server: Received keys: {list(state_dict.keys())}")

        return classifier

    def _aggregate_covariances(self):
        """Aggregate covariance matrices using parallel axis theorem"""
        logger.info("Server: Aggregating covariance matrices...")

        # Collect all class indices
        all_classes = set()
        for client_stats in self.local_statistics_buffer.values():
            all_classes.update(client_stats['means'].keys())

        embedding_dim = self.ggeur_cfg.embedding_dim

        for class_idx in all_classes:
            class_idx = int(class_idx)
            means = []
            covs = []
            counts = []

            # Collect statistics for this class from all clients
            for client_id, client_stats in self.local_statistics_buffer.items(
            ):
                if class_idx in client_stats['means']:
                    means.append(client_stats['means'][class_idx])
                    covs.append(client_stats['covs'][class_idx])
                    counts.append(client_stats['counts'][class_idx])

            if len(counts) == 0:
                self.global_cov_matrices[class_idx] = np.eye(
                    embedding_dim) * 0.01
                continue

            # Compute aggregated mean
            total_count = sum(counts)
            aggregated_mean = np.zeros(embedding_dim)
            for i, (mean, count) in enumerate(zip(means, counts)):
                aggregated_mean += count * mean
            aggregated_mean /= total_count

            # Compute aggregated covariance using parallel axis theorem
            aggregated_cov = np.zeros((embedding_dim, embedding_dim))

            # First term: weighted average of local covariances
            for i, (cov, count) in enumerate(zip(covs, counts)):
                aggregated_cov += count * cov

            # Second term: between-client variance
            for i, (mean, count) in enumerate(zip(means, counts)):
                diff = mean - aggregated_mean
                aggregated_cov += count * np.outer(diff, diff)

            aggregated_cov /= total_count

            self.global_cov_matrices[class_idx] = aggregated_cov

        logger.info(
            f"Server: Aggregated covariances for {len(self.global_cov_matrices)} classes"
        )

    def _compute_global_prototypes(self):
        """Compute global prototypes (weighted average of local means) for each class"""
        logger.info("Server: Computing global prototypes...")

        embedding_dim = self.ggeur_cfg.embedding_dim

        # Collect all class indices
        all_classes = set()
        for client_stats in self.local_statistics_buffer.values():
            all_classes.update(client_stats['means'].keys())

        for class_idx in all_classes:
            class_idx = int(class_idx)
            means = []
            counts = []

            # Collect means for this class from all clients
            for client_id, client_stats in self.local_statistics_buffer.items(
            ):
                if class_idx in client_stats['means']:
                    means.append(client_stats['means'][class_idx])
                    counts.append(client_stats['counts'][class_idx])

            if len(counts) == 0:
                continue

            # Compute weighted average mean (global prototype)
            total_count = sum(counts)
            global_mean = np.zeros(embedding_dim)
            for mean, count in zip(means, counts):
                global_mean += count * mean
            global_mean /= total_count

            self.global_prototypes[class_idx] = global_mean

        logger.info(
            f"Server: Computed global prototypes for {len(self.global_prototypes)} classes"
        )

    def _prepare_other_prototypes(self):
        """Prepare prototypes from other clients for each client"""
        other_prototypes = {}

        for client_id in self.all_prototypes.keys():
            other_prototypes[client_id] = {}

            for other_client_id, prototypes in self.all_prototypes.items():
                if other_client_id == client_id:
                    continue

                for class_idx, prototype in prototypes.items():
                    class_idx = int(class_idx)
                    if class_idx not in other_prototypes[client_id]:
                        other_prototypes[client_id][class_idx] = []
                    other_prototypes[client_id][class_idx].append(prototype)

        return other_prototypes

    def _broadcast_global_covariances(self, other_prototypes):
        """Broadcast global covariance matrices and prototypes to all clients.

        IMPORTANT: Message.transform_to_list() mutates dict values IN-PLACE,
        converting numpy arrays to Python lists (~7x memory expansion).
        We pre-serialize cov_matrices ONCE and reuse for all clients to avoid
        mutating self.global_cov_matrices and to reduce redundant conversions.
        """
        logger.info("Server: Broadcasting global covariances to clients...")

        # Pre-serialize covariance matrices once (65 × 512×512 numpy → Python lists)
        # This avoids transform_to_list mutating self.global_cov_matrices in-place
        cov_serialized = {
            k: v.tolist() if hasattr(v, 'tolist') else v
            for k, v in self.global_cov_matrices.items()
        }

        # Pre-serialize global prototypes once
        global_proto_serialized = {
            k: v.tolist() if hasattr(v, 'tolist') else v
            for k, v in self.global_prototypes.items()
        }

        for client_id in self.local_statistics_buffer.keys():
            # Pre-serialize this client's other_prototypes
            raw_other = other_prototypes.get(client_id, {})
            other_serialized = {}
            for cls_idx, protos in raw_other.items():
                if isinstance(protos, list):
                    other_serialized[cls_idx] = [
                        p.tolist() if hasattr(p, 'tolist') else p
                        for p in protos
                    ]
                else:
                    other_serialized[cls_idx] = protos

            content = {
                'cov_matrices': cov_serialized,
                'other_prototypes': {
                    client_id: other_serialized
                },
                'global_prototypes': global_proto_serialized,
            }

            self.comm_manager.send(
                Message(msg_type='global_covariances',
                        sender=self.ID,
                        receiver=[client_id],
                        state=self.state,
                        content=content))

        # Free pre-serialized data
        del cov_serialized, global_proto_serialized

        logger.info(
            f"Server: Broadcasted global covariances to {len(self.local_statistics_buffer)} clients"
        )

    def callback_funcs_model_para(self, message: Message):
        """
        Handle model parameter messages from clients.
        Aggregate using FedAvg.
        """
        round_idx = message.state
        sender = message.sender
        content = message.content

        if isinstance(content, tuple) and len(content) == 2:
            sample_size, model_para = content
        else:
            sample_size, model_para = 0, content

        # Store in message buffer
        if round_idx not in self.msg_buffer['train']:
            self.msg_buffer['train'][round_idx] = []

        self.msg_buffer['train'][round_idx].append(
            (sample_size, model_para, sender))

        logger.info(
            f"Server: Received model from client {sender} for round {round_idx} "
            f"({len(self.msg_buffer['train'][round_idx])}/{self._client_num})")

        # Check if all clients have responded
        if len(self.msg_buffer['train'][round_idx]) >= self._client_num:
            self._perform_fedavg(round_idx)

    def _perform_fedavg(self, round_idx):
        """Perform FedAvg aggregation for MLP (and optionally CNN)"""
        logger.info(
            f"Server: Performing FedAvg aggregation for round {round_idx}")

        # Collect all model parameters and clear buffer for this round
        all_params = self.msg_buffer['train'][round_idx]
        # Clear buffer to prevent memory leak (base Server.check_and_move_on
        # normally does this, but GGEUR bypasses it with custom _perform_fedavg)
        self.msg_buffer['train'][round_idx] = []

        # Filter out empty updates
        valid_params = [(s, p) for s, p, _ in all_params
                        if s > 0 and p is not None]

        if not valid_params:
            logger.warning("Server: No valid model parameters received")
            self.state = round_idx + 1
            if self.state < self._total_round_num:
                self._start_training_round()
            else:
                self._finish()
            return

        # Compute sample weights
        sample_sizes = [s for s, p in valid_params]
        total_samples = sum(sample_sizes)

        # Handle separated training mode
        if self.use_separated_training:
            self._perform_separated_fedavg(valid_params, total_samples,
                                           round_idx)
        else:
            # Check if we're in CNN distillation mode
            first_params = valid_params[0][1]
            is_combined_params = isinstance(first_params,
                                            dict) and 'mlp' in first_params

            if is_combined_params:
                # Aggregate MLP and CNN separately
                mlp_aggregated = self._aggregate_model_params(
                    [(s, p['mlp'])
                     for s, p in valid_params if p.get('mlp') is not None],
                    total_samples)
                cnn_aggregated = self._aggregate_model_params(
                    [(s, p['cnn'])
                     for s, p in valid_params if p.get('cnn') is not None],
                    total_samples)

                # Update global MLP
                if mlp_aggregated and self.global_mlp is not None:
                    try:
                        self.global_mlp.load_state_dict(mlp_aggregated)
                    except Exception as e:
                        logger.debug(f"Server: Could not load MLP params: {e}")

                # Update global CNN
                if cnn_aggregated and self.global_cnn is not None:
                    try:
                        self.global_cnn.load_state_dict(cnn_aggregated)
                    except Exception as e:
                        logger.debug(f"Server: Could not load CNN params: {e}")
            else:
                # Standard mode: only MLP
                mlp_aggregated = self._aggregate_model_params(
                    valid_params, total_samples)

                if mlp_aggregated and self.global_mlp is not None:
                    try:
                        self.global_mlp.load_state_dict(mlp_aggregated)
                    except Exception as e:
                        logger.debug(f"Server: Could not load MLP params: {e}")

        # Evaluate MLP on test sets (using CLIP features)
        test_results = self._evaluate_on_test_sets()
        if test_results:
            acc_str = ', '.join(
                [f"{k}: {v:.4f}" for k, v in test_results.items()])
            logger.info(
                f"Server: Round {round_idx} MLP Test Accuracy - {acc_str}")

        # Evaluate CNN on test sets (using original images) if enabled
        # Include separated training Phase 2
        should_eval_cnn = ((self.use_cnn_distillation
                            or self.use_feature_alignment)
                           or (self.use_separated_training
                               and self.training_phase == 'cnn_backbone'))
        if should_eval_cnn and self.global_cnn is not None:
            cnn_test_results = self._evaluate_cnn_on_test_sets()
            if cnn_test_results:
                acc_str = ', '.join(
                    [f"{k}: {v:.4f}" for k, v in cnn_test_results.items()])
                logger.info(
                    f"Server: Round {round_idx} CNN Test Accuracy - {acc_str}")

        # Log progress
        logger.info(
            f"Server: Round {round_idx} aggregation complete, total samples: {total_samples}"
        )

        # Move to next round
        self.state = round_idx + 1

        if self.state < self._total_round_num:
            self._start_training_round()
        else:
            self._finish()

    def _perform_separated_fedavg(self, valid_params, total_samples,
                                  round_idx):
        """Perform FedAvg for separated training mode"""
        first_params = valid_params[0][1]

        if self.training_phase == 'classifier':
            # Phase 1: Aggregate classifier parameters
            if isinstance(first_params, dict) and 'classifier' in first_params:
                classifier_aggregated = self._aggregate_model_params([
                    (s, p['classifier'])
                    for s, p in valid_params if p.get('classifier') is not None
                ], total_samples)
            else:
                classifier_aggregated = self._aggregate_model_params(
                    valid_params, total_samples)

            if classifier_aggregated and self.global_mlp is not None:
                try:
                    self.global_mlp.load_state_dict(classifier_aggregated)
                    logger.info(
                        f"Server: Phase 1 - Aggregated classifier from {len(valid_params)} clients"
                    )
                except Exception as e:
                    logger.debug(
                        f"Server: Could not load classifier params: {e}")

        else:
            # Phase 2: Aggregate only CNN backbone parameters (classifier is frozen)
            if isinstance(first_params,
                          dict) and 'cnn_backbone' in first_params:
                cnn_aggregated = self._aggregate_model_params(
                    [(s, p['cnn_backbone']) for s, p in valid_params
                     if p.get('cnn_backbone') is not None], total_samples)

                if cnn_aggregated and self.global_cnn is not None:
                    try:
                        self.global_cnn.load_state_dict(cnn_aggregated)
                        logger.info(
                            f"Server: Phase 2 - Aggregated CNN backbone from {len(valid_params)} clients"
                        )
                    except Exception as e:
                        logger.debug(
                            f"Server: Could not load CNN backbone params: {e}")

    def _aggregate_model_params(self, params_list, total_samples):
        """Helper function to aggregate model parameters using weighted average"""
        if not params_list:
            return None

        # Filter valid params
        valid_params = [(s, p) for s, p in params_list
                        if s > 0 and p is not None]
        if not valid_params:
            return None

        first_params = valid_params[0][1]
        aggregated_params = {}

        for key in first_params.keys():
            param_tensor = first_params[key]
            if not isinstance(param_tensor, torch.Tensor):
                param_tensor = torch.tensor(param_tensor)
            aggregated_params[key] = torch.zeros_like(param_tensor).float()

        # Weighted average
        for sample_size, params in valid_params:
            weight = sample_size / total_samples
            for key in params.keys():
                param_tensor = params[key]
                if not isinstance(param_tensor, torch.Tensor):
                    param_tensor = torch.tensor(param_tensor)
                aggregated_params[key] += weight * param_tensor.float()

        return aggregated_params

    def _load_test_images(self):
        """Load test images for CNN evaluation"""
        if self.test_images_loaded:
            return

        logger.info("Server: Loading test images for CNN evaluation...")

        data_type = self._cfg.data.type.lower()
        data_root = self._cfg.data.root

        # Get split parameters
        if hasattr(self._cfg.data, 'splits'):
            splits = tuple(self._cfg.data.splits)
        else:
            if 'office' in data_type and 'home' in data_type:
                splits = (0.7, 0.0, 0.3)
            else:
                splits = (0.8, 0.1, 0.1)

        train_ratio, val_ratio = splits[0], splits[1]
        seed = self._cfg.seed if hasattr(self._cfg, 'seed') else 123

        # Determine domains based on dataset type
        if 'pacs' in data_type:
            domains = ['photo', 'art_painting', 'cartoon', 'sketch']
            from federatedscope.cv.dataset.pacs import PACS
            dataset_class = PACS
        elif 'office' in data_type and 'home' in data_type:
            domains = ['Art', 'Clipart', 'Product', 'Real_World']
            from federatedscope.cv.dataset.office_home import OfficeHome
            dataset_class = OfficeHome
        else:
            logger.warning(
                f"Server: Unknown dataset type {data_type} for CNN evaluation")
            return

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        for domain in domains:
            try:
                test_dataset = dataset_class(root=data_root,
                                             domain=domain,
                                             split='test',
                                             transform=transform,
                                             train_ratio=train_ratio,
                                             val_ratio=val_ratio,
                                             seed=seed)

                if len(test_dataset) > 0:
                    self.test_image_loaders[domain] = DataLoader(test_dataset,
                                                                 batch_size=32,
                                                                 shuffle=False,
                                                                 num_workers=0)
                    logger.info(
                        f"Server: Loaded {len(test_dataset)} test images for {domain}"
                    )

            except Exception as e:
                logger.warning(
                    f"Server: Failed to load test images for {domain}: {e}")

        self.test_images_loaded = True

    def _evaluate_cnn_on_test_sets(self):
        """Evaluate global CNN on test sets using original images"""
        if self.global_cnn is None:
            logger.warning(
                "Server: global_cnn is None, skipping CNN evaluation")
            return {}

        # Load test images if not already loaded
        if not self.test_images_loaded:
            self._load_test_images()

        if not self.test_image_loaders:
            logger.warning("Server: No test image loaders available")
            return {}

        self.global_cnn.eval()
        results = {}

        # For separated training, we need to use the pretrained classifier
        classifier = None
        if self.use_separated_training:
            if self.pretrained_classifier is not None:
                classifier = self._build_classifier_from_state_dict(
                    self.pretrained_classifier)
                classifier.eval()
                logger.info(
                    f"Server: Using pretrained classifier for CNN evaluation")
            else:
                logger.warning(
                    "Server: Separated training mode but pretrained_classifier is None!"
                )
                return {}

        with torch.no_grad():
            for domain, dataloader in self.test_image_loaders.items():
                correct = 0
                total = 0

                for images, labels in dataloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Get CNN output
                    if self.use_separated_training and classifier is not None:
                        # Separated training: backbone -> features -> classifier -> logits
                        features = self.global_cnn(images)  # 512-dim features
                        outputs = classifier(features)  # logits
                    else:
                        # Other modes: CNN outputs logits directly
                        outputs = self.global_cnn(images)

                    _, predicted = torch.max(outputs, 1)

                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

                accuracy = correct / total if total > 0 else 0
                results[domain] = accuracy

                # Track history
                if domain not in self.cnn_test_accuracies_history:
                    self.cnn_test_accuracies_history[domain] = []
                self.cnn_test_accuracies_history[domain].append(accuracy)

        # Compute average
        if results:
            avg_accuracy = sum(results.values()) / len(results)
            results['average'] = avg_accuracy

            if 'average' not in self.cnn_test_accuracies_history:
                self.cnn_test_accuracies_history['average'] = []
            self.cnn_test_accuracies_history['average'].append(avg_accuracy)

            # Track best CNN model
            if avg_accuracy > self.best_cnn_avg_accuracy:
                self.best_cnn_avg_accuracy = avg_accuracy
                self.best_cnn_model_state = copy.deepcopy(
                    self.global_cnn.state_dict())

        return results

    def _finish(self):
        """Finish FL training"""
        logger.info("=" * 60)
        logger.info(f"Server: Training finished after {self.state} rounds")

        # Print MLP/Classifier final results
        if self.test_accuracies_history:
            logger.info("=" * 60)
            if self.use_separated_training:
                logger.info(
                    f"Classifier Final Test Results (Pretrained in {self.classifier_pretrain_rounds} rounds):"
                )
            else:
                logger.info("MLP Final Test Results:")
            for domain, acc_list in self.test_accuracies_history.items():
                if acc_list:
                    logger.info(
                        f"  {domain}: final={acc_list[-1]:.4f}, best={max(acc_list):.4f}"
                    )

            logger.info(
                f"Classifier Best Average Accuracy: {self.best_avg_accuracy:.4f}"
            )

        # Print CNN final results
        if self.use_separated_training and self.cnn_test_accuracies_history:
            logger.info("-" * 60)
            logger.info(
                "CNN Final Test Results (Separated Training - From Scratch):")
            for domain, acc_list in self.cnn_test_accuracies_history.items():
                if acc_list:
                    logger.info(
                        f"  {domain}: final={acc_list[-1]:.4f}, best={max(acc_list):.4f}"
                    )

            logger.info(
                f"CNN Best Average Accuracy: {self.best_cnn_avg_accuracy:.4f}")

        elif (self.use_cnn_distillation or
              self.use_feature_alignment) and self.cnn_test_accuracies_history:
            logger.info("-" * 60)
            logger.info("CNN Final Test Results (Feature Alignment):" if self.
                        use_feature_alignment else
                        "CNN Final Test Results (Distillation):")
            for domain, acc_list in self.cnn_test_accuracies_history.items():
                if acc_list:
                    logger.info(
                        f"  {domain}: final={acc_list[-1]:.4f}, best={max(acc_list):.4f}"
                    )

            logger.info(
                f"CNN Best Average Accuracy: {self.best_cnn_avg_accuracy:.4f}")

        logger.info("=" * 60)

        for client_id in range(1, self._client_num + 1):
            self.comm_manager.send(
                Message(msg_type='finish',
                        sender=self.ID,
                        receiver=[client_id],
                        state=self.state,
                        content=None))

        self._monitor.finish_fl()
