"""
GGEUR_Clip Trainer Implementation

A simplified trainer that:
1. Works with CLIP feature embeddings instead of raw images
2. Trains a simple MLP classifier on augmented features
3. Provides evaluation on test embeddings
"""

import logging
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.register import register_trainer

logger = logging.getLogger(__name__)


class EmbeddingDataset(Dataset):
    """Dataset for CLIP feature embeddings"""

    def __init__(self, embeddings, labels):
        if isinstance(embeddings, np.ndarray):
            self.embeddings = torch.from_numpy(embeddings).float()
        else:
            self.embeddings = embeddings.float()

        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class GGEURTrainer(GeneralTorchTrainer):
    """
    Trainer for GGEUR_Clip that works with feature embeddings.

    Key differences from standard trainer:
    1. Works with pre-extracted CLIP embeddings
    2. Uses simple MLP classifier
    3. Supports augmented data injection
    """

    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        # Note: model here might be a placeholder, we'll build MLP separately
        super(GGEURTrainer, self).__init__(model, data, device, config,
                                            only_for_eval, monitor)

        self.ggeur_cfg = config.ggeur if hasattr(config, 'ggeur') else None

        # MLP classifier for embeddings
        self.mlp_classifier = None

        # Augmented data
        self.augmented_embeddings = None
        self.augmented_labels = None
        self.augmented_loader = None

        # Test embeddings (pre-extracted CLIP features)
        self.test_embeddings = None
        self.test_labels = None
        self.test_loader = None

    def _build_mlp(self, num_classes):
        """Build MLP classifier"""
        if self.ggeur_cfg is None:
            input_dim = 512
            hidden_dim = 0
            dropout = 0.0
        else:
            input_dim = self.ggeur_cfg.embedding_dim
            hidden_dim = self.ggeur_cfg.mlp_hidden_dim
            dropout = self.ggeur_cfg.mlp_dropout

        if hidden_dim > 0:
            self.mlp_classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.mlp_classifier = nn.Linear(input_dim, num_classes)

        self.mlp_classifier = self.mlp_classifier.to(self.device)

    def setup_augmented_data(self, embeddings, labels):
        """
        Set up augmented data for training.
        Called by GGEURClient after augmentation.
        """
        self.augmented_embeddings = embeddings
        self.augmented_labels = labels

        # Create data loader
        dataset = EmbeddingDataset(embeddings, labels)
        batch_size = self.cfg.dataloader.batch_size if hasattr(self.cfg, 'dataloader') else 16

        self.augmented_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )

        # Build MLP if not done yet
        num_classes = len(np.unique(labels))
        if self.mlp_classifier is None:
            self._build_mlp(num_classes)

        logger.info(f"GGEURTrainer: Set up augmented data with {len(labels)} samples, {num_classes} classes")

    def setup_test_data(self, embeddings, labels):
        """Set up test embeddings for evaluation"""
        self.test_embeddings = embeddings
        self.test_labels = labels

        dataset = EmbeddingDataset(embeddings, labels)
        batch_size = self.cfg.dataloader.batch_size if hasattr(self.cfg, 'dataloader') else 16

        self.test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )

    def train(self):
        """Train on augmented embeddings"""
        if self.augmented_loader is None:
            logger.warning("GGEURTrainer: No augmented data available")
            return 0, self.get_model_para(), {}

        if self.mlp_classifier is None:
            num_classes = len(np.unique(self.augmented_labels))
            self._build_mlp(num_classes)

        self.mlp_classifier.train()

        optimizer = torch.optim.Adam(
            self.mlp_classifier.parameters(),
            lr=self.cfg.train.optimizer.lr if hasattr(self.cfg.train, 'optimizer') else 0.001
        )
        criterion = nn.CrossEntropyLoss()

        local_epochs = self.cfg.train.local_update_steps if hasattr(self.cfg.train, 'local_update_steps') else 1

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for embeddings, labels in self.augmented_loader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.mlp_classifier(embeddings)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * embeddings.size(0)
                _, predicted = torch.max(outputs, 1)
                epoch_correct += (predicted == labels).sum().item()
                epoch_samples += embeddings.size(0)

            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        results = {
            'train_loss': avg_loss,
            'train_acc': accuracy,
            'train_total': total_samples
        }

        return total_samples, self.get_model_para(), results

    def evaluate(self, target_data_split_name='test'):
        """Evaluate on test embeddings"""
        if self.test_loader is None:
            return {}

        if self.mlp_classifier is None:
            return {}

        self.mlp_classifier.eval()

        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for embeddings, labels in self.test_loader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)

                outputs = self.mlp_classifier(embeddings)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * embeddings.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / total if total > 0 else 0

        results = {
            f'{target_data_split_name}_acc': accuracy,
            f'{target_data_split_name}_loss': avg_loss,
            f'{target_data_split_name}_total': total
        }

        return results

    def get_model_para(self):
        """Get model parameters"""
        if self.mlp_classifier is not None:
            return copy.deepcopy(self.mlp_classifier.state_dict())
        return {}

    def update(self, model_para, strict=False):
        """Update model with parameters"""
        if self.mlp_classifier is not None and model_para:
            try:
                self.mlp_classifier.load_state_dict(model_para, strict=strict)
            except Exception as e:
                logger.debug(f"GGEURTrainer: Could not load model params: {e}")


def call_ggeur_trainer(trainer_type):
    """Factory function for GGEUR_Clip trainer"""
    if trainer_type.lower() == 'ggeur':
        return GGEURTrainer
    return None


register_trainer('ggeur', call_ggeur_trainer)
