"""
FedMIA Server Implementation.
Fully aligned with FedMIA-main repository implementation.
"""

from federatedscope.attack.worker_as_attacker.server_attacker import PassiveServer
from federatedscope.core.message import Message
from federatedscope.core.workers import Server
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import json
import random
import copy
import logging
from scipy.stats import norm
from sklearn import metrics
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader, Subset, Dataset

logger = logging.getLogger(__name__)


class FedMIAServer(PassiveServer):
    """
    FedMIA Attack Server.
    Fully aligned with FedMIA-main repository implementation.
    
    Target client: client 1 (FederatedScope uses 1-based IDs)
    Val client: client 2
    Shadow clients: clients 2 to K (excluding target)
    
    IMPORTANT - About Server Data Access:
    ========================================
    The server holds full_train_dataset and client_data_indices for ATTACK EVALUATION
    purposes only. This is NOT information leakage - it's the standard FedMIA design:
    
    1. Server uses all clients' models to score the SAME samples (target client's
       training data). This is the core of FedMIA - comparing how different models
       score the same samples.
    
    2. client_data_indices (ground truth membership) is used ONLY for computing
       AUC/TPR evaluation metrics, NOT as attack signals.
    
    3. Attack signals are:
       - Loss values (how confident each model is on each sample)
       - Gradient cosine similarity (how aligned each model's gradient is)
       - These are computed from model weights, NOT from ground truth labels
    
    4. The passive server assumption holds: server observes model updates but
       does not modify training process or directly access member labels.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Attack configuration
        # In FederatedScope, client IDs start from 1, not 0
        self.target_client_id = 1  # watch_train_client_id in original (mapped to FederatedScope's client 1)
        self.val_client_id = 2     # watch_val_client_id in original (mapped to FederatedScope's client 2)
        
        # Store model states for gradient computation
        self.global_model_state = None
        self.client_model_states = {}  # round -> {client_id -> state}
        
        # PKL storage
        self.pkl_dir = None
        self._attack_executed = False
        
        # Data indices for each client
        self.client_data_indices = {}  # client_id -> train_indices
        self.testset_idx = (50000 + np.arange(10000)).astype(int)
        
        # Parameter keys for gradient computation
        self.param_keys = None
        
        # Reference to full training dataset
        self.full_train_dataset = None
        self.full_test_dataset = None
        
        # Training configuration
        self.lr = self._cfg.train.optimizer.lr
        self.optim = self._cfg.train.optimizer.type
        
        # Try to get data references from self.data (server data)
        self._init_data_references()
        
        logger.info(f"FedMIAServer initialized: target_client={self.target_client_id}, "
                    f"val_client={self.val_client_id}")

    def _init_data_references(self):
        """Initialize references to training and test datasets."""
        # self.data is the server data, typically contains 'test' key
        if self.data is not None:
            # Try to get test dataset
            if 'test' in self.data:
                test_loader = self.data['test']
                if hasattr(test_loader, 'dataset'):
                    self.full_test_dataset = test_loader.dataset
                    logger.info(f"Got test dataset: {type(self.full_test_dataset)}")
            
            # Try to get train dataset from 'train' key if available
            if 'train' in self.data:
                train_loader = self.data['train']
                if hasattr(train_loader, 'dataset'):
                    self.full_train_dataset = train_loader.dataset
                    logger.info(f"Got train dataset from server: {type(self.full_train_dataset)}")
        
        # ---- Offline-feature mode: load all_train.pt + meta client indices ----
        data_root = getattr(self._cfg.data, 'root', None)
        if data_root is not None:
            import os
            all_train_path = os.path.join(data_root, 'all_train.pt')
            meta_path      = os.path.join(data_root, 'meta.pt')
            srv_test_path  = os.path.join(data_root, 'server_test.pt')

            if os.path.isfile(all_train_path):
                from torch.utils.data import TensorDataset
                d = torch.load(all_train_path, weights_only=False)
                self.full_train_dataset = TensorDataset(d['x'].float(), d['y'].long())
                logger.info(f"[FedMIA offline] Loaded all_train: {len(self.full_train_dataset)} samples")

            if os.path.isfile(srv_test_path) and self.full_test_dataset is None:
                from torch.utils.data import TensorDataset
                d = torch.load(srv_test_path, weights_only=False)
                self.full_test_dataset = TensorDataset(d['x'].float(), d['y'].long())
                logger.info(f"[FedMIA offline] Loaded server_test: {len(self.full_test_dataset)} samples")
                # Update testset_idx to valid range (0..9999)
                self.testset_idx = np.arange(len(self.full_test_dataset))

            if os.path.isfile(meta_path):
                meta = torch.load(meta_path, weights_only=False)
                if 'client_train_indices' in meta:
                    # meta stores {cid(int) -> list} but dict keys may be int
                    raw = meta['client_train_indices']
                    self.client_data_indices = {int(k): list(v) for k, v in raw.items()}
                    logger.info(f"[FedMIA offline] Loaded client indices for {len(self.client_data_indices)} clients")

        # If we don't have train dataset, we'll try to get it later from clients

    def link_clients(self, clients):
        """Link clients to server to get data references."""
        logger.info(f"link_clients called with {len(clients) if clients else 0} clients")
        
        # Call parent if exists
        try:
            if hasattr(super(), 'link_clients'):
                super().link_clients(clients)
        except Exception as e:
            logger.warning(f"super().link_clients failed: {e}")
        
        # Try to get train dataset and indices from clients
        try:
            if clients and len(clients) > 0:
                first_client = clients.get(1) if 1 in clients else list(clients.values())[0]
                logger.info(f"First client type: {type(first_client)}, has trainer: {hasattr(first_client, 'trainer')}")
                
                if first_client and hasattr(first_client, 'trainer'):
                    trainer = first_client.trainer
                    logger.info(f"Trainer type: {type(trainer)}")
                    
                    # Data is stored in trainer.ctx.data
                    if trainer and hasattr(trainer, 'ctx') and hasattr(trainer.ctx, 'data'):
                        client_data = trainer.ctx.data
                        logger.info(f"Client data type: {type(client_data)}")
                        
                        # client_data is a ClientData (subclass of dict)
                        # - client_data.train_data is the raw dataset
                        # - client_data['train'] is the DataLoader
                        if client_data:
                            # Only try to get train data if not already loaded from all_train.pt
                            if self.full_train_dataset is None:
                                # Try to get train data from train_data attribute first
                                if hasattr(client_data, 'train_data') and client_data.train_data is not None:
                                    train_dataset = client_data.train_data
                                    logger.info(f"Train data type: {type(train_dataset)}")
                                    # If it's a Subset, get underlying dataset and indices
                                    if hasattr(train_dataset, 'dataset'):
                                        self.full_train_dataset = train_dataset.dataset
                                        logger.info(f"Got full train dataset from train_data: {type(self.full_train_dataset)}, len: {len(self.full_train_dataset) if hasattr(self.full_train_dataset, '__len__') else 'N/A'}")
                                    else:
                                        self.full_train_dataset = train_dataset
                                        logger.info(f"Got train dataset: {type(self.full_train_dataset)}")
                                # Fallback: try to get from 'train' key (DataLoader)
                                elif 'train' in client_data:
                                    train_loader = client_data['train']
                                    if hasattr(train_loader, 'dataset'):
                                        dataset = train_loader.dataset
                                        logger.info(f"Train dataset type: {type(dataset)}")
                                        if hasattr(dataset, 'dataset'):
                                            self.full_train_dataset = dataset.dataset
                                        else:
                                            self.full_train_dataset = dataset
                                        logger.info(f"Got full train dataset from loader: {type(self.full_train_dataset)}")
                            else:
                                logger.info(f"[FedMIA] full_train_dataset already set ({len(self.full_train_dataset)} samples), skipping client override")
                            
                            # Try to get test data
                            if hasattr(client_data, 'test_data') and client_data.test_data is not None and self.full_test_dataset is None:
                                self.full_test_dataset = client_data.test_data
                                logger.info(f"Got test dataset from test_data: {type(self.full_test_dataset)}")
                            elif 'test' in client_data and self.full_test_dataset is None:
                                test_data = client_data['test']
                                if hasattr(test_data, 'dataset'):
                                    self.full_test_dataset = test_data.dataset
                                    logger.info(f"Got test dataset from client: {type(self.full_test_dataset)}")
            
            # Now try to get client data indices from splitter or from client data
            self._get_client_indices_from_data(clients)
            
        except Exception as e:
            logger.warning(f"Error in link_clients: {e}")
            import traceback
            logger.warning(traceback.format_exc())
    
    def _get_client_indices_from_data(self, clients):
        """Get training data indices for each client."""
        for client_id, client in clients.items():
            if client and hasattr(client, 'trainer'):
                trainer = client.trainer
                # Data is stored in trainer.ctx.data
                if trainer and hasattr(trainer, 'ctx') and hasattr(trainer.ctx, 'data'):
                    client_data = trainer.ctx.data
                    # client_data is a ClientData (subclass of dict)
                    # client_data.train_data is the raw dataset (Subset)
                    if client_data and hasattr(client_data, 'train_data') and client_data.train_data is not None:
                        train_dataset = client_data.train_data
                        # If it's a Subset, we can get indices
                        if hasattr(train_dataset, 'indices'):
                            self.client_data_indices[client_id] = list(train_dataset.indices)
                        # Also try to get the underlying dataset if needed
                        if hasattr(train_dataset, 'dataset') and self.full_train_dataset is None:
                            self.full_train_dataset = train_dataset.dataset
                            logger.info(f"Got full train dataset from client {client_id}: {type(self.full_train_dataset)}")
        
        logger.info(f"Got client data indices for {len(self.client_data_indices)} clients")
        for cid, indices in list(self.client_data_indices.items())[:3]:
            logger.info(f"  Client {cid}: {len(indices)} samples")

    def _get_param_keys(self, state_dict: dict) -> List[str]:
        """Get parameter keys for gradient computation (exclude BN stats)."""
        return [k for k in state_dict.keys()
                if 'running_mean' not in k and 'running_var' not in k
                and 'num_batches_tracked' not in k]

    def _initialize_pkl_dir(self):
        """Initialize PKL directory."""
        if self.pkl_dir is None:
            self.pkl_dir = self._cfg.outdir
            os.makedirs(self.pkl_dir, exist_ok=True)
            logger.info(f"PKL directory: {self.pkl_dir}")

    def callback_funcs_model_para(self, message: Message):
        """
        Process client model updates and save PKL files.
        This is called when a client sends its model update.
        """
        round_num = message.state
        sender_id = message.sender
        content = message.content
        model_para = content[1] if isinstance(content, tuple) else content
        
        # Initialize PKL directory
        self._initialize_pkl_dir()
        
        # Store global model state (before processing any client update this round)
        if round_num not in self.client_model_states:
            self.global_model_state = copy.deepcopy(self.model.state_dict())
            if self.param_keys is None:
                self.param_keys = self._get_param_keys(self.global_model_state)
            self.client_model_states[round_num] = {}
        
        # Store client model state
        self.client_model_states[round_num][sender_id] = copy.deepcopy(model_para)
        
        # Save PKL file every 10 rounds and at the final round
        is_final_round = (round_num == self.total_round_num - 1)
        if round_num % 10 == 0 or is_final_round:
            self._save_client_pkl(round_num, sender_id, model_para)
        
        # Standard aggregation
        Server.callback_funcs_model_para(self, message)
        
        # Execute attack at the end of training
        if is_final_round and not self._attack_executed:
            self._attack_executed = True
            self.run_final_attack()

    def _save_client_pkl(self, round_num: int, client_id: int, model_para: dict):
        """
        Save PKL file for a client at a specific round.
        Aligned with FedMIA-main main.py save logic.
        """
        # Initialize save dictionary
        save_dict = {}
        
        # Get device
        device = self.device
        
        # Load model with client parameters and ensure correct device
        self.model.load_state_dict(model_para, strict=False)
        self.model = self.model.to(device)
        self.model.eval()
        
        criterion_noreduce = nn.CrossEntropyLoss(reduction='none')
        
        # Compute test_acc and test_loss on test set
        test_acc, test_loss = self._compute_test_accuracy()
        save_dict['test_acc'] = test_acc
        save_dict['test_loss'] = test_loss
        
        # test_index: Fixed as 50000 + np.arange(10000) for CIFAR-100
        save_dict['test_index'] = self.testset_idx
        
        # Compute test_res
        if self.full_test_dataset is not None:
            test_loader = DataLoader(self.full_test_dataset, batch_size=64, 
                                    shuffle=False, num_workers=2)
            test_res = self._get_all_losses(test_loader, self.model, criterion_noreduce, device)
            save_dict['test_res'] = test_res
        else:
            save_dict['test_res'] = {'loss': np.array([]), 'logit': torch.tensor([]), 'labels': torch.tensor([])}
        
        # Get indices for train/val/mix
        train_indices, val_indices, mix_indices = self._get_client_indices(client_id)
        
        # train_index and train_res
        save_dict['train_index'] = np.array(train_indices) if train_indices else np.array([])
        if self.full_train_dataset is not None and len(train_indices) > 0:
            save_dict['train_res'] = self._get_all_losses_from_indexes(
                self.full_train_dataset, train_indices, self.model, device)
        else:
            save_dict['train_res'] = {'loss': np.array([]), 'logit': torch.tensor([]), 'labels': torch.tensor([])}
        
        # val_index and val_res
        save_dict['val_index'] = np.array(val_indices) if val_indices else np.array([])
        if self.full_train_dataset is not None and len(val_indices) > 0:
            save_dict['val_res'] = self._get_all_losses_from_indexes(
                self.full_train_dataset, val_indices, self.model, device)
        else:
            save_dict['val_res'] = {'loss': np.array([]), 'logit': torch.tensor([]), 'labels': torch.tensor([])}
        
        # mix_index and mix_res
        save_dict['mix_index'] = np.array(mix_indices) if mix_indices else np.array([])
        if self.full_train_dataset is not None and len(mix_indices) > 0:
            save_dict['mix_res'] = self._get_all_losses_from_indexes(
                self.full_train_dataset, mix_indices, self.model, device)
        else:
            save_dict['mix_res'] = {'loss': np.array([]), 'logit': torch.tensor([]), 'labels': torch.tensor([])}
        
        # Compute gradient features (cos, diff, norm)
        # IMPORTANT: We compute gradient features for ALL clients, not just target client.
        # This is required for FedMIA-II which needs shadow clients' cosine scores.
        # FedMIA design: Server uses all clients' models to score the SAME samples
        # (target client's training data), then compares distributions.
        if self.global_model_state is not None and self.full_train_dataset is not None:
            # Compute for all clients (not just target) - FedMIA-II needs shadow cosines
            grad_features = self._compute_gradient_features(
                model_para, train_indices, val_indices, mix_indices, device)
            
            # Note: Original FedMIA has typo 'tarin' instead of 'train'
            save_dict['tarin_cos'] = grad_features.get('train_cos', [])
            save_dict['val_cos'] = grad_features.get('val_cos', [])
            save_dict['test_cos'] = grad_features.get('test_cos', [])
            save_dict['mix_cos'] = grad_features.get('mix_cos', [])
            
            save_dict['tarin_diffs'] = grad_features.get('train_diffs', [])
            save_dict['val_diffs'] = grad_features.get('val_diffs', [])
            save_dict['test_diffs'] = grad_features.get('test_diffs', [])
            save_dict['mix_diffs'] = grad_features.get('mix_diffs', [])
            
            save_dict['tarin_grad_norm'] = grad_features.get('train_norm', [])
            save_dict['val_grad_norm'] = grad_features.get('val_norm', [])
            save_dict['test_grad_norm'] = grad_features.get('test_norm', [])
            save_dict['mix_grad_norm'] = grad_features.get('mix_norm', [])
            # Note: All clients get gradient features now, including shadow clients
        else:
            save_dict['tarin_cos'] = []
            save_dict['val_cos'] = []
            save_dict['test_cos'] = []
            save_dict['mix_cos'] = []
            save_dict['tarin_diffs'] = []
            save_dict['val_diffs'] = []
            save_dict['test_diffs'] = []
            save_dict['mix_diffs'] = []
            save_dict['tarin_grad_norm'] = []
            save_dict['val_grad_norm'] = []
            save_dict['test_grad_norm'] = []
            save_dict['mix_grad_norm'] = []
        
        # Save PKL file
        file_path = os.path.join(self.pkl_dir, f"client_{client_id}_losses_epoch{round_num}.pkl")
        torch.save(save_dict, file_path)
        logger.info(f"Saved PKL: {file_path}")

    def _compute_test_accuracy(self) -> Tuple[float, float]:
        """Compute test accuracy and loss."""
        if self.data and 'test' in self.data:
            test_loader = self.data['test']
            self.model.eval()
            correct = 0
            total = 0
            total_loss = 0.0
            criterion = nn.CrossEntropyLoss()
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            acc = correct / total if total > 0 else 0.0
            avg_loss = total_loss / total if total > 0 else 0.0
            return acc, avg_loss
        return 0.0, 0.0

    def _get_all_losses(self, dataloader, model, criterion, device):
        """Compute loss, logit, labels for a dataloader."""
        model.eval()
        losses = []
        logits = []
        labels = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                losses.append(loss.cpu().numpy())
                logits.append(outputs.cpu())
                labels.append(targets.cpu())
        
        return {
            "loss": np.concatenate(losses),
            "logit": torch.cat(logits),
            "labels": torch.cat(labels)
        }

    def _get_all_losses_from_indexes(self, dataset, indexes, model, device):
        """Compute loss, logit, labels for samples at specific indexes."""
        criterion = nn.CrossEntropyLoss(reduction='none')
        subset = Subset(dataset, indexes)
        dataloader = DataLoader(subset, batch_size=200, shuffle=False, num_workers=0)
        
        model.eval()
        losses = []
        logits = []
        labels = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                losses.append(loss.cpu().numpy())
                logits.append(outputs.cpu())
                labels.append(targets.cpu())
        
        return {
            "loss": np.concatenate(losses),
            "logit": torch.cat(logits),
            "labels": torch.cat(labels)
        }

    def _get_client_indices(self, client_id: int) -> Tuple[List, List, List]:
        """
        Get training indices for train/val/mix datasets.
        
        Returns:
            train_indices: Target client's training indices (client 0)
            val_indices: Val client's training indices (client 1)
            mix_indices: Mixed indices from shadow clients
        """
        train_indices = []
        val_indices = []
        mix_indices = []
        
        client_num = self._cfg.federate.client_num
        data_num = int(10000 / client_num)  # For mix sampling
        
        # Get indices from stored client data
        if self.target_client_id in self.client_data_indices:
            train_indices = self.client_data_indices[self.target_client_id]
        
        if self.val_client_id in self.client_data_indices:
            val_indices = self.client_data_indices[self.val_client_id]
        
        # Mix indices: sample from each shadow client
        for c_id in range(1, client_num):
            if c_id in self.client_data_indices and len(self.client_data_indices[c_id]) > 0:
                sample_num = min(data_num, len(self.client_data_indices[c_id]))
                mix_indices.extend(random.sample(list(self.client_data_indices[c_id]), sample_num))
        
        return train_indices, val_indices, mix_indices

    def _compute_gradient_features(self, local_state, train_indices, val_indices, 
                                    mix_indices, device) -> dict:
        """
        Compute gradient-based features (cos, diff, norm).
        Aligned with FedMIA-main get_all_cos function.
        Uses manual per-sample gradient computation instead of Opacus.
        """
        global_state = self.global_model_state
        
        # Compute model gradients (global - local)
        model_grads = self._get_model_grads(global_state, local_state)
        
        if len(train_indices) == 0:
            return {}
        
        # Create a copy of the model for gradient computation
        cos_model = copy.deepcopy(self.model)
        cos_model.load_state_dict(global_state)
        cos_model = cos_model.to(device)

        # Check if we are in shared-backbone mode (model is ClassifierHead)
        # In that case, inject the shared backbone so _get_cos_score can use it
        shared_backbone = getattr(self, '_shared_backbone', None)
        if shared_backbone is not None:
            # Move backbone to compute device
            shared_backbone.to(device)
        
        # Get param_keys for consistent gradient computation
        param_keys = self.param_keys
        
        # Compute features for each subset using manual per-sample gradient computation
        results = {}
        
        # Train indices
        if len(train_indices) > 0:
            train_loader = DataLoader(
                Subset(self.full_train_dataset, train_indices),
                batch_size=1, shuffle=False, num_workers=0
            )
            train_cos, train_diffs, train_norm = self._get_cos_score(
                train_loader, cos_model, device, model_grads, param_keys,
                shared_backbone=shared_backbone)
            results['train_cos'] = train_cos
            results['train_diffs'] = train_diffs
            results['train_norm'] = train_norm
        
        # Test indices (needed_test_indexs)
        needed_test_indexs = list(range(0, 10000, 10))[:len(train_indices)] if len(train_indices) < 10000 else range(0, 10000)
        if self.full_test_dataset is not None:
            test_loader = DataLoader(
                Subset(self.full_test_dataset, list(needed_test_indexs)),
                batch_size=1, shuffle=False, num_workers=0
            )
            test_cos, test_diffs, test_norm = self._get_cos_score(
                test_loader, cos_model, device, model_grads, param_keys,
                shared_backbone=shared_backbone)
            results['test_cos'] = test_cos
            results['test_diffs'] = test_diffs
            results['test_norm'] = test_norm
        
        # Mix indices
        if len(mix_indices) > 0:
            mix_loader = DataLoader(
                Subset(self.full_train_dataset, mix_indices),
                batch_size=1, shuffle=False, num_workers=0
            )
            mix_cos, mix_diffs, mix_norm = self._get_cos_score(
                mix_loader, cos_model, device, model_grads, param_keys,
                shared_backbone=shared_backbone)
            results['mix_cos'] = mix_cos
            results['mix_diffs'] = mix_diffs
            results['mix_norm'] = mix_norm
        
        return results

    def _get_model_grads(self, global_state, local_state):
        """Compute model gradient (global - local) as flattened tensor."""
        grads = []
        for name in self.param_keys:
            if name in local_state and name in global_state:
                # Move both tensors to CPU for computation
                g = global_state[name].cpu() if global_state[name].device.type != 'cpu' else global_state[name]
                l = local_state[name].cpu() if local_state[name].device.type != 'cpu' else local_state[name]
                para_diff = g - l
                grads.append(para_diff.detach().flatten())
        return torch.cat(grads) if grads else torch.zeros(1)

    def _get_cos_score(self, samples_ldr, model, device, model_grads, param_keys,
                       shared_backbone=None):
        """
        Compute cosine similarity, gradient difference, and gradient norm for each sample.
        Uses manual per-sample gradient computation (batch_size=1) instead of Opacus.
        Uses eval() mode to avoid BatchNorm issues with single sample.
        Uses param_keys to ensure consistent gradient computation with model_grads.

        Args:
            shared_backbone: If provided (SharedBackbone), images are first passed
                             through backbone before the head model. This is the case
                             when model.type = 'convnext_*_head'.
        """
        model_grads = model_grads.to(device)
        model_diff_norm = torch.norm(model_grads, p=2) ** 2
        cos_scores, grad_diffs, sample_norms = [], [], []

        # Build a mapping from name to parameter, only for params in param_keys
        param_dict = {name: p for name, p in model.named_parameters() if name in param_keys}
        # Get the ordered list of param names that exist in both param_keys and model
        ordered_param_names = [name for name in param_keys if name in param_dict]
        
        # Use eval() mode to avoid BatchNorm error with batch_size=1
        # BatchNorm will use running statistics instead of batch statistics
        model.eval()
        for x, y in samples_ldr:
            for i in range(x.shape[0]):  # 逐样本
                xi = x[i:i+1].to(device)
                yi = y[i:i+1].to(device)
                model.zero_grad()
                # If shared backbone mode: extract features first (no_grad),
                # then forward through ClassifierHead (with grad for param_keys)
                if shared_backbone is not None:
                    with torch.no_grad():
                        xi = shared_backbone(xi)   # (1, feature_dim)
                loss = F.cross_entropy(model(xi), yi)
                loss.backward()
                # Collect gradients in the same order as param_keys
                # For params with no grad, fill with zeros to match model_grads size
                grads = []
                for name in ordered_param_names:
                    p = param_dict[name]
                    if p.grad is not None:
                        grads.append(p.grad.detach().flatten())
                    else:
                        # Fill with zeros for params not used in forward pass
                        grads.append(torch.zeros_like(p).flatten())
                sample_grad = torch.cat(grads)
                cos_scores.append(F.cosine_similarity(sample_grad, model_grads, dim=0).item())
                grad_diffs.append((model_diff_norm - torch.norm(model_grads - sample_grad, p=2)**2).item())
                sample_norms.append(torch.norm(sample_grad, p=2).item() ** 2)

        return torch.tensor(cos_scores), torch.tensor(grad_diffs), torch.tensor(sample_norms)

    def run_final_attack(self):
        """
        Execute all 8 attack methods after training completes.
        Aligned with FedMIA-main mia_attack_auto.py attack_comparison function.
        """
        logger.info("=" * 20 + " FedMIA Final Attack " + "=" * 20)
        
        if self.pkl_dir is None or not os.path.exists(self.pkl_dir):
            logger.warning("No PKL files found, cannot execute attack")
            return
        
        # Load all PKL files
        pkl_data = self._load_all_pkl_files()
        if not pkl_data:
            logger.warning("Failed to load PKL data")
            return
        
        # Get all round numbers
        all_rounds = sorted(set(
            int(f.split('epoch')[1].split('.')[0])
            for f in os.listdir(self.pkl_dir)
            if f.endswith('.pkl')
        ))
        
        logger.info(f"Loaded PKL data for rounds: {all_rounds}")
        
        # Number of clients (K in FedMIA)
        K = self._cfg.federate.client_num
        
        # Execute all 8 attacks
        results = {}
        
        # Attack modes: 'test' or 'mix'
        MODE = 'mix'  # Default to mix mode as in original FedMIA
        
        # 1. Blackbox-Loss Attack
        results['Blackbox-Loss'] = self._attack_blackbox_loss(pkl_data, all_rounds, MODE, K)
        
        # 2. Grad-Cosine Attack
        results['Grad-Cosine'] = self._attack_grad_cosine(pkl_data, all_rounds, MODE, K)
        
        # 3. Grad-Diff Attack
        results['Grad-Diff'] = self._attack_grad_diff(pkl_data, all_rounds, MODE, K)
        
        # 4. Grad-Norm Attack
        results['Grad-Norm'] = self._attack_grad_norm(pkl_data, all_rounds, MODE, K)
        
        # 5. Loss-Series Attack
        results['Loss-Series'] = self._attack_loss_series(pkl_data, all_rounds, MODE, K)
        
        # 6. Avg-Cosine Attack
        results['Avg-Cosine'] = self._attack_avg_cosine(pkl_data, all_rounds, MODE, K)
        
        # 7. FedMIA-I Attack (loss-based)
        results['FedMIA-I'] = self._attack_fedmia_i(pkl_data, all_rounds, MODE, K)
        
        # 8. FedMIA-II Attack (cosine-based)
        results['FedMIA-II'] = self._attack_fedmia_ii(pkl_data, all_rounds, MODE, K)
        
        # Print and save results
        self._print_and_save_results(results, all_rounds)

    def _load_all_pkl_files(self) -> Dict:
        """Load all PKL files and organize by round."""
        pkl_data = {}  # round -> {client_id -> data}
        
        for filename in os.listdir(self.pkl_dir):
            if not filename.endswith('.pkl'):
                continue
            
            try:
                parts = filename.replace('.pkl', '').split('_')
                client_id = int(parts[1])
                round_num = int(parts[3].replace('epoch', ''))
                
                file_path = os.path.join(self.pkl_dir, filename)
                data = torch.load(file_path, weights_only=False)
                
                if round_num not in pkl_data:
                    pkl_data[round_num] = {}
                pkl_data[round_num][client_id] = data
                
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}")
        
        return pkl_data

    def _get_member_nonmember_scores(self, pkl_data, round_num, attack_mode, MODE, mix_length=1000):
        """
        Get member and non-member scores for a specific round.
        
        Member: client 0's train_res/tarin_cos
        Non-member: test_res/test_cos (test mode) or mix_res/mix_cos (mix mode)
        """
        round_data = pkl_data.get(round_num, {})
        target_data = round_data.get(self.target_client_id, {})  # Use configured target client
        
        member_scores = np.array([])
        nonmember_scores = np.array([])
        
        if attack_mode == 'loss':
            member_scores = -self._ce_loss_fn(
                target_data.get('train_res', {}).get('logit', torch.tensor([])),
                target_data.get('train_res', {}).get('labels', torch.tensor([]))
            ).cpu().numpy() if len(target_data.get('train_res', {}).get('logit', [])) > 0 else np.array([])
            
            if MODE == 'test':
                nonmember_scores = -self._ce_loss_fn(
                    target_data.get('test_res', {}).get('logit', torch.tensor([])),
                    target_data.get('test_res', {}).get('labels', torch.tensor([]))
                ).cpu().numpy() if len(target_data.get('test_res', {}).get('logit', [])) > 0 else np.array([])
            elif MODE == 'mix':
                test_logit = target_data.get('test_res', {}).get('logit', torch.tensor([]))
                test_labels = target_data.get('test_res', {}).get('labels', torch.tensor([]))
                if len(test_logit) > 0:
                    random_indices = torch.randperm(test_logit.shape[0])
                    test_scores = -self._ce_loss_fn(
                        test_logit[random_indices[:mix_length]],
                        test_labels[random_indices[:mix_length]]
                    ).cpu().numpy()
                    mix_scores = -self._ce_loss_fn(
                        target_data.get('mix_res', {}).get('logit', torch.tensor([])),
                        target_data.get('mix_res', {}).get('labels', torch.tensor([]))
                    ).cpu().numpy()
                    nonmember_scores = np.concatenate([test_scores, mix_scores])
                
        elif attack_mode == 'cos':
            member_scores = np.array([x.item() if hasattr(x, 'item') else x for x in target_data.get('tarin_cos', [])])
            
            if MODE == 'test':
                nonmember_scores = np.array([x.item() if hasattr(x, 'item') else x for x in target_data.get('test_cos', [])])
            elif MODE == 'mix':
                test_cos = target_data.get('test_cos', [])
                if len(test_cos) > 0:
                    random_indices = torch.randperm(len(test_cos))
                    test_scores = torch.tensor([test_cos[i] for i in random_indices[:mix_length]])
                    mix_scores = torch.tensor(target_data.get('mix_cos', []))
                    nonmember_scores = torch.cat([test_scores, mix_scores]).cpu().numpy()
                    
        elif attack_mode == 'diff':
            member_scores = np.array([x.item() if hasattr(x, 'item') else x for x in target_data.get('tarin_diffs', [])])
            
            if MODE == 'test':
                nonmember_scores = np.array([x.item() if hasattr(x, 'item') else x for x in target_data.get('test_diffs', [])])
            elif MODE == 'mix':
                test_diffs = target_data.get('test_diffs', [])
                if len(test_diffs) > 0:
                    random_indices = torch.randperm(len(test_diffs))
                    test_scores = torch.tensor([test_diffs[i] for i in random_indices[:mix_length]])
                    mix_scores = torch.tensor(target_data.get('mix_diffs', []))
                    nonmember_scores = torch.cat([test_scores, mix_scores]).cpu().numpy()
                    
        elif attack_mode == 'norm':
            member_scores = -np.array([x.item() if hasattr(x, 'item') else x for x in target_data.get('tarin_grad_norm', [])])
            
            if MODE == 'test':
                nonmember_scores = -np.array([x.item() if hasattr(x, 'item') else x for x in target_data.get('test_grad_norm', [])])
            elif MODE == 'mix':
                test_norm = target_data.get('test_grad_norm', [])
                if len(test_norm) > 0:
                    random_indices = torch.randperm(len(test_norm))
                    test_scores = -torch.tensor([test_norm[i] for i in random_indices[:mix_length]])
                    mix_scores = -torch.tensor(target_data.get('mix_grad_norm', []))
                    nonmember_scores = torch.cat([test_scores, mix_scores]).cpu().numpy()
        
        return member_scores, nonmember_scores

    def _get_shadow_scores(self, pkl_data, round_num, attack_mode, MODE, mix_length=1000):
        """Get shadow client scores for Q_out distribution estimation."""
        round_data = pkl_data.get(round_num, {})
        
        shadow_train_scores = []
        shadow_test_scores = []
        
        # Shadow clients are all clients except target client
        # Note: In FederatedScope, client IDs start from 1 (not 0)
        for client_id in sorted(round_data.keys()):
            if client_id == self.target_client_id:  # Skip target client (ID=1 in FederatedScope)
                continue
            
            client_data = round_data[client_id]
            
            if attack_mode == 'loss':
                train_score = -self._ce_loss_fn(
                    client_data.get('train_res', {}).get('logit', torch.tensor([])),
                    client_data.get('train_res', {}).get('labels', torch.tensor([]))
                ).cpu().numpy() if len(client_data.get('train_res', {}).get('logit', [])) > 0 else np.array([])
                
                if MODE == 'test':
                    test_score = -self._ce_loss_fn(
                        client_data.get('test_res', {}).get('logit', torch.tensor([])),
                        client_data.get('test_res', {}).get('labels', torch.tensor([]))
                    ).cpu().numpy() if len(client_data.get('test_res', {}).get('logit', [])) > 0 else np.array([])
                elif MODE == 'mix':
                    test_logit = client_data.get('test_res', {}).get('logit', torch.tensor([]))
                    test_labels = client_data.get('test_res', {}).get('labels', torch.tensor([]))
                    if len(test_logit) > 0:
                        random_indices = torch.randperm(test_logit.shape[0])
                        test_s = -self._ce_loss_fn(
                            test_logit[random_indices[:mix_length]],
                            test_labels[random_indices[:mix_length]]
                        ).cpu().numpy()
                        mix_s = -self._ce_loss_fn(
                            client_data.get('mix_res', {}).get('logit', torch.tensor([])),
                            client_data.get('mix_res', {}).get('labels', torch.tensor([]))
                        ).cpu().numpy()
                        test_score = np.concatenate([test_s, mix_s])
                    else:
                        test_score = np.array([])
                        
            elif attack_mode == 'cos':
                train_score = np.array([x.item() if hasattr(x, 'item') else x for x in client_data.get('tarin_cos', [])])
                if MODE == 'test':
                    test_score = np.array([x.item() if hasattr(x, 'item') else x for x in client_data.get('test_cos', [])])
                elif MODE == 'mix':
                    test_cos = client_data.get('test_cos', [])
                    if len(test_cos) > 0:
                        random_indices = torch.randperm(len(test_cos))
                        test_s = torch.tensor([test_cos[i] for i in random_indices[:mix_length]])
                        mix_s = torch.tensor(client_data.get('mix_cos', []))
                        test_score = torch.cat([test_s, mix_s]).cpu().numpy()
                    else:
                        test_score = np.array([])
            else:
                train_score = np.array([])
                test_score = np.array([])
            
            if len(train_score) > 0:
                shadow_train_scores.append(train_score)
            if len(test_score) > 0:
                shadow_test_scores.append(test_score)
        
        return shadow_train_scores, shadow_test_scores

    def _ce_loss_fn(self, x, y):
        """Compute cross-entropy loss without reduction."""
        if len(x) == 0 or len(y) == 0:
            return torch.tensor([])
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        return loss_fn(x, y)

    def _plot_auc(self, member_scores, nonmember_scores):
        """
        Compute AUC and TPR@FPR metrics.
        Aligned with FedMIA-main plot_auc function.
        """
        if len(member_scores) == 0 or len(nonmember_scores) == 0:
            return {'auc': float('nan'), 'log_auc': float('nan'),
                    'tpr@0.1': float('nan'), 'tpr@0.01': float('nan'), 'tpr@0.001': float('nan')}
        
        # Create labels and scores
        member_scores = np.array(member_scores).flatten()
        nonmember_scores = np.array(nonmember_scores).flatten()
        
        y_true = np.concatenate([np.zeros(len(nonmember_scores)), np.ones(len(member_scores))])
        y_scores = np.concatenate([nonmember_scores, member_scores])
        
        # Compute ROC curve
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
        auc = metrics.auc(fpr, tpr)
        
        # Log-scale AUC
        log_tpr, log_fpr = np.log10(tpr + 1e-10), np.log10(fpr + 1e-10)
        log_tpr[log_tpr < -5] = -5
        log_fpr[log_fpr < -5] = -5
        log_fpr = (log_fpr + 5) / 5.0
        log_tpr = (log_tpr + 5) / 5.0
        log_auc = metrics.auc(log_fpr, log_tpr)
        
        # TPR at specific FPR thresholds
        tprs = {}
        for fpr_thres in [10, 1, 0.1, 0.02, 0.01, 0.001, 0.0001]:
            tpr_index = np.sum(fpr < fpr_thres)
            if tpr_index > 0:
                tprs[str(fpr_thres)] = tpr[tpr_index - 1]
            else:
                tprs[str(fpr_thres)] = 0.0
        
        return {
            'auc': auc,
            'log_auc': log_auc,
            'tpr@0.1': tprs.get('0.1', 0.0),
            'tpr@0.01': tprs.get('0.01', 0.0),
            'tpr@0.001': tprs.get('0.001', 0.0),
            'tprs': tprs
        }

    def _get_last_valid_round(self, pkl_data, rounds):
        """
        Find the last round that has target client data.
        
        Since client sampling is random, target client may not be sampled
        in the final round. We need to find the last round where target
        client was actually sampled.
        
        Args:
            pkl_data: Dict of round -> client_id -> data
            rounds: List of available round numbers
            
        Returns:
            The last round number that has target client data, or rounds[-1] as fallback
        """
        for round_num in reversed(rounds):
            if self.target_client_id in pkl_data.get(round_num, {}):
                return round_num
        logger.warning(f"No valid round found with target client data, using last round {rounds[-1]}")
        return rounds[-1] if rounds else 0

    def _attack_blackbox_loss(self, pkl_data, rounds, MODE, K):
        """Blackbox-Loss Attack: Single round loss comparison."""
        round_num = self._get_last_valid_round(pkl_data, rounds)
        member_scores, nonmember_scores = self._get_member_nonmember_scores(
            pkl_data, round_num, 'loss', MODE)
        return self._plot_auc(member_scores, nonmember_scores)

    def _attack_grad_cosine(self, pkl_data, rounds, MODE, K):
        """Grad-Cosine Attack: Single round cosine similarity comparison."""
        round_num = self._get_last_valid_round(pkl_data, rounds)
        member_scores, nonmember_scores = self._get_member_nonmember_scores(
            pkl_data, round_num, 'cos', MODE)
        return self._plot_auc(member_scores, nonmember_scores)

    def _attack_grad_diff(self, pkl_data, rounds, MODE, K):
        """Grad-Diff Attack: Single round gradient difference comparison."""
        round_num = self._get_last_valid_round(pkl_data, rounds)
        member_scores, nonmember_scores = self._get_member_nonmember_scores(
            pkl_data, round_num, 'diff', MODE)
        return self._plot_auc(member_scores, nonmember_scores)

    def _attack_grad_norm(self, pkl_data, rounds, MODE, K):
        """Grad-Norm Attack: Single round gradient norm comparison."""
        round_num = self._get_last_valid_round(pkl_data, rounds)
        member_scores, nonmember_scores = self._get_member_nonmember_scores(
            pkl_data, round_num, 'norm', MODE)
        return self._plot_auc(member_scores, nonmember_scores)

    def _attack_loss_series(self, pkl_data, rounds, MODE, K):
        """Loss-Series Attack: Average loss across all rounds."""
        all_member_scores = []
        all_nonmember_scores = []
        
        for round_num in rounds:
            member_scores, nonmember_scores = self._get_member_nonmember_scores(
                pkl_data, round_num, 'loss', MODE)
            if len(member_scores) > 0:
                all_member_scores.append(member_scores)
            if len(nonmember_scores) > 0:
                all_nonmember_scores.append(nonmember_scores)
        
        if not all_member_scores or not all_nonmember_scores:
            return {'auc': float('nan'), 'log_auc': float('nan'),
                    'tpr@0.1': float('nan'), 'tpr@0.01': float('nan'), 'tpr@0.001': float('nan')}
        
        # Average across rounds
        min_len = min(len(s) for s in all_member_scores)
        min_len_non = min(len(s) for s in all_nonmember_scores)
        
        member_avg = np.mean([s[:min_len] for s in all_member_scores], axis=0)
        nonmember_avg = np.mean([s[:min_len_non] for s in all_nonmember_scores], axis=0)
        
        return self._plot_auc(member_avg, nonmember_avg)

    def _attack_avg_cosine(self, pkl_data, rounds, MODE, K):
        """Avg-Cosine Attack: Average cosine similarity across all rounds."""
        all_member_scores = []
        all_nonmember_scores = []
        
        for round_num in rounds:
            member_scores, nonmember_scores = self._get_member_nonmember_scores(
                pkl_data, round_num, 'cos', MODE)
            if len(member_scores) > 0:
                all_member_scores.append(member_scores)
            if len(nonmember_scores) > 0:
                all_nonmember_scores.append(nonmember_scores)
        
        if not all_member_scores or not all_nonmember_scores:
            return {'auc': float('nan'), 'log_auc': float('nan'),
                    'tpr@0.1': float('nan'), 'tpr@0.01': float('nan'), 'tpr@0.001': float('nan')}
        
        min_len = min(len(s) for s in all_member_scores)
        min_len_non = min(len(s) for s in all_nonmember_scores)
        
        member_avg = np.mean([s[:min_len] for s in all_member_scores], axis=0)
        nonmember_avg = np.mean([s[:min_len_non] for s in all_nonmember_scores], axis=0)
        
        return self._plot_auc(member_avg, nonmember_avg)

    def _attack_fedmia_i(self, pkl_data, rounds, MODE, K):
        """
        FedMIA-I Attack: CDF of loss across all rounds.
        Uses shadow clients to estimate Q_out distribution.
        Per-sample computation.
        """
        all_member_cdf = []
        all_nonmember_cdf = []
        
        for round_num in rounds:
            member_scores, nonmember_scores = self._get_member_nonmember_scores(
                pkl_data, round_num, 'loss', MODE)
            shadow_train_scores, shadow_test_scores = self._get_shadow_scores(
                pkl_data, round_num, 'loss', MODE)
            
            if len(member_scores) == 0 or len(nonmember_scores) == 0:
                continue
            
            # Stack shadow scores for Q_out estimation
            if shadow_train_scores:
                shadow_stack = np.vstack(shadow_train_scores)
                # Per-sample mu and var
                train_mu_out = np.mean(shadow_stack, axis=0) if shadow_stack.shape[0] > 1 else shadow_stack[0]
                train_var_out = np.var(shadow_stack, axis=0) + 1e-8 if shadow_stack.shape[0] > 1 else np.ones_like(shadow_stack[0]) * 0.1
            else:
                # Fallback: use nonmember scores
                train_mu_out = np.mean(nonmember_scores)
                train_var_out = np.var(nonmember_scores) + 1e-8
            
            # Compute CDF scores using per-sample parameters
            # Handle dimension mismatch
            if isinstance(train_mu_out, np.ndarray) and len(train_mu_out) > 1:
                min_len = min(len(member_scores), len(train_mu_out))
                member_cdf = norm.cdf(member_scores[:min_len], 
                                     train_mu_out[:min_len], 
                                     np.sqrt(train_var_out[:min_len]))
                nonmember_min = min(len(nonmember_scores), len(train_mu_out))
                nonmember_cdf = norm.cdf(nonmember_scores[:nonmember_min],
                                        train_mu_out[:nonmember_min],
                                        np.sqrt(train_var_out[:nonmember_min]))
            else:
                # Scalar mu/var
                member_cdf = norm.cdf(member_scores, train_mu_out, np.sqrt(train_var_out))
                nonmember_cdf = norm.cdf(nonmember_scores, train_mu_out, np.sqrt(train_var_out))
            
            all_member_cdf.append(member_cdf)
            all_nonmember_cdf.append(nonmember_cdf)
        
        if not all_member_cdf or not all_nonmember_cdf:
            return {'auc': float('nan'), 'log_auc': float('nan'),
                    'tpr@0.1': float('nan'), 'tpr@0.01': float('nan'), 'tpr@0.001': float('nan')}
        
        # Average CDF across rounds
        min_member_len = min(len(s) for s in all_member_cdf)
        min_nonmember_len = min(len(s) for s in all_nonmember_cdf)
        
        member_avg = np.mean([s[:min_member_len] for s in all_member_cdf], axis=0)
        nonmember_avg = np.mean([s[:min_nonmember_len] for s in all_nonmember_cdf], axis=0)
        
        return self._plot_auc(member_avg, nonmember_avg)

    def _attack_fedmia_ii(self, pkl_data, rounds, MODE, K):
        """
        FedMIA-II Attack: CDF of cosine similarity across all rounds.
        Uses shadow clients to estimate Q_out distribution.
        Per-sample computation.
        """
        all_member_cdf = []
        all_nonmember_cdf = []
        
        for round_num in rounds:
            member_scores, nonmember_scores = self._get_member_nonmember_scores(
                pkl_data, round_num, 'cos', MODE)
            shadow_train_scores, shadow_test_scores = self._get_shadow_scores(
                pkl_data, round_num, 'cos', MODE)
            
            if len(member_scores) == 0 or len(nonmember_scores) == 0:
                continue
            
            # Stack shadow scores for Q_out estimation
            if shadow_train_scores:
                shadow_stack = np.vstack(shadow_train_scores)
                # Per-sample mu and var
                train_mu_out = np.mean(shadow_stack, axis=0) if shadow_stack.shape[0] > 1 else shadow_stack[0]
                train_var_out = np.var(shadow_stack, axis=0) + 1e-8 if shadow_stack.shape[0] > 1 else np.ones_like(shadow_stack[0]) * 0.1
            else:
                # Fallback: use nonmember scores
                train_mu_out = np.mean(nonmember_scores)
                train_var_out = np.var(nonmember_scores) + 1e-8
            
            # Compute CDF scores using per-sample parameters
            if isinstance(train_mu_out, np.ndarray) and len(train_mu_out) > 1:
                min_len = min(len(member_scores), len(train_mu_out))
                member_cdf = norm.cdf(member_scores[:min_len],
                                     train_mu_out[:min_len],
                                     np.sqrt(train_var_out[:min_len]))
                nonmember_min = min(len(nonmember_scores), len(train_mu_out))
                nonmember_cdf = norm.cdf(nonmember_scores[:nonmember_min],
                                        train_mu_out[:nonmember_min],
                                        np.sqrt(train_var_out[:nonmember_min]))
            else:
                # Scalar mu/var
                member_cdf = norm.cdf(member_scores, train_mu_out, np.sqrt(train_var_out))
                nonmember_cdf = norm.cdf(nonmember_scores, train_mu_out, np.sqrt(train_var_out))
            
            all_member_cdf.append(member_cdf)
            all_nonmember_cdf.append(nonmember_cdf)
        
        if not all_member_cdf or not all_nonmember_cdf:
            return {'auc': float('nan'), 'log_auc': float('nan'),
                    'tpr@0.1': float('nan'), 'tpr@0.01': float('nan'), 'tpr@0.001': float('nan')}
        
        # Average CDF across rounds
        min_member_len = min(len(s) for s in all_member_cdf)
        min_nonmember_len = min(len(s) for s in all_nonmember_cdf)
        
        member_avg = np.mean([s[:min_member_len] for s in all_member_cdf], axis=0)
        nonmember_avg = np.mean([s[:min_nonmember_len] for s in all_nonmember_cdf], axis=0)
        
        return self._plot_auc(member_avg, nonmember_avg)

    def _print_and_save_results(self, results: Dict, rounds: List[int]):
        """Print results to log and save to file."""
        logger.info("\n" + "=" * 60)
        logger.info("FedMIA Attack Results")
        logger.info("=" * 60)
        
        output = {
            'rounds': rounds,
            'attacks': {}
        }
        
        for attack_name, metrics in results.items():
            auc = metrics.get('auc', float('nan'))
            tpr_01 = metrics.get('tpr@0.1', float('nan'))
            tpr_001 = metrics.get('tpr@0.01', float('nan'))
            tpr_0001 = metrics.get('tpr@0.001', float('nan'))
            
            logger.info(f"\n{attack_name}:")
            logger.info(f"  AUC: {auc:.4f}")
            logger.info(f"  TPR@FPR=0.1: {tpr_01:.4f}")
            logger.info(f"  TPR@FPR=0.01: {tpr_001:.4f}")
            logger.info(f"  TPR@FPR=0.001: {tpr_0001:.4f}")
            
            output['attacks'][attack_name] = {
                'auc': float(auc) if not np.isnan(auc) else None,
                'tpr@0.1': float(tpr_01) if not np.isnan(tpr_01) else None,
                'tpr@0.01': float(tpr_001) if not np.isnan(tpr_001) else None,
                'tpr@0.001': float(tpr_0001) if not np.isnan(tpr_0001) else None
            }
        
        # Save to JSON file
        results_path = os.path.join(self.pkl_dir, "attack_results.log")
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"\nResults saved to: {results_path}")
