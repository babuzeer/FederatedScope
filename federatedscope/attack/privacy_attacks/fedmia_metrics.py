"""
FedMIA metrics computation module.
Fully aligned with FedMIA-main repository implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader, Subset, Dataset


def get_all_losses(dataloader, model, criterion, device, req_logits=False):
    """
    Compute loss, logit, labels for a dataloader.
    Aligned with FedMIA-main get_all_losses function.
    
    Args:
        dataloader: DataLoader for the dataset
        model: The model
        criterion: Loss function (should be reduction='none')
        device: Device to use
        req_logits: Whether to return logits (not used, kept for compatibility)
    
    Returns:
        dict: {'loss': np.array, 'logit': tensor, 'labels': tensor}
    """
    model.eval()
    losses = []
    logits = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward
            outputs = model(inputs)
            # Evaluate
            loss = criterion(outputs, targets)
            losses.append(loss.cpu().numpy())
            logits.append(outputs.cpu())
            labels.append(targets.cpu())
    
    losses = np.concatenate(losses)
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    
    return {"loss": losses, "logit": logits, "labels": labels}


def get_all_losses_from_indexes(dataset, indexes, model, device=None):
    """
    Compute loss, logit, labels for samples at specific indexes.
    Aligned with FedMIA-main get_all_losses_from_indexes function.
    
    Args:
        dataset: Full dataset
        indexes: List of sample indexes
        model: The model
        device: Device to use
    
    Returns:
        dict: {'loss': np.array, 'logit': tensor, 'labels': tensor}
    """
    criterion = nn.CrossEntropyLoss(reduction='none')
    if device is None:
        device = next(model.parameters()).device
    
    # Create subset and dataloader
    subset = Subset(dataset, indexes)
    dataloader = DataLoader(subset, batch_size=200, shuffle=False, num_workers=0)
    
    model.eval()
    losses = []
    logits = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward
            outputs = model(inputs)
            # Evaluate
            loss = criterion(outputs, targets)
            losses.append(loss.cpu().numpy())
            logits.append(outputs.cpu())
            labels.append(targets.cpu())
    
    losses = np.concatenate(losses)
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    
    return {"loss": losses, "logit": logits, "labels": labels}


def get_all_cos(cos_model, initial_loader, test_dataloader, test_set, train_set,
                train_idxs, val_idxs, mix_idxs, needed_test_indexs, 
                model_grads, lr, optim_choice):
    """
    Compute cosine similarity, gradient difference, and gradient norm scores.
    Aligned with FedMIA-main get_all_cos function.
    
    Args:
        cos_model: Model copy for computing gradients
        initial_loader: DataLoader (not used, kept for compatibility)
        test_dataloader: DataLoader for test set (not used directly)
        test_set: Test dataset
        train_set: Training dataset
        train_idxs: Target client training indices
        val_idxs: Val client training indices (not used in original)
        mix_idxs: Mixed sample indices
        needed_test_indexs: Test set indices to use
        model_grads: Flattened model gradient tensor
        lr: Learning rate (not used, kept for compatibility)
        optim_choice: Optimizer type (not used, kept for compatibility)
    
    Returns:
        tuple: (train_cos, train_diffs, train_norm, 
                val_cos, val_diffs, val_norm,
                test_cos, test_diffs, test_norm,
                mix_cos, mix_diffs, mix_norm)
    """
    device = next(cos_model.parameters()).device
    
    # Create dataloaders for each subset (batch_size=1 for per-sample gradients)
    train_dataloader = DataLoader(Subset(train_set, train_idxs), 
                                  batch_size=1, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(Subset(test_set, needed_test_indexs), 
                                 batch_size=1, shuffle=False, num_workers=0)
    mix_dataloader = DataLoader(Subset(train_set, mix_idxs), 
                                batch_size=1, shuffle=False, num_workers=0)
    
    # Compute scores for each subset using manual per-sample gradient computation
    train_cos, train_diffs, train_norm = get_cos_score(
        train_dataloader, cos_model, device, model_grads)
    
    test_cos, test_diffs, test_norm = get_cos_score(
        test_dataloader, cos_model, device, model_grads)
    
    mix_cos, mix_diffs, mix_norm = get_cos_score(
        mix_dataloader, cos_model, device, model_grads)
    
    # val not computed in original implementation
    val_cos, val_diffs, val_norm = None, None, None
    
    return (train_cos, train_diffs, train_norm,
            val_cos, val_diffs, val_norm,
            test_cos, test_diffs, test_norm,
            mix_cos, mix_diffs, mix_norm)


def get_cos_score(samples_ldr, model, device, model_grads):
    """
    Compute cosine similarity, gradient difference, and gradient norm for each sample.
    Aligned with FedMIA-main get_cos_score function.
    
    Uses manual per-sample gradient computation (batch_size=1) instead of Opacus.
    This avoids Opacus limitations with BatchNorm layers.
    Uses eval() mode to avoid BatchNorm issues with single sample.
    
    Args:
        samples_ldr: DataLoader for samples (should have batch_size=1)
        model: The model
        device: Device to use
        model_grads: Flattened model gradient tensor
    
    Returns:
        tuple: (cos_scores, grad_diffs, grad_norms) as tensors
    """
    model_grads = model_grads.to(device)
    model_diff_norm = torch.norm(model_grads, p=2) ** 2
    cos_scores, grad_diffs, sample_norms = [], [], []

    # Use eval() mode to avoid BatchNorm error with batch_size=1
    # BatchNorm will use running statistics instead of batch statistics
    model.eval()
    for x, y in samples_ldr:
        for i in range(x.shape[0]):  # 逐样本
            xi = x[i:i+1].to(device)
            yi = y[i:i+1].to(device)
            model.zero_grad()
            loss = F.cross_entropy(model(xi), yi)
            loss.backward()
            sample_grad = torch.cat([
                p.grad.detach().flatten()
                for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ])
            cos_scores.append(F.cosine_similarity(sample_grad, model_grads, dim=0).item())
            grad_diffs.append((model_diff_norm - torch.norm(model_grads - sample_grad, p=2)**2).item())
            sample_norms.append(torch.norm(sample_grad, p=2).item() ** 2)

    return torch.tensor(cos_scores), torch.tensor(grad_diffs), torch.tensor(sample_norms)


def get_model_grads(global_state_dict, local_state_dict, param_keys=None):
    """
    Compute model gradient (global - local) for selected parameters.
    
    Args:
        global_state_dict: Global model state dict
        local_state_dict: Local (client) model state dict
        param_keys: List of parameter names to include
    
    Returns:
        Flattened tensor of model gradients
    """
    if param_keys is None:
        param_keys = [k for k in global_state_dict.keys()
                     if 'running_mean' not in k and 'running_var' not in k
                     and 'num_batches_tracked' not in k]
    
    grads = []
    for name in param_keys:
        if name in local_state_dict and name in global_state_dict:
            # Note: original FedMIA uses global - local
            para_diff = global_state_dict[name] - local_state_dict[name]
            grads.append(para_diff.detach().cpu().flatten())
    
    if not grads:
        return torch.zeros(1)
    
    return torch.cat(grads, -1)


def ce_loss_fn(x, y):
    """Compute cross-entropy loss without reduction."""
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    return loss_fn(x, y)


def hinge_loss_fn(x, y, device='cuda'):
    """Compute hinge loss."""
    x, y = x.clone().to(device), y.clone().to(device)
    mask = torch.eye(x.shape[1], device=device)[y].bool()
    tmp1 = x[mask]
    x_clone = x.clone()
    x_clone[mask] = -1e10
    tmp2 = torch.max(x_clone, dim=1)[0]
    return (tmp1 - tmp2).cpu().numpy()
