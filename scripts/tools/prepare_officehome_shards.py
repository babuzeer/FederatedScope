#!/usr/bin/env python3
"""
Utility to create Office-Home shards for distributed training.

Based on prepare_femnist_shards.py, adapted for Office-Home dataset with:
- Support for LDS (Label Distribution Skew) via Dirichlet distribution
- Per-domain client allocation
- Reproducible shard generation with manifest
"""

import argparse
import json
import os
import os.path as osp
import random
import shutil
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

DOMAINS = ['Art', 'Clipart', 'Product', 'Real World']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create Office-Home shards for distributed training')
    parser.add_argument('--root',
                        default='/root/OfficeHomeDataset_10072016',
                        help='Root directory of Office-Home dataset')
    parser.add_argument('--output',
                        default='data/officehome/shards',
                        help='Output directory for shards')
    parser.add_argument(
        '--clients',
        type=int,
        default=4,
        help='Total number of clients (must be divisible by 4 domains)')
    parser.add_argument('--lds-alpha',
                        type=float,
                        default=0.1,
                        help='Dirichlet alpha for LDS (0 to disable LDS)')
    parser.add_argument('--lds-seed',
                        type=int,
                        default=42,
                        help='Random seed for LDS')
    parser.add_argument('--min-samples-per-client',
                        type=int,
                        default=32,
                        help='Minimum number of training samples each client '
                        'must receive after domain-internal split')
    parser.add_argument(
        '--splits',
        nargs=3,
        type=float,
        default=[0.7, 0.0, 0.3],
        help='Train/val/test split ratios (default: 0.7 0.0 0.3)')
    parser.add_argument(
        '--manifest',
        default='',
        help='Output path for manifest.json (default: output_dir/manifest.json)'
    )
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='Overwrite existing shards')
    return parser.parse_args()


def validate_args(args):
    if args.clients <= 0:
        raise ValueError('--clients must be positive')

    if args.min_samples_per_client <= 0:
        raise ValueError('--min-samples-per-client must be positive')

    if len(args.splits) != 3:
        raise ValueError('--splits must contain train/val/test ratios')

    split_sum = sum(args.splits)
    if not np.isclose(split_sum, 1.0):
        raise ValueError(f'--splits must sum to 1.0, but got {args.splits} '
                         f'(sum={split_sum:.6f})')

    num_domains = len(DOMAINS)
    if args.clients % num_domains != 0:
        raise ValueError(
            f'--clients must be divisible by the number of domains '
            f'({num_domains}), but got {args.clients}')


def load_office_home_data(root: str,
                          domain: str) -> Tuple[List[str], List[int]]:
    """
    Load Office-Home data for a specific domain.
    
    Returns:
        image_paths: List of image file paths
        labels: List of corresponding labels
    """
    domain_dir = osp.join(root, domain)
    if not osp.isdir(domain_dir):
        raise FileNotFoundError(f'Domain directory not found: {domain_dir}')

    # Get all class directories
    classes = sorted([
        d for d in os.listdir(domain_dir) if osp.isdir(osp.join(domain_dir, d))
    ])

    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    image_paths = []
    labels = []

    for cls in classes:
        cls_dir = osp.join(domain_dir, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(osp.join(cls_dir, fname))
                labels.append(class_to_idx[cls])

    return image_paths, labels


def split_data(image_paths: List[str],
               labels: List[int],
               splits: Tuple[float, float, float],
               seed: int = 123):
    """
    Split data into train/val/test sets.
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        splits: (train_ratio, val_ratio, test_ratio)
        seed: Random seed
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing (paths, labels) tuples
    """
    np.random.seed(seed)

    # Group by class for stratified split
    class_indices = {}
    for idx, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    train_indices, val_indices, test_indices = [], [], []
    train_ratio, val_ratio, test_ratio = splits

    for class_idx, indices in class_indices.items():
        indices = np.array(indices)
        np.random.shuffle(indices)

        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_indices.extend(indices[:n_train].tolist())
        val_indices.extend(indices[n_train:n_train + n_val].tolist())
        test_indices.extend(indices[n_train + n_val:].tolist())

    result = {}
    for split_name, indices in [('train', train_indices), ('val', val_indices),
                                ('test', test_indices)]:
        if len(indices) > 0:
            result[split_name] = ([image_paths[i] for i in indices],
                                  [labels[i] for i in indices])
        else:
            result[split_name] = ([], [])

    return result


def apply_lds(split_data: Dict, num_clients_per_domain: int, alpha: float,
              seed: int):
    """
    Apply Label Distribution Skew using Dirichlet distribution.
    
    Args:
        split_data: Data splits from split_data()
        num_clients_per_domain: Number of clients to split among
        alpha: Dirichlet parameter (smaller = more skewed)
        seed: Random seed
        
    Returns:
        List of (paths, labels) for each client
    """
    np.random.seed(seed)

    train_paths, train_labels = split_data['train']
    if len(train_paths) == 0:
        return [([], [])] * num_clients_per_domain

    # Get number of classes
    num_classes = len(set(train_labels))

    # Generate Dirichlet distribution matrix
    # Shape: (num_clients, num_classes)
    dirichlet_matrix = np.random.dirichlet([alpha] * num_clients_per_domain,
                                           num_classes).T

    # Group samples by class
    class_samples = {}
    for path, label in zip(train_paths, train_labels):
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append((path, label))

    # Allocate samples to clients based on Dirichlet proportions
    client_data = [[] for _ in range(num_clients_per_domain)]

    for class_idx, samples in class_samples.items():
        np.random.shuffle(samples)
        proportions = dirichlet_matrix[:, class_idx]

        # Calculate number of samples for each client
        n_samples = len(samples)
        allocations = (proportions * n_samples).astype(int)

        # Adjust for rounding errors
        diff = n_samples - allocations.sum()
        if diff > 0:
            # Add remaining samples to clients with highest proportions
            top_clients = np.argsort(proportions)[-diff:]
            for client_idx in top_clients:
                allocations[client_idx] += 1

        # Distribute samples
        start_idx = 0
        for client_idx, n_alloc in enumerate(allocations):
            client_data[client_idx].extend(samples[start_idx:start_idx +
                                                   n_alloc])
            start_idx += n_alloc

    # Convert to (paths, labels) format
    result = []
    for samples in client_data:
        if len(samples) > 0:
            paths = [s[0] for s in samples]
            labels = [s[1] for s in samples]
            result.append((paths, labels))
        else:
            result.append(([], []))

    return result


def split_uniformly(split_data: Dict, num_clients_per_domain: int,
                    seed: int) -> List[Tuple[List[str], List[int]]]:
    """Uniformly split one domain's train set into multiple clients."""
    train_paths, train_labels = split_data['train']
    if len(train_paths) == 0:
        return [([], []) for _ in range(num_clients_per_domain)]

    np.random.seed(seed)
    indices = np.random.permutation(len(train_paths))
    splits = np.array_split(indices, num_clients_per_domain)
    return [([train_paths[i] for i in split], [train_labels[i] for i in split])
            for split in splits]


def enforce_min_samples_per_client(
        client_train_data: List[Tuple[List[str],
                                      List[int]]], min_samples_per_client: int,
        domain: str) -> List[Tuple[List[str], List[int]]]:
    """
    Ensure each client receives at least ``min_samples_per_client`` samples.

    The repair strategy is intentionally simple:
    - move samples from the currently largest shard to the smallest shard
    - fail fast if the total sample count cannot satisfy the constraint
    """
    total_samples = sum(len(paths) for paths, _ in client_train_data)
    required_samples = len(client_train_data) * min_samples_per_client

    if total_samples < required_samples:
        raise ValueError(
            f"Domain '{domain}' has only {total_samples} train samples, "
            f'which cannot satisfy min_samples_per_client='
            f'{min_samples_per_client} for {len(client_train_data)} clients')

    repaired = [(list(paths), list(labels))
                for paths, labels in client_train_data]

    def _sizes():
        return [len(paths) for paths, _ in repaired]

    while True:
        sizes = _sizes()
        min_size = min(sizes)
        if min_size >= min_samples_per_client:
            break

        receiver_idx = sizes.index(min_size)
        donor_idx = sizes.index(max(sizes))

        if receiver_idx == donor_idx or sizes[
                donor_idx] <= min_samples_per_client:
            raise ValueError(
                f"Unable to rebalance domain '{domain}' to satisfy "
                f'min_samples_per_client={min_samples_per_client}. '
                f'Current shard sizes: {sizes}')

        donor_paths, donor_labels = repaired[donor_idx]
        receiver_paths, receiver_labels = repaired[receiver_idx]

        move_count = min(min_samples_per_client - len(receiver_paths),
                         len(donor_paths) - min_samples_per_client)
        if move_count <= 0:
            raise ValueError(
                f"Unable to continue rebalancing domain '{domain}'. "
                f'Current shard sizes: {sizes}')

        move_indices = random.sample(range(len(donor_paths)), move_count)
        move_indices.sort(reverse=True)

        for idx in move_indices:
            receiver_paths.append(donor_paths.pop(idx))
            receiver_labels.append(donor_labels.pop(idx))

    return repaired


def build_label_hist(labels: List[int]) -> Dict[str, int]:
    """Build label histogram"""
    counter = Counter(labels)
    return {str(k): v for k, v in sorted(counter.items())}


def save_shard(output_dir: str, split_data: Dict, client_train_data: Tuple,
               domain: str):
    """
    Save shard data to JSON files.
    
    Args:
        output_dir: Client shard directory
        split_data: Full split data with val/test
        client_train_data: (train_paths, train_labels) for this client
    """
    os.makedirs(output_dir, exist_ok=True)

    train_paths, train_labels = client_train_data

    # Save train data
    if len(train_paths) > 0:
        train_data = [{
            'path': p,
            'label': int(l)
        } for p, l in zip(train_paths, train_labels)]
        with open(osp.join(output_dir, 'train.json'), 'w') as f:
            json.dump(train_data, f)

    # Save val data (shared across clients in same domain)
    val_paths, val_labels = split_data['val']
    if len(val_paths) > 0:
        val_data = [{
            'path': p,
            'label': int(l)
        } for p, l in zip(val_paths, val_labels)]
        with open(osp.join(output_dir, 'val.json'), 'w') as f:
            json.dump(val_data, f)

    # Save test data (shared across clients in same domain)
    test_paths, test_labels = split_data['test']
    if len(test_paths) > 0:
        test_data = [{
            'path': p,
            'label': int(l)
        } for p, l in zip(test_paths, test_labels)]
        with open(osp.join(output_dir, 'test.json'), 'w') as f:
            json.dump(test_data, f)

    # Save metadata
    meta = {
        'domain': domain,
        'num_samples': {
            'train': len(train_paths),
            'val': len(val_paths),
            'test': len(test_paths)
        },
        'label_hist': build_label_hist(train_labels)
    }

    with open(osp.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


def create_manifest(manifest_path: str, args, client_metas: Dict[int, Dict]):
    """Create manifest.json with all shard metadata"""
    manifest = {
        'dataset': 'office-home',
        'root': osp.abspath(args.root),
        'seed': args.lds_seed,
        'lds_alpha': args.lds_alpha if args.lds_alpha > 0 else None,
        'clients_per_domain': args.clients // len(DOMAINS),
        'domains': DOMAINS,
        'min_samples_per_client': args.min_samples_per_client,
        'splits': args.splits,
        'total_clients': args.clients,
        'clients': []
    }

    for client_id in sorted(client_metas.keys()):
        meta = client_metas[client_id]
        manifest['clients'].append({
            'client_id': client_id,
            'domain': meta['domain'],
            'shard_path': meta['shard_path'],
            'num_samples': meta['num_samples'],
            'label_hist': meta['label_hist']
        })

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\nManifest saved to {manifest_path}')


def main():
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)

    # Validate arguments
    domains = DOMAINS
    num_domains = len(domains)

    clients_per_domain = args.clients // num_domains
    use_lds = args.lds_alpha > 0

    print(f'\nOffice-Home Shard Generation')
    print(f'=' * 60)
    print(f'Root: {args.root}')
    print(f'Output: {args.output}')
    print(f'Clients: {args.clients} ({clients_per_domain} per domain)')
    print(f'Min samples/client: {args.min_samples_per_client}')
    print(
        f'Splits: train={args.splits[0]}, val={args.splits[1]}, test={args.splits[2]}'
    )
    if use_lds:
        print(f'LDS: alpha={args.lds_alpha}, seed={args.lds_seed}')
    else:
        print(f'LDS: disabled')
    print(f'=' * 60)

    # Prepare output directory
    if osp.exists(args.output):
        if args.overwrite:
            shutil.rmtree(args.output)
        else:
            print(f'\nError: Output directory {args.output} already exists')
            print(f'Use --overwrite to replace it')
            return

    os.makedirs(args.output, exist_ok=True)

    # Process each domain
    client_metas = {}
    client_id = 1

    for domain in domains:
        print(f'\nProcessing domain: {domain}')
        print(f'-' * 60)

        # Load domain data
        try:
            image_paths, labels = load_office_home_data(args.root, domain)
            print(
                f'Loaded {len(image_paths)} images with {len(set(labels))} classes'
            )
        except Exception as e:
            print(f'Error loading {domain}: {e}')
            continue

        # Split into train/val/test
        split_data_dict = split_data(image_paths,
                                     labels,
                                     tuple(args.splits),
                                     seed=args.lds_seed)

        train_size = len(split_data_dict['train'][0])
        val_size = len(split_data_dict['val'][0])
        test_size = len(split_data_dict['test'][0])
        print(f'Split: train={train_size}, val={val_size}, test={test_size}')

        # Apply LDS or uniform split
        if use_lds:
            print(f'Applying LDS with alpha={args.lds_alpha}...')
            client_train_data = apply_lds(
                split_data_dict, clients_per_domain, args.lds_alpha,
                args.lds_seed + domains.index(domain))
        else:
            client_train_data = split_uniformly(
                split_data_dict, clients_per_domain,
                args.lds_seed + domains.index(domain))

        try:
            client_train_data = enforce_min_samples_per_client(
                client_train_data, args.min_samples_per_client, domain)
        except ValueError as error:
            print(f'Error: {error}', file=sys.stderr)
            sys.exit(1)
        shard_sizes = [len(paths) for paths, _ in client_train_data]
        print(f'Final per-client train sizes for {domain}: {shard_sizes}')

        # Save shards for this domain
        for i, train_data in enumerate(client_train_data):
            shard_dir = osp.join(args.output, f'client_{client_id}')

            meta = save_shard(shard_dir, split_data_dict, train_data, domain)
            # Create full metadata dict
            full_meta = {
                'domain': domain,
                'shard_path': osp.abspath(shard_dir),
                'num_samples': meta['num_samples'],
                'label_hist': meta['label_hist']
            }
            client_metas[client_id] = full_meta

            print(
                f'  Client {client_id} ({domain}): '
                f'train={meta["num_samples"]["train"]} samples -> {shard_dir}')
            client_id += 1

    # Create manifest
    manifest_path = args.manifest or osp.join(args.output, 'manifest.json')
    create_manifest(manifest_path, args, client_metas)

    print(f'\n{"=" * 60}')
    print(f'Shard generation complete!')
    print(f'Total clients: {len(client_metas)}')
    print(f'Output directory: {args.output}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
