"""
Single-machine GGEUR head-only hierarchical FL with real feature caches.

The script does not modify or import the existing GGEUR server/client worker
classes. It uses the existing configuration system, enforces the head-only
switches, and runs:

    parameter service -> clients -> sub-servers -> root server

Round 0 always uses the configured real dataset and frozen backbone to write
augmented feature caches. Later rounds train only the MLP head from those
caches. If generation is disabled, the caches must already exist.

Metrics include wall-clock round time, training sample throughput, approximated
download/upload QPS, payload size, sub-server aggregation time, and root
aggregation time.
"""

import argparse
import copy
import hashlib
import io
import json
import math
import multiprocessing
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.configs.config import global_cfg


def build_mlp(input_dim, num_classes, hidden_dim=0, dropout=0.0):
    if hidden_dim > 0:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    return nn.Linear(input_dim, num_classes)


def clone_state(state_dict, dtype=torch.float32):
    return {
        name: tensor.detach().cpu().to(dtype).clone()
        for name, tensor in state_dict.items()
    }


def tensor_dtype(name):
    if name == 'float16':
        return torch.float16
    return torch.float32


def state_payload_bytes(state_dict):
    buffer = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in state_dict.items()}, buffer)
    return buffer.tell()


def state_checksum(state_dict):
    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode('utf-8'))
        digest.update(str(tuple(tensor.shape)).encode('utf-8'))
        digest.update(str(tensor.dtype).encode('utf-8'))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def weighted_average(updates):
    """Average a list of (sample_size, state_dict) with sample weights."""
    valid = [(int(samples), params) for samples, params in updates
             if int(samples) > 0 and params]
    if not valid:
        return {}, 0

    total_samples = sum(samples for samples, _ in valid)
    first_state = valid[0][1]
    averaged = {}
    for key in first_state.keys():
        acc = None
        for samples, state in valid:
            value = state[key].detach().cpu().float()
            weighted = value * (samples / total_samples)
            acc = weighted if acc is None else acc + weighted
        averaged[key] = acc
    return averaged, total_samples


def resolve_feature_cache_dir(cfg):
    if cfg.ggeur_headonly.feature_cache_dir:
        return cfg.ggeur_headonly.feature_cache_dir
    return str(Path(cfg.outdir) / 'ggeur_headonly_cache' /
               cfg.ggeur_headonly.feature_cache_version)


def resolve_run_dir(cfg):
    run_id = str(getattr(cfg.ggeur_headonly, 'run_id', '') or '').strip()
    if not run_id:
        run_id = time.strftime('%Y%m%d_%H%M%S')
    run_root = str(
        getattr(cfg.ggeur_headonly, 'run_root', 'exp/ggeur_headonly_runs')
    )
    return Path(run_root) / run_id, run_id


def resolve_output_json(cfg, run_dir, run_id):
    output_json = str(getattr(cfg.ggeur_headonly, 'output_json', '') or '')
    if output_json:
        output_json = output_json.format(
            run_id=run_id,
            timestamp=run_id,
        )
        return Path(output_json)
    return run_dir / 'metrics.json'


def cache_path(cache_dir, client_id):
    return Path(cache_dir) / f'client_{client_id:06d}.pt'


def manifest_path(cache_dir):
    return Path(cache_dir) / 'manifest.json'


def test_cache_path(cfg, feature_cache_dir):
    configured = str(getattr(cfg.ggeur_headonly, 'test_feature_cache', '') or '')
    if configured:
        return Path(configured)
    return Path(feature_cache_dir) / 'test_features.pt'


def load_feature_cache(cache_dir, client_id):
    if not cache_dir:
        return None

    base = Path(cache_dir)
    candidates = [
        base / f'client_{client_id}.pt',
        base / f'client_{client_id}.pth',
        base / f'client_{client_id:06d}.pt',
        base / f'client_{client_id:06d}.pth',
    ]

    existing = next((path for path in candidates if path.exists()), None)
    if existing is None:
        return None

    data = torch.load(existing, map_location='cpu')
    if isinstance(data, dict):
        metadata = data.get('metadata', {})
        if metadata.get('source') != 'real_dataset':
            raise ValueError(
                f'{existing} is not a real-data augmented feature cache. '
                'Regenerate it with ggeur_headonly.run_data_generation=True.'
            )
        features = data.get('features', data.get('augmented_features'))
        labels = data.get('labels', data.get('augmented_labels'))
    elif isinstance(data, (list, tuple)) and len(data) >= 2:
        raise ValueError(
            f'{existing} uses an old cache format without metadata. '
            'Regenerate it from the real dataset.'
        )
    else:
        raise ValueError(f'Unsupported feature cache format: {existing}')

    features = torch.as_tensor(features).float()
    labels = torch.as_tensor(labels).long()

    return features, labels


def compute_feature_statistics(features, labels, num_classes):
    input_dim = features.size(1)
    counts = torch.zeros(num_classes, dtype=torch.long)
    sums = torch.zeros(num_classes, input_dim, dtype=torch.float64)
    sq_sums = torch.zeros(num_classes, input_dim, dtype=torch.float64)
    means = torch.zeros(num_classes, input_dim, dtype=torch.float32)
    covs = torch.zeros(num_classes, input_dim, input_dim, dtype=torch.float32)

    for class_idx in range(num_classes):
        mask = labels == class_idx
        if not mask.any():
            continue
        class_features = features[mask].double()
        counts[class_idx] = class_features.size(0)
        sums[class_idx] = class_features.sum(dim=0)
        sq_sums[class_idx] = (class_features * class_features).sum(dim=0)
        mean = class_features.mean(dim=0)
        centered = class_features - mean
        cov = centered.t().matmul(centered) / class_features.size(0)
        means[class_idx] = mean.float()
        covs[class_idx] = cov.float()

    return {
        'counts': counts,
        'sums': sums,
        'sq_sums': sq_sums,
        'means': means,
        'covs': covs,
    }


def local_statistics_payload_bytes(item):
    non_empty = item['counts'] > 0
    class_count = int(non_empty.sum().item())
    input_dim = int(item['sums'].size(1))
    dtype_bytes = 4
    count_bytes = 8
    means = class_count * input_dim * dtype_bytes
    covs = class_count * input_dim * input_dim * dtype_bytes
    prototypes = class_count * input_dim * dtype_bytes
    counts = class_count * count_bytes
    return int(means + covs + prototypes + counts)


def broadcast_statistics_payload_bytes(global_covs, global_means, client_count):
    non_empty = (global_covs.abs().sum(dim=(1, 2)) > 0)
    class_count = int(non_empty.sum().item())
    input_dim = int(global_means.size(1))
    dtype_bytes = 4
    covs = class_count * input_dim * input_dim * dtype_bytes
    prototypes = class_count * input_dim * dtype_bytes
    return int(client_count * (covs + prototypes))


def build_covariance_factors(covariances):
    factors = []
    start = time.perf_counter()
    for class_idx in range(covariances.size(0)):
        cov = covariances[class_idx].float()
        jitter = 1e-6
        factor = None
        while factor is None:
            try:
                factor = torch.linalg.cholesky(
                    cov + torch.eye(cov.size(0)) * jitter
                )
            except RuntimeError:
                jitter *= 10
                if jitter > 1.0:
                    factor = torch.sqrt(
                        torch.clamp(torch.diag(cov), min=1e-6)
                    )
        factors.append(factor)
    return factors, time.perf_counter() - start


def tensor_sha256(*tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        tensor = torch.as_tensor(tensor).detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode('utf-8'))
        digest.update(str(tensor.dtype).encode('utf-8'))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def resolve_device(cfg):
    use_gpu = bool(getattr(cfg, 'use_gpu', False))
    device_idx = int(getattr(cfg, 'device', -1))
    if use_gpu and torch.cuda.is_available() and device_idx >= 0:
        return torch.device(f'cuda:{device_idx}')
    return torch.device('cpu')


def build_feature_extractor(cfg, device):
    extractor_type = getattr(cfg.ggeur, 'feature_extractor', 'clip')
    if extractor_type == 'cnn':
        from federatedscope.contrib.model.ggeur_cnn_extractor import \
            CNNFeatureExtractor
        extractor = CNNFeatureExtractor(
            model_name=getattr(cfg.ggeur, 'cnn_backbone', 'convnext_base'),
            pretrained=getattr(cfg.ggeur, 'cnn_pretrained', True),
            freeze=True,
            checkpoint_path=getattr(cfg.ggeur, 'cnn_checkpoint_path', ''),
        ).to(device)
        extractor.eval()
        return extractor, extractor, int(extractor.get_feature_dim())

    if extractor_type == 'timm':
        from federatedscope.contrib.model.ggeur_timm_extractor import \
            TimmFeatureExtractor
        extractor = TimmFeatureExtractor(
            model_name=getattr(cfg.ggeur, 'timm_model', 'gfnet_tiny'),
            pretrained=getattr(cfg.ggeur, 'timm_pretrained', True),
            freeze=True,
            checkpoint_path=getattr(cfg.ggeur, 'timm_checkpoint_path', ''),
            in_chans=getattr(cfg.ggeur, 'timm_in_chans', 3),
            global_pool=getattr(cfg.ggeur, 'timm_global_pool', 'avg'),
        ).to(device)
        extractor.eval()
        return extractor, extractor, int(extractor.get_feature_dim())

    import open_clip
    model_name = getattr(cfg.ggeur, 'clip_model', 'ViT-B-16')
    model_path = getattr(cfg.ggeur, 'clip_model_path', '')
    pretrained = model_path if model_path and Path(model_path).is_file() else \
        getattr(cfg.ggeur, 'clip_pretrained', 'openai')
    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device).eval()

    def encode(images):
        return model.encode_image(images)

    visual_dim = getattr(getattr(model, 'visual', None), 'output_dim', None)
    if visual_dim is None and hasattr(model, 'text_projection'):
        visual_dim = int(model.text_projection.shape[1])
    return model, encode, int(visual_dim or getattr(cfg.ggeur, 'embedding_dim', 512))


def extract_real_client_features(cfg, client_id, train_loader, encode, device):
    start = time.perf_counter()
    extract_time = 0.0
    sample_count = 0
    feature_chunks = []
    label_chunks = []
    use_fp16 = bool(getattr(cfg.ggeur, 'use_fp16_extraction', True)) and \
        device.type == 'cuda'

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
        for batch in train_loader:
            if len(batch) < 2:
                continue
            images, labels = batch[0], batch[1]
            if images.dim() != 4 or images.size(1) != 3:
                raise ValueError(
                    f'Client {client_id + 1}: expected image batch [N,3,H,W], '
                    f'got {tuple(images.shape)}'
                )
            images = images.to(device)
            t0 = time.perf_counter()
            features = encode(images)
            if isinstance(features, (tuple, list)):
                features = features[0]
            if features.dim() > 2:
                features = torch.flatten(features, start_dim=1)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            extract_time += time.perf_counter() - t0
            sample_count += int(features.size(0))
            feature_chunks.append(features.detach().cpu().float())
            label_chunks.append(labels.detach().cpu().long())

    if not feature_chunks:
        raise RuntimeError(f'Client {client_id + 1}: no real features extracted')

    features = torch.cat(feature_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)
    stats = compute_feature_statistics(features, labels, int(cfg.model.num_classes))
    elapsed = time.perf_counter() - start
    return {
        'client_id': client_id,
        'data_client_id': client_id + 1,
        'sub_server_id': client_id % int(cfg.ggeur_headonly.num_sub_servers),
        'sample_size': int(features.size(0)),
        'features': features,
        'labels': labels,
        'counts': stats['counts'],
        'sums': stats['sums'],
        'sq_sums': stats['sq_sums'],
        'means': stats['means'],
        'covs': stats['covs'],
        'timing': {
            'feature_generation_sec': elapsed,
            'backbone_forward_sec': extract_time,
            'feature_generation_qps': sample_count / elapsed
            if elapsed > 0 else 0.0,
            'backbone_forward_qps': sample_count / extract_time
            if extract_time > 0 else 0.0,
        },
    }


def add_statistics(acc, item):
    if acc is None:
        return {
            'counts': item['counts'].clone(),
            'sums': item['sums'].clone(),
            'sq_sums': item['sq_sums'].clone(),
            'cov_weighted_sums': (
                item['covs'].double() +
                item['means'].double().unsqueeze(2).matmul(
                    item['means'].double().unsqueeze(1))
            ) * item['counts'].double().view(-1, 1, 1),
            'sample_size': int(item['sample_size']),
        }
    acc['counts'] += item['counts']
    acc['sums'] += item['sums']
    acc['sq_sums'] += item['sq_sums']
    acc['cov_weighted_sums'] += (
        item['covs'].double() +
        item['means'].double().unsqueeze(2).matmul(
            item['means'].double().unsqueeze(1))
    ) * item['counts'].double().view(-1, 1, 1)
    acc['sample_size'] += int(item['sample_size'])
    return acc


def aggregate_generation_statistics(local_stats, num_sub_servers, epsilon):
    start = time.perf_counter()
    sub_stats = []
    for sub_server_id in range(num_sub_servers):
        acc = None
        for item in local_stats:
            if item['sub_server_id'] == sub_server_id:
                acc = add_statistics(acc, item)
        if acc is not None:
            acc['sub_server_id'] = sub_server_id
            sub_stats.append(acc)
    sub_elapsed = time.perf_counter() - start

    root_start = time.perf_counter()
    root_acc = None
    for item in sub_stats:
        root_acc = add_statistics(root_acc, item)
    if root_acc is None:
        raise RuntimeError('No local statistics were generated')

    counts = root_acc['counts'].double()
    safe_counts = counts.clamp(min=1.0).unsqueeze(1)
    means = root_acc['sums'] / safe_counts
    variances = root_acc['sq_sums'] / safe_counts - means * means
    variances = torch.clamp(variances, min=epsilon)
    second_moments = root_acc['cov_weighted_sums'] / safe_counts.view(-1, 1, 1)
    covariances = second_moments - means.unsqueeze(2).matmul(means.unsqueeze(1))
    eye = torch.eye(covariances.size(1), dtype=covariances.dtype)
    covariances = covariances + eye.unsqueeze(0) * epsilon
    means[counts == 0] = 0.0
    variances[counts == 0] = epsilon
    covariances[counts == 0] = eye * epsilon
    root_elapsed = time.perf_counter() - root_start

    return {
        'counts': counts.long(),
        'means': means.float(),
        'variances': variances.float(),
        'covariances': covariances.float(),
        'total_sample_size': int(root_acc['sample_size']),
        'sub_server_count': len(sub_stats),
        'sub_server_statistics_time_sec': sub_elapsed,
        'root_statistics_time_sec': root_elapsed,
    }


def write_augmented_feature_cache(args):
    start = time.perf_counter()
    features = args['features'].float()
    labels = args['labels'].long()
    gen = torch.Generator().manual_seed(
        args['seed'] + 2000003 * (args['client_id'] + 1)
    )
    global_means = args['global_means']
    global_vars = args['global_variances']
    global_covs = args.get('global_covariances')
    use_full_cov = bool(args.get('use_full_covariance', False)) and \
        global_covs is not None
    noise_scale = args['augmentation_noise_scale']
    per_sample = int(args['num_generated_per_sample'])
    per_proto = int(args['num_generated_per_prototype'])
    target_per_class = int(args['target_size_per_class'])

    all_features = [features]
    all_labels = [labels]

    cov_factors = {}
    if use_full_cov:
        factor_start = time.perf_counter()
        for class_idx in range(args['num_classes']):
            cov = global_covs[class_idx].float() * (noise_scale ** 2)
            jitter = 1e-6
            factor = None
            while factor is None:
                try:
                    factor = torch.linalg.cholesky(
                        cov + torch.eye(cov.size(0)) * jitter
                    )
                except RuntimeError:
                    jitter *= 10
                    if jitter > 1.0:
                        factor = torch.sqrt(
                            torch.clamp(torch.diag(cov), min=1e-6)
                        )
            cov_factors[class_idx] = factor
        factor_elapsed = time.perf_counter() - factor_start
    else:
        factor_elapsed = 0.0

    def sample_gaussian(center, class_idx, count):
        if count <= 0:
            return None
        if use_full_cov:
            factor = cov_factors[class_idx]
            z = torch.randn(
                (count, center.size(-1)),
                generator=gen,
                dtype=center.dtype,
            )
            if factor.dim() == 1:
                return center.unsqueeze(0) + z * factor.unsqueeze(0)
            return center.unsqueeze(0) + z.matmul(factor.t())
        std = torch.sqrt(global_vars[class_idx]).to(center.dtype) * noise_scale
        z = torch.randn(
            (count, center.size(-1)),
            generator=gen,
            dtype=center.dtype,
        )
        return center.unsqueeze(0) + z * std.unsqueeze(0)

    if per_sample > 0:
        generated_features = []
        generated_labels = []
        for feat, label in zip(features, labels):
            class_idx = int(label.item())
            generated = sample_gaussian(feat, class_idx, per_sample)
            if generated is None:
                continue
            generated_features.append(generated)
            generated_labels.append(
                torch.full((per_sample,), class_idx, dtype=torch.long)
            )
        if generated_features:
            all_features.append(torch.cat(generated_features, dim=0))
            all_labels.append(torch.cat(generated_labels, dim=0))

    current_counts = torch.bincount(labels, minlength=args['num_classes']).long()
    proto_features = []
    proto_labels = []
    for class_idx in range(args['num_classes']):
        need = max(per_proto, target_per_class - int(current_counts[class_idx]))
        if need <= 0:
            continue
        generated = sample_gaussian(global_means[class_idx], class_idx, need)
        if generated is None:
            continue
        proto_features.append(generated)
        proto_labels.append(
            torch.full((need,), class_idx, dtype=torch.long)
        )
    if proto_features:
        all_features.append(torch.cat(proto_features, dim=0))
        all_labels.append(torch.cat(proto_labels, dim=0))

    augmented_features = torch.cat(all_features, dim=0).float()
    augmented_labels = torch.cat(all_labels, dim=0).long()

    output_path = cache_path(args['feature_cache_dir'], args['client_id'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'features': augmented_features,
            'labels': augmented_labels,
            'metadata': {
                'source': 'real_dataset',
                'client_id': args['client_id'],
                'data_client_id': args['data_client_id'],
                'feature_cache_version': args['feature_cache_version'],
                'raw_sample_size': int(features.size(0)),
                'augmented_sample_size': int(augmented_features.size(0)),
                'embedding_dim': args['input_dim'],
                'num_classes': args['num_classes'],
                'num_generated_per_sample': per_sample,
                'num_generated_per_prototype': per_proto,
                'target_size_per_class': target_per_class,
                'use_full_covariance': use_full_cov,
                'dataset': args['dataset'],
                'data_root': args['data_root'],
                'feature_extractor': args['feature_extractor'],
                'checksum': tensor_sha256(augmented_features,
                                          augmented_labels),
            },
        },
        output_path,
    )

    elapsed = time.perf_counter() - start
    return {
        'client_id': args['client_id'],
        'sub_server_id': args['client_id'] % args['num_sub_servers'],
        'raw_sample_size': int(features.size(0)),
        'augmented_sample_size': int(augmented_features.size(0)),
        'feature_cache_path': str(output_path),
        'timing': {
            'augmentation_sec': elapsed,
            'covariance_factorization_sec': factor_elapsed,
            'augmentation_qps': augmented_features.size(0) / elapsed
            if elapsed > 0 else 0.0,
        },
    }


def run_worker_jobs(worker_fn, job_args, max_workers, timeout=0.0):
    if max_workers <= 1:
        results = []
        failed = []
        deadline = None if timeout <= 0 else time.perf_counter() + timeout
        for args in job_args:
            if deadline is not None and time.perf_counter() >= deadline:
                failed.append({
                    'client_id': args.get('client_id', -1),
                    'error': 'timeout',
                })
                continue
            try:
                results.append(worker_fn(args))
            except Exception as exc:
                failed.append({
                    'client_id': args.get('client_id', -1),
                    'error': str(exc),
                })
        return results, failed

    results = []
    failed = []
    timeout_arg = None if timeout <= 0 else timeout
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context('spawn'),
    ) as executor:
        futures = {
            executor.submit(worker_fn, args): args.get('client_id', -1)
            for args in job_args
        }
        done, not_done = wait(futures.keys(), timeout=timeout_arg)
        for future in not_done:
            future.cancel()
            failed.append({'client_id': futures[future], 'error': 'timeout'})
        for future in done:
            client_id = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failed.append({'client_id': client_id, 'error': str(exc)})
    return results, failed


def load_or_make_client_data(args):
    cached = load_feature_cache(args['feature_cache_dir'], args['client_id'])
    if cached is not None:
        return cached
    raise FileNotFoundError(
        f"No real augmented feature cache found for client "
        f"{args['client_id']} in {args['feature_cache_dir']!r}. "
        "Run once with ggeur_headonly.run_data_generation=True."
    )


def client_worker(args):
    torch.set_num_threads(1)
    start = time.perf_counter()

    download_start = time.perf_counter()
    model = build_mlp(
        args['input_dim'],
        args['num_classes'],
        args['hidden_dim'],
        args['dropout'],
    )
    model.load_state_dict(args['global_state'])
    download_elapsed = time.perf_counter() - download_start

    features, labels = load_or_make_client_data(args)
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=args['batch_size'],
        shuffle=True,
    )

    train_start = time.perf_counter()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for _ in range(args['local_update_steps']):
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            batch_size = batch_features.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (
                outputs.argmax(dim=1) == batch_labels
            ).sum().item()
            total_samples += batch_size

    train_elapsed = time.perf_counter() - train_start

    upload_start = time.perf_counter()
    upload_state = clone_state(model.state_dict(), tensor_dtype(args['param_dtype']))
    payload_bytes = state_payload_bytes(upload_state)
    upload_elapsed = time.perf_counter() - upload_start

    return {
        'client_id': args['client_id'],
        'sub_server_id': args['sub_server_id'],
        'round': args['round'],
        'base_version': args['base_version'],
        'sample_size': int(total_samples),
        'feature_cache_version': args['feature_cache_version'],
        'mlp_state_dict': upload_state,
        'payload_bytes': payload_bytes,
        'metrics': {
            'train_loss': total_loss / total_samples if total_samples else 0.0,
            'train_acc': total_correct / total_samples if total_samples else 0.0,
        },
        'timing': {
            'download_elapsed_sec': download_elapsed,
            'train_elapsed_sec': train_elapsed,
            'upload_elapsed_sec': upload_elapsed,
            'client_elapsed_sec': time.perf_counter() - start,
        },
    }


class ParameterService:
    def __init__(self, initial_state, metadata, param_dtype='float32'):
        self._versions = {}
        self._latest = 0
        self._param_dtype = param_dtype
        self.publish(initial_state, metadata)

    def latest_version(self):
        return self._latest

    def get(self, version=None):
        if version is None:
            version = self._latest
        record = self._versions[version]
        return clone_state(record['state_dict'], tensor_dtype(self._param_dtype)), copy.deepcopy(record['metadata'])

    def publish(self, state_dict, metadata):
        version = int(metadata.get('version', self._latest))
        state = clone_state(state_dict, tensor_dtype(self._param_dtype))
        full_metadata = copy.deepcopy(metadata)
        full_metadata.update({
            'version': version,
            'checksum': state_checksum(state),
            'payload_bytes': state_payload_bytes(state),
        })
        self._versions[version] = {
            'state_dict': state,
            'metadata': full_metadata,
        }
        self._latest = version
        return full_metadata


def aggregate_sub_servers(updates, cfg, round_idx, base_version):
    start = time.perf_counter()
    sub_results = []
    dropped = []

    by_sub_server = {idx: [] for idx in range(cfg.ggeur_headonly.num_sub_servers)}
    for update in updates:
        by_sub_server[update['sub_server_id']].append(update)

    for sub_server_id, sub_updates in by_sub_server.items():
        expected_sampled = sum(
            1 for update in updates if update['sub_server_id'] == sub_server_id
        )
        min_required = math.ceil(
            expected_sampled * cfg.ggeur_headonly.min_received_ratio
        )

        valid = []
        invalid_clients = []
        for update in sub_updates:
            if update['base_version'] != base_version:
                invalid_clients.append(update['client_id'])
                continue
            if update['feature_cache_version'] != cfg.ggeur_headonly.feature_cache_version:
                invalid_clients.append(update['client_id'])
                continue
            valid.append((update['sample_size'], update['mlp_state_dict']))

        if len(valid) < min_required:
            dropped.extend(update['client_id'] for update in sub_updates)
            continue

        aggregated, total_samples = weighted_average(valid)
        sub_results.append({
            'sub_server_id': sub_server_id,
            'round': round_idx,
            'base_version': base_version,
            'client_count': len(valid),
            'total_sample_size': total_samples,
            'aggregated_mlp_state_dict': aggregated,
            'dropped_clients': invalid_clients,
            'late_clients': [],
        })
        dropped.extend(invalid_clients)

    return sub_results, dropped, time.perf_counter() - start


def run_round(round_idx, cfg, parameter_service, feature_cache_dir):
    round_start = time.perf_counter()
    rng = random.Random(cfg.seed + round_idx)
    sampled = rng.sample(
        range(cfg.ggeur_headonly.client_total),
        cfg.ggeur_headonly.sample_clients_per_round,
    )

    version = parameter_service.latest_version()
    global_state, metadata = parameter_service.get(version)
    input_dim = int(cfg.ggeur.embedding_dim)
    num_classes = int(cfg.model.num_classes)
    max_workers = int(cfg.ggeur_headonly.process_workers)
    if max_workers <= 0:
        max_workers = min(len(sampled), os.cpu_count() or 1)

    worker_args = []
    for client_id in sampled:
        worker_args.append({
            'client_id': client_id,
            'sub_server_id': client_id % cfg.ggeur_headonly.num_sub_servers,
            'round': round_idx,
            'base_version': version,
            'feature_cache_version': cfg.ggeur_headonly.feature_cache_version,
            'global_state': global_state,
            'input_dim': input_dim,
            'num_classes': num_classes,
            'hidden_dim': int(cfg.ggeur.mlp_hidden_dim),
            'dropout': float(cfg.ggeur.mlp_dropout),
            'batch_size': int(cfg.dataloader.batch_size),
            'local_update_steps': int(cfg.train.local_update_steps),
            'lr': float(cfg.train.optimizer.lr),
            'param_dtype': cfg.ggeur_headonly.param_dtype,
            'feature_cache_dir': feature_cache_dir,
            'seed': int(cfg.seed),
        })

    timeout = float(cfg.ggeur_headonly.round_timeout)
    timeout_arg = None if timeout <= 0 else timeout
    updates = []
    late_clients = []
    failed_clients = []

    if max_workers == 1:
        deadline = None if timeout <= 0 else time.perf_counter() + timeout
        for args in worker_args:
            if deadline is not None and time.perf_counter() >= deadline:
                late_clients.append(args['client_id'])
                continue
            try:
                updates.append(client_worker(args))
            except Exception as exc:
                failed_clients.append({
                    'client_id': args['client_id'],
                    'error': str(exc),
                })
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(client_worker, args): args['client_id']
                for args in worker_args
            }
            done, not_done = wait(futures.keys(), timeout=timeout_arg)
            late_clients = [futures[future] for future in not_done]
            for future in not_done:
                future.cancel()
            for future in done:
                client_id = futures[future]
                try:
                    updates.append(future.result())
                except Exception as exc:
                    failed_clients.append({
                        'client_id': client_id,
                        'error': str(exc),
                    })

    sub_results, dropped_clients, sub_elapsed = aggregate_sub_servers(
        updates, cfg, round_idx, version
    )

    root_start = time.perf_counter()
    root_updates = [
        (result['total_sample_size'], result['aggregated_mlp_state_dict'])
        for result in sub_results
    ]
    new_state, total_samples = weighted_average(root_updates)
    root_elapsed = time.perf_counter() - root_start

    publish_elapsed = 0.0
    next_metadata = None
    if new_state:
        publish_start = time.perf_counter()
        next_metadata = parameter_service.publish(
            new_state,
            {
                'version': version + 1,
                'round': round_idx,
                'param_type': 'mlp_head',
                'embedding_dim': input_dim,
                'num_classes': num_classes,
                'feature_cache_version': cfg.ggeur_headonly.feature_cache_version,
            },
        )
        publish_elapsed = time.perf_counter() - publish_start

    round_elapsed = time.perf_counter() - round_start
    valid_clients = sum(result['client_count'] for result in sub_results)
    payloads = [update['payload_bytes'] for update in updates]
    train_times = [update['timing']['train_elapsed_sec'] for update in updates]
    download_times = [update['timing']['download_elapsed_sec'] for update in updates]
    upload_times = [update['timing']['upload_elapsed_sec'] for update in updates]
    train_accs = [update['metrics']['train_acc'] for update in updates]
    train_losses = [update['metrics']['train_loss'] for update in updates]

    parallel_download_time = sum(download_times) / max_workers if download_times else 0.0
    parallel_upload_time = sum(upload_times) / max_workers if upload_times else 0.0

    return {
        'round': round_idx,
        'base_version': version,
        'new_version': parameter_service.latest_version(),
        'sampled_clients': len(sampled),
        'received_updates': len(updates),
        'valid_clients': valid_clients,
        'late_clients': late_clients,
        'dropped_clients': dropped_clients,
        'failed_clients': failed_clients,
        'sub_server_count': len(sub_results),
        'total_samples': total_samples,
        'round_time_sec': round_elapsed,
        'train_qps_samples_per_sec': total_samples / round_elapsed if round_elapsed > 0 else 0.0,
        'download_qps_req_per_sec': (
            len(updates) / parallel_download_time if parallel_download_time > 0 else 0.0
        ),
        'upload_qps_req_per_sec': (
            len(updates) / parallel_upload_time if parallel_upload_time > 0 else 0.0
        ),
        'avg_train_loss': statistics.mean(train_losses) if train_losses else 0.0,
        'avg_train_acc': statistics.mean(train_accs) if train_accs else 0.0,
        'avg_payload_bytes': statistics.mean(payloads) if payloads else 0.0,
        'max_payload_bytes': max(payloads) if payloads else 0,
        'avg_client_train_time_sec': statistics.mean(train_times) if train_times else 0.0,
        'sub_server_aggregation_time_sec': sub_elapsed,
        'root_aggregation_time_sec': root_elapsed,
        'parameter_publish_time_sec': publish_elapsed,
        'parameter_payload_bytes': (
            next_metadata or metadata
        ).get('payload_bytes', 0),
        'parameter_checksum': (
            next_metadata or metadata
        ).get('checksum', ''),
    }


def generation_manifest(cfg):
    return {
        'source': 'real_dataset',
        'feature_cache_version': cfg.ggeur_headonly.feature_cache_version,
        'dataset': cfg.data.type,
        'data_root': cfg.data.root,
        'data_splits': list(cfg.data.splits) if hasattr(cfg.data, 'splits') else [],
        'seed': int(cfg.seed),
        'client_total': int(cfg.ggeur_headonly.client_total),
        'num_sub_servers': int(cfg.ggeur_headonly.num_sub_servers),
        'feature_extractor': getattr(cfg.ggeur, 'feature_extractor', 'clip'),
        'embedding_dim': int(cfg.ggeur.embedding_dim),
        'num_classes': int(cfg.model.num_classes),
        'num_generated_per_sample': int(cfg.ggeur.num_generated_per_sample),
        'num_generated_per_prototype': int(cfg.ggeur.num_generated_per_prototype),
        'target_size_per_class': int(cfg.ggeur.target_size_per_class),
        'augmentation_noise_scale': float(
            cfg.ggeur_headonly.augmentation_noise_scale
        ),
    }


def existing_real_cache_metrics(cfg, feature_cache_dir, expected):
    path = manifest_path(feature_cache_dir)
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None
    if manifest.get('config') != expected:
        return None

    raw_samples = 0
    augmented_samples = 0
    cache_bytes = 0
    clients = int(cfg.ggeur_headonly.client_total)
    for client_id in range(clients):
        path = cache_path(feature_cache_dir, client_id)
        if not path.exists():
            return None
        data = torch.load(path, map_location='cpu')
        metadata = data.get('metadata', {}) if isinstance(data, dict) else {}
        if metadata.get('source') != 'real_dataset':
            return None
        if metadata.get('feature_cache_version') != \
                cfg.ggeur_headonly.feature_cache_version:
            return None
        raw_samples += int(metadata.get('raw_sample_size', 0))
        augmented_samples += int(metadata.get('augmented_sample_size', 0))
        cache_bytes += path.stat().st_size

    return {
        'feature_cache_dir': feature_cache_dir,
        'feature_cache_version': cfg.ggeur_headonly.feature_cache_version,
        'client_total': clients,
        'raw_samples': raw_samples,
        'augmented_samples': augmented_samples,
        'cache_bytes': cache_bytes,
        'total_time_sec': 0.0,
        'real_feature_extraction_time_sec': 0.0,
        'augmentation_time_sec': 0.0,
        'sub_server_statistics_time_sec': 0.0,
        'root_statistics_time_sec': 0.0,
        'raw_feature_qps_samples_per_sec': 0.0,
        'augmentation_qps_samples_per_sec': 0.0,
        'end_to_end_generation_qps_samples_per_sec': 0.0,
        'avg_client_raw_feature_qps': 0.0,
        'avg_client_augmentation_qps': 0.0,
        'reused_existing_cache': True,
        'manifest_path': str(manifest_path(feature_cache_dir)),
        'global_statistics': manifest.get('global_statistics', {}),
    }


def run_data_generation(cfg, feature_cache_dir):
    start = time.perf_counter()
    cache_dir = Path(feature_cache_dir)
    expected = generation_manifest(cfg)

    if cfg.ggeur_headonly.reuse_feature_cache and \
            not cfg.ggeur_headonly.overwrite_feature_cache:
        cached = existing_real_cache_metrics(cfg, feature_cache_dir, expected)
        if cached is not None:
            return cached

    if cfg.ggeur_headonly.overwrite_feature_cache and cache_dir.exists():
        for path in cache_dir.glob('client_*.pt'):
            path.unlink()
        manifest_path(cache_dir).unlink(missing_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    input_dim = int(cfg.ggeur.embedding_dim)
    num_classes = int(cfg.model.num_classes)
    device = resolve_device(cfg)
    data_cfg = cfg.clone()
    data_cfg.defrost()
    data_cfg.federate.mode = 'standalone'
    data_cfg.federate.client_num = int(cfg.ggeur_headonly.client_total)
    data_cfg.federate.sample_client_num = int(
        cfg.ggeur_headonly.sample_clients_per_round)
    data, modified_cfg = get_data(data_cfg)
    if data is None:
        raise RuntimeError('Real dataset loader returned no data')
    if int(modified_cfg.model.num_classes) != num_classes:
        raise ValueError(
            f'Configured model.num_classes={num_classes}, but real dataset '
            f'loader discovered {modified_cfg.model.num_classes}. Update the '
            'HeadOnly config before generation.'
        )

    extractor, encode, real_dim = build_feature_extractor(cfg, device)
    if real_dim != input_dim:
        raise ValueError(
            f'Configured ggeur.embedding_dim={input_dim}, but extractor '
            f'produces {real_dim}. Update the config before generation.'
        )

    stats_start = time.perf_counter()
    local_stats = []
    for client_id in range(int(cfg.ggeur_headonly.client_total)):
        data_client_id = client_id + 1
        if data_client_id not in data:
            raise KeyError(f'Real dataset has no client {data_client_id}')
        train_loader = data[data_client_id].get('train')
        if train_loader is None:
            raise KeyError(f'Real dataset client {data_client_id} has no train loader')
        local_stats.append(
            extract_real_client_features(
                cfg,
                client_id,
                train_loader,
                encode,
                device,
            )
        )
    stats_elapsed = time.perf_counter() - stats_start

    if device.type == 'cuda':
        extractor.cpu()
        del extractor
        torch.cuda.empty_cache()

    global_stats = aggregate_generation_statistics(
        local_stats,
        int(cfg.ggeur_headonly.num_sub_servers),
        float(cfg.ggeur_headonly.covariance_epsilon),
    )

    aug_start = time.perf_counter()
    aug_results = []
    for item in local_stats:
        aug_results.append(
            write_augmented_feature_cache({
                'client_id': item['client_id'],
                'data_client_id': item['data_client_id'],
                'features': item['features'],
                'labels': item['labels'],
                'input_dim': input_dim,
                'num_classes': num_classes,
                'seed': int(cfg.seed),
                'num_sub_servers': int(cfg.ggeur_headonly.num_sub_servers),
                'global_means': global_stats['means'],
                'global_variances': global_stats['variances'],
                'augmentation_noise_scale': float(
                    cfg.ggeur_headonly.augmentation_noise_scale),
                'num_generated_per_sample': int(
                    cfg.ggeur.num_generated_per_sample),
                'num_generated_per_prototype': int(
                    cfg.ggeur.num_generated_per_prototype),
                'target_size_per_class': int(cfg.ggeur.target_size_per_class),
                'feature_cache_dir': feature_cache_dir,
                'feature_cache_version':
                    cfg.ggeur_headonly.feature_cache_version,
                'dataset': cfg.data.type,
                'data_root': cfg.data.root,
                'feature_extractor': getattr(
                    cfg.ggeur, 'feature_extractor', 'clip'),
            })
        )
    aug_elapsed = time.perf_counter() - aug_start

    total_elapsed = time.perf_counter() - start
    raw_samples = sum(item['sample_size'] for item in local_stats)
    augmented_samples = sum(item['augmented_sample_size'] for item in aug_results)
    cache_bytes = sum(
        cache_path(feature_cache_dir, client_id).stat().st_size
        for client_id in range(int(cfg.ggeur_headonly.client_total))
    )
    feature_qps_values = [
        item['timing']['feature_generation_qps'] for item in local_stats
    ]
    aug_qps_values = [
        item['timing']['augmentation_qps'] for item in aug_results
    ]

    global_statistics = {
        'total_sample_size': global_stats['total_sample_size'],
        'sub_server_count': global_stats['sub_server_count'],
        'non_empty_classes': int((global_stats['counts'] > 0).sum().item()),
    }
    manifest_path(cache_dir).write_text(
        json.dumps({
            'config': expected,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
            'global_statistics': global_statistics,
            'raw_samples': raw_samples,
            'augmented_samples': augmented_samples,
            'cache_bytes': cache_bytes,
        }, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    return {
        'feature_cache_dir': feature_cache_dir,
        'feature_cache_version': cfg.ggeur_headonly.feature_cache_version,
        'client_total': int(cfg.ggeur_headonly.client_total),
        'raw_samples': raw_samples,
        'augmented_samples': augmented_samples,
        'cache_bytes': cache_bytes,
        'total_time_sec': total_elapsed,
        'real_feature_extraction_time_sec': stats_elapsed,
        'augmentation_time_sec': aug_elapsed,
        'sub_server_statistics_time_sec':
            global_stats['sub_server_statistics_time_sec'],
        'root_statistics_time_sec': global_stats['root_statistics_time_sec'],
        'raw_feature_qps_samples_per_sec':
            raw_samples / stats_elapsed if stats_elapsed > 0 else 0.0,
        'augmentation_qps_samples_per_sec':
            augmented_samples / aug_elapsed if aug_elapsed > 0 else 0.0,
        'end_to_end_generation_qps_samples_per_sec':
            augmented_samples / total_elapsed if total_elapsed > 0 else 0.0,
        'avg_client_raw_feature_qps':
            statistics.mean(feature_qps_values) if feature_qps_values else 0.0,
        'avg_client_augmentation_qps':
            statistics.mean(aug_qps_values) if aug_qps_values else 0.0,
        'reused_existing_cache': False,
        'manifest_path': str(manifest_path(feature_cache_dir)),
        'global_statistics': global_statistics,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run GGEUR head-only hierarchical FL on one machine.'
    )
    parser.add_argument('--cfg', '--cfg_file', dest='cfg_file', default='')
    parser.add_argument(
        'opts',
        nargs=argparse.REMAINDER,
        help='Optional KEY VALUE config overrides.',
    )
    return parser.parse_args()


def build_cfg(args):
    cfg = global_cfg.clone()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    if args.opts:
        cfg.merge_from_list(args.opts)

    if not (cfg.ggeur_headonly.use or cfg.ggeur.head_only_mode):
        raise ValueError(
            'Set ggeur.head_only_mode=True or ggeur_headonly.use=True to '
            'run the head-only hierarchical training script.'
        )

    cfg.freeze(inform=False, save=False)
    return cfg


def run(cfg):
    input_dim = int(cfg.ggeur.embedding_dim)
    num_classes = int(cfg.model.num_classes)
    feature_cache_dir = resolve_feature_cache_dir(cfg)
    run_dir, run_id = resolve_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    generation_metrics = None

    if cfg.ggeur_headonly.run_data_generation:
        print(
            'GGEUR head-only round-0 data generation: '
            f'clients={cfg.ggeur_headonly.client_total}, '
            f'cache_dir={feature_cache_dir}'
        )
        generation_metrics = run_data_generation(cfg, feature_cache_dir)
        print(
            'generation '
            f"raw_samples={generation_metrics['raw_samples']} "
            f"augmented_samples={generation_metrics['augmented_samples']} "
            f"time={generation_metrics['total_time_sec']:.4f}s "
            f"raw_qps={generation_metrics['raw_feature_qps_samples_per_sec']:.2f} samples/s "
            f"aug_qps={generation_metrics['augmentation_qps_samples_per_sec']:.2f} samples/s "
            f"end_to_end_qps={generation_metrics['end_to_end_generation_qps_samples_per_sec']:.2f} samples/s"
        )

    model = build_mlp(
        input_dim,
        num_classes,
        int(cfg.ggeur.mlp_hidden_dim),
        float(cfg.ggeur.mlp_dropout),
    )
    initial_state = clone_state(model.state_dict(), tensor_dtype(cfg.ggeur_headonly.param_dtype))
    parameter_service = ParameterService(
        initial_state,
        {
            'version': 0,
            'round': 0,
            'param_type': 'mlp_head',
            'embedding_dim': input_dim,
            'num_classes': num_classes,
            'feature_cache_version': cfg.ggeur_headonly.feature_cache_version,
        },
        param_dtype=cfg.ggeur_headonly.param_dtype,
    )

    round_metrics = []
    total_start = time.perf_counter()
    total_rounds = int(cfg.federate.total_round_num)

    print(
        'GGEUR head-only hierarchical training: '
        f'clients={cfg.ggeur_headonly.client_total}, '
        f'sample_per_round={cfg.ggeur_headonly.sample_clients_per_round}, '
        f'sub_servers={cfg.ggeur_headonly.num_sub_servers}, '
        f'rounds={total_rounds}, '
        f'feature_cache_dir={feature_cache_dir}'
    )

    for round_idx in range(1, total_rounds + 1):
        metrics = run_round(round_idx, cfg, parameter_service, feature_cache_dir)
        round_metrics.append(metrics)
        print(
            f"round={metrics['round']} "
            f"version={metrics['base_version']}->{metrics['new_version']} "
            f"valid={metrics['valid_clients']}/{metrics['sampled_clients']} "
            f"samples={metrics['total_samples']} "
            f"time={metrics['round_time_sec']:.4f}s "
            f"train_qps={metrics['train_qps_samples_per_sec']:.2f} samples/s "
            f"download_qps={metrics['download_qps_req_per_sec']:.2f} req/s "
            f"upload_qps={metrics['upload_qps_req_per_sec']:.2f} req/s "
            f"loss={metrics['avg_train_loss']:.4f} "
            f"acc={metrics['avg_train_acc']:.4f}"
        )

    total_elapsed = time.perf_counter() - total_start
    total_samples = sum(item['total_samples'] for item in round_metrics)
    generation_elapsed = (
        generation_metrics.get('total_time_sec', 0.0)
        if generation_metrics else 0.0
    )
    end_to_end_elapsed = generation_elapsed + total_elapsed
    summary = {
        'run_id': run_id,
        'run_dir': str(run_dir),
        'feature_cache_dir': feature_cache_dir,
        'generation': generation_metrics,
        'total_rounds': total_rounds,
        'total_time_sec': total_elapsed,
        'end_to_end_total_time_sec': end_to_end_elapsed,
        'total_samples': total_samples,
        'overall_train_qps_samples_per_sec': (
            total_samples / total_elapsed if total_elapsed > 0 else 0.0
        ),
        'end_to_end_train_qps_with_generation_samples_per_sec': (
            total_samples / end_to_end_elapsed if end_to_end_elapsed > 0 else 0.0
        ),
        'avg_round_time_sec': statistics.mean(
            item['round_time_sec'] for item in round_metrics
        ) if round_metrics else 0.0,
        'avg_train_qps_samples_per_sec': statistics.mean(
            item['train_qps_samples_per_sec'] for item in round_metrics
        ) if round_metrics else 0.0,
        'avg_download_qps_req_per_sec': statistics.mean(
            item['download_qps_req_per_sec'] for item in round_metrics
        ) if round_metrics else 0.0,
        'avg_upload_qps_req_per_sec': statistics.mean(
            item['upload_qps_req_per_sec'] for item in round_metrics
        ) if round_metrics else 0.0,
        'rounds': round_metrics,
    }

    print('summary=' + json.dumps(summary, ensure_ascii=False, indent=2))

    output_path = resolve_output_json(cfg, run_dir, run_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    try:
        resolved_config = cfg.dump()
    except Exception as error:
        resolved_config = (
            '# Failed to serialize resolved config through cfg.dump().\n'
            f'# Error: {error}\n'
        )
    (run_dir / 'resolved_config.yaml').write_text(
        resolved_config,
        encoding='utf-8',
    )

    return summary


def main():
    args = parse_args()
    cfg = build_cfg(args)
    run(cfg)


if __name__ == '__main__':
    main()
