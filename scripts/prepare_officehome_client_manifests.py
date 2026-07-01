#!/usr/bin/env python3
"""Prepare exact per-client OfficeHome manifests for real distributed runs."""

import argparse
import json
import os
import shutil
from pathlib import Path

from torchvision import transforms

from federatedscope.cv.dataset.office_home import (
    OfficeHome, load_office_home_domain_data)
from federatedscope.contrib.data.ggeur_data import (
    _generate_dirichlet_matrix, _split_dataset_for_clients,
    _split_dataset_with_lds, _split_subset_for_clients)


def iter_records(dataset):
    from torch.utils.data import Subset

    if isinstance(dataset, Subset):
        base = dataset.dataset
        for idx in dataset.indices:
            yield base.data[idx], int(base.targets[idx])
        return

    for path, label in zip(dataset.data, dataset.targets):
        yield path, int(label)


def materialize(path, label, source_root, client_root, mode):
    rel_path = os.path.relpath(path, source_root)
    if mode == 'manifest-only':
        return {'path': rel_path.replace('\\', '/'), 'label': label}

    dst = client_root / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        if mode == 'copy':
            shutil.copy2(path, dst)
        elif mode == 'hardlink':
            os.link(path, dst)
        else:
            os.symlink(path, dst)
    return {'path': rel_path.replace('\\', '/'), 'label': label}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--client-num', type=int, required=True)
    parser.add_argument('--splits', default='0.7,0.0,0.3')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-lds', action='store_true')
    parser.add_argument('--lds-alpha', type=float, default=0.1)
    parser.add_argument('--lds-seed', type=int, default=42)
    parser.add_argument('--domains', default='')
    parser.add_argument('--mode',
                        choices=['symlink', 'copy', 'hardlink',
                                 'manifest-only'],
                        default='symlink')
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    splits = tuple(float(x) for x in args.splits.split(','))
    domains = [x.strip() for x in args.domains.split(',') if x.strip()] \
        or OfficeHome.DOMAINS
    invalid = [domain for domain in domains if domain not in OfficeHome.DOMAINS]
    if invalid:
        raise ValueError(f"Invalid OfficeHome domains: {invalid}")
    if args.client_num % len(domains) != 0:
        raise ValueError(
            f"client-num={args.client_num} must be divisible by "
            f"domain count={len(domains)}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    clients_per_domain = args.client_num // len(domains)
    dirichlet_matrix = None
    if args.use_lds:
        dirichlet_matrix = _generate_dirichlet_matrix(
            len(domains), len(OfficeHome.CLASSES), args.lds_alpha,
            args.lds_seed)

    matrix = []
    client_id = 1
    for domain_idx, domain in enumerate(domains):
        domain_data = load_office_home_domain_data(
            root=str(source_root),
            domain=domain,
            splits=splits,
            transform=transform,
            seed=args.seed)
        train_dataset = domain_data['train']
        if args.use_lds:
            lds_subset, _ = _split_dataset_with_lds(
                train_dataset,
                dirichlet_matrix[domain_idx],
                seed=args.seed + domain_idx)
            train_subsets = _split_subset_for_clients(
                lds_subset, clients_per_domain, seed=args.seed + domain_idx)
        elif clients_per_domain > 1:
            train_subsets = _split_dataset_for_clients(
                train_dataset, clients_per_domain, seed=args.seed)
        else:
            train_subsets = [train_dataset]

        for train_subset in train_subsets:
            client_root = output_root / f'client_{client_id:06d}'
            client_root.mkdir(parents=True, exist_ok=True)
            manifest = {
                'dataset': 'office-home',
                'client_id': client_id,
                'domain': domain,
                'domains': [domain],
                'classes': OfficeHome.CLASSES,
                'source_root': str(source_root),
                'root': str(source_root) if args.mode == 'manifest-only'
                else '.',
                'splits': {
                    'train': [],
                    'val': [],
                    'test': [],
                },
            }
            for split_name, dataset in (
                    ('train', train_subset),
                    ('val', domain_data['val']),
                    ('test', domain_data['test'])):
                manifest['splits'][split_name] = [
                    materialize(path, label, str(source_root), client_root,
                                args.mode)
                    for path, label in iter_records(dataset)
                ]

            manifest_path = client_root / 'client_manifest.json'
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding='utf-8')
            matrix.append({
                'client_id': client_id,
                'domain': domain,
                'root': str(client_root),
                'manifest': str(manifest_path),
                'train': len(manifest['splits']['train']),
                'val': len(manifest['splits']['val']),
                'test': len(manifest['splits']['test']),
            })
            client_id += 1

    summary = {
        'source_root': str(source_root),
        'output_root': str(output_root),
        'client_num': args.client_num,
        'domains': domains,
        'splits': splits,
        'use_lds': args.use_lds,
        'lds_alpha': args.lds_alpha,
        'lds_seed': args.lds_seed,
        'mode': args.mode,
        'clients': matrix,
    }
    summary_path = output_root / 'manifest_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding='utf-8')
    print(summary_path)


if __name__ == '__main__':
    main()
