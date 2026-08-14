#!/usr/bin/env python3
"""
Generate real multi-machine GGEUR distributed configs.

The generated configs are consumed by federatedscope/main.py. They support a
two-host deployment where one server machine runs the server process and one
client machine simulates multiple client processes.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import yaml


DATASETS = {
    'officehome': {
        'data_type': 'office-home',
        'root': '/root/autodl-tmp/datasets/OfficeHomeDataset_10072016',
        'num_classes': 65,
        'splits': [0.7, 0.0, 0.3],
    },
    'domainnet': {
        'data_type': 'domainnet',
        'root': '/root/autodl-tmp/datasets/DomainNet',
        'num_classes': 345,
        'splits': [0.7, 0.0, 0.3],
        'domainnet_domains': ['clipart', 'painting', 'real', 'sketch'],
    },
}

MODELS = {
    'vit': {
        'feature_extractor': 'clip',
        'embedding_dim': 512,
        'train_lr': 0.0001,
        'local_update_steps': 1,
        'mlp_hidden_dim': 0,
        'mlp_dropout': 0.0,
        'clip_model': 'ViT-B-16',
        'clip_pretrained': 'openai',
        'clip_model_path': '/root/autodl-tmp/models/open_clip_vitb16.bin',
    },
    'cnn': {
        'feature_extractor': 'cnn',
        'embedding_dim': 1024,
        'train_lr': 0.001,
        'local_update_steps': 20,
        'mlp_hidden_dim': 256,
        'mlp_dropout': 0.2,
        'cnn_backbone': 'convnext_base',
        'cnn_pretrained': True,
    },
    'mixer': {
        'feature_extractor': 'timm',
        'embedding_dim': 768,
        'train_lr': 0.001,
        'local_update_steps': 10,
        'mlp_hidden_dim': 0,
        'mlp_dropout': 0.0,
        'timm_model': 'mixer_b16_224',
        'timm_pretrained': False,
        'timm_checkpoint_path': '/root/autodl-tmp/models/mixer_b16_224_complete.pth',
        'timm_in_chans': 3,
        'timm_global_pool': 'avg',
    },
}

METHODS = ['fedavg', 'fedprox', 'fedopt', 'moon', 'fedproto', 'promptfl']


def split_csv(value):
    return [item.strip() for item in str(value or '').split(',')
            if item.strip()]


def pick_from_csv(items, index, default='', mode='block', total=1):
    if not items:
        return default
    if len(items) == 1:
        return items[0]
    if mode == 'round-robin':
        return items[(index - 1) % len(items)]
    block = max(1, (int(total) + len(items) - 1) // len(items))
    return items[min((index - 1) // block, len(items) - 1)]


def pick_port(explicit_ports, index, base_port, total=1):
    if explicit_ports:
        return int(pick_from_csv(explicit_ports, index, '', 'block', total))
    return int(base_port) + index


def path_str(path):
    return str(path).replace('\\', '/')


def deep_set(cfg, dotted_key, value):
    node = cfg
    parts = dotted_key.split('.')
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def base_config(args, dataset, model, method):
    ds = copy.deepcopy(DATASETS[dataset])
    model_cfg = copy.deepcopy(MODELS[model])
    rounds = args.promptfl_rounds if method == 'promptfl' else args.rounds

    cfg = {
        'use_gpu': args.use_gpu,
        'device': args.device,
        'seed': args.seed,
        'verbose': 1,
        'federate': {
            'method': 'ggeur',
            'mode': 'distributed',
            'client_num': args.clients,
            'sample_client_num': args.sample_clients or args.clients,
            'total_round_num': rounds,
            'make_global_eval': False,
            'online_aggr': False,
        },
        'distribute': {
            'use': True,
            'server_host': args.server_host,
            'server_port': args.server_port,
            'join_timeout_seconds': args.join_timeout_seconds,
            'grpc_max_send_message_length':
                args.grpc_max_send_message_length,
            'grpc_max_receive_message_length':
                args.grpc_max_receive_message_length,
            'grpc_enable_http_proxy': False,
            'grpc_compression': args.grpc_compression,
        },
        'data': {
            'type': ds['data_type'],
            'root': args.data_root or ds['root'],
            'splits': args.splits or ds['splits'],
        },
        'dataloader': {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
        },
        'model': {
            'type': 'ggeur_mlp',
            'num_classes': ds['num_classes'],
        },
        'train': {
            'local_update_steps': model_cfg['local_update_steps'],
            'optimizer': {
                'type': 'Adam',
                'lr': model_cfg['train_lr'],
            },
        },
        'eval': {
            'freq': args.eval_freq,
            'metrics': ['acc'],
            'split': ['test'],
            'best_res_update_round_wise_key': 'test_acc',
        },
        'trainer': {'type': 'ggeur'},
        'ggeur': {
            'use': True,
            'feature_extractor': model_cfg['feature_extractor'],
            'embedding_dim': model_cfg['embedding_dim'],
            'freeze_backbone': True,
            'use_feature_cache': True,
            'feature_cache_dir': args.feature_cache_dir,
            'unload_extractor_after_cache': True,
            'use_fp16_extraction': True,
            'extract_batch_size': args.extract_batch_size,
            'num_generated_per_sample': args.num_generated_per_sample,
            'num_generated_per_prototype': args.num_generated_per_prototype,
            'target_size_per_class': args.target_size_per_class,
            'mlp_hidden_dim': model_cfg['mlp_hidden_dim'],
            'mlp_dropout': model_cfg['mlp_dropout'],
            'use_cross_client_prototypes': True,
            'statistics_round': 0,
            'headonly_eval_mode': args.headonly_eval_mode,
            'use_fedproto': False,
            'use_lds': args.use_lds,
            'lds_alpha': args.lds_alpha,
            'lds_seed': args.seed,
            'use_cnn_distillation': False,
            'use_feature_alignment': False,
            'use_separated_training': False,
            'use_end_to_end_finetune': False,
            'use_moon': False,
            'use_promptfl': False,
            'distributed_stage_timeout': args.stage_timeout,
            'headonly_cache_version': args.headonly_cache_version,
            'headonly_skip_round0_if_augmented_cache_exists':
                args.headonly_skip_round0_if_augmented_cache_exists,
            'officehome_manifest_path': '',
            'officehome_domains': split_csv(args.officehome_domains),
            'min_statistics_clients': args.min_statistics_clients,
            'min_augmentation_clients': args.min_augmentation_clients,
            'min_train_updates': args.min_train_updates,
        },
    }

    if dataset == 'domainnet':
        cfg['ggeur']['domainnet_domains'] = ds['domainnet_domains']
        cfg['ggeur']['domainnet_shared_classes_only'] = False

    for key, value in model_cfg.items():
        if key in {
            'feature_extractor', 'embedding_dim', 'train_lr',
            'local_update_steps', 'mlp_hidden_dim', 'mlp_dropout'
        }:
            continue
        cfg['ggeur'][key] = value

    apply_method(cfg, method)
    if args.head_only:
        apply_head_only(cfg, args)

    return cfg


def apply_method(cfg, method):
    if method == 'fedavg':
        return
    if method == 'fedprox':
        cfg.setdefault('fedprox', {})['use'] = True
        cfg['fedprox']['mu'] = 0.1
    elif method == 'fedopt':
        cfg.setdefault('fedopt', {})['use'] = True
        cfg['fedopt']['optimizer'] = {'type': 'SGD', 'lr': 0.01}
        cfg['fedopt']['annealing'] = False
    elif method == 'moon':
        cfg['ggeur']['use_moon'] = True
        cfg['ggeur']['moon_mu'] = 1.0
        cfg['ggeur']['moon_temperature'] = 0.5
    elif method == 'fedproto':
        cfg['ggeur']['use_fedproto'] = True
        cfg['ggeur']['proto_weight'] = 0.1
        cfg['ggeur']['proto_distance'] = 'cosine'
        cfg['ggeur']['proto_temperature'] = 0.1
    elif method == 'promptfl':
        cfg['ggeur']['use_promptfl'] = True
        cfg['ggeur']['prompt_length'] = 4
        cfg['ggeur']['prompt_local_epochs'] = 1
        cfg['ggeur']['prompt_max_train_batches'] = 1
        cfg['ggeur']['prompt_samples_per_proto'] = 1
        cfg['ggeur']['prompt_lr'] = 0.002
        cfg['ggeur']['prompt_temperature'] = 0.07
    else:
        raise ValueError(f'Unknown method: {method}')


def apply_head_only(cfg, args):
    cfg['ggeur']['head_only_mode'] = True
    cfg['ggeur']['freeze_backbone'] = True
    cfg['ggeur']['use_feature_cache'] = True
    cfg['ggeur']['unload_extractor_after_cache'] = True
    cfg['ggeur']['use_cnn_distillation'] = False
    cfg['ggeur']['use_feature_alignment'] = False
    cfg['ggeur']['use_separated_training'] = False
    cfg['ggeur']['use_end_to_end_finetune'] = False
    # PromptFL is not an MLP-head-only path. Head-only matrix skips it unless
    # the caller explicitly disables strict checking.
    if cfg['ggeur'].get('use_promptfl') and not args.allow_promptfl_in_head_only:
        raise ValueError('promptfl is incompatible with strict head-only mode')
    cfg.setdefault('ggeur_headonly', {})
    cfg['ggeur_headonly'].update({
        'use': True,
        'client_total': cfg['federate']['client_num'],
        'sample_clients_per_round': cfg['federate']['sample_client_num'],
        'num_sub_servers': args.sub_servers,
        'feature_cache_dir': args.feature_cache_dir,
        'feature_cache_version': args.headonly_cache_version,
    })


def write_yaml(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def generate_case(args, dataset, model, method, out_root):
    case_name = f'{dataset}_{model}_{method}'
    if args.head_only:
        case_name += '_headonly'
    case_dir = out_root / case_name
    cfg_dir = case_dir / 'configs'
    cfg_dir.mkdir(parents=True, exist_ok=True)

    server_cfg = base_config(args, dataset, model, method)
    server_cfg['distribute']['role'] = 'server'
    server_cfg['distribute']['server_host'] = args.server_bind_host
    server_cfg['outdir'] = path_str(case_dir / 'server_out')
    server_cfg['expname'] = f'{case_name}_server'
    write_yaml(cfg_dir / 'server.yaml', server_cfg)

    client_cfgs = []
    client_hosts = split_csv(args.client_hosts)
    client_bind_hosts = split_csv(args.client_bind_hosts)
    client_advertise_ports = split_csv(args.client_advertise_ports)
    client_bind_ports = split_csv(args.client_bind_ports)
    manifest_base = Path(args.officehome_manifest_base) \
        if args.officehome_manifest_base else None
    for client_id in range(1, args.clients + 1):
        client_cfg = base_config(args, dataset, model, method)
        advertise_host = pick_from_csv(
            client_hosts,
            client_id,
            args.client_host,
            args.client_host_assignment,
            args.clients,
        )
        bind_host = pick_from_csv(
            client_bind_hosts,
            client_id,
            args.client_bind_host,
            args.client_host_assignment,
            args.clients,
        )
        bind_port = pick_port(client_bind_ports,
                              client_id,
                              args.client_bind_port_base,
                              args.clients)
        advertise_port = pick_port(client_advertise_ports,
                                   client_id,
                                   args.client_advertise_port_base,
                                   args.clients)
        client_cfg['distribute']['role'] = 'client'
        client_cfg['distribute']['server_host'] = args.server_host
        client_cfg['distribute']['client_host'] = bind_host
        client_cfg['distribute']['client_port'] = bind_port
        client_cfg['distribute']['client_advertise_host'] = advertise_host
        client_cfg['distribute']['client_advertise_port'] = advertise_port
        client_cfg['distribute']['data_idx'] = client_id
        client_cfg['outdir'] = path_str(case_dir / f'client_{client_id}_out')
        client_cfg['expname'] = f'{case_name}_client_{client_id}'
        if manifest_base is not None and dataset == 'officehome':
            client_root = manifest_base / f'client_{client_id:06d}'
            manifest_path = client_root / 'client_manifest.json'
            client_cfg['data']['root'] = path_str(client_root)
            client_cfg['ggeur']['officehome_manifest_path'] = path_str(
                manifest_path)
        path = cfg_dir / f'client_{client_id}.yaml'
        write_yaml(path, client_cfg)
        client_cfgs.append(path_str(path))

    manifest = {
        'case': case_name,
        'dataset': dataset,
        'model': model,
        'method': method,
        'head_only': args.head_only,
        'server_host': args.server_host,
        'server_bind_host': args.server_bind_host,
        'client_host': args.client_host,
        'client_hosts': client_hosts or [args.client_host],
        'client_bind_host': args.client_bind_host,
        'server_port': args.server_port,
        'client_port_base': args.client_port_base,
        'client_bind_port_base': args.client_bind_port_base,
        'client_advertise_port_base': args.client_advertise_port_base,
        'clients': args.clients,
        'officehome_manifest_base': path_str(manifest_base)
        if manifest_base is not None else '',
        'feature_cache_dir': args.feature_cache_dir,
        'headonly_cache_version': args.headonly_cache_version,
        'server_config': path_str(cfg_dir / 'server.yaml'),
        'client_configs': client_cfgs,
        'log_dir': path_str(case_dir / 'logs'),
        'pid_dir': path_str(case_dir / 'pids'),
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
    }
    (case_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', default='')
    parser.add_argument('--run-id', default=time.strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--dataset', choices=list(DATASETS) + ['all'], default='officehome')
    parser.add_argument('--model', choices=list(MODELS) + ['all'], default='vit')
    parser.add_argument('--method', choices=METHODS + ['all'], default='fedavg')
    parser.add_argument('--head-only', action='store_true')
    parser.add_argument('--allow-promptfl-in-head-only', action='store_true')
    parser.add_argument('--server-host', required=True)
    parser.add_argument('--server-bind-host', default='0.0.0.0')
    parser.add_argument('--client-host', required=True)
    parser.add_argument('--client-hosts', default='')
    parser.add_argument('--client-host-assignment',
                        choices=['block', 'round-robin'],
                        default='block')
    parser.add_argument('--client-bind-host', default='0.0.0.0')
    parser.add_argument('--client-bind-hosts', default='')
    parser.add_argument('--client-advertise-port-base', type=int, default=0)
    parser.add_argument('--client-advertise-ports', default='')
    parser.add_argument('--server-port', type=int, default=55051)
    parser.add_argument('--client-port-base', type=int, default=56000)
    parser.add_argument('--client-bind-port-base', type=int, default=0)
    parser.add_argument('--client-bind-ports', default='')
    parser.add_argument('--clients', type=int, default=4)
    parser.add_argument('--sample-clients', type=int, default=0)
    parser.add_argument('--sub-servers', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--promptfl-rounds', type=int, default=1)
    parser.add_argument('--stage-timeout', type=int, default=3600)
    parser.add_argument('--join-timeout-seconds', type=int, default=600)
    parser.add_argument('--grpc-max-send-message-length', type=int,
                        default=512 * 1024 * 1024)
    parser.add_argument('--grpc-max-receive-message-length', type=int,
                        default=512 * 1024 * 1024)
    parser.add_argument('--grpc-compression', default='nocompression',
                        choices=['nocompression', 'deflate', 'gzip'])
    parser.add_argument('--use-gpu', action='store_true', default=True)
    parser.add_argument('--no-use-gpu', dest='use_gpu', action='store_false')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--extract-batch-size', type=int, default=64)
    parser.add_argument('--eval-freq', type=int, default=1)
    parser.add_argument('--data-root', default='')
    parser.add_argument('--splits', nargs=3, type=float)
    parser.add_argument('--num-generated-per-sample', type=int, default=2)
    parser.add_argument('--num-generated-per-prototype', type=int, default=2)
    parser.add_argument('--target-size-per-class', type=int, default=2)
    parser.add_argument('--use-lds', action='store_true', default=True)
    parser.add_argument('--no-use-lds', dest='use_lds', action='store_false')
    parser.add_argument('--lds-alpha', type=float, default=0.1)
    parser.add_argument('--feature-cache-dir',
                        default='exp/headonly_system/cache/officehome_headonly')
    parser.add_argument('--headonly-cache-version', default='fcache_v1')
    parser.add_argument('--feature-cache-version', default='',
                        help='Backward-compatible alias for '
                             '--headonly-cache-version')
    parser.add_argument('--headonly-skip-round0-if-augmented-cache-exists',
                        action='store_true', default=True)
    parser.add_argument('--no-headonly-skip-round0-if-augmented-cache-exists',
                        dest='headonly_skip_round0_if_augmented_cache_exists',
                        action='store_false')
    parser.add_argument('--officehome-manifest-base', default='')
    parser.add_argument('--officehome-domains',
                        default='Art,Clipart,Product,Real_World')
    parser.add_argument('--min-statistics-clients', type=int, default=0)
    parser.add_argument('--min-augmentation-clients', type=int, default=0)
    parser.add_argument('--min-train-updates', type=int, default=0)
    parser.add_argument('--headonly-eval-mode',
                        choices=['server', 'client'],
                        default='server')
    args = parser.parse_args()
    if args.feature_cache_version:
        args.headonly_cache_version = args.feature_cache_version
    if args.client_bind_port_base <= 0:
        args.client_bind_port_base = args.client_port_base
    if args.client_advertise_port_base <= 0:
        args.client_advertise_port_base = args.client_port_base
    return args


def selected(value, keys):
    return keys if value == 'all' else [value]


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    out_root = Path(args.output_root) if args.output_root else script_dir / 'runs' / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    manifests = []
    for dataset in selected(args.dataset, list(DATASETS)):
        for model in selected(args.model, list(MODELS)):
            for method in selected(args.method, METHODS):
                if args.head_only and method == 'promptfl' and not args.allow_promptfl_in_head_only:
                    continue
                port_offset = len(manifests) * 20
                case_args = copy.copy(args)
                case_args.server_port = args.server_port + port_offset
                case_args.client_port_base = args.client_port_base + port_offset
                manifests.append(generate_case(case_args, dataset, model, method, out_root))

    matrix = {
        'run_id': args.run_id,
        'output_root': path_str(out_root),
        'case_count': len(manifests),
        'cases': manifests,
    }
    (out_root / 'matrix_manifest.json').write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(matrix, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
