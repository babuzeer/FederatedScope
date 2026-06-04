"""
Run a real single-machine multi-process GGEUR head-only training job.

This is not the synthetic load-test path. It launches the existing
FederatedScope server/client entrypoint with real dataset configs, so round 0
uses the existing GGEUR feature extraction, statistics aggregation and feature
augmentation code. The script only manages local processes and log parsing.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


def deep_set(cfg, dotted_key, value):
    node = cfg
    parts = dotted_key.split('.')
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def dump_yaml(path, cfg):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def prepare_configs(args, run_dir):
    task_dir = Path(args.task_dir)
    cfg_dir = run_dir / 'configs'
    cfg_dir.mkdir(parents=True, exist_ok=True)

    common_updates = {
        'federate.total_round_num': args.rounds,
        'federate.client_num': args.clients,
        'federate.sample_client_num': args.clients,
        'distribute.server_host': args.host,
        'distribute.server_port': args.port,
        'ggeur.head_only_mode': True,
        'ggeur.freeze_backbone': True,
        'ggeur.use_feature_cache': True,
        'ggeur.unload_extractor_after_cache': True,
        'ggeur.use_cnn_distillation': False,
        'ggeur.use_feature_alignment': False,
        'ggeur.use_separated_training': False,
        'ggeur.use_end_to_end_finetune': False,
        'ggeur.use_promptfl': False,
        'ggeur.num_generated_per_sample': args.num_generated_per_sample,
        'ggeur.num_generated_per_prototype': args.num_generated_per_prototype,
        'ggeur.target_size_per_class': args.target_size_per_class,
        'ggeur_headonly.use': True,
        'ggeur_headonly.client_total': args.clients,
        'ggeur_headonly.sample_clients_per_round': args.clients,
        'ggeur_headonly.num_sub_servers': args.sub_servers,
    }

    server_cfg = load_yaml(task_dir / 'officehome_vit_ggeur_fedavg_server.yaml')
    for key, value in common_updates.items():
        deep_set(server_cfg, key, value)
    deep_set(server_cfg, 'distribute.role', 'server')
    deep_set(server_cfg, 'outdir', str(run_dir / 'server_out'))
    deep_set(server_cfg, 'expname', 'headonly_real_server')
    server_cfg_path = cfg_dir / 'server.yaml'
    dump_yaml(server_cfg_path, server_cfg)

    client_cfg_paths = []
    for client_id in range(1, args.clients + 1):
        src = task_dir / f'officehome_vit_ggeur_fedavg_client_{client_id}.yaml'
        if not src.exists():
            raise FileNotFoundError(
                f'Missing client config {src}; this runner currently expects '
                'the existing OfficeHome 4-client sample configs.'
            )
        client_cfg = load_yaml(src)
        for key, value in common_updates.items():
            deep_set(client_cfg, key, value)
        deep_set(client_cfg, 'distribute.role', 'client')
        deep_set(client_cfg, 'distribute.client_host', args.host)
        deep_set(client_cfg, 'distribute.client_port', args.port + client_id)
        deep_set(client_cfg, 'distribute.data_idx', client_id)
        deep_set(client_cfg, 'outdir', str(run_dir / f'client_{client_id}_out'))
        deep_set(client_cfg, 'expname', f'headonly_real_client_{client_id}')
        path = cfg_dir / f'client_{client_id}.yaml'
        dump_yaml(path, client_cfg)
        client_cfg_paths.append(path)

    return server_cfg_path, client_cfg_paths


def start_process(name, python_bin, root_dir, cfg_path, log_path):
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{root_dir}:{env.get('PYTHONPATH', '')}"
    log_f = open(log_path, 'w', encoding='utf-8')
    proc = subprocess.Popen(
        [python_bin, str(Path(root_dir) / 'federatedscope' / 'main.py'),
         '--cfg', str(cfg_path)],
        cwd=root_dir,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )
    return {'name': name, 'proc': proc, 'log_file': log_f, 'log_path': log_path}


def terminate_processes(processes):
    for item in processes:
        proc = item['proc']
        if proc.poll() is None:
            proc.terminate()
    time.sleep(3)
    for item in processes:
        proc = item['proc']
        if proc.poll() is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass


def close_logs(processes):
    for item in processes:
        item['log_file'].close()


def wait_processes(processes, timeout):
    start = time.perf_counter()
    while True:
        if all(item['proc'].poll() is not None for item in processes):
            return True, time.perf_counter() - start
        if time.perf_counter() - start > timeout:
            return False, time.perf_counter() - start
        time.sleep(5)


def grep_lines(path, patterns):
    if not path.exists():
        return []
    hits = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if any(pattern in line for pattern in patterns):
                hits.append(line.strip())
    return hits


def parse_logs(run_dir, processes, elapsed):
    server_log = run_dir / 'server.log'
    client_logs = sorted(run_dir.glob('client_*.log'))

    round_metrics = []
    round_re = re.compile(
        r'Round\s+(\d+).*?aggregation complete.*?total samples:\s*(\d+)'
        r'.*?round time:\s*([0-9.]+)s',
        re.IGNORECASE,
    )
    for line in grep_lines(server_log, ['aggregation complete']):
        match = round_re.search(line)
        if not match:
            continue
        round_id = int(match.group(1))
        samples = int(match.group(2))
        seconds = float(match.group(3))
        round_metrics.append({
            'round': round_id,
            'total_samples': samples,
            'round_time_sec': seconds,
            'train_qps_samples_per_sec': samples / seconds
            if seconds > 0 else 0.0,
            'line': line,
        })

    feature_qps = []
    feature_re = re.compile(r'Feature extraction QPS[=:]\s*([0-9.]+)')
    augmented_samples = []
    aug_re = re.compile(r'Augmented data.*?([0-9]+)\s+samples')
    for log_path in client_logs:
        for line in grep_lines(log_path, ['Feature extraction QPS', 'Augmented data']):
            qps_match = feature_re.search(line)
            if qps_match:
                feature_qps.append(float(qps_match.group(1)))
            aug_match = aug_re.search(line)
            if aug_match:
                augmented_samples.append(int(aug_match.group(1)))

    return_codes = {
        item['name']: item['proc'].returncode
        for item in processes
    }
    success_markers = grep_lines(server_log, ['Training finished', 'Server finished'])
    errors = []
    for log_path in [server_log] + client_logs:
        errors.extend(
            f'{log_path.name}: {line}'
            for line in grep_lines(log_path, ['Traceback', 'ERROR', 'Error'])
        )

    total_round_samples = sum(item['total_samples'] for item in round_metrics)
    total_round_time = sum(item['round_time_sec'] for item in round_metrics)

    return {
        'elapsed_wall_time_sec': elapsed,
        'return_codes': return_codes,
        'success_markers': success_markers[-10:],
        'rounds': round_metrics,
        'total_round_samples': total_round_samples,
        'sum_round_time_sec': total_round_time,
        'avg_train_qps_samples_per_sec': (
            sum(item['train_qps_samples_per_sec'] for item in round_metrics) /
            len(round_metrics)
        ) if round_metrics else 0.0,
        'overall_train_qps_by_round_time_samples_per_sec': (
            total_round_samples / total_round_time
            if total_round_time > 0 else 0.0
        ),
        'avg_feature_extraction_qps_img_per_sec': (
            sum(feature_qps) / len(feature_qps)
            if feature_qps else 0.0
        ),
        'feature_extraction_qps_values': feature_qps,
        'augmented_samples_by_client': augmented_samples,
        'errors_tail': errors[-30:],
    }


def write_text_report(path, summary):
    lines = [
        '# Head-only Real Local Training Report',
        '',
        f"run_id: {summary['run_id']}",
        f"root_dir: {summary['root_dir']}",
        f"status: {summary['status']}",
        f"elapsed_wall_time_sec: {summary['metrics']['elapsed_wall_time_sec']:.3f}",
        '',
        '## Metrics',
        '',
        f"avg_train_qps_samples_per_sec: {summary['metrics']['avg_train_qps_samples_per_sec']:.2f}",
        f"overall_train_qps_by_round_time_samples_per_sec: {summary['metrics']['overall_train_qps_by_round_time_samples_per_sec']:.2f}",
        f"avg_feature_extraction_qps_img_per_sec: {summary['metrics']['avg_feature_extraction_qps_img_per_sec']:.2f}",
        '',
        '## Rounds',
    ]
    for item in summary['metrics']['rounds']:
        lines.append(
            f"- round {item['round']}: samples={item['total_samples']}, "
            f"time={item['round_time_sec']:.3f}s, "
            f"qps={item['train_qps_samples_per_sec']:.2f}"
        )
    lines.extend(['', '## Logs', ''])
    for name, path_str in summary['logs'].items():
        lines.append(f"- {name}: {path_str}")
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        '--task-dir',
        default='scripts/distributed_scripts/ggeur_officehome_vit_fedavg',
    )
    parser.add_argument('--python-bin', default=sys.executable)
    parser.add_argument('--run-id', default=time.strftime('headonly_real_%Y%m%d_%H%M%S'))
    parser.add_argument('--log-root', default='exp/headonly_real_local')
    parser.add_argument('--clients', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=52051)
    parser.add_argument('--sub-servers', type=int, default=2)
    parser.add_argument('--server-start-wait', type=float, default=8.0)
    parser.add_argument('--client-start-gap', type=float, default=2.0)
    parser.add_argument('--timeout', type=float, default=3600.0)
    parser.add_argument('--num-generated-per-sample', type=int, default=2)
    parser.add_argument('--num-generated-per-prototype', type=int, default=2)
    parser.add_argument('--target-size-per-class', type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    args.task_dir = str((root_dir / args.task_dir).resolve())
    run_dir = (root_dir / args.log_root / args.run_id).resolve()
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    server_cfg, client_cfgs = prepare_configs(args, run_dir)
    processes = []
    start = time.perf_counter()
    status = 'unknown'
    try:
        processes.append(
            start_process(
                'server',
                args.python_bin,
                root_dir,
                server_cfg,
                run_dir / 'server.log',
            )
        )
        time.sleep(args.server_start_wait)
        for idx, cfg_path in enumerate(client_cfgs, 1):
            processes.append(
                start_process(
                    f'client_{idx}',
                    args.python_bin,
                    root_dir,
                    cfg_path,
                    run_dir / f'client_{idx}.log',
                )
            )
            time.sleep(args.client_start_gap)

        finished, elapsed = wait_processes(processes, args.timeout)
        if not finished:
            status = 'timeout'
            terminate_processes(processes)
        else:
            status = 'completed'
    except KeyboardInterrupt:
        status = 'interrupted'
        terminate_processes(processes)
        elapsed = time.perf_counter() - start
        raise
    except Exception:
        status = 'failed_to_launch'
        terminate_processes(processes)
        elapsed = time.perf_counter() - start
        raise
    finally:
        close_logs(processes)

    metrics = parse_logs(run_dir, processes, elapsed)
    if status == 'completed' and any(code not in (0, None)
                                     for code in metrics['return_codes'].values()):
        status = 'completed_with_errors'

    summary = {
        'run_id': args.run_id,
        'root_dir': str(root_dir),
        'run_dir': str(run_dir),
        'status': status,
        'configs': {
            'server': str(server_cfg),
            'clients': [str(p) for p in client_cfgs],
        },
        'logs': {
            'server': str(run_dir / 'server.log'),
            **{
                f'client_{idx}': str(run_dir / f'client_{idx}.log')
                for idx in range(1, len(client_cfgs) + 1)
            },
        },
        'metrics': metrics,
    }
    (run_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    write_text_report(run_dir / 'report.md', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
