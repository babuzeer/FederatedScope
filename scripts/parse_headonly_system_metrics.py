"""Parse GGEUR HeadOnly system logs into a compact metrics JSON."""

import argparse
import json
import re
from pathlib import Path


ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
ROUND_RE = re.compile(
    r"Round (?P<round>\d+) aggregation complete, total samples: "
    r"(?P<samples>\d+), round time: (?P<time>[0-9.]+)s, "
    r"train_qps=(?P<qps>[0-9.]+) samples/s, "
    r"valid_updates=(?P<valid>\d+)/(?P<clients>\d+)"
)
TEST_RE = re.compile(
    r"Round (?P<round>\d+) MLP Test Accuracy - (?P<body>.+)$"
)
FINAL_RE = re.compile(r"\s+(?P<name>[^:]+): final=(?P<final>[0-9.]+), best=(?P<best>[0-9.]+)")
FEATURE_RE = re.compile(
    r"Client (?P<client>\d+): Feature extraction QPS=(?P<qps>[0-9.]+) img/s "
    r"\((?P<samples>\d+) samples in (?P<time>[0-9.]+)s\)"
)
STATS_PAYLOAD_RE = re.compile(
    r"Client (?P<client>\d+): Local statistics payload bytes=(?P<bytes>\d+) "
    r"\(classes=(?P<classes>\d+), embedding_dim=(?P<dim>\d+)\)"
)
AUG_RE = re.compile(
    r"Client (?P<client>\d+): Augmentation timing - samples=(?P<samples>\d+), "
    r"time=(?P<time>[0-9.]+)s, qps=(?P<qps>[0-9.]+) samples/s"
)
TRAIN_RE = re.compile(
    r"Client (?P<client>\d+): Train loss=(?P<loss>[0-9.]+).*accuracy=(?P<acc>[0-9.]+)"
)


def parse_test_body(body):
    body = ANSI_RE.sub('', body)
    result = {}
    for part in body.split(','):
        if ':' not in part:
            continue
        key, value = part.rsplit(':', 1)
        key = key.strip()
        try:
            result[key] = float(value.strip())
        except ValueError:
            continue
    return result


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def parse_server(path):
    rounds = []
    tests = []
    finals = {}
    if not path.exists():
        return rounds, tests, finals
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = ANSI_RE.sub('', line)
        match = ROUND_RE.search(line)
        if match:
            rounds.append({
                'round': int(match.group('round')),
                'total_samples': int(match.group('samples')),
                'round_time_sec': float(match.group('time')),
                'train_qps_samples_per_sec': float(match.group('qps')),
                'valid_updates': int(match.group('valid')),
                'expected_clients': int(match.group('clients')),
            })
            continue
        match = TEST_RE.search(line)
        if match:
            tests.append({
                'round': int(match.group('round')),
                'accuracy': parse_test_body(match.group('body')),
            })
            continue
        match = FINAL_RE.search(line)
        if match:
            finals[match.group('name').strip()] = {
                'final': float(match.group('final')),
                'best': float(match.group('best')),
            }
    return rounds, tests, finals


def parse_clients(log_dir):
    feature = []
    stats_payloads = []
    augmentation = []
    train = []
    paths = sorted(log_dir.glob('client*.log'))
    server_log = log_dir / 'server.log'
    if server_log.exists():
        paths.append(server_log)
    for path in paths:
        for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
            line = ANSI_RE.sub('', line)
            match = FEATURE_RE.search(line)
            if match:
                feature.append({
                    'client': int(match.group('client')),
                    'samples': int(match.group('samples')),
                    'time_sec': float(match.group('time')),
                    'qps_img_per_sec': float(match.group('qps')),
                })
                continue
            match = STATS_PAYLOAD_RE.search(line)
            if match:
                stats_payloads.append({
                    'client': int(match.group('client')),
                    'payload_bytes': int(match.group('bytes')),
                    'classes': int(match.group('classes')),
                    'embedding_dim': int(match.group('dim')),
                })
                continue
            match = AUG_RE.search(line)
            if match:
                augmentation.append({
                    'client': int(match.group('client')),
                    'samples': int(match.group('samples')),
                    'time_sec': float(match.group('time')),
                    'qps_samples_per_sec': float(match.group('qps')),
                })
                continue
            match = TRAIN_RE.search(line)
            if match:
                train.append({
                    'client': int(match.group('client')),
                    'loss': float(match.group('loss')),
                    'accuracy': float(match.group('acc')),
                })
    return feature, stats_payloads, augmentation, train


def build_summary(log_dir):
    server_rounds, test_acc, final_acc = parse_server(log_dir / 'server.log')
    feature, stats_payloads, augmentation, train = parse_clients(log_dir)
    qps_values = [item['train_qps_samples_per_sec'] for item in server_rounds]
    stable_qps_values = qps_values[1:] if len(qps_values) > 1 else qps_values
    total_samples = sum(item['total_samples'] for item in server_rounds)
    total_round_time = sum(item['round_time_sec'] for item in server_rounds)
    latest_test = test_acc[-1]['accuracy'] if test_acc else {}
    best_average = max(
        (item['accuracy'].get('average', 0.0) for item in test_acc),
        default=0.0,
    )
    return {
        'log_dir': str(log_dir),
        'round_count': len(server_rounds),
        'total_train_samples': total_samples,
        'total_round_time_sec': total_round_time,
        'overall_train_qps_samples_per_sec': (
            total_samples / total_round_time if total_round_time > 0 else 0.0
        ),
        'avg_train_qps_samples_per_sec': mean(qps_values),
        'stable_avg_train_qps_samples_per_sec': mean(stable_qps_values),
        'min_train_qps_samples_per_sec': min(qps_values) if qps_values else 0.0,
        'max_train_qps_samples_per_sec': max(qps_values) if qps_values else 0.0,
        'final_test_accuracy': latest_test,
        'best_average_test_accuracy': best_average,
        'final_accuracy_summary': final_acc,
        'rounds': server_rounds,
        'test_accuracy_by_round': test_acc,
        'round0': {
            'feature_extract_total_samples': sum(item['samples'] for item in feature),
            'feature_extract_avg_qps_img_per_sec': mean(
                item['qps_img_per_sec'] for item in feature),
            'statistics_total_payload_bytes': sum(
                item['payload_bytes'] for item in stats_payloads),
            'statistics_avg_payload_bytes': mean(
                item['payload_bytes'] for item in stats_payloads),
            'augmentation_total_samples': sum(item['samples'] for item in augmentation),
            'augmentation_avg_qps_samples_per_sec': mean(
                item['qps_samples_per_sec'] for item in augmentation),
            'feature_extract_clients': feature,
            'statistics_payloads': stats_payloads,
            'augmentation_clients': augmentation,
        },
        'client_train_log_points': train,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir')
    parser.add_argument('--output', default='')
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    summary = build_summary(log_dir)
    output = Path(args.output) if args.output else log_dir / 'metrics_summary.json'
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                      encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
