#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path


ROUND_RE = re.compile(
    r'Round\s+(\d+).*?aggregation complete.*?total samples:\s*(\d+)'
    r'.*?round time:\s*([0-9.]+)s',
    re.IGNORECASE,
)


def grep(path, patterns):
    if not path.exists():
        return []
    hits = []
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        if any(p in line for p in patterns):
            hits.append(line)
    return hits


def main():
    if len(sys.argv) != 2:
        print('Usage: summarize_case.py <case_dir>')
        raise SystemExit(1)
    case_dir = Path(sys.argv[1])
    log_dir = case_dir / 'logs'
    server_log = log_dir / 'server.log'
    rounds = []
    for line in grep(server_log, ['aggregation complete']):
        match = ROUND_RE.search(line)
        if match:
            samples = int(match.group(2))
            seconds = float(match.group(3))
            rounds.append({
                'round': int(match.group(1)),
                'total_samples': samples,
                'round_time_sec': seconds,
                'train_qps_samples_per_sec': samples / seconds if seconds > 0 else 0.0,
            })
    total_samples = sum(r['total_samples'] for r in rounds)
    total_seconds = sum(r['round_time_sec'] for r in rounds)
    errors = []
    for log in log_dir.glob('*.log'):
        errors.extend(f'{log.name}: {line}' for line in grep(log, ['Traceback', 'ERROR', 'Error']))
    summary = {
        'case_dir': str(case_dir),
        'server_log': str(server_log),
        'status_markers': grep(server_log, ['Training finished'])[-5:],
        'rounds': rounds,
        'total_round_samples': total_samples,
        'sum_round_time_sec': total_seconds,
        'overall_train_qps_samples_per_sec': total_samples / total_seconds if total_seconds > 0 else 0.0,
        'errors_tail': errors[-20:],
    }
    out = case_dir / 'summary.json'
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
