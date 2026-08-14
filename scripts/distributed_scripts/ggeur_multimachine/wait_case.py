#!/usr/bin/env python3
import sys
import time
from pathlib import Path


def alive(pid):
    try:
        Path(f'/proc/{pid}').stat()
        return True
    except FileNotFoundError:
        return False


def main():
    if len(sys.argv) != 3:
        print('Usage: wait_case.py <case_dir> <timeout_sec>')
        raise SystemExit(1)
    case_dir = Path(sys.argv[1])
    timeout = float(sys.argv[2])
    pid_dir = case_dir / 'pids'
    start = time.time()
    while time.time() - start <= timeout:
        pid_files = list(pid_dir.glob('*.pid')) if pid_dir.exists() else []
        if pid_files and not any(alive(int(p.read_text().strip())) for p in pid_files):
            print('all tracked processes exited')
            return
        time.sleep(5)
    print(f'timed out waiting for processes in {pid_dir}')
    raise SystemExit(1)


if __name__ == '__main__':
    main()
