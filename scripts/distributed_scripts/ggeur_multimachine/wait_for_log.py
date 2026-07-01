#!/usr/bin/env python3
import sys
import time
from pathlib import Path


def main():
    if len(sys.argv) != 4:
        print('Usage: wait_for_log.py <log_path> <pattern> <timeout_sec>')
        raise SystemExit(1)
    path = Path(sys.argv[1])
    pattern = sys.argv[2]
    timeout = float(sys.argv[3])
    start = time.time()
    while time.time() - start <= timeout:
        if path.exists():
            text = path.read_text(encoding='utf-8', errors='ignore')
            if pattern in text:
                print(f'found pattern: {pattern}')
                return
        time.sleep(2)
    print(f'timed out waiting for {pattern} in {path}')
    raise SystemExit(1)


if __name__ == '__main__':
    main()
