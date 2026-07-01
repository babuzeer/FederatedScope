#!/usr/bin/env python3
"""Summarize per-subserver parameter-read QPS events for one case."""

import argparse
import json
from pathlib import Path


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def fmt(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def summarize_file(path):
    events = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("event") == "parameter_read_qps":
                events.append(item)

    valid = [
        item for item in events
        if float(item.get("subserver_parameter_read_qps") or 0.0) > 0.0
        and int(item.get("responses") or 0) > 0
    ]
    qps_values = [
        float(item.get("subserver_parameter_read_qps") or 0.0)
        for item in valid
    ]
    last = valid[-1] if valid else {}
    return {
        "subserver": path.stem.replace("_events", ""),
        "event_file": str(path),
        "rounds_total": len(events),
        "rounds_valid": len(valid),
        "responses_total": sum(int(item.get("responses") or 0) for item in valid),
        "avg_qps_nonzero": (sum(qps_values) / len(qps_values)) if qps_values else 0.0,
        "max_qps": max(qps_values) if qps_values else 0.0,
        "last_valid_round": last.get("round", ""),
        "last_valid_qps": float(last.get("subserver_parameter_read_qps") or 0.0) if last else 0.0,
        "last_valid_responses": int(last.get("responses") or 0) if last else 0,
        "last_valid_denominator_sec": float(last.get("qps_denominator_response_window_sec") or 0.0) if last else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "case_dir",
        help="Case directory under scripts/distributed_scripts/headonly_hierarchical_qps/runs",
    )
    parser.add_argument(
        "--log-dir",
        default="",
        help="Override root/subserver log directory; by default read root_log_dir from manifest.json",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of TSV")
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        manifest = read_json(case_dir / "manifest.json")
        log_dir = Path(manifest["root_log_dir"])

    rows = [summarize_file(path) for path in sorted(log_dir.glob("subserver_*_events.jsonl"))]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    fields = [
        "subserver",
        "rounds_total",
        "rounds_valid",
        "responses_total",
        "avg_qps_nonzero",
        "max_qps",
        "last_valid_round",
        "last_valid_qps",
        "last_valid_responses",
        "last_valid_denominator_sec",
    ]
    print("\t".join(fields))
    for row in rows:
        print("\t".join(fmt(row.get(field, "")) for field in fields))


if __name__ == "__main__":
    main()
