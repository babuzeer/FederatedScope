#!/usr/bin/env python3
"""Summarize HeadOnly QPS benchmark result directories."""

import argparse
import json
from pathlib import Path


FIELDS = [
    "run_id",
    "ack_mode",
    "subservers",
    "clients_per_subserver",
    "expected_clients",
    "ack_clients",
    "first_chunk_ack_clients",
    "success_ratio",
    "dispatch_start_window_sec",
    "dispatch_start_qps",
    "latency_p50_sec",
    "latency_p95_sec",
    "latency_p99_sec",
    "subserver_qps_min",
    "subserver_qps_avg",
    "subserver_qps_max",
    "payload_bytes_per_download",
    "first_chunk_bytes",
    "total_network_bytes",
    "total_model_payload_bytes",
    "global_ack_qps",
    "global_ack_window_sec",
]


def read_result(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    global_item = data.get("global", {})
    run_id = path.parent.name
    row = {
        "run_id": run_id,
        "ack_mode": global_item.get("ack_mode", data.get("ack_mode", "")),
        "subservers": data.get("subservers", ""),
        "clients_per_subserver": data.get("clients_per_subserver", ""),
    }
    for key in FIELDS:
        if key not in row:
            row[key] = global_item.get(key, "")
    return row


def fmt(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def numeric(row, key):
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        help="QPS output root, e.g. exp/headonly_download_qps",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Only include run dirs starting with this prefix",
    )
    parser.add_argument("--output", default="", help="Optional TSV output")
    args = parser.parse_args()

    root = Path(args.root)
    results = []
    for path in sorted(root.glob("*/download_qps_summary.json")):
        if args.prefix and not path.parent.name.startswith(args.prefix):
            continue
        results.append(read_result(path))

    lines = ["\t".join(FIELDS)]
    for row in results:
        lines.append("\t".join(fmt(row.get(field, "")) for field in FIELDS))

    successful = [
        row for row in results
        if numeric(row, "success_ratio") >= 1.0
        and numeric(row, "first_chunk_ack_clients") > 0
    ]
    if successful:
        best = max(successful,
                   key=lambda row: numeric(row, "dispatch_start_qps"))
        lines.append("")
        lines.append("# best_successful_dispatch_start_run")
        lines.append("\t".join(FIELDS))
        lines.append("\t".join(fmt(best.get(field, ""))
                               for field in FIELDS))

    text = "\n".join(lines) + "\n"
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
