#!/usr/bin/env python3
"""Parse GGEUR timing log events into detail and aggregate CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from statistics import mean
from typing import Dict, Iterable, List, Tuple


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^ \n\r]+)")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_value(value: str):
    if value in {"True", "False"}:
        return value == "True"
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_events(log_path: Path) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            line = strip_ansi(raw)
            if "GGEUR_TIMING_CLIENT " in line:
                payload = line.split("GGEUR_TIMING_CLIENT ", 1)[1]
                source = "client"
            elif "GGEUR_TIMING_SERVER " in line:
                payload = line.split("GGEUR_TIMING_SERVER ", 1)[1]
                source = "server"
            else:
                continue
            row: Dict[str, object] = {"source": source, "line_no": line_no}
            for key, value in KV_RE.findall(payload):
                row[key] = parse_value(value)
            events.append(row)
    return events


def fieldnames(rows: Iterable[Dict[str, object]]) -> List[str]:
    preferred = [
        "run_label",
        "source",
        "stage",
        "client",
        "line_no",
        "dataset",
        "extractor",
        "total_sec",
    ]
    keys = set()
    for row in rows:
        keys.update(row.keys())
    ordered = [key for key in preferred if key in keys]
    ordered.extend(sorted(keys - set(ordered)))
    return ordered


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def numeric_items(row: Dict[str, object]) -> Iterable[Tuple[str, float]]:
    for key, value in row.items():
        if key in {"line_no", "client"}:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            yield key, float(value)


def aggregate(events: List[Dict[str, object]], run_label: str) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, object], List[Dict[str, object]]] = {}
    for event in events:
        key = (event.get("source", ""), event.get("stage", ""))
        groups.setdefault(key, []).append(event)

    rows: List[Dict[str, object]] = []
    for (source, stage), group in sorted(groups.items()):
        numeric_keys = sorted({key for row in group for key, _ in numeric_items(row)})
        base = {
            "run_label": run_label,
            "source": source,
            "stage": stage,
            "events": len(group),
        }
        for key in numeric_keys:
            values = [
                float(row[key])
                for row in group
                if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
            ]
            if not values:
                continue
            base[f"{key}_sum"] = sum(values)
            base[f"{key}_mean"] = mean(values)
            base[f"{key}_max"] = max(values)
            base[f"{key}_min"] = min(values)
        rows.append(base)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-label", default="")
    args = parser.parse_args()

    log_path = Path(args.log)
    output_dir = Path(args.output_dir)
    run_label = args.run_label or log_path.parent.name

    events = parse_events(log_path)
    for event in events:
        event["run_label"] = run_label

    detail_path = output_dir / f"{run_label}_timing_detail.csv"
    aggregate_path = output_dir / f"{run_label}_timing_aggregate.csv"
    json_path = output_dir / f"{run_label}_timing.json"

    write_csv(detail_path, events)
    agg_rows = aggregate(events, run_label)
    write_csv(aggregate_path, agg_rows)
    json_path.write_text(
        json.dumps(
            {
                "run_label": run_label,
                "log": str(log_path),
                "event_count": len(events),
                "aggregate": agg_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"events={len(events)}")
    print(f"detail={detail_path}")
    print(f"aggregate={aggregate_path}")
    print(f"json={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
