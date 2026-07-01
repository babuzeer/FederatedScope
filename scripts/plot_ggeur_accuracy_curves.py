#!/usr/bin/env python3
"""Plot per-round average accuracy curves from GGEUR stdout logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ROUND_AVG_RE = re.compile(
    r"Round\s+(\d+)\s+MLP\s+Test\s+Accuracy\s+-.*?\baverage:\s*([0-9.]+)",
    re.IGNORECASE,
)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_curve(log_path: Path) -> List[Tuple[int, float]]:
    points: Dict[int, float] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = strip_ansi(raw)
            match = ROUND_AVG_RE.search(line)
            if match:
                points[int(match.group(1))] = float(match.group(2))
    return sorted(points.items())


def method_name(case_dir: Path) -> str:
    return case_dir.name.split("__")[-1]


def iter_logs(input_dir: Path, dataset: str, model: str,
              methods: Iterable[str], allow_missing: bool) -> List[Tuple[str, Path]]:
    pairs: List[Tuple[str, Path]] = []
    for method in methods:
        log_path = input_dir / f"{dataset}__{model}__{method}" / "stdout.log"
        if not log_path.exists():
            if allow_missing:
                print(f"missing: {method} {log_path}")
                continue
            raise FileNotFoundError(log_path)
        pairs.append((method, log_path))
    return pairs


def write_curve_csv(path: Path, points: List[Tuple[int, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "average_accuracy"])
        writer.writerows(points)


def plot_one(method: str, points: List[Tuple[int, float]], output_path: Path,
             dataset: str, model: str) -> None:
    import matplotlib.pyplot as plt

    rounds = [round_id for round_id, _ in points]
    values = [acc for _, acc in points]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    ax.plot(rounds, values, linewidth=2.0)
    ax.scatter(rounds[-1:], values[-1:], s=28)
    ax.set_title(f"{dataset} / {model} / {method} average accuracy")
    ax.set_xlabel("Round")
    ax.set_ylabel("Average accuracy")
    ax.set_xlim(min(rounds), max(rounds))
    ax.set_ylim(0, max(1.0, max(values) * 1.05))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.text(
        rounds[-1],
        values[-1],
        f" final={values[-1]:.4f}",
        ha="left",
        va="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_combined(curves: Dict[str, List[Tuple[int, float]]], output_path: Path,
                  dataset: str, model: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=160)
    for method, points in curves.items():
        rounds = [round_id for round_id, _ in points]
        values = [acc for _, acc in points]
        ax.plot(rounds, values, linewidth=2.0, label=method)
    ax.set_title(f"{dataset} / {model} average accuracy comparison")
    ax.set_xlabel("Round")
    ax.set_ylabel("Average accuracy")
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="officehome")
    parser.add_argument("--model", default="mixer")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ggeur", "fedavg", "fedprox", "fedproto", "fedopt", "moon"],
    )
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curves: Dict[str, List[Tuple[int, float]]] = {}
    for method, log_path in iter_logs(input_dir, args.dataset, args.model,
                                      args.methods, args.allow_missing):
        points = parse_curve(log_path)
        if not points:
            if args.allow_missing:
                print(f"no_points: {method} {log_path}")
                continue
            raise RuntimeError(f"no round accuracy points found in {log_path}")
        curves[method] = points
        write_curve_csv(output_dir / f"{args.dataset}_{args.model}_{method}_curve.csv", points)
        plot_one(
            method,
            points,
            output_dir / f"{args.dataset}_{args.model}_{method}_accuracy.png",
            args.dataset,
            args.model,
        )
        print(f"{method}: points={len(points)}, final={points[-1][1]:.4f}")

    if args.combined and curves:
        plot_combined(
            curves,
            output_dir / f"{args.dataset}_{args.model}_all_methods_accuracy.png",
            args.dataset,
            args.model,
        )
    if not curves:
        raise RuntimeError("no curves were parsed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
