"""
Plot accuracy comparison curves for DomainNet ViT experiments.

This script automatically finds the latest log for each method under:
  exp/domainnet_4domains/vit/<method>/

Supported methods by default:
  - ggeur_fedavg
  - fedavg
  - fedprox
  - fedproto
  - fedopt
  - moon

Usage:
  python scripts/plot_domainnet_vit_accuracy.py
  python scripts/plot_domainnet_vit_accuracy.py --show
  python scripts/plot_domainnet_vit_accuracy.py --methods fedavg fedprox fedopt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP_ROOT = REPO_ROOT / "exp" / "domainnet_4domains" / "vit"
DEFAULT_PLOT_DIR = DEFAULT_EXP_ROOT / "plots"

DEFAULT_METHODS = [
    "ggeur_fedavg",
    "fedavg",
    "fedprox",
    "fedproto",
    "fedopt",
    "moon",
]

METHOD_LABELS = {
    "ggeur_fedavg": "GGEUR+FedAvg",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedproto": "FedProto",
    "fedopt": "FedOpt",
    "moon": "MOON",
}

METHOD_STYLES = {
    "GGEUR+FedAvg": dict(color="#D32F2F", marker="o", linewidth=2.4),
    "FedAvg": dict(color="#455A64", marker="s", linewidth=2.0),
    "FedProx": dict(color="#388E3C", marker="^", linewidth=2.0),
    "FedProto": dict(color="#F57C00", marker="D", linewidth=2.0),
    "FedOpt": dict(color="#7B1FA2", marker="v", linewidth=2.0),
    "MOON": dict(color="#1976D2", marker="P", linewidth=2.0),
}

ROUND_PATTERN = re.compile(r"Server: Round (\d+) MLP Test Accuracy - (.+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot DomainNet ViT method accuracy comparison curves."
    )
    parser.add_argument(
        "--exp-root",
        type=Path,
        default=DEFAULT_EXP_ROOT,
        help="Experiment root directory, default: exp/domainnet_4domains/vit",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="Output directory for generated plots.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Method directories to include.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show matplotlib windows after saving plots.",
    )
    return parser.parse_args()


def resolve_latest_log(method_dir: Path) -> Path:
    if not method_dir.exists():
        raise FileNotFoundError(f"Method directory not found: {method_dir}")

    candidates = sorted(
        list(method_dir.rglob("exp_print.log")) +
        list(method_dir.rglob("*_exp_print.log")),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No log file found under: {method_dir}")
    return candidates[-1]


def parse_metric_blob(metric_blob: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for item in metric_blob.split(","):
        item = item.strip()
        if ": " not in item:
            continue
        key, value = item.split(": ", 1)
        try:
            metrics[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return metrics


def parse_log_file(log_path: Path) -> Dict[str, List[float]]:
    rounds: List[int] = []
    series: Dict[str, List[float]] = {}

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = ROUND_PATTERN.search(line)
            if not match:
                continue
            round_idx = int(match.group(1))
            metrics = parse_metric_blob(match.group(2))
            if not metrics:
                continue

            rounds.append(round_idx)
            for key, value in metrics.items():
                series.setdefault(key, []).append(value)

    result: Dict[str, List[float]] = {"rounds": rounds}
    result.update(series)
    return result


def collect_all_data(exp_root: Path, methods: List[str]) -> Dict[str, Dict[str, List[float]]]:
    all_data: Dict[str, Dict[str, List[float]]] = {}
    print("Using logs:")
    for method in methods:
        method_dir = exp_root / method
        log_path = resolve_latest_log(method_dir)
        label = METHOD_LABELS.get(method, method)
        print(f"  - {label}: {log_path}")
        data = parse_log_file(log_path)
        if not data.get("rounds"):
            raise ValueError(f"No per-round accuracy found in log: {log_path}")
        all_data[label] = data
    return all_data


def infer_domains(all_data: Dict[str, Dict[str, List[float]]]) -> List[str]:
    for data in all_data.values():
        keys = [k for k in data.keys() if k != "rounds"]
        if "average" in keys:
            keys.remove("average")
            return ["average"] + keys
        return keys
    return ["average"]


def plot_average(all_data: Dict[str, Dict[str, List[float]]], output_path: Path) -> None:
    plt.figure(figsize=(10.5, 6.2))

    max_round = 0
    for label, data in all_data.items():
        rounds = data["rounds"]
        avg = data.get("average")
        if not avg:
            continue

        style = METHOD_STYLES.get(label, dict(color="#333333", marker="x", linewidth=2.0))
        y = np.array(avg) * 100
        plt.plot(
            rounds,
            y,
            label=f"{label} (Final: {y[-1]:.2f}%)",
            markersize=4,
            alpha=0.9,
            markevery=max(1, len(rounds) // 12),
            **style,
        )
        max_round = max(max_round, max(rounds))

    plt.xlim(0, max_round + 1 if max_round else 100)
    plt.ylim(bottom=0)
    plt.xlabel("Communication Round", fontsize=12)
    plt.ylabel("Average Test Accuracy (%)", fontsize=12)
    plt.title("DomainNet 4 Domains - ViT Methods Accuracy Comparison", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_domains(all_data: Dict[str, Dict[str, List[float]]], output_path: Path) -> None:
    domains = infer_domains(all_data)
    if not domains:
        return

    metric_domains = domains[:]
    if "average" not in metric_domains:
        metric_domains = ["average"] + metric_domains

    panel_domains = metric_domains
    n_panels = len(panel_domains)
    n_cols = 3
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.1 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle("DomainNet 4 Domains - ViT Per-domain Accuracy Comparison", fontsize=14)

    for idx, domain in enumerate(panel_domains):
        ax = axes[idx]
        for label, data in all_data.items():
            if domain not in data:
                continue
            style = METHOD_STYLES.get(label, dict(color="#333333", marker="x", linewidth=2.0))
            ax.plot(
                data["rounds"],
                np.array(data[domain]) * 100,
                label=label,
                markersize=3.5,
                alpha=0.9,
                markevery=max(1, len(data["rounds"]) // 12),
                **style,
            )
        ax.set_title("Average" if domain == "average" else domain)
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, loc="lower right")

    for idx in range(n_panels, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def maybe_show(show: bool) -> None:
    if show:
        plt.show()


def main() -> int:
    args = parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    all_data = collect_all_data(args.exp_root, args.methods)

    avg_path = args.plot_dir / "domainnet_vit_methods_average.png"
    domains_path = args.plot_dir / "domainnet_vit_methods_domains.png"

    plot_average(all_data, avg_path)
    plot_domains(all_data, domains_path)
    maybe_show(args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
