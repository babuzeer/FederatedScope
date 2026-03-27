"""
Plot CNN baseline curves (FedAvg/FedProx/FedProto/FedOpt) using the latest
experiment outputs, and keep an Our_method curve for reference.

Usage:
  python plot_accuracy_comparison.py

Outputs:
  exp/ggeur_baselines/officehome_lds/plots/cnn_baselines_with_our_method_average.png
  exp/ggeur_baselines/officehome_lds/plots/cnn_baselines_with_our_method_domains.png
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent

# Latest CNN baselines live here:
#   exp/ggeur_baselines/officehome_lds/cnn/<method>/sub_exp_YYYYMMDDHHMMSS/exp_print.log
CNN_BASELINES_DIR = REPO_ROOT / "exp" / "ggeur_baselines" / "officehome_lds" / "cnn"

# Keep an existing Our_method curve (update if you have a newer log path)
GGEUR_LOG = (
    REPO_ROOT
    / "exp"
    / "GGEUR_CNN_convnext"
    / "ggeur_alpha0.1_clientsnum60_exp_print.log"
)

PLOTS_DIR = REPO_ROOT / "exp" / "ggeur_baselines" / "officehome_lds" / "plots"

SHOW_PLOTS = False

OUR_METHOD_LABEL = "Our_method"

LOG_PATTERN = re.compile(
    r"Server: Round (\d+) MLP Test Accuracy - "
    r"Art: ([\d.]+), Clipart: ([\d.]+), Product: ([\d.]+), Real_World: ([\d.]+), average: ([\d.]+)"
)


def _find_latest_sub_exp(method_dir: Path) -> Path:
    sub_exps = [p for p in method_dir.iterdir() if p.is_dir() and p.name.startswith("sub_exp_")]
    if not sub_exps:
        raise FileNotFoundError(f"No sub_exp_* found under {method_dir}")
    return max(sub_exps, key=lambda p: p.name)


def _resolve_latest_baseline_log(method: str) -> Path:
    method_dir = CNN_BASELINES_DIR / method
    if not method_dir.exists():
        raise FileNotFoundError(f"Missing method dir: {method_dir}")
    sub_exp_dir = _find_latest_sub_exp(method_dir)
    log_path = sub_exp_dir / "exp_print.log"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing exp_print.log: {log_path}")
    return log_path


def parse_log_file(log_path: Path) -> Dict[str, List[float]]:
    data: Dict[str, List[float]] = {
        "rounds": [],
        "average": [],
        "Art": [],
        "Clipart": [],
        "Product": [],
        "Real_World": [],
    }

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if not m:
                continue
            data["rounds"].append(int(m.group(1)))
            data["Art"].append(float(m.group(2)))
            data["Clipart"].append(float(m.group(3)))
            data["Product"].append(float(m.group(4)))
            data["Real_World"].append(float(m.group(5)))
            data["average"].append(float(m.group(6)))

    return data


def _maybe_show() -> None:
    if SHOW_PLOTS:
        plt.show()


def plot_average_comparison(all_data: Dict[str, Dict[str, List[float]]], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))

    styles = {
        "FedAvg": ("#607D8B", "o"),
        "FedProx": ("#8BC34A", "s"),
        "FedProto": ("#FF9800", "^"),
        "FedOpt": ("#9C27B0", "D"),
        OUR_METHOD_LABEL: ("#F44336", "*"),
    }

    all_rounds: List[int] = []
    for name, d in all_data.items():
        if not d["rounds"]:
            continue
        all_rounds.extend(d["rounds"])
        color, marker = styles.get(name, ("#333333", "x"))
        y = np.array(d["average"]) * 100
        plt.plot(
            d["rounds"],
            y,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=3,
            alpha=0.9,
            label=f"{name} (Final: {d['average'][-1]*100:.2f}%)",
        )

    max_round = max(all_rounds) if all_rounds else 100
    plt.xlim([0, max_round + 1])
    plt.ylim([0, 80])
    plt.xlabel("Communication Round", fontsize=12)
    plt.ylabel("Average Test Accuracy (%)", fontsize=12)
    plt.title(f"CNN Baselines vs {OUR_METHOD_LABEL} (Average Test Accuracy)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved: {output_path}")
    _maybe_show()
    plt.close()


def plot_all_domains(all_data: Dict[str, Dict[str, List[float]]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"CNN Baselines vs {OUR_METHOD_LABEL} (Per-domain Test Accuracy)", fontsize=14)

    domains = ["average", "Art", "Clipart", "Product", "Real_World"]
    titles = ["Average", "Art", "Clipart", "Product", "Real_World"]
    colors = {
        "FedAvg": "#607D8B",
        "FedProx": "#8BC34A",
        "FedProto": "#FF9800",
        "FedOpt": "#9C27B0",
        OUR_METHOD_LABEL: "#F44336",
    }

    for idx, (domain, title) in enumerate(zip(domains, titles)):
        ax = axes[idx // 3, idx % 3]
        for name, d in all_data.items():
            if d["rounds"] and d.get(domain):
                ax.plot(
                    d["rounds"],
                    np.array(d[domain]) * 100,
                    "-",
                    label=name,
                    linewidth=2,
                    alpha=0.9,
                    color=colors.get(name, None),
                )
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 90])
        ax.legend(fontsize=9, loc="lower right")

    axes[1, 2].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved: {output_path}")
    _maybe_show()
    plt.close()


def main() -> int:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    logs: Dict[str, Path] = {
        "FedAvg": _resolve_latest_baseline_log("fedavg"),
        "FedProx": _resolve_latest_baseline_log("fedprox"),
        "FedProto": _resolve_latest_baseline_log("fedproto"),
        "FedOpt": _resolve_latest_baseline_log("fedopt"),
        OUR_METHOD_LABEL: GGEUR_LOG,
    }

    print("Using logs:")
    for name, path in logs.items():
        print(f"  - {name}: {path}")

    all_data = {name: parse_log_file(path) for name, path in logs.items()}

    avg_plot = PLOTS_DIR / "cnn_baselines_with_our_method_average.png"
    dom_plot = PLOTS_DIR / "cnn_baselines_with_our_method_domains.png"

    plot_average_comparison(all_data, avg_plot)
    plot_all_domains(all_data, dom_plot)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
