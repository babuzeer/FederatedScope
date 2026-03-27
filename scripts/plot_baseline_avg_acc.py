#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot average test accuracy curves for the baseline comparisons (plus Our_method).

This script:
1) Finds the latest run output under each experiment directory (supports
   FederatedScope's `sub_exp_YYYYMMDDHHMMSS/` folders).
2) Parses `exp_print.log` lines like:
   "Server: Round X MLP Test Accuracy - ... average: 0.1234"
3) Produces 2 figures: one for ViT branch and one for CNN branch.

Default expected directory structure:
  exp/ggeur_baselines/officehome_lds/vit/{fedavg,fedprox,fedopt,fedproto}/...
  exp/ggeur_baselines/officehome_lds/cnn/{fedavg,fedprox,fedopt,fedproto}/...
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple


BASELINE_METHODS = ["fedavg", "fedprox", "fedopt", "fedproto"]

OUR_METHOD_KEY = "ggeur"
OUR_METHOD_LABEL = "Our_method"

PLOT_STYLES = {
    # Use different linestyles + markers + markevery offsets so that even if
    # curves overlap, you can still see multiple methods.
    "fedavg": dict(linestyle="-", marker="o", markevery=(0, 10), linewidth=2.0, alpha=0.85),
    "fedprox": dict(linestyle="--", marker="s", markevery=(2, 10), linewidth=2.0, alpha=0.85),
    "fedopt": dict(linestyle="-.", marker="^", markevery=(4, 10), linewidth=2.0, alpha=0.85),
    "fedproto": dict(linestyle=":", marker="D", markevery=(6, 10), linewidth=2.0, alpha=0.85),
    OUR_METHOD_KEY: dict(linestyle="-", marker="*", markevery=(1, 12), linewidth=2.6, alpha=0.95),
}

_ACC_LINE_RE = re.compile(
    r"Server:\s*Round\s+(?P<round>\d+)\s+MLP Test Accuracy\s*-\s*(?P<rest>.*)$"
)

_SUB_EXP_RE = re.compile(r"^sub_exp_(\d{14})$")


def _sub_exp_key_from_log(log_path: Path) -> Tuple[int, float]:
    """
    Prefer selecting the "latest" run by sub_exp folder name timestamp
    (sub_exp_YYYYMMDDHHMMSS). Fall back to filesystem mtime.
    """
    parent = log_path.parent.name
    m = _SUB_EXP_RE.match(parent)
    ts = int(m.group(1)) if m else -1
    return ts, log_path.stat().st_mtime


def _iter_candidate_exp_logs(exp_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    direct = exp_dir / "exp_print.log"
    if direct.is_file():
        candidates.append(direct)
    candidates.extend(sorted(exp_dir.glob("sub_exp_*/exp_print.log")))
    return [p for p in candidates if p.is_file()]


def find_latest_exp_print_log(exp_dir: Path) -> Path:
    # FederatedScope behavior:
    # - normal case: logs are written directly under exp_dir
    # - when exp_dir already exists: logs are written under sub_exp_YYYY.../
    #
    # We prefer the latest sub_exp run if any exist; otherwise use the
    # (single) exp_print.log directly under exp_dir.
    sub_logs = [p for p in exp_dir.glob("sub_exp_*/exp_print.log") if p.is_file()]
    if sub_logs:
        # Choose by (timestamp in folder name, mtime) to be robust to copying
        return max(sub_logs, key=_sub_exp_key_from_log)

    direct = exp_dir / "exp_print.log"
    if direct.is_file():
        return direct

    candidates = _iter_candidate_exp_logs(exp_dir)
    if not candidates:  # fallback
        raise FileNotFoundError(f"No exp_print.log found under: {exp_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_kv_metrics(rest: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for part in rest.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def parse_avg_acc_curve(exp_print_log: Path) -> Tuple[List[int], List[float]]:
    curve: Dict[int, float] = {}
    with exp_print_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _ACC_LINE_RE.search(line)
            if not m:
                continue
            round_idx = int(m.group("round"))
            metrics = _parse_kv_metrics(m.group("rest"))

            avg = None
            for k in ("average", "avg", "mean"):
                if k in metrics:
                    avg = metrics[k]
                    break
                # sometimes keys are capitalized
                for mk, mv in metrics.items():
                    if mk.lower() == k:
                        avg = mv
                        break
                if avg is not None:
                    break

            if avg is None and metrics:
                values = [v for k, v in metrics.items() if k.lower() not in {"average", "avg", "mean"}]
                if values:
                    avg = sum(values) / len(values)

            if avg is not None:
                curve[round_idx] = avg

    if not curve:
        raise ValueError(
            f"Failed to parse any 'MLP Test Accuracy' lines from: {exp_print_log}"
        )

    rounds = sorted(curve.keys())
    accs = [curve[r] for r in rounds]
    return rounds, accs


def _normalize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}

    # Keep original keys (e.g. Art/Clipart/...), but normalize average-like keys
    # to a consistent "average".
    for k, v in metrics.items():
        out[k] = v
        if k.lower() in {"average", "avg", "mean"}:
            out["average"] = v

    if "average" not in out:
        values = [v for k, v in out.items() if k.lower() not in {"average", "avg", "mean"}]
        if values:
            out["average"] = sum(values) / len(values)

    return out


def parse_acc_curves(exp_print_log: Path) -> Tuple[List[int], Dict[str, List[float]]]:
    per_round: Dict[int, Dict[str, float]] = {}
    with exp_print_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _ACC_LINE_RE.search(line)
            if not m:
                continue
            round_idx = int(m.group("round"))
            metrics = _normalize_metrics(_parse_kv_metrics(m.group("rest")))
            if metrics:
                per_round[round_idx] = metrics

    if not per_round:
        raise ValueError(
            f"Failed to parse any 'MLP Test Accuracy' lines from: {exp_print_log}"
        )

    rounds = sorted(per_round.keys())
    keys = sorted(
        {k for metrics in per_round.values() for k in metrics.keys()},
        key=lambda k: (0 if k.lower() in {"average", "avg", "mean"} else 1, k.lower()),
    )

    nan = float("nan")
    curves = {k: [per_round[r].get(k, nan) for r in rounds] for k in keys}
    return rounds, curves


def plot_group(root: Path, group: str, out_dir: Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required. Install it with: pip install matplotlib"
        ) from e

    group_dir = root / group
    if not group_dir.is_dir():
        raise FileNotFoundError(f"Group directory not found: {group_dir}")

    series = {}
    chosen_logs = {}

    # 1) baseline methods from the directory tree
    for method in BASELINE_METHODS:
        exp_dir = group_dir / method
        if not exp_dir.is_dir():
            continue
        override_key = f"{group}/{method}"
        if override_key in plot_group.overrides:
            log_path = Path(plot_group.overrides[override_key])
            if not log_path.is_file():
                raise FileNotFoundError(
                    f"Override log does not exist for {override_key}: {log_path}"
                )
        else:
            # Special case requested: for ViT FedAvg, always prefer the top-level
            # exp_print.log if it exists (instead of the latest sub_exp_* run).
            if group == "vit" and method == "fedavg":
                preferred = exp_dir / "exp_print.log"
                log_path = preferred if preferred.is_file() else find_latest_exp_print_log(exp_dir)
            else:
                log_path = find_latest_exp_print_log(exp_dir)
        rounds, curves = parse_acc_curves(log_path)
        series[method] = (rounds, curves)
        chosen_logs[method] = log_path

    # 2) add Our_method curve (log outside baseline tree)
    ggeur_key = f"{group}/{OUR_METHOD_KEY}"
    ggeur_log = None
    if ggeur_key in plot_group.overrides:
        ggeur_log = Path(plot_group.overrides[ggeur_key])
    else:
        ggeur_log = plot_group.ggeur_logs.get(group)  # type: ignore[attr-defined]

    if ggeur_log is not None:
        ggeur_log = Path(ggeur_log)
        if ggeur_log.is_file():
            rounds, curves = parse_acc_curves(ggeur_log)
            series[OUR_METHOD_KEY] = (rounds, curves)
            chosen_logs[OUR_METHOD_KEY] = ggeur_log

    if not series:
        raise FileNotFoundError(
            f"No valid experiment logs found under: {group_dir}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{group}_avg_accuracy.png"
    avg_struct_out_path = out_dir / f"{group}_baselines_with_our_method_average.png"
    dom_out_path = out_dir / f"{group}_baselines_with_our_method_domains.png"

    plt.figure(figsize=(10, 5))
    for method in (BASELINE_METHODS + [OUR_METHOD_KEY]):
        if method not in series:
            continue
        rounds, curves = series[method]
        if "average" not in curves:
            continue
        style = PLOT_STYLES.get(method, {})
        label = OUR_METHOD_LABEL if method == OUR_METHOD_KEY else method

        y = [v * 100 for v in curves["average"]]
        final = None
        for v in reversed(y):
            if not math.isnan(v):
                final = v
                break
        if final is None:
            continue

        plt.plot(rounds, y, label=f"{label} (Final: {final:.2f}%)", **style)

    plt.title(f"{group.upper()} Baselines vs {OUR_METHOD_LABEL} (Average Test Accuracy)")
    plt.xlabel("Round")
    plt.ylabel("Average Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Method", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.savefig(avg_struct_out_path, dpi=200)
    plt.close()

    print(f"[{group}] saved: {out_path}")
    print(f"[{group}] saved: {avg_struct_out_path}")

    # Per-domain subplot figure (make ViT look like CNN plotting structure)
    preferred_domains = ["average", "Art", "Clipart", "Product", "Real_World"]
    available_domains: List[str] = []
    for domain in preferred_domains:
        if any((domain in curves) for _, curves in series.values()):
            available_domains.append(domain)

    if len(available_domains) > 1:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            f"{group.upper()} Baselines vs {OUR_METHOD_LABEL} (Per-domain Test Accuracy)",
            fontsize=14,
        )
        axes = axes.flatten()

        colors = {
            "fedavg": "#607D8B",
            "fedprox": "#8BC34A",
            "fedproto": "#FF9800",
            "fedopt": "#9C27B0",
            OUR_METHOD_KEY: "#F44336",
        }

        for idx, domain in enumerate(available_domains):
            ax = axes[idx]
            for method in (BASELINE_METHODS + [OUR_METHOD_KEY]):
                if method not in series:
                    continue
                rounds, curves = series[method]
                if domain not in curves:
                    continue
                label = OUR_METHOD_LABEL if method == OUR_METHOD_KEY else method
                ax.plot(
                    rounds,
                    [v * 100 for v in curves[domain]],
                    "-",
                    label=label,
                    linewidth=2,
                    alpha=0.9,
                    color=colors.get(method, None),
                )

            ax.set_title("Average" if domain == "average" else domain)
            ax.set_xlabel("Round")
            ax.set_ylabel("Accuracy (%)")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=9, loc="lower right")

        for j in range(len(available_domains), 6):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(dom_out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[{group}] saved: {dom_out_path}")
    for method in sorted(chosen_logs.keys()):
        label = OUR_METHOD_LABEL if method == OUR_METHOD_KEY else method
        print(f"  - {label}: {chosen_logs[method]}")

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="exp/ggeur_baselines/officehome_lds",
        help="Root directory containing vit/ and cnn/ subfolders",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="exp/ggeur_baselines/officehome_lds/plots",
        help="Output directory for the images",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["vit", "cnn"],
        default=["vit", "cnn"],
        help="Which groups to plot (default: vit cnn).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a log selection: '<group>/<method>=<path_to_exp_print.log>' "
             "(e.g. 'vit/fedavg=exp/ggeur_baselines/officehome_lds/vit/fedavg/exp_print.log' "
             "or 'vit/ggeur=exp/GGEUR_Clip/ggeur_alpha0.1_clientsnum60_exp_print.log' for Our_method). "
             "Can be specified multiple times.",
    )
    parser.add_argument(
        "--ggeur-vit-log",
        type=str,
        default="exp/GGEUR_Clip/ggeur_alpha0.1_clientsnum60_exp_print.log",
        help="Our_method (ViT/CLIP) exp_print.log path to plot as an extra curve.",
    )
    parser.add_argument(
        "--ggeur-cnn-log",
        type=str,
        default="exp/GGEUR_CNN_convnext/ggeur_alpha0.1_clientsnum60_exp_print.log",
        help="Our_method (CNN/ConvNeXt) exp_print.log path to plot as an extra curve.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)

    overrides = {}
    for item in args.override:
        if "=" not in item:
            raise SystemExit(f"Invalid --override (missing '='): {item}")
        key, path = item.split("=", 1)
        key = key.strip()
        path = path.strip().strip("\"'")
        if "/" not in key:
            raise SystemExit(f"Invalid --override key (expected '<group>/<method>'): {key}")
        overrides[key] = path

    plot_group.overrides = overrides  # type: ignore[attr-defined]
    plot_group.ggeur_logs = {  # type: ignore[attr-defined]
        "vit": args.ggeur_vit_log,
        "cnn": args.ggeur_cnn_log,
    }

    for group in args.groups:
        plot_group(root, group, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
