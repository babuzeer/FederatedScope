#!/usr/bin/env python3
"""Run the historical GGEUR accuracy matrix sequentially.

The queue is for single-machine accuracy reruns.  It intentionally does not use
the distributed or QPS validation entrypoints.  Each YAML is executed through
``federatedscope/main.py`` with its original hyper-parameters, and each case gets
an isolated output directory plus a full stdout log.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence


METHODS = ("ggeur", "fedavg", "fedprox", "fedproto", "fedopt", "moon")


class Case(NamedTuple):
    dataset: str
    model: str
    method: str
    config: Path

    @property
    def slug(self) -> str:
        return f"{self.dataset}__{self.model}__{self.method}"


CASES: Sequence[Case] = (
    Case("officehome", "cnn", "ggeur", Path("scripts/example_configs/ggeur_baseline_cnn/officehome_lds_cnn_ggeur_fedavg.yaml")),
    Case("officehome", "cnn", "fedavg", Path("scripts/example_configs/ggeur_baseline_cnn/officehome_lds_cnn_fedavg.yaml")),
    Case("officehome", "cnn", "fedprox", Path("scripts/example_configs/ggeur_baseline_cnn/officehome_lds_cnn_fedprox.yaml")),
    Case("officehome", "cnn", "fedproto", Path("scripts/example_configs/ggeur_baseline_cnn/officehome_lds_cnn_fedproto.yaml")),
    Case("officehome", "cnn", "fedopt", Path("scripts/example_configs/ggeur_baseline_cnn/officehome_lds_cnn_fedopt.yaml")),
    Case("officehome", "cnn", "moon", Path("scripts/example_configs/ggeur_baseline_cnn/officehome_lds_cnn_moon.yaml")),
    Case("officehome", "mixer", "ggeur", Path("scripts/example_configs/ggeur_baseline_mixer/officehome_lds_mixer_ggeur_fedavg_local_weights.yaml")),
    Case("officehome", "mixer", "fedavg", Path("scripts/example_configs/ggeur_baseline_mixer/officehome_lds_mixer_fedavg_baseline_local_weights.yaml")),
    Case("officehome", "mixer", "fedprox", Path("scripts/example_configs/ggeur_baseline_mixer/officehome_lds_mixer_fedprox.yaml")),
    Case("officehome", "mixer", "fedproto", Path("scripts/example_configs/ggeur_baseline_mixer/officehome_lds_mixer_fedproto.yaml")),
    Case("officehome", "mixer", "fedopt", Path("scripts/example_configs/ggeur_baseline_mixer/officehome_lds_mixer_fedopt.yaml")),
    Case("officehome", "mixer", "moon", Path("scripts/example_configs/ggeur_baseline_mixer/officehome_lds_mixer_moon.yaml")),
    Case("officehome", "vit", "ggeur", Path("scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_ggeur_fedavg.yaml")),
    Case("officehome", "vit", "fedavg", Path("scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_fedavg.yaml")),
    Case("officehome", "vit", "fedprox", Path("scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_fedprox.yaml")),
    Case("officehome", "vit", "fedproto", Path("scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_fedproto.yaml")),
    Case("officehome", "vit", "fedopt", Path("scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_fedopt.yaml")),
    Case("officehome", "vit", "moon", Path("scripts/example_configs/ggeur_baseline_vit/officehome_lds_vit_moon.yaml")),
    Case("domainnet", "cnn", "ggeur", Path("scripts/example_configs/ggeur_baseline_cnn_domainnet/domainnet_4domains_cnn_ggeur_fedavg.yaml")),
    Case("domainnet", "cnn", "fedavg", Path("scripts/example_configs/ggeur_baseline_cnn_domainnet/domainnet_4domains_cnn_fedavg.yaml")),
    Case("domainnet", "cnn", "fedprox", Path("scripts/example_configs/ggeur_baseline_cnn_domainnet/domainnet_4domains_cnn_fedprox.yaml")),
    Case("domainnet", "cnn", "fedproto", Path("scripts/example_configs/ggeur_baseline_cnn_domainnet/domainnet_4domains_cnn_fedproto.yaml")),
    Case("domainnet", "cnn", "fedopt", Path("scripts/example_configs/ggeur_baseline_cnn_domainnet/domainnet_4domains_cnn_fedopt.yaml")),
    Case("domainnet", "cnn", "moon", Path("scripts/example_configs/ggeur_baseline_cnn_domainnet/domainnet_4domains_cnn_moon.yaml")),
    Case("domainnet", "mlp", "ggeur", Path("scripts/example_configs/ggeur_baseline_mlp_domainnet/domainnet_4domains_mlp_ggeur_fedavg.yaml")),
    Case("domainnet", "mlp", "fedavg", Path("scripts/example_configs/ggeur_baseline_mlp_domainnet/domainnet_4domains_mlp_fedavg.yaml")),
    Case("domainnet", "mlp", "fedprox", Path("scripts/example_configs/ggeur_baseline_mlp_domainnet/domainnet_4domains_mlp_fedprox.yaml")),
    Case("domainnet", "mlp", "fedproto", Path("scripts/example_configs/ggeur_baseline_mlp_domainnet/domainnet_4domains_mlp_fedproto.yaml")),
    Case("domainnet", "mlp", "fedopt", Path("scripts/example_configs/ggeur_baseline_mlp_domainnet/domainnet_4domains_mlp_fedopt.yaml")),
    Case("domainnet", "mlp", "moon", Path("scripts/example_configs/ggeur_baseline_mlp_domainnet/domainnet_4domains_mlp_moon.yaml")),
    Case("domainnet", "vit", "ggeur", Path("scripts/example_configs/ggeur_baseline_vit_domainnet/domainnet_4domains_vit_ggeur_fedavg.yaml")),
    Case("domainnet", "vit", "fedavg", Path("scripts/example_configs/ggeur_baseline_vit_domainnet/domainnet_4domains_vit_fedavg.yaml")),
    Case("domainnet", "vit", "fedprox", Path("scripts/example_configs/ggeur_baseline_vit_domainnet/domainnet_4domains_vit_fedprox.yaml")),
    Case("domainnet", "vit", "fedproto", Path("scripts/example_configs/ggeur_baseline_vit_domainnet/domainnet_4domains_vit_fedproto.yaml")),
    Case("domainnet", "vit", "fedopt", Path("scripts/example_configs/ggeur_baseline_vit_domainnet/domainnet_4domains_vit_fedopt.yaml")),
    Case("domainnet", "vit", "moon", Path("scripts/example_configs/ggeur_baseline_vit_domainnet/domainnet_4domains_vit_moon.yaml")),
    Case("pacs", "cnn", "ggeur", Path("scripts/example_configs/ggeur_baseline_cnn_pacs/pacs_lds_cnn_ggeur_fedavg.yaml")),
    Case("pacs", "cnn", "fedavg", Path("scripts/example_configs/ggeur_baseline_cnn_pacs/pacs_lds_cnn_fedavg.yaml")),
    Case("pacs", "cnn", "fedprox", Path("scripts/example_configs/ggeur_baseline_cnn_pacs/pacs_lds_cnn_fedprox.yaml")),
    Case("pacs", "cnn", "fedproto", Path("scripts/example_configs/ggeur_baseline_cnn_pacs/pacs_lds_cnn_fedproto.yaml")),
    Case("pacs", "cnn", "fedopt", Path("scripts/example_configs/ggeur_baseline_cnn_pacs/pacs_lds_cnn_fedopt.yaml")),
    Case("pacs", "cnn", "moon", Path("scripts/example_configs/ggeur_baseline_cnn_pacs/pacs_lds_cnn_moon.yaml")),
    Case("pacs", "mixer", "ggeur", Path("scripts/example_configs/ggeur_baseline_mixer_pacs/pacs_lds_mixer_ggeur_fedavg.yaml")),
    Case("pacs", "mixer", "fedavg", Path("scripts/example_configs/ggeur_baseline_mixer_pacs/pacs_lds_mixer_fedavg.yaml")),
    Case("pacs", "mixer", "fedprox", Path("scripts/example_configs/ggeur_baseline_mixer_pacs/pacs_lds_mixer_fedprox.yaml")),
    Case("pacs", "mixer", "fedproto", Path("scripts/example_configs/ggeur_baseline_mixer_pacs/pacs_lds_mixer_fedproto.yaml")),
    Case("pacs", "mixer", "fedopt", Path("scripts/example_configs/ggeur_baseline_mixer_pacs/pacs_lds_mixer_fedopt.yaml")),
    Case("pacs", "mixer", "moon", Path("scripts/example_configs/ggeur_baseline_mixer_pacs/pacs_lds_mixer_moon.yaml")),
    Case("pacs", "vit", "ggeur", Path("scripts/example_configs/ggeur_baseline_vit_pacs/pacs_lds_vit_ggeur_fedavg.yaml")),
    Case("pacs", "vit", "fedavg", Path("scripts/example_configs/ggeur_baseline_vit_pacs/pacs_lds_vit_fedavg.yaml")),
    Case("pacs", "vit", "fedprox", Path("scripts/example_configs/ggeur_baseline_vit_pacs/pacs_lds_vit_fedprox.yaml")),
    Case("pacs", "vit", "fedproto", Path("scripts/example_configs/ggeur_baseline_vit_pacs/pacs_lds_vit_fedproto.yaml")),
    Case("pacs", "vit", "fedopt", Path("scripts/example_configs/ggeur_baseline_vit_pacs/pacs_lds_vit_fedopt.yaml")),
    Case("pacs", "vit", "moon", Path("scripts/example_configs/ggeur_baseline_vit_pacs/pacs_lds_vit_moon.yaml")),
)


ACCURACY_PATTERNS = (
    re.compile(
        r"Round\s+\d+\s+(?:MLP|CNN|Prompt)?\s*Test\s+Accuracy\s+-.*?\baverage:\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    re.compile(r"test[_ ]acc(?:uracy)?[^0-9]*(\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"best[_ ]result[^0-9]*(\d+(?:\.\d+)?)", re.IGNORECASE),
)


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def command_to_text(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def load_state(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["slug"]: item for item in data.get("records", [])}


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def write_state(path: Path, records: Dict[str, dict]) -> None:
    write_json_atomic(path, {
        "updated_at": now_iso(),
        "records": [records[k] for k in sorted(records)],
    })


def append_markdown(path: Path, title: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n\n{text.rstrip()}\n")


def selected_cases(args: argparse.Namespace) -> List[Case]:
    datasets = set(args.dataset or [])
    models = set(args.model or [])
    methods = set(args.method or [])
    only = [x.lower() for x in args.only]
    exclude = [x.lower() for x in args.exclude]
    cases: List[Case] = []
    for case in CASES:
        key = f"{case.dataset}/{case.model}/{case.method}/{case.config.as_posix()}".lower()
        if datasets and case.dataset not in datasets:
            continue
        if models and case.model not in models:
            continue
        if methods and case.method not in methods:
            continue
        if only and not any(token in key for token in only):
            continue
        if exclude and any(token in key for token in exclude):
            continue
        cases.append(case)
    if args.max_runs > 0:
        cases = cases[:args.max_runs]
    return cases


def parse_last_accuracy(log_path: Path) -> Optional[float]:
    if not log_path.exists():
        return None
    value: Optional[float] = None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        for pattern in ACCURACY_PATTERNS:
            match = pattern.search(line)
            if match:
                value = float(match.group(1))
                break
    return value


def validate_cases(repo: Path, cases: Sequence[Case]) -> None:
    missing = [case.config.as_posix() for case in cases if not (repo / case.config).exists()]
    if missing:
        raise FileNotFoundError("missing config files:\n" + "\n".join(missing))


def federatedscope_command(args: argparse.Namespace, case: Case, case_dir: Path) -> List[str]:
    cmd = [
        args.python_bin,
        "federatedscope/main.py",
        "--cfg",
        case.config.as_posix(),
        "outdir",
        case_dir.as_posix(),
        "expname",
        "run",
    ]
    for item in args.opt:
        if "=" in item:
            left, right = item.split("=", 1)
            cmd.extend([left, right])
        else:
            cmd.append(item)
    return cmd


def run_one(args: argparse.Namespace, case: Case, run_dir: Path,
            records: Dict[str, dict], state_path: Path,
            experiment_log: Path) -> int:
    case_dir = run_dir / "cases" / case.slug
    stdout_log = case_dir / "stdout.log"
    record = records.get(case.slug, {})

    if args.resume and record.get("status") == "success":
        print(f"[skip] {case.slug} already succeeded")
        return 0

    case_dir.mkdir(parents=True, exist_ok=True)
    cmd = federatedscope_command(args, case, case_dir)
    if args.dry_run:
        print(command_to_text(cmd))
        return 0

    start = time.time()
    started_at = now_iso()
    records[case.slug] = {
        "slug": case.slug,
        "dataset": case.dataset,
        "model": case.model,
        "method": case.method,
        "config": case.config.as_posix(),
        "case_dir": case_dir.as_posix(),
        "stdout_log": stdout_log.as_posix(),
        "status": "running",
        "started_at": started_at,
        "command": command_to_text(cmd),
    }
    write_state(state_path, records)
    append_markdown(
        experiment_log,
        f"START {case.slug}",
        "\n".join([
            f"- time: {started_at}",
            f"- dataset: {case.dataset}",
            f"- model: {case.model}",
            f"- method: {case.method}",
            f"- config: `{case.config.as_posix()}`",
            f"- output: `{case_dir.as_posix()}`",
            f"- command: `{command_to_text(cmd)}`",
        ]),
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print(f"[run] {case.slug}")
    with stdout_log.open("w", encoding="utf-8", errors="replace") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=args.repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
        rc = proc.wait()

    elapsed = time.time() - start
    finished_at = now_iso()
    status = "success" if rc == 0 else "failed"
    records[case.slug].update({
        "status": status,
        "returncode": rc,
        "finished_at": finished_at,
        "elapsed_sec": round(elapsed, 3),
        "last_accuracy": parse_last_accuracy(stdout_log),
    })
    write_state(state_path, records)
    append_markdown(
        experiment_log,
        f"END {case.slug}",
        "\n".join([
            f"- time: {finished_at}",
            f"- status: {status}",
            f"- returncode: {rc}",
            f"- elapsed_sec: {elapsed:.3f}",
            f"- last_accuracy: {records[case.slug].get('last_accuracy')}",
            f"- stdout: `{stdout_log.as_posix()}`",
        ]),
    )
    print(f"[done] {case.slug}: {status}, elapsed={elapsed:.1f}s")
    return rc


def write_manifest(run_dir: Path, repo: Path, cases: Sequence[Case],
                   args: argparse.Namespace) -> None:
    manifest = {
        "created_at": now_iso(),
        "repo_dir": str(repo),
        "run_dir": str(run_dir),
        "purpose": "single-machine accuracy rerun",
        "case_count": len(cases),
        "datasets": sorted({case.dataset for case in cases}),
        "models": sorted({case.model for case in cases}),
        "methods": list(METHODS),
        "notes": [
            "Uses the original YAML hyper-parameters, including client_num, total_round_num, lr and local_update_steps.",
            "This queue does not run distributed or QPS validation code.",
        ],
        "extra_opts": args.opt,
        "cases": [
            {
                "slug": case.slug,
                "dataset": case.dataset,
                "model": case.model,
                "method": case.method,
                "config": case.config.as_posix(),
            }
            for case in cases
        ],
    }
    write_json_atomic(run_dir / "manifest.json", manifest)


def write_summary(run_dir: Path, records: Dict[str, dict]) -> None:
    rows = [records[k] for k in sorted(records)]
    summary = {
        "finished_at": now_iso(),
        "total_records": len(rows),
        "success": sum(1 for row in rows if row.get("status") == "success"),
        "failed": sum(1 for row in rows if row.get("status") == "failed"),
        "running": sum(1 for row in rows if row.get("status") == "running"),
        "records": rows,
    }
    write_json_atomic(run_dir / "summary.json", summary)

    lines = ["dataset,model,method,status,last_accuracy,elapsed_sec,stdout_log"]
    for row in rows:
        lines.append(",".join([
            str(row.get("dataset", "")),
            str(row.get("model", "")),
            str(row.get("method", "")),
            str(row.get("status", "")),
            "" if row.get("last_accuracy") is None else str(row.get("last_accuracy")),
            "" if row.get("elapsed_sec") is None else str(row.get("elapsed_sec")),
            str(row.get("stdout_log", "")),
        ]))
    (run_dir / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default=".", help="FederatedScope repo")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", default="exp/ggeur_accuracy_reruns")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset", action="append", choices=("officehome", "domainnet", "pacs"))
    parser.add_argument("--model", action="append", choices=("cnn", "mixer", "mlp", "vit"))
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--only", action="append", default=[], help="Substring filter")
    parser.add_argument("--exclude", action="append", default=[], help="Substring filter")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument(
        "--opt",
        action="append",
        default=[],
        help="Extra FederatedScope override, e.g. train.local_update_steps=1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.repo_dir = str(Path(args.repo_dir).resolve())
    repo = Path(args.repo_dir)
    if not (repo / "federatedscope/main.py").exists():
        raise FileNotFoundError(f"not a FederatedScope repo: {repo}")

    cases = selected_cases(args)
    validate_cases(repo, cases)

    run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = repo / args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    experiment_log = run_dir / "experiment_log.md"
    records = load_state(state_path)
    write_manifest(run_dir, repo, cases, args)

    if not experiment_log.exists():
        append_markdown(
            experiment_log,
            "Experiment Queue",
            "\n".join([
                f"- created_at: {now_iso()}",
                f"- purpose: single-machine accuracy rerun",
                f"- case_count: {len(cases)}",
                f"- output_root: `{run_dir.as_posix()}`",
            ]),
        )

    print(f"[queue] cases={len(cases)} run_dir={run_dir}")
    failures = 0
    for case in cases:
        rc = run_one(args, case, run_dir, records, state_path, experiment_log)
        write_summary(run_dir, records)
        if rc != 0:
            failures += 1
            if args.stop_on_fail:
                break

    write_summary(run_dir, records)
    print((run_dir / "summary.json").read_text(encoding="utf-8"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
