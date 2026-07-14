#!/usr/bin/env python3
"""Run one third-party accuracy case selected from a YAML matrix."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment issue
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc


ACCURACY_PATTERNS = (
    re.compile(
        r"Round\s+\d+\s+(?:MLP|CNN|Prompt)?\s*Test\s+Accuracy\s+-.*?\baverage:\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    re.compile(r"test[_ ]acc(?:uracy)?[^0-9]*(\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"best[_ ]result[^0-9]*(\d+(?:\.\d+)?)", re.IGNORECASE),
)


class Case(NamedTuple):
    name: str
    dataset: str
    model: str
    method: str
    config: Path
    enabled: bool
    opts: Mapping[str, Any]

    @property
    def slug(self) -> str:
        return self.name


class Matrix(NamedTuple):
    path: Path
    defaults: Mapping[str, Any]
    methods: Sequence[str]
    cases: Sequence[Case]


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def command_to_text(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in cmd)


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    return str(value)


def flatten_opts(opts: Any) -> List[str]:
    if not opts:
        return []
    if isinstance(opts, Mapping):
        flattened: List[str] = []
        for key, value in opts.items():
            flattened.extend([str(key), format_value(value)])
        return flattened
    if isinstance(opts, Sequence) and not isinstance(opts, (str, bytes)):
        flattened = []
        for item in opts:
            text = str(item)
            if "=" in text:
                key, value = text.split("=", 1)
                flattened.extend([key, value])
            else:
                flattened.append(text)
        return flattened
    raise TypeError(f"unsupported opts type: {type(opts)!r}")


def resolve_existing_path(repo: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (repo / path).resolve()


def load_matrix(path: Path) -> Matrix:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"matrix must be a mapping: {path}")

    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, Mapping):
        raise ValueError("matrix defaults must be a mapping")

    methods = [str(item) for item in raw.get("methods", [])]
    raw_cases = raw.get("cases") or []
    if not isinstance(raw_cases, Sequence):
        raise ValueError("matrix cases must be a list")

    cases: List[Case] = []
    seen = set()
    for index, item in enumerate(raw_cases, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"case #{index} must be a mapping")
        dataset = str(item["dataset"])
        model = str(item["model"])
        method = str(item["method"])
        name = str(item.get("name") or f"{dataset}__{model}__{method}")
        if name in seen:
            raise ValueError(f"duplicate case name: {name}")
        seen.add(name)
        cases.append(
            Case(
                name=name,
                dataset=dataset,
                model=model,
                method=method,
                config=Path(str(item["config"])),
                enabled=bool(item.get("enabled", True)),
                opts=item.get("opts") or {},
            )
        )
    return Matrix(path=path, defaults=defaults, methods=methods, cases=cases)


def list_cases(cases: Sequence[Case]) -> None:
    rows = [("case", "dataset", "model", "method", "config")]
    rows.extend(
        (case.slug, case.dataset, case.model, case.method, case.config.as_posix())
        for case in cases
        if case.enabled
    )
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def selected_cases(args: argparse.Namespace, matrix: Matrix) -> List[Case]:
    names = set(args.case or [])
    datasets = set(args.dataset or [])
    models = set(args.model or [])
    methods = set(args.method or [])
    cases: List[Case] = []
    for case in matrix.cases:
        if not case.enabled and not args.include_disabled:
            continue
        if names and case.slug not in names:
            continue
        if datasets and case.dataset not in datasets:
            continue
        if models and case.model not in models:
            continue
        if methods and case.method not in methods:
            continue
        cases.append(case)
    return cases


def validate_cases(repo: Path, cases: Sequence[Case]) -> None:
    missing = [
        case.config.as_posix()
        for case in cases
        if not (repo / case.config).exists()
    ]
    if missing:
        raise FileNotFoundError("missing config files:\n" + "\n".join(missing))


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
                candidate = float(match.group(1))
                if 0.0 <= candidate <= 1.0:
                    value = candidate
                break
    return value


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
        "records": [records[key] for key in sorted(records)],
    })


def append_markdown(path: Path, title: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n\n{text.rstrip()}\n")


def write_manifest(run_dir: Path, repo: Path, matrix: Matrix,
                   cases: Sequence[Case], args: argparse.Namespace,
                   opts: Sequence[str]) -> None:
    manifest = {
        "created_at": now_iso(),
        "repo_dir": str(repo),
        "run_dir": str(run_dir),
        "matrix": matrix.path.as_posix(),
        "purpose": "single-case third-party accuracy rerun",
        "case_count": len(cases),
        "datasets": sorted({case.dataset for case in cases}),
        "models": sorted({case.model for case in cases}),
        "methods": list(matrix.methods),
        "extra_opts": list(opts),
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
    rows = [records[key] for key in sorted(records)]
    summary = {
        "finished_at": now_iso(),
        "total_records": len(rows),
        "success": sum(1 for row in rows if row.get("status") == "success"),
        "failed": sum(1 for row in rows if row.get("status") == "failed"),
        "running": sum(1 for row in rows if row.get("status") == "running"),
        "records": rows,
    }
    write_json_atomic(run_dir / "summary.json", summary)

    csv_path = run_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "model",
            "method",
            "status",
            "last_accuracy",
            "elapsed_sec",
            "stdout_log",
        ])
        for row in rows:
            writer.writerow([
                row.get("dataset", ""),
                row.get("model", ""),
                row.get("method", ""),
                row.get("status", ""),
                row.get("last_accuracy", ""),
                row.get("elapsed_sec", ""),
                row.get("stdout_log", ""),
            ])


def federatedscope_command(args: argparse.Namespace, case: Case,
                           case_dir: Path, opts: Sequence[str]) -> List[str]:
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
    cmd.extend(opts)
    return cmd


def run_one(args: argparse.Namespace, case: Case, opts: Sequence[str],
            run_dir: Path, records: Dict[str, dict], state_path: Path,
            experiment_log: Path) -> int:
    case_dir = run_dir / "cases" / case.slug
    stdout_log = case_dir / "stdout.log"
    record = records.get(case.slug, {})

    if args.resume and record.get("status") == "success":
        print(f"[skip] {case.slug} already succeeded")
        return 0

    case_dir.mkdir(parents=True, exist_ok=True)
    cmd = federatedscope_command(args, case, case_dir, opts)
    if args.dry_run:
        print(command_to_text(cmd))
        return 0

    started_at = now_iso()
    started = time.time()
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

    elapsed = time.time() - started
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


def group_methods(matrix: Matrix, dataset: str, model: str) -> List[str]:
    available = {
        case.method
        for case in matrix.cases
        if case.enabled and case.dataset == dataset and case.model == model
    }
    ordered = [method for method in matrix.methods if method in available]
    return ordered or sorted(available)


def plot_group(args: argparse.Namespace, matrix: Matrix, run_dir: Path,
               case: Case) -> int:
    methods = group_methods(matrix, case.dataset, case.model)
    if not methods:
        print(f"[plot] no methods configured for {case.dataset}/{case.model}")
        return 0
    cmd = [
        args.python_bin,
        "scripts/plot_ggeur_accuracy_curves.py",
        "--input-dir",
        (run_dir / "cases").as_posix(),
        "--output-dir",
        (run_dir / "plots").as_posix(),
        "--dataset",
        case.dataset,
        "--model",
        case.model,
        "--methods",
        *methods,
        "--combined",
        "--allow-missing",
    ]
    print(f"[plot] {command_to_text(cmd)}")
    if args.dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=args.repo_dir)
    if proc.returncode != 0:
        print(f"[plot] failed rc={proc.returncode}")
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exactly one third-party accuracy case from YAML by default."
    )
    parser.add_argument("--matrix", default="scripts/thirdparty_accuracy_cases.yaml")
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--python-bin", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--case", action="append", help="Case name, e.g. officehome__cnn__ggeur")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--model", action="append")
    parser.add_argument("--method", action="append")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--allow-multiple", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--plot-group", action="store_true")
    parser.add_argument(
        "--opt",
        action="append",
        default=[],
        help="Extra FederatedScope override, e.g. train.local_update_steps=1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo_dir).resolve()
    if not (repo / "federatedscope/main.py").exists():
        raise FileNotFoundError(f"not a FederatedScope repo: {repo}")
    args.repo_dir = str(repo)

    matrix_path = resolve_existing_path(repo, args.matrix)
    matrix = load_matrix(matrix_path)
    defaults = matrix.defaults

    args.python_bin = args.python_bin or str(defaults.get("python_bin") or sys.executable)
    args.output_root = args.output_root or str(defaults.get("output_root") or "exp/ggeur_accuracy_reruns")
    args.run_id = args.run_id or str(defaults.get("run_id") or dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    args.cuda_visible_devices = args.cuda_visible_devices or str(defaults.get("cuda_visible_devices") or "")

    if args.list_cases:
        list_cases(matrix.cases)
        return 0

    selected = selected_cases(args, matrix)
    if not selected:
        raise SystemExit("no case matched; use --list-cases to inspect available case names")
    if len(selected) != 1 and not args.allow_multiple:
        names = ", ".join(case.slug for case in selected)
        raise SystemExit(
            f"{len(selected)} cases matched: {names}. "
            "Select one --case, or pass --allow-multiple intentionally."
        )

    validate_cases(repo, selected)
    run_dir = repo / args.output_root / args.run_id
    state_path = run_dir / "state.json"
    experiment_log = run_dir / "experiment_log.md"
    records = load_state(state_path)

    common_opts = flatten_opts(defaults.get("opts") or {})
    run_dir.mkdir(parents=True, exist_ok=True)

    if not experiment_log.exists():
        append_markdown(
            experiment_log,
            "Experiment Queue",
            "\n".join([
                f"- created_at: {now_iso()}",
                "- purpose: single-case third-party accuracy rerun",
                f"- matrix: `{matrix.path.as_posix()}`",
                f"- output_root: `{run_dir.as_posix()}`",
            ]),
        )

    failures = 0
    last_case: Optional[Case] = None
    for case in selected:
        case_opts = [*common_opts, *flatten_opts(case.opts), *flatten_opts(args.opt)]
        write_manifest(run_dir, repo, matrix, selected, args, case_opts)
        rc = run_one(args, case, case_opts, run_dir, records, state_path, experiment_log)
        write_summary(run_dir, records)
        last_case = case
        if rc != 0:
            failures += 1
            if args.stop_on_fail:
                break

    if args.plot_group and last_case is not None:
        plot_group(args, matrix, run_dir, last_case)

    write_summary(run_dir, records)
    print((run_dir / "summary.json").read_text(encoding="utf-8"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
