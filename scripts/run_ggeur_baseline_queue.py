#!/usr/bin/env python3
"""Run the GGEUR OfficeHome baseline config queue sequentially.

This runner is intentionally separate from the existing server/client and
training code.  It enumerates the historical CNN/Mixer/ViT baseline YAML files,
runs each config through ``federatedscope/main.py``, and records enough state to
resume after an interruption.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Dict, Iterable, List


CONFIG_DIRS = [
    Path("scripts/example_configs/ggeur_baseline_cnn"),
    Path("scripts/example_configs/ggeur_baseline_mixer"),
    Path("scripts/example_configs/ggeur_baseline_vit"),
]


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def slug_for_config(path: Path) -> str:
    model = path.parent.name.replace("ggeur_baseline_", "")
    return f"{model}__{path.stem}"


def iter_configs(patterns: Iterable[str], excludes: Iterable[str]) -> List[Path]:
    configs: List[Path] = []
    include = [p.lower() for p in patterns if p]
    exclude = [p.lower() for p in excludes if p]
    for cfg_dir in CONFIG_DIRS:
        for cfg in sorted(cfg_dir.glob("*.yaml")):
            key = str(cfg).replace("\\", "/").lower()
            if include and not any(p in key for p in include):
                continue
            if exclude and any(p in key for p in exclude):
                continue
            configs.append(cfg)
    return configs


def load_state(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["slug"]: item for item in data.get("records", [])}


def write_state(path: Path, records: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": now_iso(),
        "records": [records[k] for k in sorted(records)],
    }
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def append_experiment_log(path: Path, title: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n\n{text.rstrip()}\n")


def command_to_text(cmd: List[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def run_one(args: argparse.Namespace, cfg: Path, run_dir: Path,
            records: Dict[str, dict], state_path: Path,
            experiment_log: Path) -> int:
    slug = slug_for_config(cfg)
    case_dir = run_dir / "cases" / slug
    case_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = case_dir / "stdout.log"
    record = records.get(slug, {})

    if args.resume and record.get("status") == "success":
        print(f"[skip] {slug} already succeeded")
        return 0

    cmd = [
        args.python_bin,
        "federatedscope/main.py",
        "--cfg",
        str(cfg).replace("\\", "/"),
        "outdir",
        str(case_dir).replace("\\", "/"),
        "expname",
        "run",
    ]
    for item in args.opt:
        cmd.extend(item.split("=", 1) if "=" in item else [item])

    if args.dry_run:
        print(command_to_text(cmd))
        return 0

    start = time.time()
    start_iso = now_iso()
    records[slug] = {
        "slug": slug,
        "config": str(cfg).replace("\\", "/"),
        "case_dir": str(case_dir).replace("\\", "/"),
        "stdout_log": str(stdout_log).replace("\\", "/"),
        "status": "running",
        "started_at": start_iso,
        "command": command_to_text(cmd),
    }
    write_state(state_path, records)
    append_experiment_log(
        experiment_log,
        f"START {slug}",
        "\n".join([
            f"- time: {start_iso}",
            f"- config: `{cfg.as_posix()}`",
            f"- output: `{case_dir.as_posix()}`",
            f"- command: `{command_to_text(cmd)}`",
        ]),
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print(f"[run] {slug}")
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
    finished_iso = now_iso()
    status = "success" if rc == 0 else "failed"
    records[slug].update({
        "status": status,
        "returncode": rc,
        "finished_at": finished_iso,
        "elapsed_sec": round(elapsed, 3),
    })
    write_state(state_path, records)
    append_experiment_log(
        experiment_log,
        f"END {slug}",
        "\n".join([
            f"- time: {finished_iso}",
            f"- status: {status}",
            f"- returncode: {rc}",
            f"- elapsed_sec: {elapsed:.3f}",
            f"- stdout: `{stdout_log.as_posix()}`",
        ]),
    )
    print(f"[done] {slug}: {status}, elapsed={elapsed:.1f}s")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default=".", help="FederatedScope repo")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", default="exp/ggeur_baseline_reruns")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument(
        "--opt",
        action="append",
        default=[],
        help="Extra FederatedScope override, e.g. train.local_update_steps=1",
    )
    args = parser.parse_args()
    args.repo_dir = str(Path(args.repo_dir).resolve())
    repo = Path(args.repo_dir)
    if not (repo / "federatedscope/main.py").exists():
        raise FileNotFoundError(f"not a FederatedScope repo: {repo}")

    run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = repo / args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    experiment_log = run_dir / "experiment_log.md"
    records = load_state(state_path)

    configs = iter_configs(args.only, args.exclude)
    if args.max_runs > 0:
        configs = configs[:args.max_runs]

    manifest = {
        "created_at": now_iso(),
        "repo_dir": str(repo),
        "run_dir": str(run_dir),
        "config_count": len(configs),
        "configs": [c.as_posix() for c in configs],
    }
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if not experiment_log.exists():
        append_experiment_log(
            experiment_log,
            "Experiment Queue",
            "\n".join([
                f"- created_at: {manifest['created_at']}",
                f"- config_count: {len(configs)}",
                f"- output_root: `{run_dir.as_posix()}`",
            ]),
        )

    failures = 0
    for cfg in configs:
        rc = run_one(args, cfg, run_dir, records, state_path, experiment_log)
        if rc != 0:
            failures += 1
            if args.stop_on_fail:
                break

    summary = {
        "finished_at": now_iso(),
        "total": len(configs),
        "success": sum(1 for r in records.values() if r.get("status") == "success"),
        "failed": sum(1 for r in records.values() if r.get("status") == "failed"),
        "running": sum(1 for r in records.values() if r.get("status") == "running"),
        "state": str(state_path),
        "experiment_log": str(experiment_log),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
