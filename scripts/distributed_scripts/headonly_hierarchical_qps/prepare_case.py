#!/usr/bin/env python3
"""Prepare a two-machine three-level HeadOnly parameter-QPS case."""

import argparse
import json
import time
from pathlib import Path


def path_str(path):
    return str(path).replace("\\", "/")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_subserver_assignments(client_num, subserver_num):
    if client_num <= 0:
        raise ValueError("--client-num must be positive")
    if subserver_num <= 0:
        raise ValueError("--subserver-num must be positive")
    if subserver_num > client_num:
        raise ValueError("--subserver-num cannot exceed --client-num")

    base = client_num // subserver_num
    remainder = client_num % subserver_num
    assignments = []
    start = 1
    for idx in range(subserver_num):
        count = base + (1 if idx < remainder else 0)
        end = start + count - 1
        assignments.append(
            {
                "subserver_id": idx + 1,
                "client_start": start,
                "client_end": end,
                "expected_clients": count,
            }
        )
        start = end + 1
    return assignments


def find_assignment(assignments, client_id):
    for item in assignments:
        if item["client_start"] <= client_id <= item["client_end"]:
            return item
    raise ValueError(f"client {client_id} was not assigned to a subserver")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="scripts/distributed_scripts/headonly_hierarchical_qps/runs")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--experiment-summary", default="headonly60c-1sub-paramqps")
    parser.add_argument("--date", default=time.strftime("%Y%m%d"))
    parser.add_argument("--client-num", type=int, default=60)
    parser.add_argument("--subserver-num", type=int, default=1)
    parser.add_argument("--total-rounds", type=int, default=20)
    parser.add_argument("--root-bind-host", default="0.0.0.0")
    parser.add_argument("--root-host", default="127.0.0.1")
    parser.add_argument("--root-port", type=int, default=58051)
    parser.add_argument("--subserver-bind-host", default="0.0.0.0")
    parser.add_argument("--subserver-advertise-host", required=True)
    parser.add_argument("--subserver-port", type=int, default=58061)
    parser.add_argument("--upload-timeout-sec", type=float, default=10.0)
    parser.add_argument("--qps-window-sec", type=float, default=1.0)
    parser.add_argument("--root-rpc-timeout-sec", type=float, default=600.0)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-version", default="officehome_vitb16_60c_gen20_fcache_v1")
    parser.add_argument("--dataset", default="office-home")
    parser.add_argument("--num-classes", type=int, default=65)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--train-timeout-sec", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--root-log-base", default="/root/autodl-tmp/FederatedScope/exp/headonly_hierarchical_qps")
    parser.add_argument("--client-log-base", default="D:/Projects/FederatedScope/exp/headonly_hierarchical_qps_clients")
    args = parser.parse_args()

    run_id = args.run_id or f"{args.experiment_summary}-{args.date}"
    case_dir = Path(args.output_root) / run_id
    cfg_dir = case_dir / "configs"
    client_cfg_dir = cfg_dir / "clients"
    root_log_dir = Path(args.root_log_base) / f"{args.experiment_summary}-{args.date}"
    client_log_dir = Path(args.client_log_base) / run_id

    subserver_assignments = build_subserver_assignments(
        args.client_num, args.subserver_num
    )

    root_cfg = {
        "role": "root",
        "experiment_summary": args.experiment_summary,
        "run_id": run_id,
        "listen_host": args.root_bind_host,
        "listen_port": args.root_port,
        "expected_subservers": args.subserver_num,
        "total_rounds": args.total_rounds,
        "log_dir": path_str(root_log_dir),
    }
    write_json(cfg_dir / "root_server.json", root_cfg)

    subserver_cfgs = []
    for assignment in subserver_assignments:
        subserver_id = int(assignment["subserver_id"])
        sub_cfg = {
            "role": "subserver",
            "experiment_summary": args.experiment_summary,
            "run_id": run_id,
            "subserver_id": subserver_id,
            "listen_host": args.subserver_bind_host,
            "listen_port": args.subserver_port + subserver_id - 1,
            "advertise_host": args.subserver_advertise_host,
            "root_host": args.root_host,
            "root_port": args.root_port,
            "expected_clients": assignment["expected_clients"],
            "client_start": assignment["client_start"],
            "client_end": assignment["client_end"],
            "total_rounds": args.total_rounds,
            "upload_timeout_sec": args.upload_timeout_sec,
            "qps_window_sec": args.qps_window_sec,
            "root_rpc_timeout_sec": args.root_rpc_timeout_sec,
            "log_dir": path_str(root_log_dir),
        }
        sub_path = cfg_dir / f"subserver_{subserver_id}.json"
        write_json(sub_path, sub_cfg)
        subserver_cfgs.append(path_str(sub_path))

    client_cfgs = []
    for client_id in range(1, args.client_num + 1):
        assignment = find_assignment(subserver_assignments, client_id)
        subserver_id = int(assignment["subserver_id"])
        client_cfg = {
            "role": "client",
            "experiment_summary": args.experiment_summary,
            "run_id": run_id,
            "client_id": client_id,
            "subserver_id": subserver_id,
            "subserver_host": args.subserver_advertise_host,
            "subserver_port": args.subserver_port + subserver_id - 1,
            "total_rounds": args.total_rounds,
            "cache_dir": args.cache_dir,
            "cache_version": args.cache_version,
            "dataset": args.dataset,
            "num_classes": args.num_classes,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "local_epochs": args.local_epochs,
            "max_batches": args.max_batches,
            "train_timeout_sec": args.train_timeout_sec,
            "lr": args.lr,
            "device": args.device,
            "log_dir": path_str(client_log_dir),
            "rpc_timeout_sec": 600,
            "param_wait_timeout_sec": max(1800, int(args.upload_timeout_sec * 4)),
        }
        path = client_cfg_dir / f"client_{client_id:06d}.json"
        write_json(path, client_cfg)
        client_cfgs.append(path_str(path))

    manifest = {
        "run_id": run_id,
        "experiment_summary": args.experiment_summary,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "case_dir": path_str(case_dir),
        "root_log_dir": path_str(root_log_dir),
        "client_log_dir": path_str(client_log_dir),
        "client_num": args.client_num,
        "subserver_num": args.subserver_num,
        "subserver_assignments": subserver_assignments,
        "total_rounds": args.total_rounds,
        "qps_definition": (
            "subserver_parameter_read_qps = number of client parameter-read requests "
            "ACKed by the subserver divided by the elapsed response window from the "
            "first read request received to the last ACK sent after root aggregation "
            "completes and global-ready is released; ACK only, no parameter payload "
            "on measured path. One-second window counts are logged as auxiliary fields."
        ),
        "root_config": path_str(cfg_dir / "root_server.json"),
        "subserver_configs": subserver_cfgs,
        "subserver_config": subserver_cfgs[0] if subserver_cfgs else "",
        "client_configs": client_cfgs,
    }
    write_json(case_dir / "manifest.json", manifest)
    print(path_str(case_dir))


if __name__ == "__main__":
    main()
