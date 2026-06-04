#!/usr/bin/env python3
"""Estimate HeadOnly MLP parameter request QPS and network capacity.

This estimates the post-round-0 HeadOnly FL phase where each sampled client:
1. receives one MLP parameter message from the server;
2. uploads one locally trained MLP parameter message back to the server.

The default values match the OfficeHome + ViT-B-16 HeadOnly run:
input_dim=512, num_classes=65, hidden_dim=0.
"""

import argparse
import json
import math


def mlp_param_count(input_dim, num_classes, hidden_dim):
    if hidden_dim and hidden_dim > 0:
        return (input_dim * hidden_dim + hidden_dim +
                hidden_dim * num_classes + num_classes)
    return input_dim * num_classes + num_classes


def fmt_bytes(value):
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(value)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=60,
                        help="Sampled clients per training round.")
    parser.add_argument("--round-seconds", type=float, default=5.07,
                        help="Measured wall time per post-round-0 round.")
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=65)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--dtype-bytes", type=int, default=4)
    parser.add_argument("--serialization-multiplier", type=float, default=1.3,
                        help="Approximate pickle/gRPC/message overhead multiplier.")
    parser.add_argument("--target-qps", type=float, default=10000.0,
                        help="Target total request QPS, counting upload+download.")
    args = parser.parse_args()

    params = mlp_param_count(args.input_dim, args.num_classes, args.hidden_dim)
    raw_payload_bytes = params * args.dtype_bytes
    estimated_payload_bytes = raw_payload_bytes * args.serialization_multiplier

    total_requests_per_round = args.clients * 2
    one_way_requests_per_round = args.clients
    total_request_qps = total_requests_per_round / args.round_seconds
    one_way_request_qps = one_way_requests_per_round / args.round_seconds

    total_bandwidth_bps = estimated_payload_bytes * total_request_qps
    one_way_bandwidth_bps = estimated_payload_bytes * one_way_request_qps

    clients_for_target_total = math.ceil(args.target_qps * args.round_seconds / 2)
    clients_for_target_one_way = math.ceil(args.target_qps * args.round_seconds)
    round_seconds_for_target_current_clients = (
        total_requests_per_round / args.target_qps
        if args.target_qps > 0 else 0.0
    )

    result = {
        "clients": args.clients,
        "round_seconds": args.round_seconds,
        "input_dim": args.input_dim,
        "num_classes": args.num_classes,
        "hidden_dim": args.hidden_dim,
        "mlp_param_count": params,
        "raw_payload_bytes_per_request": raw_payload_bytes,
        "estimated_payload_bytes_per_request": estimated_payload_bytes,
        "requests_per_round_total_upload_plus_download":
            total_requests_per_round,
        "request_qps_total_upload_plus_download": total_request_qps,
        "request_qps_one_way": one_way_request_qps,
        "bandwidth_bytes_per_sec_total_upload_plus_download":
            total_bandwidth_bps,
        "bandwidth_bytes_per_sec_one_way": one_way_bandwidth_bps,
        "target_qps_total_upload_plus_download": args.target_qps,
        "clients_needed_for_target_total_at_same_round_seconds":
            clients_for_target_total,
        "clients_needed_for_target_one_way_at_same_round_seconds":
            clients_for_target_one_way,
        "round_seconds_needed_for_target_with_current_clients":
            round_seconds_for_target_current_clients,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()
    print("Human summary")
    print(f"- MLP params: {params:,}")
    print(f"- Raw payload/request: {fmt_bytes(raw_payload_bytes)}")
    print(f"- Estimated serialized payload/request: "
          f"{fmt_bytes(estimated_payload_bytes)}")
    print(f"- Total request QPS, upload+download: {total_request_qps:.2f}")
    print(f"- One-way request QPS: {one_way_request_qps:.2f}")
    print(f"- Total bandwidth, upload+download: "
          f"{fmt_bytes(total_bandwidth_bps)}/s")
    print(f"- One-way bandwidth: {fmt_bytes(one_way_bandwidth_bps)}/s")
    print(f"- Clients needed for {args.target_qps:.0f} total req/s at "
          f"{args.round_seconds:.2f}s/round: {clients_for_target_total:,}")
    print(f"- Clients needed for {args.target_qps:.0f} one-way req/s at "
          f"{args.round_seconds:.2f}s/round: {clients_for_target_one_way:,}")
    print(f"- Round time needed for current {args.clients} clients to reach "
          f"{args.target_qps:.0f} total req/s: "
          f"{round_seconds_for_target_current_clients:.4f}s")


if __name__ == "__main__":
    main()
