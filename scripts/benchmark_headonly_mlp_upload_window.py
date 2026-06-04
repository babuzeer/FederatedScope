#!/usr/bin/env python3
"""Benchmark synchronized HeadOnly MLP parameter upload windows.

This script sends one real-shape HeadOnly MLP payload per logical client after
the server broadcasts a one-byte "go" signal. It measures the concentrated
upload window separately from local training time.
"""

import argparse
import asyncio
import json
import struct
import time


def build_payload_bytes(input_dim, num_classes, hidden_dim, dtype_bytes,
                        overhead_multiplier):
    if hidden_dim and hidden_dim > 0:
        params = (input_dim * hidden_dim + hidden_dim +
                  hidden_dim * num_classes + num_classes)
    else:
        params = input_dim * num_classes + num_classes
    raw_bytes = params * dtype_bytes
    payload_bytes = int(raw_bytes * overhead_multiplier)
    return params, raw_bytes, payload_bytes, b"x" * payload_bytes


def fmt_bytes(value):
    units = ["B", "KiB", "MiB", "GiB"]
    value = float(value)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} GiB"


class UploadWindowServer:
    def __init__(self, expected_clients):
        self.expected_clients = expected_clients
        self.writers = []
        self.connected_times = []
        self.complete_times = []
        self.total_bytes = 0
        self.go_time = None
        self.done = asyncio.Event()
        self._go_sent = False

    async def handle_client(self, reader, writer):
        self.writers.append(writer)
        self.connected_times.append(time.perf_counter())
        if len(self.writers) == self.expected_clients and not self._go_sent:
            self._go_sent = True
            self.go_time = time.perf_counter()
            for item in self.writers:
                item.write(b"G")
            await asyncio.gather(
                *(item.drain() for item in self.writers),
                return_exceptions=True,
            )

        header = await reader.readexactly(8)
        payload_len = struct.unpack("!Q", header)[0]
        remaining = payload_len
        while remaining:
            chunk = await reader.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise ConnectionError("connection closed before payload end")
            remaining -= len(chunk)
        self.complete_times.append(time.perf_counter())
        self.total_bytes += payload_len
        writer.write(b"A")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

        if len(self.complete_times) >= self.expected_clients:
            self.done.set()

    def summary(self, payload_bytes):
        if not self.complete_times:
            return {}
        first_complete = min(self.complete_times)
        last_complete = max(self.complete_times)
        first_connect = min(self.connected_times)
        last_connect = max(self.connected_times)
        go_time = self.go_time or first_complete
        upload_window_sec = max(last_complete - first_complete, 1e-9)
        go_to_last_sec = max(last_complete - go_time, 1e-9)
        received = len(self.complete_times)
        return {
            "expected_clients": self.expected_clients,
            "received_clients": received,
            "payload_bytes_per_upload": payload_bytes,
            "total_payload_bytes": self.total_bytes,
            "connection_ready_span_sec": last_connect - first_connect,
            "go_to_first_upload_complete_sec": first_complete - go_time,
            "go_to_last_upload_complete_sec": go_to_last_sec,
            "first_to_last_upload_complete_sec": upload_window_sec,
            "upload_qps_go_to_last": received / go_to_last_sec,
            "upload_qps_first_to_last": received / upload_window_sec,
            "payload_bytes_per_sec_go_to_last":
                self.total_bytes / go_to_last_sec,
            "payload_bytes_per_sec_first_to_last":
                self.total_bytes / upload_window_sec,
        }


async def run_server(host, port, expected_clients, payload_bytes, output):
    state = UploadWindowServer(expected_clients)
    server = await asyncio.start_server(
        state.handle_client,
        host,
        port,
        backlog=max(expected_clients, 128),
        limit=1024 * 1024,
    )
    async with server:
        await state.done.wait()
    result = state.summary(payload_bytes)
    emit_result(result, output)
    return result


async def run_clients(host, port, clients, payload, connect_timeout):
    async def one_client():
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, limit=1024 * 1024),
            timeout=connect_timeout,
        )
        await reader.readexactly(1)
        writer.write(struct.pack("!Q", len(payload)))
        writer.write(payload)
        await writer.drain()
        await reader.readexactly(1)
        writer.close()
        await writer.wait_closed()

    tasks = [asyncio.create_task(one_client()) for _ in range(clients)]
    await asyncio.gather(*tasks)


async def run_local(args, payload):
    server_task = asyncio.create_task(
        run_server(args.host, args.port, args.clients, len(payload),
                   args.output))
    await asyncio.sleep(0.2)
    await run_clients(args.host, args.port, args.clients, payload,
                      args.connect_timeout)
    return await server_task


def emit_result(result, output):
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if output:
        with open(output, "w", encoding="utf-8") as file:
            file.write(text + "\n")
    print(text)
    print()
    print("Human summary")
    print(f"- received: {result['received_clients']}/"
          f"{result['expected_clients']}")
    print(f"- payload/upload: "
          f"{fmt_bytes(result['payload_bytes_per_upload'])}")
    print(f"- go_to_last: "
          f"{result['go_to_last_upload_complete_sec']:.6f}s, "
          f"qps={result['upload_qps_go_to_last']:.2f}")
    print(f"- first_to_last: "
          f"{result['first_to_last_upload_complete_sec']:.6f}s, "
          f"qps={result['upload_qps_first_to_last']:.2f}")
    print(f"- bandwidth(go_to_last): "
          f"{fmt_bytes(result['payload_bytes_per_sec_go_to_last'])}/s")


def add_common(parser):
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=39201)
    parser.add_argument("--clients", type=int, default=60)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=65)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--dtype-bytes", type=int, default=4)
    parser.add_argument("--overhead-multiplier", type=float, default=1.0)
    parser.add_argument("--output", default="")
    parser.add_argument("--connect-timeout", type=float, default=30.0)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    local_parser = subparsers.add_parser("local")
    add_common(local_parser)
    server_parser = subparsers.add_parser("server")
    add_common(server_parser)
    client_parser = subparsers.add_parser("client")
    add_common(client_parser)
    args = parser.parse_args()

    params, raw_bytes, payload_bytes, payload = build_payload_bytes(
        args.input_dim,
        args.num_classes,
        args.hidden_dim,
        args.dtype_bytes,
        args.overhead_multiplier,
    )
    print(f"MLP params={params}, raw_bytes={raw_bytes}, "
          f"payload_bytes={payload_bytes}")

    if args.mode == "local":
        asyncio.run(run_local(args, payload))
    elif args.mode == "server":
        asyncio.run(
            run_server(args.host, args.port, args.clients, payload_bytes,
                       args.output))
    elif args.mode == "client":
        asyncio.run(
            run_clients(args.host, args.port, args.clients, payload,
                        args.connect_timeout))


if __name__ == "__main__":
    main()
