#!/usr/bin/env python3
"""Benchmark HeadOnly hierarchical dispatch-start QPS.

The benchmark models the first-stage acceptance target documented in
docs/HeadOnly_*.md:

    coordinator -> subserver processes -> logical clients

The primary metric is Dispatch-Start QPS. A successful dispatch-start means:

1. a client requests the current model parameters from its assigned subserver;
2. the subserver returns model metadata;
3. the subserver sends the first model parameter chunk;
4. the client ACKs immediately after receiving that first chunk.

The default mode is therefore ``ack_mode=first-chunk``. ``header`` and ``full``
are retained as auxiliary modes for control-plane and full-payload reporting.
"""

import argparse
import asyncio
import json
import multiprocessing as mp
import queue
import statistics
import struct
import time
from pathlib import Path


ACK_MODES = ("header", "first-chunk", "full")
REQUEST_BYTE = b"R"
ACK_BYTE = b"A"
META_LEN_STRUCT = struct.Struct("!I")


def parse_connect_ports(base_port, subservers, connect_ports=""):
    if not connect_ports:
        return [base_port + idx for idx in range(subservers)]
    ports = [int(item.strip()) for item in connect_ports.split(",")
             if item.strip()]
    if len(ports) != subservers:
        raise ValueError(
            "--connect-ports must contain exactly one port per subserver: "
            f"got {len(ports)}, expected {subservers}",
        )
    return ports


def parse_source_hosts(subservers, source_host="", source_hosts=""):
    if source_hosts:
        hosts = [item.strip() for item in source_hosts.split(",")
                 if item.strip()]
        if len(hosts) != subservers:
            raise ValueError(
                "--source-hosts must contain exactly one host per subserver: "
                f"got {len(hosts)}, expected {subservers}",
            )
        return hosts
    if source_host:
        return [source_host for _ in range(subservers)]
    return ["" for _ in range(subservers)]


def build_payload_bytes(input_dim, num_classes, hidden_dim, dtype_bytes,
                        overhead_multiplier, payload_file=""):
    if hidden_dim and hidden_dim > 0:
        params = (input_dim * hidden_dim + hidden_dim +
                  hidden_dim * num_classes + num_classes)
    else:
        params = input_dim * num_classes + num_classes
    raw_bytes = params * dtype_bytes
    if payload_file:
        payload = Path(payload_file).read_bytes()
        return params, raw_bytes, len(payload), payload
    payload_bytes = int(raw_bytes * overhead_multiplier)
    return params, raw_bytes, payload_bytes, b"x" * payload_bytes


def fmt_bytes(value):
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(value)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def percentile(values, pct):
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    weight = rank - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def mode_ack_counts(ack_mode, completed):
    return {
        "ack_clients": completed,
        "completed_clients": completed,
        "header_ack_clients": completed if ack_mode == "header" else 0,
        "first_chunk_ack_clients":
            completed if ack_mode == "first-chunk" else 0,
        "full_payload_ack_clients": completed if ack_mode == "full" else 0,
    }


def primary_metric_name(ack_mode):
    if ack_mode == "header":
        return "header_response_qps"
    if ack_mode == "full":
        return "full_payload_download_qps"
    return "dispatch_start_qps"


def make_metadata_frame(model_version, payload_size, first_chunk_size,
                        ack_mode, subserver_id):
    metadata = {
        "model_version": model_version,
        "payload_size": payload_size,
        "chunk_index": 0,
        "first_chunk_size": first_chunk_size,
        "ack_mode": ack_mode,
        "subserver_id": subserver_id,
    }
    body = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return META_LEN_STRUCT.pack(len(body)) + body, len(body)


class DownloadSubserver:
    def __init__(self, subserver_id, expected_clients, listen_host, port,
                 payload, ack_mode, first_chunk_bytes, model_version,
                 request_timeout, write_chunk_bytes, write_buffer_bytes):
        self.subserver_id = subserver_id
        self.expected_clients = expected_clients
        self.listen_host = listen_host
        self.port = port
        self.payload = payload
        self.ack_mode = ack_mode
        self.first_chunk_bytes = first_chunk_bytes
        self.model_version = model_version
        self.request_timeout = request_timeout
        self.write_chunk_bytes = write_chunk_bytes
        self.write_buffer_bytes = write_buffer_bytes

        self.connections = []
        self.request_times = []
        self.release_time = None
        self.first_ack_time = None
        self.last_ack_time = None
        self.ack_times = []
        self.client_latencies = []
        self.total_network_bytes = 0
        self.total_model_payload_bytes = 0
        self.total_first_chunk_payload_bytes = 0
        self.connected = asyncio.Event()
        self._server = None

    @property
    def actual_first_chunk_bytes(self):
        if not self.payload:
            return 0
        return min(max(self.first_chunk_bytes, 1), len(self.payload))

    async def start(self):
        self._server = await asyncio.start_server(
            self._handle_client,
            self.listen_host,
            self.port,
            backlog=max(self.expected_clients * 2, 128),
            limit=1024 * 1024,
        )
        return self._server

    async def _handle_client(self, reader, writer):
        try:
            request = await asyncio.wait_for(
                reader.readexactly(1),
                timeout=self.request_timeout,
            )
            if request != REQUEST_BYTE:
                raise ValueError(f"invalid request byte: {request!r}")
            self.connections.append((reader, writer))
            self.request_times.append(time.perf_counter())
            if len(self.connections) >= self.expected_clients:
                self.connected.set()
        except Exception:
            writer.close()
            await writer.wait_closed()

    async def wait_connected(self, timeout):
        await asyncio.wait_for(self.connected.wait(), timeout=timeout)

    async def _write_full_payload_tail(self, writer, start_offset):
        offset = start_offset
        while offset < len(self.payload):
            end = min(offset + self.write_chunk_bytes, len(self.payload))
            writer.write(self.payload[offset:end])
            offset = end
            if writer.transport.get_write_buffer_size() >= \
                    self.write_buffer_bytes:
                await writer.drain()

    async def send_payloads(self, ack_timeout):
        self.release_time = time.perf_counter()
        first_len = self.actual_first_chunk_bytes
        first_chunk = self.payload[:first_len]
        metadata_frame, metadata_body_bytes = make_metadata_frame(
            self.model_version,
            len(self.payload),
            first_len,
            self.ack_mode,
            self.subserver_id,
        )
        first_chunk_frame = META_LEN_STRUCT.pack(first_len) + first_chunk

        async def one_client(reader, writer):
            network_bytes = 0
            model_payload_bytes = 0
            first_chunk_payload_bytes = 0

            writer.write(metadata_frame)
            network_bytes += len(metadata_frame)

            if self.ack_mode in {"first-chunk", "full"}:
                writer.write(first_chunk_frame)
                network_bytes += len(first_chunk_frame)
                first_chunk_payload_bytes += first_len
                model_payload_bytes += first_len

            if self.ack_mode == "full":
                await self._write_full_payload_tail(writer, first_len)
                network_bytes += max(len(self.payload) - first_len, 0)
                model_payload_bytes = len(self.payload)

            await writer.drain()
            await asyncio.wait_for(reader.readexactly(1), timeout=ack_timeout)
            complete_time = time.perf_counter()
            self.ack_times.append(complete_time)
            self.client_latencies.append(complete_time - self.release_time)
            self.total_network_bytes += network_bytes
            self.total_model_payload_bytes += model_payload_bytes
            self.total_first_chunk_payload_bytes += first_chunk_payload_bytes
            writer.close()
            await writer.wait_closed()

        await asyncio.gather(
            *(one_client(reader, writer)
              for reader, writer in self.connections[:self.expected_clients]),
            return_exceptions=False,
        )
        if self.ack_times:
            self.first_ack_time = min(self.ack_times)
            self.last_ack_time = max(self.ack_times)

        # Keep the variable visible in summaries without writing it per client.
        self.metadata_body_bytes = metadata_body_bytes

    async def close(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    def summary(self):
        completed = len(self.ack_times)
        release_time = self.release_time or time.perf_counter()
        first_ack_time = self.first_ack_time or release_time
        last_ack_time = self.last_ack_time or release_time
        ack_window = max(last_ack_time - release_time, 1e-9)
        first_to_last = max(last_ack_time - first_ack_time, 1e-9)
        ready_span = (
            max(self.request_times) - min(self.request_times)
            if self.request_times else 0.0
        )
        ack_qps = completed / ack_window
        counts = mode_ack_counts(self.ack_mode, completed)
        dispatch_window = ack_window if self.ack_mode == "first-chunk" else 0.0
        dispatch_qps = ack_qps if self.ack_mode == "first-chunk" else 0.0
        return {
            "subserver_id": self.subserver_id,
            "listen_host": self.listen_host,
            "port": self.port,
            "ack_mode": self.ack_mode,
            "model_version": self.model_version,
            "expected_clients": self.expected_clients,
            **counts,
            "success_ratio": (
                completed / self.expected_clients
                if self.expected_clients else 0.0
            ),
            "connection_ready_span_sec": ready_span,
            "release_to_first_ack_sec": first_ack_time - release_time,
            "ack_window_sec": ack_window,
            "ack_first_to_last_window_sec": first_to_last,
            "ack_qps": ack_qps,
            "primary_metric": primary_metric_name(self.ack_mode),
            "primary_qps": ack_qps,
            "dispatch_start_window_sec": dispatch_window,
            "dispatch_start_qps": dispatch_qps,
            "download_window_go_to_last_sec": ack_window,
            "download_window_first_to_last_sec": first_to_last,
            "download_qps_go_to_last": ack_qps,
            "download_qps_first_to_last": completed / first_to_last,
            "payload_bytes_per_download": len(self.payload),
            "first_chunk_bytes": self.actual_first_chunk_bytes,
            "metadata_body_bytes": getattr(self, "metadata_body_bytes", 0),
            "total_network_bytes": self.total_network_bytes,
            "total_model_payload_bytes": self.total_model_payload_bytes,
            "total_first_chunk_payload_bytes":
                self.total_first_chunk_payload_bytes,
            "payload_bytes_per_sec_go_to_last":
                self.total_model_payload_bytes / ack_window,
            "network_bytes_per_sec_go_to_last":
                self.total_network_bytes / ack_window,
            "latency_p50_sec": percentile(self.client_latencies, 50),
            "latency_p95_sec": percentile(self.client_latencies, 95),
            "latency_p99_sec": percentile(self.client_latencies, 99),
            "dispatch_start_latency_p50_sec": (
                percentile(self.client_latencies, 50)
                if self.ack_mode == "first-chunk" else 0.0
            ),
            "dispatch_start_latency_p95_sec": (
                percentile(self.client_latencies, 95)
                if self.ack_mode == "first-chunk" else 0.0
            ),
            "dispatch_start_latency_p99_sec": (
                percentile(self.client_latencies, 99)
                if self.ack_mode == "first-chunk" else 0.0
            ),
            "release_time": release_time,
            "first_ack_time": first_ack_time,
            "last_ack_time": last_ack_time,
            "_client_latencies_sec": list(self.client_latencies),
        }


def global_summary(sub_summaries, payload_bytes):
    if not sub_summaries:
        return {}
    ack_modes = sorted({item["ack_mode"] for item in sub_summaries})
    ack_mode = ack_modes[0] if len(ack_modes) == 1 else "mixed"
    global_release = min(item["release_time"] for item in sub_summaries)
    global_last = max(item["last_ack_time"] for item in sub_summaries)
    global_first = min(item["first_ack_time"] for item in sub_summaries)
    global_window = max(global_last - global_release, 1e-9)
    first_to_last = max(global_last - global_first, 1e-9)
    ack_clients = sum(item["ack_clients"] for item in sub_summaries)
    expected = sum(item["expected_clients"] for item in sub_summaries)
    header_ack_clients = sum(item["header_ack_clients"]
                             for item in sub_summaries)
    first_chunk_ack_clients = sum(item["first_chunk_ack_clients"]
                                  for item in sub_summaries)
    full_payload_ack_clients = sum(item["full_payload_ack_clients"]
                                   for item in sub_summaries)
    total_network = sum(item["total_network_bytes"]
                        for item in sub_summaries)
    total_model_payload = sum(item["total_model_payload_bytes"]
                              for item in sub_summaries)
    total_first_chunk_payload = sum(item["total_first_chunk_payload_bytes"]
                                    for item in sub_summaries)
    sub_qps = [item["ack_qps"] for item in sub_summaries]
    latencies = []
    for item in sub_summaries:
        latencies.extend(item.get("_client_latencies_sec", []))
    ack_qps = ack_clients / global_window
    is_dispatch_start = ack_mode == "first-chunk"
    return {
        "ack_mode": ack_mode,
        "primary_metric": primary_metric_name(ack_mode)
        if ack_mode in ACK_MODES else "mixed_ack_qps",
        "expected_clients": expected,
        "ack_clients": ack_clients,
        "completed_clients": ack_clients,
        "header_ack_clients": header_ack_clients,
        "first_chunk_ack_clients": first_chunk_ack_clients,
        "full_payload_ack_clients": full_payload_ack_clients,
        "success_ratio": ack_clients / expected if expected else 0.0,
        "payload_bytes_per_download": payload_bytes,
        "first_chunk_bytes": (
            sub_summaries[0].get("first_chunk_bytes", 0)
            if sub_summaries else 0
        ),
        "total_network_bytes": total_network,
        "total_model_payload_bytes": total_model_payload,
        "total_first_chunk_payload_bytes": total_first_chunk_payload,
        "global_ack_window_sec": global_window,
        "global_ack_first_to_last_window_sec": first_to_last,
        "global_ack_qps": ack_qps,
        "primary_qps": ack_qps,
        "dispatch_start_window_sec": (
            global_window if is_dispatch_start else 0.0
        ),
        "dispatch_start_qps": ack_qps if is_dispatch_start else 0.0,
        "global_download_window_go_to_last_sec": global_window,
        "global_download_window_first_to_last_sec": first_to_last,
        "global_download_qps_go_to_last": ack_qps,
        "global_download_qps_first_to_last": ack_clients / first_to_last,
        "global_payload_bytes_per_sec_go_to_last":
            total_model_payload / global_window,
        "global_network_bytes_per_sec_go_to_last":
            total_network / global_window,
        "latency_p50_sec": percentile(latencies, 50),
        "latency_p95_sec": percentile(latencies, 95),
        "latency_p99_sec": percentile(latencies, 99),
        "dispatch_start_latency_p50_sec": (
            percentile(latencies, 50) if is_dispatch_start else 0.0
        ),
        "dispatch_start_latency_p95_sec": (
            percentile(latencies, 95) if is_dispatch_start else 0.0
        ),
        "dispatch_start_latency_p99_sec": (
            percentile(latencies, 99) if is_dispatch_start else 0.0
        ),
        "subserver_qps_min": min(sub_qps),
        "subserver_qps_avg": statistics.mean(sub_qps),
        "subserver_qps_max": max(sub_qps),
    }


def public_subserver_summaries(sub_summaries):
    public = []
    for item in sub_summaries:
        copied = dict(item)
        copied.pop("_client_latencies_sec", None)
        public.append(copied)
    return public


def build_result(mode, args, summaries, payload):
    return {
        "mode": mode,
        "benchmark": "headonly_dispatch_start_qps",
        "ack_mode": args.ack_mode,
        "subservers": args.subservers,
        "clients_per_subserver": args.clients_per_subserver,
        "global": global_summary(summaries, len(payload)),
        "subserver_summaries": public_subserver_summaries(summaries),
    }


def build_subserver(args, subserver_id, port, payload):
    return DownloadSubserver(
        subserver_id=subserver_id,
        expected_clients=args.clients_per_subserver,
        listen_host=args.listen_host,
        port=port,
        payload=payload,
        ack_mode=args.ack_mode,
        first_chunk_bytes=args.first_chunk_bytes,
        model_version=args.model_version,
        request_timeout=args.request_timeout,
        write_chunk_bytes=args.write_chunk_bytes,
        write_buffer_bytes=args.write_buffer_bytes,
    )


async def run_server(args, payload):
    subservers = [
        build_subserver(args, idx + 1, args.base_port + idx, payload)
        for idx in range(args.subservers)
    ]
    await asyncio.gather(*(item.start() for item in subservers))
    print(
        f"Listening on {args.listen_host}:{args.base_port}-"
        f"{args.base_port + args.subservers - 1}; "
        f"expected_clients={args.subservers * args.clients_per_subserver}; "
        f"ack_mode={args.ack_mode}",
        flush=True,
    )
    await asyncio.gather(
        *(item.wait_connected(args.ready_timeout) for item in subservers))
    print("All client requests are ready; coordinator releases model.",
          flush=True)
    await asyncio.gather(
        *(item.send_payloads(args.ack_timeout) for item in subservers))
    summaries = [item.summary() for item in subservers]
    result = build_result("server", args, summaries, payload)
    emit_result(result, args.output)
    await asyncio.gather(*(item.close() for item in subservers))
    return result


async def run_clients(args):
    async def read_exactly_in_chunks(reader, remaining):
        while remaining:
            chunk = await reader.read(min(remaining, args.read_chunk_bytes))
            if not chunk:
                raise ConnectionError("connection closed before payload end")
            remaining -= len(chunk)

    async def one_client(host, port, source_host):
        local_addr = (source_host, 0) if source_host else None
        deadline = time.perf_counter() + args.connect_timeout
        last_error = None
        while True:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(
                        host,
                        port,
                        limit=1024 * 1024,
                        local_addr=local_addr,
                    ),
                    timeout=min(5.0,
                                max(deadline - time.perf_counter(), 0.1)),
                )
                break
            except (OSError, asyncio.TimeoutError) as error:
                last_error = error
                if time.perf_counter() >= deadline:
                    raise TimeoutError(
                        f"Could not connect to {host}:{port} within "
                        f"{args.connect_timeout}s") from last_error
                await asyncio.sleep(args.connect_retry_interval)

        writer.write(REQUEST_BYTE)
        await writer.drain()

        meta_len = META_LEN_STRUCT.unpack(
            await reader.readexactly(META_LEN_STRUCT.size))[0]
        metadata = json.loads(
            (await reader.readexactly(meta_len)).decode("utf-8"))
        ack_mode = metadata.get("ack_mode", args.ack_mode)
        if ack_mode == "header":
            writer.write(ACK_BYTE)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return

        first_chunk_len = META_LEN_STRUCT.unpack(
            await reader.readexactly(META_LEN_STRUCT.size))[0]
        if first_chunk_len:
            await reader.readexactly(first_chunk_len)
        if ack_mode == "first-chunk":
            writer.write(ACK_BYTE)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return

        remaining = int(metadata["payload_size"]) - first_chunk_len
        await read_exactly_in_chunks(reader, remaining)
        writer.write(ACK_BYTE)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    tasks = []
    ports = parse_connect_ports(
        args.base_port,
        args.subservers,
        getattr(args, "connect_ports", ""),
    )
    source_hosts = parse_source_hosts(
        args.subservers,
        getattr(args, "source_host", ""),
        getattr(args, "source_hosts", ""),
    )
    print(f"Connecting to {args.connect_host} ports={ports} "
          f"source_hosts={source_hosts}", flush=True)
    for port, source_host in zip(ports, source_hosts):
        for _ in range(args.clients_per_subserver):
            tasks.append(asyncio.create_task(
                one_client(args.connect_host, port, source_host)))
    await asyncio.gather(*tasks)


async def run_single_subserver_worker(args, payload, result_queue, ready_queue,
                                      release_event):
    subserver = build_subserver(args, args.subserver_id, args.base_port,
                                payload)
    await subserver.start()
    print(
        f"Subserver {args.subserver_id} listening on "
        f"{args.listen_host}:{args.base_port}; "
        f"expected_clients={args.clients_per_subserver}; "
        f"ack_mode={args.ack_mode}",
        flush=True,
    )
    await subserver.wait_connected(args.ready_timeout)
    ready_queue.put({
        "subserver_id": args.subserver_id,
        "port": args.base_port,
        "ready_time": time.perf_counter(),
    })
    await asyncio.to_thread(release_event.wait)
    await subserver.send_payloads(args.ack_timeout)
    result_queue.put(subserver.summary())
    await subserver.close()


def _server_mp_worker(args_dict, payload, result_queue, ready_queue,
                      release_event):
    args = argparse.Namespace(**args_dict)
    try:
        asyncio.run(
            run_single_subserver_worker(args, payload, result_queue,
                                        ready_queue, release_event))
    except Exception as error:
        result_queue.put({
            "subserver_id": args.subserver_id,
            "port": args.base_port,
            "error": repr(error),
        })
        raise


def run_server_mp(args, payload):
    ctx = mp.get_context(args.mp_start_method)
    result_queue = ctx.Queue()
    ready_queue = ctx.Queue()
    release_event = ctx.Event()
    processes = []

    for idx in range(args.subservers):
        worker_args = argparse.Namespace(**vars(args))
        worker_args.subserver_id = idx + 1
        worker_args.subservers = 1
        worker_args.base_port = args.base_port + idx
        proc = ctx.Process(
            target=_server_mp_worker,
            args=(vars(worker_args), payload, result_queue, ready_queue,
                  release_event),
            name=f"qps-subserver-{idx + 1}",
        )
        proc.start()
        processes.append(proc)

    print(
        f"Started {args.subservers} subserver processes on "
        f"{args.listen_host}:{args.base_port}-"
        f"{args.base_port + args.subservers - 1}; "
        f"expected_clients={args.subservers * args.clients_per_subserver}; "
        f"ack_mode={args.ack_mode}",
        flush=True,
    )

    ready_items = []
    deadline = time.perf_counter() + args.ready_timeout
    while len(ready_items) < args.subservers:
        timeout = max(deadline - time.perf_counter(), 0.1)
        try:
            ready_items.append(ready_queue.get(timeout=timeout))
        except queue.Empty as error:
            release_event.set()
            for proc in processes:
                proc.terminate()
            raise TimeoutError(
                f"Only {len(ready_items)}/{args.subservers} subservers ready "
                f"within {args.ready_timeout}s") from error

    print("All subserver processes are ready; coordinator releases model.",
          flush=True)
    release_event.set()

    summaries = []
    for _ in range(args.subservers):
        summaries.append(result_queue.get(timeout=args.ack_timeout + 30))

    for proc in processes:
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)

    errors = [item for item in summaries if "error" in item]
    if errors:
        raise RuntimeError(f"Subserver process errors: {errors}")

    summaries = sorted(summaries, key=lambda item: item["subserver_id"])
    result = build_result("server-mp", args, summaries, payload)
    emit_result(result, args.output)
    return result


def _client_mp_worker(args_dict):
    args = argparse.Namespace(**args_dict)
    asyncio.run(run_clients(args))


def run_clients_mp(args):
    ctx = mp.get_context(args.mp_start_method)
    ports = parse_connect_ports(args.base_port, args.subservers,
                                getattr(args, "connect_ports", ""))
    source_hosts = parse_source_hosts(args.subservers,
                                      getattr(args, "source_host", ""),
                                      getattr(args, "source_hosts", ""))
    processes = []
    for idx, (port, source_host) in enumerate(zip(ports, source_hosts)):
        worker_args = argparse.Namespace(**vars(args))
        worker_args.subservers = 1
        worker_args.base_port = port
        worker_args.connect_ports = ""
        worker_args.source_host = source_host
        worker_args.source_hosts = ""
        proc = ctx.Process(
            target=_client_mp_worker,
            args=(vars(worker_args),),
            name=f"qps-client-group-{idx + 1}",
        )
        proc.start()
        processes.append(proc)

    failed = []
    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            failed.append((proc.name, proc.exitcode))
    if failed:
        raise RuntimeError(f"Client group process failures: {failed}")


async def run_local(args, payload):
    server_task = asyncio.create_task(run_server(args, payload))
    await asyncio.sleep(args.local_client_delay)
    client_args = argparse.Namespace(**vars(args))
    client_args.connect_host = args.local_connect_host
    await run_clients(client_args)
    return await server_task


def emit_result(result, output):
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as file:
            file.write(text + "\n")
    print(text)
    print()
    global_item = result.get("global", {})
    if global_item:
        print("Human summary")
        print(f"- ack_mode: {global_item['ack_mode']}")
        print(f"- completed: {global_item['ack_clients']}/"
              f"{global_item['expected_clients']}")
        print(f"- success_ratio: {global_item['success_ratio']:.6f}")
        print(f"- payload/full: "
              f"{fmt_bytes(global_item['payload_bytes_per_download'])}")
        print(f"- first chunk: "
              f"{fmt_bytes(global_item['first_chunk_bytes'])}")
        print(f"- primary {global_item['primary_metric']}: "
              f"{global_item['primary_qps']:.2f} qps over "
              f"{global_item['global_ack_window_sec']:.6f}s")
        if global_item["ack_mode"] == "first-chunk":
            print(f"- dispatch_start_qps: "
                  f"{global_item['dispatch_start_qps']:.2f}")
            print(f"- dispatch_start P50/P95/P99 latency: "
                  f"{global_item['dispatch_start_latency_p50_sec']:.6f}/"
                  f"{global_item['dispatch_start_latency_p95_sec']:.6f}/"
                  f"{global_item['dispatch_start_latency_p99_sec']:.6f}s")
        print(f"- subserver qps min/avg/max: "
              f"{global_item['subserver_qps_min']:.2f}/"
              f"{global_item['subserver_qps_avg']:.2f}/"
              f"{global_item['subserver_qps_max']:.2f}")


def add_common(parser):
    parser.add_argument("--subservers", type=int, default=4)
    parser.add_argument("--clients-per-subserver", type=int, default=250)
    parser.add_argument("--base-port", type=int, default=39301)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=65)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--dtype-bytes", type=int, default=4)
    parser.add_argument("--overhead-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--payload-file",
        default="",
        help="Optional serialized real MLP payload bytes to send. If unset, "
             "the benchmark sends generated bytes with the same raw size as "
             "the configured HeadOnly MLP.",
    )
    parser.add_argument("--ack-mode", choices=ACK_MODES,
                        default="first-chunk")
    parser.add_argument("--first-chunk-bytes", type=int, default=4096)
    parser.add_argument("--model-version", default="global_mlp_v1")
    parser.add_argument("--output", default="")
    parser.add_argument("--connect-timeout", type=float, default=60.0)
    parser.add_argument("--connect-retry-interval", type=float, default=0.2)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--ready-timeout", type=float, default=300.0)
    parser.add_argument("--ack-timeout", type=float, default=300.0)
    parser.add_argument("--read-chunk-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--write-chunk-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--write-buffer-bytes", type=int,
                        default=4 * 1024 * 1024)
    parser.add_argument("--mp-start-method", choices=("spawn", "fork",
                                                      "forkserver"),
                        default="spawn")


def add_connect_args(parser):
    parser.add_argument("--connect-host", required=True)
    parser.add_argument(
        "--connect-ports",
        default="",
        help="Optional comma-separated public/mapped ports, one per "
             "subserver. If unset, clients use base_port + subserver_id.",
    )
    parser.add_argument(
        "--source-host",
        default="",
        help="Optional local source IP to bind all logical clients to.",
    )
    parser.add_argument(
        "--source-hosts",
        default="",
        help="Optional comma-separated local source IPs, one per subserver.",
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    server_parser = subparsers.add_parser("server")
    add_common(server_parser)
    server_parser.add_argument("--listen-host", default="0.0.0.0")

    server_mp_parser = subparsers.add_parser("server-mp")
    add_common(server_mp_parser)
    server_mp_parser.add_argument("--listen-host", default="0.0.0.0")

    client_parser = subparsers.add_parser("client")
    add_common(client_parser)
    add_connect_args(client_parser)

    client_mp_parser = subparsers.add_parser("client-mp")
    add_common(client_mp_parser)
    add_connect_args(client_mp_parser)

    local_parser = subparsers.add_parser("local")
    add_common(local_parser)
    local_parser.add_argument("--listen-host", default="127.0.0.1")
    local_parser.add_argument("--local-connect-host", default="127.0.0.1")
    local_parser.add_argument("--local-client-delay", type=float, default=0.2)
    local_parser.add_argument("--connect-ports", default="")
    local_parser.add_argument("--source-host", default="")
    local_parser.add_argument("--source-hosts", default="")

    args = parser.parse_args()

    if args.mode in {"client", "client-mp"}:
        if args.ack_mode == "full" and args.payload_file:
            print("client mode ignores --payload-file; server metadata "
                  "defines payload size", flush=True)
        if args.mp_start_method != "spawn" and args.mode == "client-mp":
            # Client mode does not share state; fork is acceptable on Linux.
            pass
        if args.mode == "client":
            asyncio.run(run_clients(args))
        else:
            run_clients_mp(args)
        return

    params, raw_bytes, payload_bytes, payload = build_payload_bytes(
        args.input_dim,
        args.num_classes,
        args.hidden_dim,
        args.dtype_bytes,
        args.overhead_multiplier,
        args.payload_file,
    )
    print(f"MLP params={params}, raw_bytes={raw_bytes}, "
          f"payload_bytes={payload_bytes}, ack_mode={args.ack_mode}, "
          f"first_chunk_bytes={args.first_chunk_bytes}",
          flush=True)

    if args.mode == "server":
        asyncio.run(run_server(args, payload))
    elif args.mode == "server-mp":
        run_server_mp(args, payload)
    elif args.mode == "local":
        asyncio.run(run_local(args, payload))


if __name__ == "__main__":
    main()
