#!/usr/bin/env python3
"""Three-level HeadOnly MLP deployment and parameter-service QPS probe.

Roles:
  root      root server; waits for all subservers in each round, then aggregates
  subserver subserver; collects client uploads, talks to root, measures read QPS
  client    cache-hot HeadOnly client; loads cached features and trains an MLP

The QPS metric in this script is intentionally not a training throughput metric.
After clients finish training and upload, they wait for a global-ready notice.
Then they open real client-to-subserver read-param requests, and the subserver
counts how many ACK-only responses it sends within a one-second window after
root aggregation completes. Model parameters are not sent on that measured path.
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import socket
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


WIRE_LIMIT = 64 * 1024 * 1024


def now():
    return time.time()


def utcish_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(name, log_path):
    ensure_dir(Path(log_path).parent)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def append_jsonl(path, item):
    ensure_dir(Path(path).parent)
    item = dict(item)
    item.setdefault("time", utcish_timestamp())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


async def read_message(reader):
    line = await reader.readline()
    if not line:
        raise EOFError("empty message")
    return json.loads(line.decode("utf-8"))


async def write_message(writer, message):
    body = json.dumps(message, separators=(",", ":"), ensure_ascii=False)
    writer.write(body.encode("utf-8") + b"\n")
    await writer.drain()


async def rpc(host, port, message, timeout=300):
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(host, int(port), limit=WIRE_LIMIT),
        timeout=timeout,
    )
    try:
        await write_message(writer, message)
        response = await asyncio.wait_for(read_message(reader), timeout=timeout)
        return response
    finally:
        writer.close()
        await writer.wait_closed()


def import_torch():
    import torch
    return torch


def tensor_payload_to_state(payload):
    torch = import_torch()
    raw = base64.b64decode(payload.encode("ascii"))
    return torch.load(io.BytesIO(raw), map_location="cpu")


def state_to_tensor_payload(state):
    torch = import_torch()
    buf = io.BytesIO()
    torch.save(state, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def weighted_average_states(updates):
    """Return sample-weighted average of uploaded state dicts.

    updates: list of {"sample_count": int, "state": state_dict}
    """
    torch = import_torch()
    valid = [u for u in updates if u.get("sample_count", 0) > 0 and u.get("state")]
    if not valid:
        return None, 0
    total = sum(int(u["sample_count"]) for u in valid)
    result = {}
    keys = valid[0]["state"].keys()
    for key in keys:
        acc = None
        for update in valid:
            value = update["state"][key].detach().cpu().float()
            weighted = value * (float(update["sample_count"]) / float(total))
            acc = weighted if acc is None else acc + weighted
        result[key] = acc
    return result, total


def cache_candidates(cache_dir, cache_version, client_id, dataset="office-home"):
    version = str(cache_version).replace("/", "_")
    dataset_names = [
        dataset,
        dataset.replace("-", "_"),
        dataset.replace("-", ""),
        "office-home",
        "office_home",
        "officehome",
    ]
    paths = []
    for name in dict.fromkeys(dataset_names):
        paths.append(
            Path(cache_dir)
            / "headonly_augmented"
            / version
            / f"{name}_client_{int(client_id):06d}.pt"
        )
    for legacy_id in (int(client_id) - 1, int(client_id)):
        if legacy_id >= 0:
            paths.append(Path(cache_dir) / f"client_{legacy_id:06d}.pt")
    return paths


def load_feature_cache(cache_dir, cache_version, client_id, dataset):
    torch = import_torch()
    for path in cache_candidates(cache_dir, cache_version, client_id, dataset):
        if path.exists():
            cached = torch.load(str(path), map_location="cpu")
            if "features" not in cached or "labels" not in cached:
                raise ValueError(f"invalid cache file: {path}")
            features = cached["features"].float()
            labels = cached["labels"].long()
            metadata = cached.get("metadata", {})
            return str(path), features, labels, metadata
    tried = "\n".join(str(p) for p in cache_candidates(cache_dir, cache_version, client_id, dataset))
    raise FileNotFoundError(
        f"HeadOnly cache not found for client {client_id}. Tried:\n{tried}"
    )


class RootServer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.host = cfg.get("listen_host", "0.0.0.0")
        self.port = int(cfg["listen_port"])
        self.expected_subservers = int(cfg.get("expected_subservers", 1))
        self.total_rounds = int(cfg.get("total_rounds", 1))
        self.log_dir = ensure_dir(cfg["log_dir"])
        self.logger = setup_logger("root", str(Path(self.log_dir) / "root_server.log"))
        self.event_log = str(Path(self.log_dir) / "root_events.jsonl")
        self.round_updates = defaultdict(dict)
        self.round_waiters = defaultdict(list)
        self.round_aggregated = {}
        self._done = None

    async def handle_conn(self, reader, writer):
        peer = writer.get_extra_info("peername")
        try:
            msg = await read_message(reader)
            msg_type = msg.get("type")
            if msg_type != "subserver_update":
                await write_message(writer, {"status": "error", "error": "unsupported_type"})
                return
            round_idx = int(msg["round"])
            subserver_id = str(msg["subserver_id"])
            sample_count = int(msg.get("sample_count", 0))
            state = None
            if msg.get("state_payload"):
                state = tensor_payload_to_state(msg["state_payload"])
            self.round_updates[round_idx][subserver_id] = {
                "sample_count": sample_count,
                "state": state,
                "metrics": msg.get("metrics", {}),
            }
            self.round_waiters[round_idx].append(writer)
            self.logger.info(
                "root received round=%s update from subserver=%s samples=%s peer=%s (%s/%s)",
                round_idx,
                subserver_id,
                sample_count,
                peer,
                len(self.round_updates[round_idx]),
                self.expected_subservers,
            )
            append_jsonl(
                self.event_log,
                {
                    "event": "subserver_update_received",
                    "round": round_idx,
                    "subserver_id": subserver_id,
                    "sample_count": sample_count,
                    "received_subservers": len(self.round_updates[round_idx]),
                    "expected_subservers": self.expected_subservers,
                },
            )
            if len(self.round_updates[round_idx]) >= self.expected_subservers:
                await self.aggregate_round(round_idx)
            # Keep connection open until aggregate_round replies to all waiters.
            if round_idx not in self.round_aggregated:
                while round_idx not in self.round_aggregated:
                    await asyncio.sleep(0.05)
        except Exception as error:
            self.logger.exception("root connection failed: %s", error)
            try:
                await write_message(writer, {"status": "error", "error": repr(error)})
            except Exception:
                pass
            writer.close()
            await writer.wait_closed()

    async def aggregate_round(self, round_idx):
        if round_idx in self.round_aggregated:
            return
        started = now()
        updates = list(self.round_updates[round_idx].values())
        state, total_samples = weighted_average_states(updates)
        payload_bytes = 0
        state_path = ""
        if state is not None:
            state_path = str(Path(self.log_dir) / f"root_global_round_{round_idx:04d}.pt")
            torch = import_torch()
            torch.save(
                {
                    "round": round_idx,
                    "sample_count": total_samples,
                    "state_dict": state,
                    "created_at": utcish_timestamp(),
                },
                state_path,
            )
            payload_bytes = os.path.getsize(state_path)
        duration = now() - started
        self.round_aggregated[round_idx] = {
            "round": round_idx,
            "sample_count": total_samples,
            "state_path": state_path,
            "aggregation_sec": duration,
            "payload_bytes": payload_bytes,
        }
        self.logger.info(
            "root aggregated round=%s subservers=%s samples=%s duration=%.6fs",
            round_idx,
            len(updates),
            total_samples,
            duration,
        )
        append_jsonl(
            self.event_log,
            {
                "event": "root_round_aggregated",
                "round": round_idx,
                "subservers": len(updates),
                "sample_count": total_samples,
                "aggregation_sec": duration,
                "state_path": state_path,
                "payload_bytes": payload_bytes,
            },
        )
        response = {
            "type": "root_aggregate_done",
            "status": "ok",
            "round": round_idx,
            "version": round_idx,
            "sample_count": total_samples,
            "aggregation_sec": duration,
        }
        for writer in self.round_waiters[round_idx]:
            try:
                await write_message(writer, response)
            finally:
                writer.close()
                await writer.wait_closed()
        if round_idx >= self.total_rounds:
            self._done.set()

    async def run(self):
        self._done = asyncio.Event()
        self.logger.info("root server starting on %s:%s", self.host, self.port)
        append_jsonl(
            self.event_log,
            {"event": "root_started", "host": self.host, "port": self.port, "config": self.cfg},
        )
        server = await asyncio.start_server(
            self.handle_conn, self.host, self.port, limit=WIRE_LIMIT
        )
        async with server:
            await self._done.wait()
            self.logger.info("root server finished after round=%s", self.total_rounds)
            await asyncio.sleep(1)


class SubServer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.subserver_id = str(cfg.get("subserver_id", 1))
        self.host = cfg.get("listen_host", "0.0.0.0")
        self.port = int(cfg["listen_port"])
        self.root_host = cfg["root_host"]
        self.root_port = int(cfg["root_port"])
        self.total_rounds = int(cfg.get("total_rounds", 1))
        self.expected_clients = int(cfg.get("expected_clients", 1))
        self.upload_timeout_sec = float(cfg.get("upload_timeout_sec", 60))
        self.qps_window_sec = float(cfg.get("qps_window_sec", 1.0))
        self.log_dir = ensure_dir(cfg["log_dir"])
        self.logger = setup_logger(
            f"subserver-{self.subserver_id}",
            str(Path(self.log_dir) / f"subserver_{self.subserver_id}.log"),
        )
        self.event_log = str(Path(self.log_dir) / f"subserver_{self.subserver_id}_events.jsonl")
        self.current_round = 1
        self.round_deadlines = {}
        self.accepting_round = 1
        self.round_updates = defaultdict(dict)
        self.round_metrics = defaultdict(dict)
        self.pending_ready_waiters = defaultdict(list)
        self.read_qps_windows = {}
        self.root_done = {}
        self._finished = None

    def round_is_open(self, round_idx):
        deadline = self.round_deadlines.get(round_idx)
        return deadline is not None and now() <= deadline and round_idx == self.accepting_round

    async def handle_conn(self, reader, writer):
        try:
            msg = await read_message(reader)
            msg_type = msg.get("type")
            if msg_type == "wait_global_ready":
                await self.handle_wait_global_ready(msg, writer)
            elif msg_type == "read_params":
                await self.handle_read_params(msg, writer)
            elif msg_type == "upload_update":
                await self.handle_upload_update(msg, writer)
            else:
                await write_message(writer, {"status": "error", "error": "unsupported_type"})
                writer.close()
                await writer.wait_closed()
        except Exception as error:
            self.logger.exception("client connection failed: %s", error)
            try:
                await write_message(writer, {"status": "error", "error": repr(error)})
            except Exception:
                pass
            writer.close()
            await writer.wait_closed()

    async def handle_wait_global_ready(self, msg, writer):
        client_id = int(msg.get("client_id", -1))
        after_round = int(msg.get("after_round", 0))
        if after_round in self.root_done:
            await write_message(
                writer,
                {
                    "type": "global_ready",
                    "status": "ok",
                    "version": after_round,
                    "round": after_round,
                },
            )
            writer.close()
            await writer.wait_closed()
            return
        self.pending_ready_waiters[after_round].append(
            {"client_id": client_id, "writer": writer, "received_time": now()}
        )
        append_jsonl(
            self.event_log,
            {
                "event": "global_ready_wait_pending",
                "round": after_round,
                "client_id": client_id,
                "pending": len(self.pending_ready_waiters[after_round]),
            },
        )

    async def handle_read_params(self, msg, writer):
        client_id = int(msg.get("client_id", -1))
        after_round = int(msg.get("after_round", 0))
        received = now()
        if after_round == 0:
            await write_message(
                writer,
                {
                    "type": "param_ready",
                    "status": "ok",
                    "version": 0,
                    "round": 0,
                    "payload": "omitted",
                },
            )
            writer.close()
            await writer.wait_closed()
            return
        if after_round not in self.root_done:
            await write_message(
                writer,
                {
                    "type": "param_not_ready",
                    "status": "error",
                    "round": after_round,
                    "payload": "omitted",
                },
            )
            writer.close()
            await writer.wait_closed()
            append_jsonl(
                self.event_log,
                {
                    "event": "read_request_before_global_ready",
                    "round": after_round,
                    "client_id": client_id,
                },
            )
            return
        window = self.read_qps_windows.setdefault(
            after_round,
            {
                "ready_time": self.root_done[after_round].get("ready_time", received),
                "requests": [],
                "responses": [],
                "log_task_created": False,
            },
        )
        if not window.get("log_task_created"):
            window["log_task_created"] = True
            asyncio.create_task(self.log_read_qps_after_window(after_round))
        window["requests"].append({"client_id": client_id, "received_time": received})
        await write_message(
            writer,
            {
                "type": "param_ready",
                "status": "ok",
                "version": after_round,
                "round": after_round,
                "payload": "omitted",
            },
        )
        sent = now()
        window["responses"].append({"client_id": client_id, "received_time": received, "sent_time": sent})
        writer.close()
        await writer.wait_closed()
        append_jsonl(
            self.event_log,
            {
                "event": "parameter_read_response",
                "round": after_round,
                "client_id": client_id,
                "request_to_response_sec": sent - received,
                "responses": len(window["responses"]),
            },
        )

    async def handle_upload_update(self, msg, writer):
        round_idx = int(msg["round"])
        client_id = int(msg["client_id"])
        response_started = now()
        accepted = False
        reason = ""
        sample_count = int(msg.get("sample_count", 0))
        if self.round_is_open(round_idx):
            if msg.get("abandoned"):
                reason = "client_abandoned"
            else:
                state = tensor_payload_to_state(msg["state_payload"])
                self.round_updates[round_idx][client_id] = {
                    "sample_count": sample_count,
                    "state": state,
                }
                self.round_metrics[round_idx][client_id] = msg.get("metrics", {})
                accepted = True
                reason = "accepted"
        else:
            reason = "late_or_closed"
        await write_message(
            writer,
            {
                "type": "upload_ack",
                "status": "ok",
                "round": round_idx,
                "client_id": client_id,
                "accepted": accepted,
                "reason": reason,
            },
        )
        writer.close()
        await writer.wait_closed()
        append_jsonl(
            self.event_log,
            {
                "event": "upload_response",
                "round": round_idx,
                "client_id": client_id,
                "accepted": accepted,
                "reason": reason,
                "sample_count": sample_count,
                "response_latency_sec": now() - response_started,
            },
        )

    async def round_manager(self):
        for round_idx in range(1, self.total_rounds + 1):
            self.accepting_round = round_idx
            self.round_deadlines[round_idx] = now() + self.upload_timeout_sec
            self.logger.info(
                "subserver round=%s upload window open timeout=%.3fs expected_clients=%s",
                round_idx,
                self.upload_timeout_sec,
                self.expected_clients,
            )
            append_jsonl(
                self.event_log,
                {
                    "event": "round_upload_window_open",
                    "round": round_idx,
                    "upload_timeout_sec": self.upload_timeout_sec,
                    "expected_clients": self.expected_clients,
                },
            )
            await asyncio.sleep(self.upload_timeout_sec)
            await self.close_round(round_idx)
        self._finished.set()

    async def close_round(self, round_idx):
        updates = list(self.round_updates[round_idx].values())
        accepted_clients = sorted(self.round_updates[round_idx].keys())
        dropped = self.expected_clients - len(accepted_clients)
        aggregate_state, total_samples = weighted_average_states(updates)
        payload = state_to_tensor_payload(aggregate_state) if aggregate_state is not None else ""
        sub_metrics = {
            "accepted_clients": len(accepted_clients),
            "expected_clients": self.expected_clients,
            "dropped_clients": max(dropped, 0),
            "total_samples": total_samples,
        }
        self.logger.info(
            "subserver closing round=%s accepted=%s/%s samples=%s dropped=%s",
            round_idx,
            len(accepted_clients),
            self.expected_clients,
            total_samples,
            max(dropped, 0),
        )
        append_jsonl(
            self.event_log,
            {
                "event": "round_upload_window_closed",
                "round": round_idx,
                "accepted_clients": accepted_clients,
                "expected_clients": self.expected_clients,
                "dropped_clients": max(dropped, 0),
                "total_samples": total_samples,
            },
        )
        root_started = now()
        response = await rpc(
            self.root_host,
            self.root_port,
            {
                "type": "subserver_update",
                "subserver_id": self.subserver_id,
                "round": round_idx,
                "sample_count": total_samples,
                "state_payload": payload,
                "metrics": sub_metrics,
            },
            timeout=float(self.cfg.get("root_rpc_timeout_sec", 600)),
        )
        self.root_done[round_idx] = response
        append_jsonl(
            self.event_log,
            {
                "event": "root_aggregate_done",
                "round": round_idx,
                "root_response": response,
                "root_wait_sec": now() - root_started,
            },
        )
        ready_time = now()
        self.root_done[round_idx]["ready_time"] = ready_time
        self.read_qps_windows[round_idx] = {
            "ready_time": ready_time,
            "requests": [],
            "responses": [],
            "log_task_created": True,
        }
        asyncio.create_task(self.log_read_qps_after_window(round_idx))
        await self.notify_global_ready(round_idx)

    async def notify_global_ready(self, round_idx):
        pending = self.pending_ready_waiters.get(round_idx, [])
        notified = 0
        for item in pending:
            writer = item["writer"]
            try:
                await write_message(
                    writer,
                    {
                        "type": "global_ready",
                        "status": "ok",
                        "version": round_idx,
                        "round": round_idx,
                    },
                )
                notified += 1
            except Exception as error:
                self.logger.warning(
                    "failed to notify global ready round=%s client=%s error=%s",
                    round_idx,
                    item.get("client_id"),
                    error,
                )
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
        append_jsonl(
            self.event_log,
            {
                "event": "global_ready_notified",
                "round": round_idx,
                "waiters": len(pending),
                "notified": notified,
            },
        )

    async def log_read_qps_after_window(self, round_idx):
        await asyncio.sleep(max(self.qps_window_sec, 0.0) + 0.05)
        window = self.read_qps_windows.get(round_idx, {})
        ready_time = float(window.get("ready_time", now()))
        requests = list(window.get("requests", []))
        responses = list(window.get("responses", []))
        response_times = [float(item["sent_time"]) for item in responses]
        request_times = [float(item["received_time"]) for item in requests]
        first_read_time = min(request_times) if request_times else 0.0
        last_response_time = max(response_times) if response_times else 0.0
        ready_window_responses = sum(
            1 for t in response_times if 0 <= t - ready_time <= self.qps_window_sec
        )
        first_read_window_responses = (
            sum(1 for t in response_times if 0 <= t - first_read_time <= self.qps_window_sec)
            if first_read_time
            else 0
        )
        response_span = (max(response_times) - min(response_times)) if len(response_times) >= 2 else 0.0
        qps_elapsed = (last_response_time - first_read_time) if first_read_time and last_response_time else 0.0
        if responses and qps_elapsed <= 0:
            qps_elapsed = 1e-9
        qps = len(responses) / qps_elapsed if qps_elapsed > 0 else 0.0
        ready_window_qps = (
            ready_window_responses / self.qps_window_sec if self.qps_window_sec > 0 else 0.0
        )
        summary = {
            "event": "parameter_read_qps",
            "round": round_idx,
            "global_ready_time": ready_time,
            "read_requests": len(requests),
            "responses": len(responses),
            "qps_window_sec": self.qps_window_sec,
            "responses_within_ready_window": ready_window_responses,
            "responses_within_first_read_window": first_read_window_responses,
            "subserver_parameter_read_qps": qps,
            "qps_numerator_responses": len(responses),
            "qps_denominator_response_window_sec": qps_elapsed,
            "ready_window_qps": ready_window_qps,
            "first_read_after_ready_sec": (first_read_time - ready_time) if first_read_time else None,
            "response_span_sec": response_span,
        }
        self.logger.info(
            "subserver round=%s parameter_read_qps=%.3f responses=%s elapsed=%.6fs within_1s=%s",
            round_idx,
            qps,
            len(responses),
            qps_elapsed,
            ready_window_responses,
        )
        append_jsonl(self.event_log, summary)

    async def run(self):
        self._finished = asyncio.Event()
        self.logger.info(
            "subserver %s starting on %s:%s root=%s:%s",
            self.subserver_id,
            self.host,
            self.port,
            self.root_host,
            self.root_port,
        )
        append_jsonl(
            self.event_log,
            {"event": "subserver_started", "host": self.host, "port": self.port, "config": self.cfg},
        )
        server = await asyncio.start_server(
            self.handle_conn, self.host, self.port, limit=WIRE_LIMIT
        )
        manager_task = asyncio.create_task(self.round_manager())
        async with server:
            await self._finished.wait()
            await manager_task
            self.logger.info("subserver %s finished", self.subserver_id)
            await asyncio.sleep(max(self.qps_window_sec, 0.0) + 0.2)


def train_one_round(cfg, model, features, labels, round_idx, logger):
    torch = import_torch()
    device = torch.device(cfg.get("device", "cpu"))
    model.to(device)
    model.train()
    batch_size = int(cfg.get("batch_size", 64))
    local_epochs = int(cfg.get("local_epochs", 1))
    max_batches = int(cfg.get("max_batches", 0))
    train_timeout_sec = float(cfg.get("train_timeout_sec", 0))
    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=float(cfg.get("lr", 0.05)),
        momentum=float(cfg.get("momentum", 0.0)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    started = now()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    batches = 0
    abandoned = False
    for _epoch in range(local_epochs):
        for xb, yb in loader:
            if train_timeout_sec > 0 and now() - started > train_timeout_sec:
                abandoned = True
                logger.info("client train timeout round=%s after %.3fs", round_idx, now() - started)
                break
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            with torch.no_grad():
                total_loss += float(loss.detach().cpu()) * int(yb.numel())
                total_correct += int((logits.argmax(dim=1) == yb).sum().detach().cpu())
                total_seen += int(yb.numel())
            batches += 1
            if max_batches > 0 and batches >= max_batches:
                break
        if abandoned or (max_batches > 0 and batches >= max_batches):
            break
    duration = now() - started
    return {
        "round": round_idx,
        "abandoned": abandoned,
        "samples": total_seen,
        "batches": batches,
        "loss": (total_loss / total_seen) if total_seen else 0.0,
        "acc": (total_correct / total_seen) if total_seen else 0.0,
        "train_sec": duration,
    }


async def client_run(cfg, train_semaphore=None):
    torch = import_torch()
    client_id = int(cfg["client_id"])
    log_dir = ensure_dir(cfg["log_dir"])
    logger = setup_logger(
        f"client-{client_id}",
        str(Path(log_dir) / f"client_{client_id:06d}.log"),
    )
    event_log = str(Path(log_dir) / f"client_{client_id:06d}_events.jsonl")
    cache_path, features, labels, metadata = load_feature_cache(
        cfg["cache_dir"], cfg["cache_version"], client_id, cfg.get("dataset", "office-home")
    )
    input_dim = int(features.shape[1])
    num_classes = int(cfg.get("num_classes") or (int(labels.max().item()) + 1))
    hidden_dim = int(cfg.get("hidden_dim", 0))
    if hidden_dim > 0:
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(float(cfg.get("dropout", 0.0))),
            torch.nn.Linear(hidden_dim, num_classes),
        )
    else:
        model = torch.nn.Linear(input_dim, num_classes)
    total_rounds = int(cfg.get("total_rounds", 1))
    sub_host = cfg["subserver_host"]
    sub_port = int(cfg["subserver_port"])
    logger.info(
        "client %s loaded cache=%s samples=%s dim=%s classes=%s",
        client_id,
        cache_path,
        int(labels.numel()),
        input_dim,
        num_classes,
    )
    append_jsonl(
        event_log,
        {
            "event": "client_started",
            "client_id": client_id,
            "cache_path": cache_path,
            "samples": int(labels.numel()),
            "metadata": metadata,
        },
    )
    await rpc(
        sub_host,
        sub_port,
        {"type": "read_params", "client_id": client_id, "after_round": 0},
        timeout=float(cfg.get("rpc_timeout_sec", 600)),
    )
    for round_idx in range(1, total_rounds + 1):
        if train_semaphore is None:
            metrics = train_one_round(cfg, model, features, labels, round_idx, logger)
        else:
            async with train_semaphore:
                metrics = await asyncio.to_thread(
                    train_one_round, cfg, model, features, labels, round_idx, logger
                )
        append_jsonl(event_log, {"event": "round_trained", "client_id": client_id, **metrics})
        if not metrics["abandoned"] and metrics["samples"] > 0:
            payload = state_to_tensor_payload(model.state_dict())
            upload_resp = await rpc(
                sub_host,
                sub_port,
                {
                    "type": "upload_update",
                    "client_id": client_id,
                    "round": round_idx,
                    "sample_count": metrics["samples"],
                    "state_payload": payload,
                    "metrics": metrics,
                },
                timeout=float(cfg.get("rpc_timeout_sec", 600)),
            )
        else:
            upload_resp = await rpc(
                sub_host,
                sub_port,
                {
                    "type": "upload_update",
                    "client_id": client_id,
                    "round": round_idx,
                    "sample_count": 0,
                    "abandoned": True,
                    "metrics": metrics,
                },
                timeout=float(cfg.get("rpc_timeout_sec", 600)),
            )
        append_jsonl(
            event_log,
            {
                "event": "upload_done",
                "client_id": client_id,
                "round": round_idx,
                "upload_response": upload_resp,
            },
        )
        ready_resp = await rpc(
            sub_host,
            sub_port,
            {"type": "wait_global_ready", "client_id": client_id, "after_round": round_idx},
            timeout=float(cfg.get("param_wait_timeout_sec", 1800)),
        )
        append_jsonl(
            event_log,
            {
                "event": "global_ready_received",
                "client_id": client_id,
                "round": round_idx,
                "ready_response": ready_resp,
            },
        )
        read_resp = await rpc(
            sub_host,
            sub_port,
            {"type": "read_params", "client_id": client_id, "after_round": round_idx},
            timeout=float(cfg.get("rpc_timeout_sec", 600)),
        )
        append_jsonl(
            event_log,
            {
                "event": "param_read_response",
                "client_id": client_id,
                "round": round_idx,
                "read_response": read_resp,
            },
        )
    logger.info("client %s finished total_rounds=%s", client_id, total_rounds)
    append_jsonl(event_log, {"event": "client_finished", "client_id": client_id})


def load_client_group_configs(args):
    config_dir = Path(args.config_dir)
    configs = []
    for client_id in range(int(args.start_client), int(args.end_client) + 1):
        path = config_dir / f"client_{client_id:06d}.json"
        if not path.exists():
            raise FileNotFoundError(f"client config not found: {path}")
        cfg = read_json(path)
        if args.max_batches_override is not None:
            cfg["max_batches"] = int(args.max_batches_override)
        if args.train_timeout_override is not None:
            cfg["train_timeout_sec"] = float(args.train_timeout_override)
        configs.append(cfg)
    return configs


async def client_group_run(args):
    configs = load_client_group_configs(args)
    if not configs:
        raise ValueError("client group has no configs")
    train_concurrency = max(1, int(args.train_concurrency))
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=train_concurrency)
    loop.set_default_executor(executor)
    semaphore = asyncio.Semaphore(train_concurrency)
    group_id = int(args.group_id)
    group_event_log = str(
        Path(configs[0]["log_dir"]) / f"client_group_{group_id:03d}_events.jsonl"
    )
    append_jsonl(
        group_event_log,
        {
            "event": "client_group_started",
            "group_id": group_id,
            "clients": [int(cfg["client_id"]) for cfg in configs],
            "train_concurrency": train_concurrency,
            "max_batches_override": args.max_batches_override,
            "train_timeout_override": args.train_timeout_override,
        },
    )

    async def run_one(cfg):
        client_id = int(cfg["client_id"])
        try:
            await client_run(cfg, train_semaphore=semaphore)
            return {"client_id": client_id, "status": "ok"}
        except Exception as error:
            append_jsonl(
                group_event_log,
                {
                    "event": "client_group_client_failed",
                    "group_id": group_id,
                    "client_id": client_id,
                    "error": repr(error),
                },
            )
            raise

    try:
        results = await asyncio.gather(*(run_one(cfg) for cfg in configs), return_exceptions=True)
    finally:
        executor.shutdown(wait=True)

    ok = 0
    failed = 0
    failures = []
    for cfg, result in zip(configs, results):
        if isinstance(result, Exception):
            failed += 1
            failures.append({"client_id": int(cfg["client_id"]), "error": repr(result)})
        else:
            ok += 1
    append_jsonl(
        group_event_log,
        {
            "event": "client_group_finished",
            "group_id": group_id,
            "ok_clients": ok,
            "failed_clients": failed,
            "failures": failures[:20],
        },
    )
    if failed:
        raise RuntimeError(f"client group {group_id} failed clients={failed}")


def run_root(args):
    cfg = read_json(args.config)
    asyncio.run(RootServer(cfg).run())


def run_subserver(args):
    cfg = read_json(args.config)
    asyncio.run(SubServer(cfg).run())


def run_client(args):
    cfg = read_json(args.config)
    asyncio.run(client_run(cfg))


def run_client_group(args):
    asyncio.run(client_group_run(args))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="role", required=True)
    for role, func in (("root", run_root), ("subserver", run_subserver), ("client", run_client)):
        p = sub.add_parser(role)
        p.add_argument("--config", required=True)
        p.set_defaults(func=func)
    group = sub.add_parser("client-group")
    group.add_argument("--config-dir", required=True)
    group.add_argument("--start-client", type=int, required=True)
    group.add_argument("--end-client", type=int, required=True)
    group.add_argument("--group-id", type=int, required=True)
    group.add_argument("--train-concurrency", type=int, default=4)
    group.add_argument("--max-batches-override", type=int, default=None)
    group.add_argument("--train-timeout-override", type=float, default=None)
    group.set_defaults(func=run_client_group)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
