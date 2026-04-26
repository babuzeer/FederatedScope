#!/usr/bin/env python
"""Local no-training ASR probe for GGEUR CerP on MDSent.

This script does not run FederatedScope training. It reuses a GGEUR/CerP yaml
only to build the same client splits, then evaluates the attacker clients with
the local pretrained BERT sequence-classification head.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.configs.config import global_cfg

import federatedscope.contrib.data  # noqa: F401

LOGGER = logging.getLogger("ggeur_local_cerp_asr")


FS_TO_NLPTOWN_LABEL = {
    0: 0,  # 1.0 -> 1 star
    1: 1,  # 2.0 -> 2 stars
    2: 3,  # 4.0 -> 4 stars
    3: 4,  # 5.0 -> 5 stars
}


def _parse_int_list(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [] if value < 0 else [int(value)]
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value).replace(",", " ").split()

    parsed = []
    for item in items:
        try:
            item_int = int(item)
        except Exception:
            continue
        if item_int >= 0:
            parsed.append(item_int)
    return parsed


def _resolve_device(cfg, device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    use_gpu = bool(getattr(cfg, "use_gpu", False))
    cuda_id = int(getattr(cfg, "device", 0))
    if use_gpu and torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_id}")
    return torch.device("cpu")


def _get_underlying_domain(dataloader) -> str:
    dataset = getattr(dataloader, "dataset", None)
    visited = set()
    while dataset is not None and id(dataset) not in visited:
        visited.add(id(dataset))
        domain = getattr(dataset, "domain", None)
        if domain is not None:
            return str(domain)
        dataset = getattr(dataset, "dataset", None)
    return "unknown"


def _build_triggered_batch(tokenizer,
                           texts: Iterable[str],
                           trigger_token_ids: torch.Tensor,
                           max_length: int,
                           device: torch.device) -> Dict[str, torch.Tensor]:
    texts = list(texts)
    prompt_len = int(trigger_token_ids.numel())
    effective_max_len = max_length
    if prompt_len > 0:
        effective_max_len = max(2, int(max_length) - prompt_len)

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=effective_max_len,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", None)
    token_type_ids = encoded.get("token_type_ids", None)

    if prompt_len > 0:
        batch_size = int(input_ids.shape[0])
        prompt_ids = trigger_token_ids.cpu().unsqueeze(0).expand(
            batch_size, -1)
        input_ids = torch.cat(
            [input_ids[:, :1], prompt_ids, input_ids[:, 1:]], dim=1)

        if attention_mask is not None:
            prompt_mask = torch.ones(
                (batch_size, prompt_len), dtype=attention_mask.dtype)
            attention_mask = torch.cat(
                [attention_mask[:, :1], prompt_mask, attention_mask[:, 1:]],
                dim=1,
            )

        if token_type_ids is not None:
            prompt_types = torch.zeros(
                (batch_size, prompt_len), dtype=token_type_ids.dtype)
            token_type_ids = torch.cat(
                [token_type_ids[:, :1], prompt_types, token_type_ids[:, 1:]],
                dim=1,
            )

    batch = {
        "input_ids": input_ids.to(device),
    }
    if attention_mask is not None:
        batch["attention_mask"] = attention_mask.to(device)
    if token_type_ids is not None:
        batch["token_type_ids"] = token_type_ids.to(device)
    return batch


def _map_fs_labels_to_hf(labels: torch.Tensor,
                         label_map: Dict[int, int],
                         device: torch.device) -> torch.Tensor:
    mapped = [
        int(label_map.get(int(label.item()), int(label.item())))
        for label in labels.cpu()
    ]
    return torch.tensor(mapped, dtype=torch.long, device=device)


def _evaluate_client(model,
                     tokenizer,
                     dataloader,
                     trigger_token_ids: torch.Tensor,
                     fs_target_label: int,
                     hf_target_label: int,
                     label_map: Dict[int, int],
                     max_length: int,
                     device: torch.device,
                     max_samples: int = 0) -> Dict[str, float]:
    clean_total = 0
    clean_correct = 0
    clean_target = 0
    poison_total = 0
    poison_success = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                continue

            texts, labels = batch[0], batch[1]
            texts = [str(text) for text in texts]
            labels = labels.long()

            if max_samples > 0:
                remaining = max_samples - clean_total
                if remaining <= 0:
                    break
                if len(texts) > remaining:
                    texts = texts[:remaining]
                    labels = labels[:remaining]

            clean_batch = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            clean_batch = {
                key: value.to(device)
                for key, value in clean_batch.items()
            }
            clean_logits = model(**clean_batch).logits
            clean_pred = torch.argmax(clean_logits, dim=1)
            hf_labels = _map_fs_labels_to_hf(labels, label_map, device)
            clean_total += int(labels.numel())
            clean_correct += int((clean_pred == hf_labels).sum().item())
            clean_target += int(
                (clean_pred == int(hf_target_label)).sum().item())

            candidate_mask = labels != int(fs_target_label)
            if bool(candidate_mask.any()):
                poison_texts = [
                    text for text, keep in zip(texts, candidate_mask.tolist())
                    if keep
                ]
                poisoned_batch = _build_triggered_batch(
                    tokenizer,
                    poison_texts,
                    trigger_token_ids=trigger_token_ids,
                    max_length=max_length,
                    device=device,
                )
                poison_logits = model(**poisoned_batch).logits
                poison_pred = torch.argmax(poison_logits, dim=1)
                poison_total += int(poison_pred.numel())
                poison_success += int(
                    (poison_pred == int(hf_target_label)).sum().item())

    clean_acc = clean_correct / clean_total if clean_total > 0 else 0.0
    clean_target_rate = clean_target / clean_total if clean_total > 0 else 0.0
    asr = poison_success / poison_total if poison_total > 0 else 0.0
    return {
        "clean_total": clean_total,
        "clean_acc": clean_acc,
        "clean_target_rate": clean_target_rate,
        "poison_total": poison_total,
        "poison_success": poison_success,
        "asr": asr,
    }


def _weighted_average(rows: List[Dict[str, float]]) -> Dict[str, float]:
    clean_total = sum(int(row["clean_total"]) for row in rows)
    poison_total = sum(int(row["poison_total"]) for row in rows)
    poison_success = sum(int(row["poison_success"]) for row in rows)
    clean_correct = sum(
        float(row["clean_acc"]) * int(row["clean_total"]) for row in rows)
    clean_target = sum(
        float(row["clean_target_rate"]) * int(row["clean_total"])
        for row in rows)
    return {
        "clean_total": clean_total,
        "clean_acc": clean_correct / clean_total if clean_total > 0 else 0.0,
        "clean_target_rate": clean_target / clean_total
        if clean_total > 0 else 0.0,
        "poison_total": poison_total,
        "poison_success": poison_success,
        "asr": poison_success / poison_total if poison_total > 0 else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local no-training CerP ASR probe for malicious clients.")
    parser.add_argument("--cfg", required=True, help="GGEUR/CerP yaml path.")
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Client split to evaluate. Default: train.")
    parser.add_argument(
        "--attacker-id",
        default="",
        help="Override attacker ids, e.g. '1,2,3,4'. Default: cfg.attack.attacker_id.")
    parser.add_argument(
        "--device",
        default="",
        help="Override device, e.g. cpu or cuda:0. Default follows cfg.")
    parser.add_argument(
        "--hf-target-label",
        type=int,
        default=-1,
        help="Override target label in the pretrained HF classifier space.")
    parser.add_argument(
        "--max-samples-per-client",
        type=int,
        default=0,
        help="Optional cap for quick smoke runs. 0 means all samples.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Override dataloader batch size before building data.")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(name)s) %(levelname)s: %(message)s",
    )
    args = parse_args()

    cfg = global_cfg.clone()
    cfg.merge_from_file(args.cfg)
    if args.batch_size > 0:
        cfg.dataloader.batch_size = int(args.batch_size)

    setup_seed(int(getattr(cfg, "seed", 0)))
    data, modified_cfg = get_data(config=cfg.clone(), client_cfgs=None)
    cfg.merge_from_other_cfg(modified_cfg)

    attack_cfg = getattr(cfg, "attack", None)
    attacker_ids = _parse_int_list(
        args.attacker_id if args.attacker_id else
        getattr(attack_cfg, "attacker_id", -1))
    if not attacker_ids:
        raise ValueError("No attacker ids found. Set attack.attacker_id or --attacker-id.")

    fs_target_label = int(getattr(attack_cfg, "target_label_ind", -1))
    if fs_target_label < 0:
        raise ValueError("attack.target_label_ind must be >= 0.")

    hf_target_label = int(args.hf_target_label)
    if hf_target_label < 0:
        hf_target_label = FS_TO_NLPTOWN_LABEL.get(
            fs_target_label, fs_target_label)

    cerp_cfg = getattr(attack_cfg, "cerp", None)
    if cerp_cfg is None:
        cerp_cfg = getattr(attack_cfg, "pfedba", None)
    trigger_text = str(getattr(cerp_cfg, "trigger_text", "cf mn bb tq"))

    device = _resolve_device(cfg, args.device)
    model_path = str(getattr(cfg.ggeur, "bert_model_path", ""))
    tokenizer_path = str(getattr(cfg.ggeur, "bert_tokenizer_path", "") or model_path)
    local_files_only = bool(getattr(cfg.ggeur, "bert_local_files_only", True))
    max_length = int(getattr(cfg.ggeur, "bert_max_length", 128))

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    LOGGER.info("Loading tokenizer/model from %s on %s", model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=local_files_only).to(device)

    trigger_ids = tokenizer.encode(trigger_text, add_special_tokens=False)
    if not trigger_ids:
        raise ValueError(f"Trigger text produced no token ids: {trigger_text!r}")
    trigger_token_ids = torch.tensor(trigger_ids, dtype=torch.long)

    LOGGER.info(
        "Local no-training attack probe: split=%s, attackers=%s, "
        "fs_target=%s, hf_target=%s, trigger=%r, trigger_tokens=%s",
        args.split,
        attacker_ids,
        fs_target_label,
        hf_target_label,
        trigger_text,
        trigger_ids,
    )

    rows = []
    for client_id in attacker_ids:
        client_data = data.get(int(client_id), None)
        if client_data is None:
            LOGGER.warning("Client %s not found in data.", client_id)
            continue
        dataloader = client_data.get(args.split, None)
        if dataloader is None:
            LOGGER.warning("Client %s has no split=%s.", client_id, args.split)
            continue

        result = _evaluate_client(
            model=model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            trigger_token_ids=trigger_token_ids,
            fs_target_label=fs_target_label,
            hf_target_label=hf_target_label,
            label_map=FS_TO_NLPTOWN_LABEL,
            max_length=max_length,
            device=device,
            max_samples=max(int(args.max_samples_per_client), 0),
        )
        result["client_id"] = int(client_id)
        result["domain"] = _get_underlying_domain(dataloader)
        rows.append(result)
        LOGGER.info(
            "Client %s (%s): clean_total=%d, clean_acc=%.4f, "
            "clean_target_rate=%.4f, poison_total=%d, ASR=%.4f (%d/%d)",
            client_id,
            result["domain"],
            int(result["clean_total"]),
            float(result["clean_acc"]),
            float(result["clean_target_rate"]),
            int(result["poison_total"]),
            float(result["asr"]),
            int(result["poison_success"]),
            int(result["poison_total"]),
        )

    if not rows:
        raise RuntimeError("No attacker client was evaluated.")

    avg = _weighted_average(rows)
    LOGGER.info(
        "Malicious clients weighted average: clean_total=%d, clean_acc=%.4f, "
        "clean_target_rate=%.4f, poison_total=%d, ASR=%.4f (%d/%d)",
        int(avg["clean_total"]),
        float(avg["clean_acc"]),
        float(avg["clean_target_rate"]),
        int(avg["poison_total"]),
        float(avg["asr"]),
        int(avg["poison_success"]),
        int(avg["poison_total"]),
    )


if __name__ == "__main__":
    main()
