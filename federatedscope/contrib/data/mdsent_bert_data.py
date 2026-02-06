"""
Multi-Domain Sentiment (Rating) with BERT Embeddings

This loader is a drop-in alternative to `mdsent` that converts texts to
fixed-size sentence embeddings using a (frozen) BERT encoder, so that
standard FL trainers (e.g., FedProtoTrainer) can consume tensor inputs.

It reuses (and can populate) the same `.npz` cache format as the GGEUR BERT
feature cache:
  - keys: sample ids (strings)
  - values: float32 embedding vectors
  - file format: np.savez(paths=[...], features=[...])

Expected BERT-related configs are read from `cfg.ggeur.*` (for convenience
and compatibility with existing GGEUR configs).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from federatedscope.register import register_data

logger = logging.getLogger(__name__)


class MDSentEmbeddingDataset(Dataset):
    """In-memory embedding dataset with stable ids and `.targets`."""

    def __init__(self, embeddings: np.ndarray, labels: List[int], ids: List[str], domain: str):
        if len(embeddings) != len(labels) or len(labels) != len(ids):
            raise ValueError("embeddings/labels/ids length mismatch")

        self.embeddings = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
        self.targets = [int(y) for y in labels]
        self.ids = [str(_id) for _id in ids]
        self.domain = str(domain)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.embeddings[idx], int(self.targets[idx])

    def get_id(self, idx: int) -> str:
        return self.ids[idx]


@dataclass(frozen=True)
class _BertCfg:
    model_path: str
    tokenizer_path: str
    max_length: int
    pooling: str
    batch_size: int
    local_files_only: bool
    use_pretrained_weights: bool
    use_feature_cache: bool
    feature_cache_dir: str
    seed: int


def _get_bert_cfg(cfg) -> _BertCfg:
    ggeur = getattr(cfg, "ggeur", None)
    if ggeur is None:
        raise ValueError("`mdsent_bert` requires `ggeur.*` BERT settings in cfg")

    model_path = str(getattr(ggeur, "bert_model_path", "") or "")
    if not model_path:
        raise ValueError("Please set `ggeur.bert_model_path` for `mdsent_bert`")

    tokenizer_path = str(getattr(ggeur, "bert_tokenizer_path", "") or "") or model_path
    max_length = int(getattr(ggeur, "bert_max_length", 128))
    pooling = str(getattr(ggeur, "bert_pooling", "cls") or "cls").lower()
    batch_size = int(getattr(ggeur, "bert_batch_size", 32))
    local_files_only = bool(getattr(ggeur, "bert_local_files_only", True))
    use_pretrained_weights = bool(getattr(ggeur, "bert_use_pretrained_weights", True))

    use_feature_cache = bool(getattr(ggeur, "use_feature_cache", True))
    feature_cache_dir = str(getattr(ggeur, "feature_cache_dir", "") or "")
    if not feature_cache_dir:
        feature_cache_dir = os.path.join(os.path.dirname(str(cfg.data.root)), "text_feature_cache")

    seed = int(getattr(cfg, "seed", 42))
    return _BertCfg(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        pooling=pooling,
        batch_size=batch_size,
        local_files_only=local_files_only,
        use_pretrained_weights=use_pretrained_weights,
        use_feature_cache=use_feature_cache,
        feature_cache_dir=feature_cache_dir,
        seed=seed,
    )


def _cache_path(bert_cfg: _BertCfg, domain: str) -> Optional[str]:
    if not bert_cfg.use_feature_cache:
        return None

    os.makedirs(bert_cfg.feature_cache_dir, exist_ok=True)

    model_name = os.path.basename(str(bert_cfg.model_path).rstrip("/\\"))
    model_name = model_name or "bert"
    mode_str = "pre" if bert_cfg.use_pretrained_weights else "rand"
    if bert_cfg.use_pretrained_weights:
        model_str = f"{model_name}_maxlen{bert_cfg.max_length}_{bert_cfg.pooling}_{mode_str}"
    else:
        model_str = f"{model_name}_maxlen{bert_cfg.max_length}_{bert_cfg.pooling}_{mode_str}_seed{bert_cfg.seed}"

    dataset_name_for_cache = "mdsent"
    domain_str = str(domain).lower().replace(" ", "_").replace("/", "_").replace("\\", "_")
    cache_filename = f"{dataset_name_for_cache}_{domain_str}_bert_{model_str}.npz"
    return os.path.join(bert_cfg.feature_cache_dir, cache_filename)


def _load_feature_cache(path: Optional[str]) -> Dict[str, np.ndarray]:
    if not path or not os.path.exists(path):
        return {}
    try:
        data = np.load(path, allow_pickle=True)
        if "paths" in data and "features" in data:
            paths = data["paths"]
            feats = data["features"]
            return {str(p): f for p, f in zip(paths, feats)}
    except Exception as e:
        logger.warning("Failed to load BERT cache %s: %s", path, e)
    return {}


def _save_feature_cache(path: Optional[str], cache: Dict[str, np.ndarray]) -> None:
    if not path:
        return
    try:
        paths = list(cache.keys())
        features = np.asarray([cache[p] for p in paths], dtype=np.float32)
        np.savez(path, paths=np.asarray(paths), features=features)
    except Exception as e:
        logger.warning("Failed to save BERT cache %s: %s", path, e)


def _load_bert(model_path: str, tokenizer_path: str, local_only: bool, use_pretrained: bool, device: torch.device):
    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except Exception as e:
        raise ImportError("transformers is required for `mdsent_bert` (pip install transformers)") from e

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=local_only)
    if use_pretrained:
        model = AutoModel.from_pretrained(model_path, local_files_only=local_only)
    else:
        cfg = AutoConfig.from_pretrained(model_path, local_files_only=local_only)
        model = AutoModel.from_config(cfg)

    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def _ensure_embeddings_cached(
    dataset,
    indices: List[int],
    model,
    tokenizer,
    device: torch.device,
    bert_cfg: _BertCfg,
    cache: Dict[str, np.ndarray],
) -> bool:
    updated = False
    missing: List[int] = []
    for idx in indices:
        sample_id = str(dataset.get_id(idx)) if hasattr(dataset, "get_id") else str(idx)
        if sample_id not in cache:
            missing.append(idx)

    if not missing:
        return False

    for i in range(0, len(missing), max(1, int(bert_cfg.batch_size))):
        batch_indices = missing[i:i + int(bert_cfg.batch_size)]
        texts: List[str] = []
        ids: List[str] = []
        for idx in batch_indices:
            text, _label = dataset[idx]
            texts.append(str(text))
            ids.append(str(dataset.get_id(idx)) if hasattr(dataset, "get_id") else str(idx))

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(bert_cfg.max_length),
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        outputs = model(**encoded)
        hidden = outputs.last_hidden_state  # (B, T, H)

        if bert_cfg.pooling == "mean":
            attn = encoded.get("attention_mask", None)
            if attn is None:
                emb = hidden.mean(dim=1)
            else:
                mask = attn.unsqueeze(-1).float()
                emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            emb = hidden[:, 0, :]

        emb_np = emb.detach().cpu().numpy().astype(np.float32)
        for sid, vec in zip(ids, emb_np):
            cache[sid] = vec
            updated = True

    return updated


def _embed_text_dataset(text_dataset, domain: str, bert_cfg: _BertCfg, model, tokenizer, device: torch.device):
    if text_dataset is None or len(text_dataset) == 0:
        return None

    cache_path = _cache_path(bert_cfg, domain)
    cache = _load_feature_cache(cache_path)

    indices = list(range(len(text_dataset)))
    updated = _ensure_embeddings_cached(
        dataset=text_dataset,
        indices=indices,
        model=model,
        tokenizer=tokenizer,
        device=device,
        bert_cfg=bert_cfg,
        cache=cache,
    )
    if updated:
        _save_feature_cache(cache_path, cache)

    ids = [str(text_dataset.get_id(i)) if hasattr(text_dataset, "get_id") else str(i) for i in indices]
    labels = [int(text_dataset.targets[i]) for i in indices]
    if not ids:
        return None

    try:
        first_vec = cache[ids[0]]
        emb_dim = int(np.asarray(first_vec).shape[-1])
    except Exception:
        raise RuntimeError("BERT cache is empty or invalid; failed to build embeddings")

    emb = np.zeros((len(indices), emb_dim), dtype=np.float32)
    for j, sid in enumerate(ids):
        vec = cache.get(sid, None)
        if vec is None:
            raise RuntimeError(f"Missing BERT embedding for sample id: {sid}")
        emb[j] = np.asarray(vec, dtype=np.float32)

    return MDSentEmbeddingDataset(embeddings=emb, labels=labels, ids=ids, domain=domain)


def load_mdsent_bert_data(config, client_cfgs=None):
    data_type = str(config.data.type).lower()
    if data_type not in {"mdsent_bert"}:
        return None

    from federatedscope.contrib.data.mdsent_data import (
        _RATING_TO_CLASS,
        _balance_dataset_by_class,
        _domain_seed,
        _get_mdsent_args,
        _load_domain_all,
        _maybe_cap_domain_samples,
        _split_train_val_test,
    )

    data_root = config.data.root
    args = _get_mdsent_args(config)
    bert_cfg = _get_bert_cfg(config)

    default_domains = ["books", "dvd", "electronics", "kitchen"]
    domains = [d for d in default_domains if os.path.isdir(os.path.join(data_root, d))]
    if not domains:
        raise FileNotFoundError(f"No sentiment domains found under: {data_root}")

    num_domains = len(domains)
    num_classes = len(set(_RATING_TO_CLASS.values()))

    configured_client_num = int(getattr(config.federate, "client_num", num_domains))
    if configured_client_num <= 0:
        configured_client_num = num_domains

    if configured_client_num % num_domains != 0:
        logger.warning(
            "client_num (%d) is not divisible by num_domains (%d). Adjusting to %d clients (one per domain).",
            configured_client_num,
            num_domains,
            num_domains,
        )
        clients_per_domain = 1
        total_clients = num_domains
    else:
        clients_per_domain = configured_client_num // num_domains
        total_clients = configured_client_num

    splits = tuple(getattr(config.data, "splits", [0.8, 0.1, 0.1]))
    alpha = float(getattr(config.data, "dirichlet_alpha", 0.0))
    min_samples = int(getattr(config.data, "min_samples_per_client", 5))

    batch_size = int(getattr(config.dataloader, "batch_size", 32))
    num_workers = int(getattr(config.dataloader, "num_workers", 0))

    device = (
        torch.device(f"cuda:{int(getattr(config, 'device', 0))}")
        if getattr(config, "use_gpu", False) and torch.cuda.is_available()
        else torch.device("cpu")
    )
    model, tokenizer = _load_bert(
        model_path=bert_cfg.model_path,
        tokenizer_path=bert_cfg.tokenizer_path,
        local_only=bert_cfg.local_files_only,
        use_pretrained=bert_cfg.use_pretrained_weights,
        device=device,
    )

    data_dict: Dict[int, Dict[str, DataLoader]] = {}
    client_id = 1

    from federatedscope.cv.dataloader.dataloader import split_data_by_dirichlet

    for domain in domains:
        logger.info("Loading MDSent-BERT domain '%s' (include_unlabeled=%s)", domain, args.include_unlabeled)

        full_dataset = _load_domain_all(data_root=data_root, args=args, domain=domain)
        full_dataset = _maybe_cap_domain_samples(full_dataset, args=args, domain=domain)
        train_dataset, val_dataset, test_dataset = _split_train_val_test(
            full_dataset, splits=splits, seed=_domain_seed(args, domain)
        )

        if getattr(args, "balance_test", False) or int(getattr(args, "test_samples_per_class", 0) or 0) > 0:
            test_dataset = _balance_dataset_by_class(
                test_dataset,
                num_classes=num_classes,
                seed=_domain_seed(args, domain) + 97,
                per_class=int(getattr(args, "test_samples_per_class", 0) or 0),
            )

        train_emb = _embed_text_dataset(
            train_dataset, domain=domain, bert_cfg=bert_cfg, model=model, tokenizer=tokenizer, device=device
        )
        val_emb = (
            _embed_text_dataset(
                val_dataset, domain=domain, bert_cfg=bert_cfg, model=model, tokenizer=tokenizer, device=device
            )
            if val_dataset is not None and len(val_dataset) > 0
            else None
        )
        test_emb = _embed_text_dataset(
            test_dataset, domain=domain, bert_cfg=bert_cfg, model=model, tokenizer=tokenizer, device=device
        )

        if clients_per_domain <= 1:
            client_splits = [list(range(len(train_emb)))]
        else:
            if alpha > 0:
                client_splits = split_data_by_dirichlet(
                    train_emb,
                    num_clients=clients_per_domain,
                    num_classes=num_classes,
                    alpha=alpha,
                    seed=_domain_seed(args, domain) + 17,
                    min_samples_per_client=min_samples,
                )
            else:
                rng = np.random.RandomState(_domain_seed(args, domain) + 17)
                indices = np.arange(len(train_emb))
                rng.shuffle(indices)
                client_splits = [indices[i::clients_per_domain].tolist() for i in range(clients_per_domain)]

        for local_idx_list in client_splits:
            train_subset = Subset(train_emb, local_idx_list)
            data_dict[client_id] = {
                "train": DataLoader(
                    train_subset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    drop_last=False,
                ),
                "val": DataLoader(val_emb, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                if val_emb is not None
                else None,
                "test": DataLoader(test_emb, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            }
            logger.info(
                "  Client %d (%s): train=%d, val=%d, test=%d",
                client_id,
                domain,
                len(train_subset),
                len(val_emb) if val_emb is not None else 0,
                len(test_emb),
            )
            client_id += 1

    config.federate.client_num = total_clients
    config.model.num_classes = int(getattr(config.model, "num_classes", num_classes))
    logger.info(
        "MDSent-BERT data loaded: %d clients, domains=%s, clients_per_domain=%d, dirichlet_alpha=%.4f, num_classes=%d",
        len(data_dict),
        domains,
        clients_per_domain,
        alpha,
        num_classes,
    )

    return data_dict, config


register_data("mdsent_bert", load_mdsent_bert_data)
