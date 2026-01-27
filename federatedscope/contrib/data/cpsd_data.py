"""
CPSD (Cross-domain Polarity Sentiment Dataset) Data Loader

Domains:
- Sentiment140
- Yelp Review Polarity
- IMDb (aclImdb)

This loader builds a cross-domain federated dataset:
- Each domain is treated as an independent source distribution
- Within each domain, clients are created by Dirichlet split (label skew)

Notes:
- For practicality, this loader supports sampling a subset of each domain via
  `cfg.data.args[0]`, e.g.:
    data:
      args: [{'max_train_samples_per_domain': 20000,
              'max_test_samples_per_domain': 5000,
              'seed': 42}]
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from torch.utils.data import DataLoader, Dataset, Subset

from federatedscope.register import register_data

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CPSDArgs:
    max_train_samples_per_domain: int = 20000
    max_test_samples_per_domain: int = 5000
    seed: int = 42
    cache_dir: str = ""  # empty -> <data.root>/cpsd_cache
    use_cache: bool = True


class CPSDTextDataset(Dataset):
    """In-memory text dataset with stable ids and `.targets`."""

    def __init__(self,
                 texts: List[str],
                 labels: List[int],
                 ids: List[str],
                 domain: str):
        assert len(texts) == len(labels) == len(ids)
        self.texts = texts
        self.targets = labels
        self.ids = ids
        self.domain = domain

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.texts[idx], int(self.targets[idx])

    def get_id(self, idx: int) -> str:
        return self.ids[idx]


def _get_cpsd_args(cfg) -> _CPSDArgs:
    raw_args = cfg.data.args[0] if getattr(cfg.data, "args", None) else {}
    if raw_args is None:
        raw_args = {}

    def _get_int(key: str, default: int) -> int:
        v = raw_args.get(key, default)
        try:
            return int(v)
        except Exception:
            return default

    def _get_bool(key: str, default: bool) -> bool:
        v = raw_args.get(key, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "y"}
        return default

    def _get_str(key: str, default: str) -> str:
        v = raw_args.get(key, default)
        return str(v) if v is not None else default

    return _CPSDArgs(
        max_train_samples_per_domain=_get_int("max_train_samples_per_domain", 20000),
        max_test_samples_per_domain=_get_int("max_test_samples_per_domain", 5000),
        seed=_get_int("seed", int(getattr(cfg, "seed", 42))),
        cache_dir=_get_str("cache_dir", ""),
        use_cache=_get_bool("use_cache", True),
    )


def _cache_dir(data_root: str, args: _CPSDArgs) -> str:
    if args.cache_dir:
        return args.cache_dir
    return os.path.join(data_root, "cpsd_cache")


def _cache_path(data_root: str, args: _CPSDArgs, domain: str, split: str,
                max_samples: int) -> str:
    os.makedirs(_cache_dir(data_root, args), exist_ok=True)
    ms = "all" if max_samples <= 0 else str(max_samples)
    filename = f"cpsd_{domain.lower()}_{split}_{ms}_seed{args.seed}.jsonl"
    return os.path.join(_cache_dir(data_root, args), filename)


def _load_cached_jsonl(path: str) -> Tuple[List[str], List[int], List[str]]:
    texts: List[str] = []
    labels: List[int] = []
    ids: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(int(obj["label"]))
            ids.append(obj["id"])
    return texts, labels, ids


def _save_cached_jsonl(path: str, texts: List[str], labels: List[int], ids: List[str]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for t, y, _id in zip(texts, labels, ids):
            f.write(json.dumps({"id": _id, "text": t, "label": int(y)}, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def _stratified_reservoir_sample_csv(csv_path: str,
                                     label_map: Dict[str, int],
                                     label_col: int,
                                     text_col: int,
                                     max_samples: int,
                                     seed: int,
                                     encoding: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Stratified reservoir sampling for large CSV files (2-class by design).
    Returns sampled (texts, labels, ids).
    """
    if max_samples <= 0:
        raise ValueError("Loading the full CSV is disabled for CPSD. Please set max_samples > 0.")

    # Allocate as evenly as possible across classes (deterministic)
    classes = sorted(set(label_map.values()))
    num_classes = len(classes)
    per_class = max_samples // num_classes
    remainder = max_samples - per_class * num_classes
    per_class_limit = {c: per_class for c in classes}
    for i in range(remainder):
        per_class_limit[classes[i]] += 1

    rng = random.Random(seed)
    reservoirs: Dict[int, List[Tuple[str, int, str]]] = {c: [] for c in set(label_map.values())}
    seen: Dict[int, int] = {c: 0 for c in reservoirs.keys()}

    with open(csv_path, "r", encoding=encoding, errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            if len(row) <= max(label_col, text_col):
                continue
            raw_label = row[label_col]
            if raw_label not in label_map:
                continue
            y = int(label_map[raw_label])
            text = row[text_col]
            if not text:
                continue

            # Per-sample stable id
            sample_id = f"{os.path.basename(csv_path)}:{row_idx}"

            seen[y] += 1
            limit = per_class_limit.get(y, per_class)

            if len(reservoirs[y]) < limit:
                reservoirs[y].append((text, y, sample_id))
            else:
                j = rng.randrange(seen[y])
                if j < limit:
                    reservoirs[y][j] = (text, y, sample_id)

    sampled: List[Tuple[str, int, str]] = []
    for c in sorted(reservoirs.keys()):
        sampled.extend(reservoirs[c])
    rng.shuffle(sampled)

    texts = [t for t, _, _ in sampled]
    labels = [y for _, y, _ in sampled]
    ids = [_id for _, _, _id in sampled]
    return texts, labels, ids


def _sample_imdb(root: str, split: str, max_samples: int, seed: int) -> Tuple[List[str], List[int], List[str]]:
    if max_samples <= 0:
        raise ValueError("Loading full IMDb is disabled for CPSD. Please set max_samples > 0.")

    pos_dir = os.path.join(root, "IMDb", "aclImdb", split, "pos")
    neg_dir = os.path.join(root, "IMDb", "aclImdb", split, "neg")

    pos_files = [os.path.join(pos_dir, fn) for fn in os.listdir(pos_dir) if fn.endswith(".txt")]
    neg_files = [os.path.join(neg_dir, fn) for fn in os.listdir(neg_dir) if fn.endswith(".txt")]

    rng = random.Random(seed)

    per_class = max_samples // 2
    pos_sel = rng.sample(pos_files, k=min(per_class, len(pos_files)))
    neg_sel = rng.sample(neg_files, k=min(max_samples - len(pos_sel), len(neg_files)))

    samples: List[Tuple[str, int, str]] = []
    for fp in pos_sel:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        rel = os.path.relpath(fp, root).replace("\\", "/")
        samples.append((text, 1, rel))
    for fp in neg_sel:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        rel = os.path.relpath(fp, root).replace("\\", "/")
        samples.append((text, 0, rel))

    rng.shuffle(samples)
    texts = [t for t, _, _ in samples]
    labels = [y for _, y, _ in samples]
    ids = [_id for _, _, _id in samples]
    return texts, labels, ids


def _load_domain_split(data_root: str, args: _CPSDArgs, domain: str, split: str,
                       max_samples: int) -> CPSDTextDataset:
    cache_path = _cache_path(data_root, args, domain, split, max_samples)
    if args.use_cache and os.path.exists(cache_path):
        texts, labels, ids = _load_cached_jsonl(cache_path)
        return CPSDTextDataset(texts=texts, labels=labels, ids=ids, domain=domain)

    if domain.lower() == "sentiment140":
        if split == "train":
            csv_path = os.path.join(data_root, "Sentiment140", "training.1600000.processed.noemoticon.csv")
            max_s = max_samples
        else:
            csv_path = os.path.join(data_root, "Sentiment140", "testdata.manual.2009.06.14.csv")
            max_s = max_samples
        label_map = {"0": 0, "4": 1}
        texts, labels, ids = _stratified_reservoir_sample_csv(
            csv_path=csv_path,
            label_map=label_map,
            label_col=0,
            text_col=5,
            max_samples=max_s,
            seed=args.seed,
            encoding="latin-1",
        )

    elif domain.lower() == "yelp":
        if split == "train":
            csv_path = os.path.join(data_root, "Yelp", "yelp_review_polarity_csv", "train.csv")
        else:
            csv_path = os.path.join(data_root, "Yelp", "yelp_review_polarity_csv", "test.csv")
        label_map = {"1": 0, "2": 1}
        texts, labels, ids = _stratified_reservoir_sample_csv(
            csv_path=csv_path,
            label_map=label_map,
            label_col=0,
            text_col=1,
            max_samples=max_samples,
            seed=args.seed,
            encoding="utf-8",
        )

    elif domain.lower() == "imdb":
        texts, labels, ids = _sample_imdb(
            root=data_root,
            split=split,
            max_samples=max_samples,
            seed=args.seed,
        )

    else:
        raise ValueError(f"Unknown CPSD domain: {domain}")

    if args.use_cache:
        _save_cached_jsonl(cache_path, texts, labels, ids)
    return CPSDTextDataset(texts=texts, labels=labels, ids=ids, domain=domain)


def _split_train_val(dataset: CPSDTextDataset, val_ratio: float,
                     seed: int) -> Tuple[CPSDTextDataset, Optional[CPSDTextDataset]]:
    if val_ratio <= 0:
        return dataset, None

    n = len(dataset)
    n_val = int(n * val_ratio)
    if n_val <= 0:
        return dataset, None

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    def _subset(idx_list: List[int]) -> CPSDTextDataset:
        texts = [dataset.texts[i] for i in idx_list]
        labels = [dataset.targets[i] for i in idx_list]
        ids = [dataset.ids[i] for i in idx_list]
        return CPSDTextDataset(texts=texts, labels=labels, ids=ids, domain=dataset.domain)

    return _subset(train_idx), _subset(val_idx)


def _uniform_split_indices(n: int, num_clients: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    splits = [indices[i::num_clients] for i in range(num_clients)]
    return splits


def load_cpsd_data(config, client_cfgs=None):
    data_type = config.data.type.lower()
    if data_type not in {"cpsd"}:
        return None

    data_root = config.data.root
    args = _get_cpsd_args(config)

    domains = ["Sentiment140", "Yelp", "IMDb"]
    num_domains = len(domains)
    num_classes = 2

    configured_client_num = int(getattr(config.federate, "client_num", num_domains))
    if configured_client_num <= 0:
        configured_client_num = num_domains

    if configured_client_num % num_domains != 0:
        logger.warning(
            f"client_num ({configured_client_num}) is not divisible by num_domains ({num_domains}). "
            f"Adjusting to {num_domains} clients (one per domain)."
        )
        clients_per_domain = 1
        total_clients = num_domains
    else:
        clients_per_domain = configured_client_num // num_domains
        total_clients = configured_client_num

    # Train/val split inside each domain's train split
    splits = tuple(getattr(config.data, "splits", [0.9, 0.1, 0.0]))
    val_ratio = float(splits[1]) if len(splits) > 1 else 0.0

    alpha = float(getattr(config.data, "dirichlet_alpha", 0.0))
    min_samples = int(getattr(config.data, "min_samples_per_client", 5))

    batch_size = int(getattr(config.dataloader, "batch_size", 32))
    num_workers = int(getattr(config.dataloader, "num_workers", 0))

    data_dict: Dict[int, Dict[str, DataLoader]] = {}
    client_id = 1

    # Import Dirichlet splitter (generic, lives in cv module but works with any Dataset.targets)
    from federatedscope.cv.dataloader.dataloader import split_data_by_dirichlet

    for domain_idx, domain in enumerate(domains):
        logger.info(f"Loading CPSD domain '{domain}'")

        train_full = _load_domain_split(
            data_root=data_root,
            args=args,
            domain=domain,
            split="train",
            max_samples=args.max_train_samples_per_domain,
        )
        train_dataset, val_dataset = _split_train_val(
            train_full, val_ratio=val_ratio, seed=int(getattr(config, "seed", 42)) + domain_idx
        )
        test_dataset = _load_domain_split(
            data_root=data_root,
            args=args,
            domain=domain,
            split="test",
            max_samples=args.max_test_samples_per_domain,
        )

        # Split train dataset into clients within this domain
        if clients_per_domain <= 1:
            client_splits = [list(range(len(train_dataset)))]
        else:
            if alpha > 0:
                client_splits = split_data_by_dirichlet(
                    train_dataset,
                    num_clients=clients_per_domain,
                    num_classes=num_classes,
                    alpha=alpha,
                    seed=int(getattr(config, "seed", 42)) + 10 * domain_idx,
                    min_samples_per_client=min_samples,
                )
            else:
                client_splits = _uniform_split_indices(
                    n=len(train_dataset),
                    num_clients=clients_per_domain,
                    seed=int(getattr(config, "seed", 42)) + 10 * domain_idx,
                )

        for local_idx_list in client_splits:
            train_subset = Subset(train_dataset, local_idx_list)
            data_dict[client_id] = {
                "train": DataLoader(
                    train_subset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    drop_last=False,
                ),
                "val": DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                ) if val_dataset is not None and len(val_dataset) > 0 else None,
                "test": DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                ),
            }
            logger.info(f"  Client {client_id} ({domain}): train={len(train_subset)}, test={len(test_dataset)}")
            client_id += 1

    config.federate.client_num = total_clients
    logger.info(
        f"CPSD data loaded: {len(data_dict)} clients, domains={domains}, "
        f"clients_per_domain={clients_per_domain}, dirichlet_alpha={alpha}"
    )

    return data_dict, config


register_data("cpsd", load_cpsd_data)
