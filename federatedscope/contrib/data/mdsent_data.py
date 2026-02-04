"""
Multi-Domain Sentiment (Rating) Data Loader

Dataset: Multi-Domain Sentiment Dataset (Blitzer et al., ACL 2007)
Local layout (default): <cfg.data.root>/{books,dvd,electronics,kitchen}/*.review

Labels:
  - Use the `<rating>` field as multi-class labels.
  - This dataset typically contains ratings {1.0, 2.0, 4.0, 5.0} (no 3.0),
    which we map to 4 classes:
      1.0 -> 0, 2.0 -> 1, 4.0 -> 2, 5.0 -> 3

Federated construction:
  - Each domain is treated as an independent source distribution
  - Within each domain, clients are created by Dirichlet split (label skew)
  - Train/val/test splits are created per-domain using `cfg.data.splits`

Optional args (cfg.data.args[0]):
  - include_unlabeled: bool (default True)  # `unlabeled.review` still has ratings
  - balance_test: bool (default False)      # make test set class-balanced (by undersampling)
  - test_samples_per_class: int (default 0) # >0: force N per class; 0: auto=min class count
  - seed: int (default cfg.seed or 42)
  - use_cache: bool (default True)
  - cache_dir: str (default <data.root>/mdsent_cache)
"""

from __future__ import annotations

import json
import logging
import os
import random
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from torch.utils.data import DataLoader, Dataset, Subset

from federatedscope.register import register_data

logger = logging.getLogger(__name__)


_RATING_TO_CLASS = {"1.0": 0, "2.0": 1, "4.0": 2, "5.0": 3}


@dataclass(frozen=True)
class _MDSentArgs:
    include_unlabeled: bool = True
    max_samples_per_domain: int = 0  # <=0 means use all
    balance_test: bool = False
    test_samples_per_class: int = 0  # <=0 means auto (min class count)
    seed: int = 42
    cache_dir: str = ""  # empty -> <data.root>/mdsent_cache
    use_cache: bool = True


class MDSentTextDataset(Dataset):
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


def _get_mdsent_args(cfg) -> _MDSentArgs:
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

    return _MDSentArgs(
        include_unlabeled=_get_bool("include_unlabeled", True),
        max_samples_per_domain=_get_int("max_samples_per_domain", 0),
        balance_test=_get_bool("balance_test", False),
        test_samples_per_class=_get_int("test_samples_per_class", 0),
        seed=_get_int("seed", int(getattr(cfg, "seed", 42))),
        cache_dir=_get_str("cache_dir", ""),
        use_cache=_get_bool("use_cache", True),
    )


def _cache_dir(data_root: str, args: _MDSentArgs) -> str:
    if args.cache_dir:
        return args.cache_dir
    return os.path.join(data_root, "mdsent_cache")


def _cache_path_all(data_root: str, args: _MDSentArgs, domain: str) -> str:
    os.makedirs(_cache_dir(data_root, args), exist_ok=True)
    include_u = 1 if args.include_unlabeled else 0
    domain_str = str(domain).lower().replace(" ", "_").replace("/", "_").replace("\\", "_")
    filename = f"mdsent_{domain_str}_all_includeu{include_u}.jsonl"
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


def _domain_seed(args: _MDSentArgs, domain: str) -> int:
    # Stable across runs (avoid Python hash randomization)
    crc = zlib.crc32(str(domain).lower().encode("utf-8")) & 0xFFFFFFFF
    return int(args.seed) + int(crc % 100000)


def _iter_reviews(review_path: str, domain: str) -> Tuple[str, str, str]:
    """
    Yield (text, rating_str, sample_id) from one *.review file.
    """
    file_name = os.path.basename(review_path)
    review_idx = -1
    in_review = False
    current_tag = None

    title_lines: List[str] = []
    text_lines: List[str] = []
    rating_value: Optional[str] = None

    def _flush():
        nonlocal title_lines, text_lines, rating_value
        if rating_value is None:
            return None
        rating_str = str(rating_value).strip()
        if not text_lines and not title_lines:
            return None
        parts: List[str] = []
        if title_lines:
            parts.append(" ".join([t.strip() for t in title_lines if str(t).strip()]))
        if text_lines:
            parts.append("\n".join([t.rstrip("\n") for t in text_lines]).strip())
        text = "\n\n".join([p for p in parts if p.strip()]).strip()
        if not text:
            return None
        sample_id = f"{domain}/{file_name}:{review_idx}"
        return text, rating_str, sample_id

    with open(review_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == "<review>":
                in_review = True
                review_idx += 1
                current_tag = None
                title_lines = []
                text_lines = []
                rating_value = None
                continue
            if not in_review:
                continue
            if line == "</review>":
                item = _flush()
                if item is not None:
                    yield item
                in_review = False
                current_tag = None
                continue

            if line in {"<title>", "<review_text>", "<rating>"}:
                current_tag = line.strip("<>").lower()
                continue
            if line in {"</title>", "</review_text>", "</rating>"}:
                current_tag = None
                continue

            if current_tag == "rating":
                if rating_value is None and line:
                    rating_value = line
                continue
            if current_tag == "title":
                if line:
                    title_lines.append(line)
                continue
            if current_tag == "review_text":
                # Keep original formatting as much as possible
                text_lines.append(raw_line.rstrip("\n"))
                continue


def _load_domain_all(data_root: str, args: _MDSentArgs, domain: str) -> MDSentTextDataset:
    cache_path = _cache_path_all(data_root, args, domain)
    if args.use_cache and os.path.exists(cache_path):
        texts, labels, ids = _load_cached_jsonl(cache_path)
        return MDSentTextDataset(texts=texts, labels=labels, ids=ids, domain=domain)

    domain_dir = os.path.join(data_root, str(domain))
    if not os.path.isdir(domain_dir):
        raise FileNotFoundError(f"Domain directory not found: {domain_dir}")

    files = ["positive.review", "negative.review"]
    if args.include_unlabeled and os.path.exists(os.path.join(domain_dir, "unlabeled.review")):
        files.append("unlabeled.review")

    texts: List[str] = []
    labels: List[int] = []
    ids: List[str] = []

    skipped = 0
    for fn in files:
        fp = os.path.join(domain_dir, fn)
        if not os.path.exists(fp):
            continue
        for text, rating_str, sample_id in _iter_reviews(fp, domain=str(domain)):
            cls = _RATING_TO_CLASS.get(rating_str)
            if cls is None:
                skipped += 1
                continue
            texts.append(text)
            labels.append(int(cls))
            ids.append(sample_id)

    if skipped > 0:
        logger.warning(f"MDSent: Skipped {skipped} reviews with unsupported ratings in domain {domain}")

    if args.use_cache:
        _save_cached_jsonl(cache_path, texts, labels, ids)

    return MDSentTextDataset(texts=texts, labels=labels, ids=ids, domain=str(domain))


def _split_train_val_test(dataset: MDSentTextDataset, splits: Tuple[float, float, float],
                          seed: int) -> Tuple[MDSentTextDataset, Optional[MDSentTextDataset], MDSentTextDataset]:
    s = list(splits) if splits is not None else [0.8, 0.1, 0.1]
    while len(s) < 3:
        s.append(0.0)
    train_r, val_r, test_r = [float(x) for x in s[:3]]
    total = train_r + val_r + test_r
    if total <= 0:
        raise ValueError(f"Invalid splits={splits}, sum must be > 0")
    train_r, val_r, test_r = train_r / total, val_r / total, test_r / total

    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(n * train_r)
    n_val = int(n * val_r)
    if n_train + n_val > n:
        n_train = max(0, min(n, n_train))
        n_val = max(0, min(n - n_train, n_val))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def _subset(idx_list: List[int]) -> MDSentTextDataset:
        texts = [dataset.texts[i] for i in idx_list]
        labels = [dataset.targets[i] for i in idx_list]
        ids = [dataset.ids[i] for i in idx_list]
        return MDSentTextDataset(texts=texts, labels=labels, ids=ids, domain=dataset.domain)

    train_set = _subset(train_idx)
    val_set = _subset(val_idx) if len(val_idx) > 0 else None
    test_set = _subset(test_idx)
    return train_set, val_set, test_set


def _balance_dataset_by_class(dataset: MDSentTextDataset,
                              num_classes: int,
                              seed: int,
                              per_class: int = 0) -> MDSentTextDataset:
    """
    Create a class-balanced subset of `dataset` by undersampling each class.

    Notes:
      - This is intended for evaluation splits (e.g., test) where we want
        equal samples per class.
      - If any class is missing, we keep the original dataset unchanged.
      - If `per_class<=0`, we use the minimum class count as the per-class size.
    """
    if dataset is None or len(dataset) == 0:
        return dataset

    num_classes = int(num_classes) if int(num_classes) > 0 else 0
    if num_classes <= 0:
        return dataset

    label_to_indices: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    for idx, y in enumerate(dataset.targets):
        y = int(y)
        if y in label_to_indices:
            label_to_indices[y].append(idx)

    counts = {c: len(v) for c, v in label_to_indices.items()}
    if any(counts[c] <= 0 for c in range(num_classes)):
        logger.warning(
            f"MDSent: Cannot balance split for domain={getattr(dataset, 'domain', '')}, "
            f"missing classes in split counts={counts}. Keeping original split."
        )
        return dataset

    min_cnt = min(counts.values())
    if int(per_class) > 0:
        k = min(int(per_class), int(min_cnt))
    else:
        k = int(min_cnt)

    if k <= 0:
        return dataset

    rng = random.Random(int(seed))
    selected_indices: List[int] = []
    for c in range(num_classes):
        idxs = list(label_to_indices[c])
        rng.shuffle(idxs)
        selected_indices.extend(idxs[:k])

    rng.shuffle(selected_indices)

    texts = [dataset.texts[i] for i in selected_indices]
    labels = [int(dataset.targets[i]) for i in selected_indices]
    ids = [dataset.ids[i] for i in selected_indices]
    return MDSentTextDataset(texts=texts, labels=labels, ids=ids, domain=dataset.domain)


def _maybe_cap_domain_samples(dataset: MDSentTextDataset, args: _MDSentArgs, domain: str) -> MDSentTextDataset:
    max_n = int(getattr(args, "max_samples_per_domain", 0) or 0)
    if max_n <= 0 or len(dataset) <= max_n:
        return dataset

    rng = random.Random(_domain_seed(args, domain) + 101)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    sel = indices[:max_n]

    texts = [dataset.texts[i] for i in sel]
    labels = [dataset.targets[i] for i in sel]
    ids = [dataset.ids[i] for i in sel]
    return MDSentTextDataset(texts=texts, labels=labels, ids=ids, domain=dataset.domain)


def load_mdsent_data(config, client_cfgs=None):
    data_type = str(config.data.type).lower()
    if data_type not in {"mdsent"}:
        return None

    data_root = config.data.root
    args = _get_mdsent_args(config)

    # Default domains for the dataset
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
            f"client_num ({configured_client_num}) is not divisible by num_domains ({num_domains}). "
            f"Adjusting to {num_domains} clients (one per domain)."
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

    data_dict: Dict[int, Dict[str, DataLoader]] = {}
    client_id = 1

    # Dirichlet splitter (generic, requires Dataset.targets)
    from federatedscope.cv.dataloader.dataloader import split_data_by_dirichlet

    for domain in domains:
        logger.info(f"Loading MDSent domain '{domain}' (include_unlabeled={args.include_unlabeled})")

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
                    seed=_domain_seed(args, domain) + 17,
                    min_samples_per_client=min_samples,
                )
            else:
                rng = random.Random(_domain_seed(args, domain) + 17)
                indices = list(range(len(train_dataset)))
                rng.shuffle(indices)
                client_splits = [indices[i::clients_per_domain] for i in range(clients_per_domain)]

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
            logger.info(
                f"  Client {client_id} ({domain}): train={len(train_subset)}, val={len(val_dataset) if val_dataset else 0}, test={len(test_dataset)}"
            )
            client_id += 1

    config.federate.client_num = total_clients
    config.model.num_classes = int(getattr(config.model, "num_classes", num_classes))
    logger.info(
        f"MDSent data loaded: {len(data_dict)} clients, domains={domains}, "
        f"clients_per_domain={clients_per_domain}, dirichlet_alpha={alpha}, num_classes={num_classes}"
    )

    return data_dict, config


register_data("mdsent", load_mdsent_data)
