"""
Data loader for offline-extracted CIFAR-100 feature vectors.

This bypasses ConvNeXt completely during FL training: the features were
pre-extracted by `extract_features.py` and saved as .pt files.

Usage in YAML:
    data:
      type: 'cifar100_features'
      root: 'data/cifar100_features/'   # directory containing client_*.pt files
      client_num: 10

The loader reads:
    {root}/meta.pt
    {root}/client_{id}_train.pt   -> {'x': Tensor[N, 1024], 'y': Tensor[N]}
    {root}/client_{id}_val.pt
    {root}/client_{id}_test.pt
    {root}/server_test.pt         -> {'x': Tensor[M, 1024], 'y': Tensor[M]}
"""

import os
import logging

import torch
from torch.utils.data import TensorDataset

from federatedscope.register import register_data
from federatedscope.core.data import StandaloneDataDict, ClientData
from federatedscope.core.data.utils import convert_data_mode
from federatedscope.core.auxiliaries.utils import setup_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_pt(path):
    """Load a .pt dict and return (x_tensor, y_tensor)."""
    if not os.path.isfile(path):
        return None, None
    d = torch.load(path, map_location='cpu', weights_only=False)
    x = d['x'].float()   # Tensor [N, feature_dim]
    y = d['y'].long()    # Tensor [N]
    return x, y


def _to_dataset(x, y):
    """Wrap tensors into a TensorDataset (compatible with DataLoader)."""
    if x is None or len(x) == 0:
        return None
    return TensorDataset(x, y)


# ---------------------------------------------------------------------------
# Main loader function
# ---------------------------------------------------------------------------

def load_cifar100_features(config, client_cfgs=None):
    """
    Build StandaloneDataDict from pre-extracted feature .pt files.

    Each client gets a ClientData with train/val/test TensorDatasets.
    Server (key 0) gets a ClientData with only a test set (server_test.pt or
    aggregated from all clients' test splits).
    """
    root = config.data.root
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"[cifar100_features] Feature directory not found: {root}\n"
            f"Run `python extract_features.py --output_dir {root}` first."
        )

    client_num = config.federate.client_num

    # ---- meta info ----
    meta_path = os.path.join(root, 'meta.pt')
    if os.path.isfile(meta_path):
        meta = torch.load(meta_path, map_location='cpu', weights_only=False)
        feature_dim = meta.get('feature_dim', 1024)
        logger.info(
            f"[cifar100_features] meta: {meta}"
        )
    else:
        feature_dim = 1024
        logger.warning("[cifar100_features] meta.pt not found, assuming feature_dim=1024")

    # ---- build per-client data ----
    datadict = {}
    for cid in range(1, client_num + 1):
        splits = {}
        for split in ['train', 'val', 'test']:
            path = os.path.join(root, f'client_{cid}_{split}.pt')
            x, y = _load_pt(path)
            ds = _to_dataset(x, y)
            if ds is not None:
                splits[split] = ds
                logger.debug(f"  client_{cid}_{split}: {len(ds)} samples, feat_dim={x.shape[1]}")
            else:
                logger.warning(f"  [cifar100_features] Missing or empty: {path}")

        # Choose correct config for this client
        if client_cfgs is not None:
            c_cfg = config.clone()
            c_cfg.merge_from_other_cfg(client_cfgs.get(f'client_{cid}'))
        else:
            c_cfg = config

        datadict[cid] = ClientData(
            c_cfg,
            train=splits.get('train'),
            val=splits.get('val'),
            test=splits.get('test'),
        )

    # ---- server test set (key = 0) ----
    server_path = os.path.join(root, 'server_test.pt')
    x_srv, y_srv = _load_pt(server_path)
    if x_srv is not None:
        srv_ds = _to_dataset(x_srv, y_srv)
        logger.info(f"[cifar100_features] server_test: {len(srv_ds)} samples")
    else:
        # Fall back: concatenate all clients' test sets
        logger.warning("[cifar100_features] server_test.pt not found; merging client test sets.")
        xs, ys = [], []
        for cid in range(1, client_num + 1):
            path = os.path.join(root, f'client_{cid}_test.pt')
            x, y = _load_pt(path)
            if x is not None:
                xs.append(x); ys.append(y)
        if xs:
            srv_ds = _to_dataset(torch.cat(xs), torch.cat(ys))
        else:
            srv_ds = None

    datadict[0] = ClientData(config, test=srv_ds)

    # ---- wrap into StandaloneDataDict ----
    data = StandaloneDataDict(datadict, config)
    data = convert_data_mode(data, config)

    setup_seed(config.seed)
    return data, config


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def call_cifar100_features(config, client_cfgs):
    if config.data.type.lower() == 'cifar100_features':
        return load_cifar100_features(config, client_cfgs)
    return None


register_data('cifar100_features', call_cifar100_features)
