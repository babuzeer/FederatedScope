"""
Offline Feature Extraction for CIFAR-100 with ConvNeXt-Base.

Usage:
    conda activate federatedscope
    python extract_features.py [--data_root data/] [--output_dir data/cifar100_features/]
                               [--client_num 10] [--alpha 0.5] [--seed 12345]
                               [--batch_size 128] [--device cuda:0]
                               [--model convnext_base] [--split iid]

Output layout:
    data/cifar100_features/
        meta.pt          # {client_num, alpha, seed, feature_dim, split}
        client_1_train.pt  # {'x': Tensor[N,1024], 'y': Tensor[N]}
        client_1_val.pt
        client_1_test.pt
        ...
        client_10_train.pt
        server_test.pt   # global test set (for make_global_eval)

The .pt files contain dicts that are directly consumed by
federatedscope.contrib.data.cifar100_features.
"""

import os
import argparse
import logging
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Offline ConvNeXt feature extraction for CIFAR-100")
    p.add_argument('--data_root',   default='data/',                       help='Root dir for torchvision datasets')
    p.add_argument('--output_dir',  default='data/cifar100_features/',     help='Where to save .pt feature files')
    p.add_argument('--client_num',  type=int,   default=10,                help='Number of FL clients')
    p.add_argument('--alpha',       type=float, default=0.5,               help='LDA alpha for non-IID split (ignored when --split iid)')
    p.add_argument('--splits',      type=float, nargs=3, default=[0.8, 0.1, 0.1], help='train/val/test ratio per client')
    p.add_argument('--seed',        type=int,   default=12345)
    p.add_argument('--batch_size',  type=int,   default=128)
    p.add_argument('--device',      default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--model',       default='convnext_base',
                   choices=['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'])
    p.add_argument('--split',       default='lda', choices=['lda', 'iid'],
                   help='lda: non-IID Dirichlet split; iid: uniform random split')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_cifar100(data_root):
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.CIFAR100(root=data_root, train=True,  transform=transform, download=True)
    test_ds  = datasets.CIFAR100(root=data_root, train=False, transform=transform, download=True)
    return train_ds, test_ds


def iid_split(n_samples, client_num, seed):
    """Return list of index arrays, one per client (uniform IID split)."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    return [indices[i::client_num] for i in range(client_num)]


def lda_split(labels, client_num, alpha, seed):
    """Return list of index arrays via Dirichlet LDA (same logic as LDASplitter)."""
    from federatedscope.core.splitters.utils import (
        dirichlet_distribution_noniid_slice,
    )
    idx_slices = dirichlet_distribution_noniid_slice(
        np.array(labels), client_num, alpha, prior=None
    )
    return idx_slices


def train_val_test_split(indices, splits, seed):
    """Split one client's indices into train/val/test."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(indices)
    n = len(indices)
    n_train = int(n * splits[0])
    n_val   = int(n * splits[1])
    return (
        indices[:n_train],
        indices[n_train:n_train + n_val],
        indices[n_train + n_val:],
    )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

FEATURE_DIMS = {
    'convnext_tiny':  768,
    'convnext_small': 768,
    'convnext_base':  1024,
    'convnext_large': 1536,
}


def build_backbone(model_name, device):
    """Load ConvNeXt, strip classifier, freeze, move to device."""
    logger.info(f"Loading {model_name} backbone (pretrained=True) ...")
    from torchvision.models import (
        convnext_tiny,  ConvNeXt_Tiny_Weights,
        convnext_small, ConvNeXt_Small_Weights,
        convnext_base,  ConvNeXt_Base_Weights,
        convnext_large, ConvNeXt_Large_Weights,
    )
    builder_map = {
        'convnext_tiny':  (convnext_tiny,  ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
        'convnext_small': (convnext_small, ConvNeXt_Small_Weights.IMAGENET1K_V1),
        'convnext_base':  (convnext_base,  ConvNeXt_Base_Weights.IMAGENET1K_V1),
        'convnext_large': (convnext_large, ConvNeXt_Large_Weights.IMAGENET1K_V1),
    }
    builder, weights = builder_map[model_name]
    raw = builder(weights=weights)

    class BackboneOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = raw.features
            self.avgpool  = raw.avgpool
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            return x.flatten(1)

    backbone = BackboneOnly().to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    n = sum(p.numel() for p in backbone.parameters())
    logger.info(f"Backbone params: {n:,}  |  device: {device}")
    return backbone


@torch.no_grad()
def extract_all(dataset, backbone, batch_size, device):
    """Extract features for an entire dataset. Returns (features, labels) tensors."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    all_feats = []
    all_labels = []
    n_batches = len(loader)
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        feat = backbone(x).cpu()
        all_feats.append(feat)
        all_labels.append(y)
        if (i + 1) % 20 == 0 or (i + 1) == n_batches:
            logger.info(f"  [{i+1}/{n_batches}] extracted {len(feat)*( i+1)} samples ...")
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ---- load CIFAR-100 ----
    logger.info("Loading CIFAR-100 ...")
    train_ds, test_ds = get_cifar100(args.data_root)
    train_labels = [y for _, y in train_ds]

    # ---- build backbone ----
    backbone = build_backbone(args.model, device)
    feature_dim = FEATURE_DIMS[args.model]

    # ---- extract ALL train/test features (one pass each) ----
    logger.info("Extracting train features ...")
    train_feats, train_lbls = extract_all(train_ds, backbone, args.batch_size, device)
    logger.info(f"Train features: {train_feats.shape}")

    logger.info("Extracting test features ...")
    test_feats, test_lbls = extract_all(test_ds, backbone, args.batch_size, device)
    logger.info(f"Test features:  {test_feats.shape}")

    # ---- DONE with backbone, release GPU memory ----
    del backbone
    torch.cuda.empty_cache()
    logger.info("Backbone released from GPU.")

    # ---- split train into clients ----
    if args.split == 'lda':
        logger.info(f"LDA split: client_num={args.client_num}, alpha={args.alpha}")
        client_idx_list = lda_split(train_labels, args.client_num, args.alpha, args.seed)
    else:
        logger.info(f"IID split: client_num={args.client_num}")
        client_idx_list = iid_split(len(train_ds), args.client_num, args.seed)

    # ---- save per-client train/val/test + record indices ----
    client_train_indices = {}  # cid -> array of indices into all_train (50000)
    client_val_indices   = {}
    client_test_local_indices = {}  # indices within the client's local split

    for cid, idx_all in enumerate(client_idx_list, start=1):
        idx_train, idx_val, idx_test = train_val_test_split(idx_all, args.splits, args.seed + cid)

        # Convert numpy arrays to lists for JSON-serializable meta
        client_train_indices[cid] = idx_train.tolist() if hasattr(idx_train, 'tolist') else list(idx_train)
        client_val_indices[cid]   = idx_val.tolist()   if hasattr(idx_val,   'tolist') else list(idx_val)

        for split_name, idx in [('train', idx_train), ('val', idx_val), ('test', idx_test)]:
            out_path = os.path.join(args.output_dir, f'client_{cid}_{split_name}.pt')
            torch.save({'x': train_feats[idx], 'y': train_lbls[idx], 'indices': idx}, out_path)
            logger.info(f"  Saved client_{cid}_{split_name}: {len(idx)} samples -> {out_path}")

    # ---- save full train features (needed by FedMIA for gradient computation) ----
    all_train_path = os.path.join(args.output_dir, 'all_train.pt')
    torch.save({'x': train_feats, 'y': train_lbls}, all_train_path)
    logger.info(f"Saved all_train: {len(train_feats)} samples -> {all_train_path}")

    # ---- save global server test set (all test features) ----
    server_path = os.path.join(args.output_dir, 'server_test.pt')
    torch.save({'x': test_feats, 'y': test_lbls}, server_path)
    logger.info(f"Saved server_test: {len(test_feats)} samples -> {server_path}")

    # ---- save meta (includes client index mappings for FedMIA) ----
    meta = {
        'client_num':           args.client_num,
        'alpha':                args.alpha,
        'seed':                 args.seed,
        'feature_dim':          feature_dim,
        'split':                args.split,
        'model':                args.model,
        'splits':               args.splits,
        'client_train_indices': client_train_indices,  # {cid -> [idx in 0..49999]}
        'client_val_indices':   client_val_indices,    # {cid -> [idx in 0..49999]}
        'n_train':              len(train_feats),
        'n_test':               len(test_feats),
    }
    torch.save(meta, os.path.join(args.output_dir, 'meta.pt'))
    logger.info(f"Done. Feature dim = {feature_dim}. All files saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
