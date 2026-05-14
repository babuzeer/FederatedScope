"""
DomainNet dataset utilities for multi-domain federated learning.

This loader assumes the extracted dataset follows the common layout:
    root/domain/class_name/image.xxx

Example:
    /root/autodl-tmp/datasets/DomainNet/clipart/aircraft_carrier/xxx.png
"""

import logging
import os
import os.path as osp

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


KNOWN_DOMAINS = [
    'clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'
]


def discover_domainnet_metadata(root,
                                selected_domains=None,
                                shared_classes_only=True):
    """Discover available domains and classes from extracted folders."""
    if not osp.exists(root):
        raise FileNotFoundError(f"DomainNet root not found: {root}")

    if selected_domains:
        domains = [d for d in selected_domains if osp.isdir(osp.join(root, d))]
        missing = [d for d in selected_domains if d not in domains]
        if missing:
            raise FileNotFoundError(
                f"Selected DomainNet domains not found under {root}: {missing}")
    else:
        domains = []
        for entry in sorted(os.listdir(root)):
            entry_path = osp.join(root, entry)
            if osp.isdir(entry_path) and entry.lower() in KNOWN_DOMAINS:
                domains.append(entry)

    if not domains:
        raise ValueError(
            f"No DomainNet domain folders found under {root}. "
            f"Expected folders like: {KNOWN_DOMAINS}")

    class_sets = []
    for domain in domains:
        domain_dir = osp.join(root, domain)
        classes = {
            entry for entry in os.listdir(domain_dir)
            if osp.isdir(osp.join(domain_dir, entry))
        }
        if not classes:
            raise ValueError(f"No class folders found in DomainNet domain: {domain_dir}")
        class_sets.append(classes)

    if shared_classes_only:
        classes = sorted(set.intersection(*class_sets))
        if not classes:
            raise ValueError(
                f"No shared class folders across selected DomainNet domains: {domains}")
    else:
        classes = sorted(set.union(*class_sets))
        if not classes:
            raise ValueError(
                f"No class folders found across selected DomainNet domains: {domains}")

    return domains, classes


class DomainNet(Dataset):
    """Single-domain DomainNet dataset with deterministic train/val/test split."""

    def __init__(self,
                 root,
                 domain,
                 classes,
                 split='train',
                 transform=None,
                 train_ratio=0.7,
                 val_ratio=0.0,
                 seed=123,
                 exclude_indices=None):
        assert split in ['train', 'val', 'test'], "Split must be train, val, or test"

        self.root = root
        self.domain = domain
        self.classes = list(classes)
        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(self.classes)
        }
        self.split = split
        self.transform = transform
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.exclude_indices = exclude_indices or set()

        self.data, self.targets = self._load_data()
        logger.info(
            f"Loaded {len(self.data)} images from DomainNet/{domain} ({split} split)")

    def _load_data(self):
        domain_dir = osp.join(self.root, self.domain)
        if not osp.exists(domain_dir):
            raise FileNotFoundError(f"DomainNet domain directory not found: {domain_dir}")

        all_images = []
        all_labels = []

        for class_name in self.classes:
            class_dir = osp.join(domain_dir, class_name)
            if not osp.exists(class_dir):
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            image_files = [
                f for f in os.listdir(class_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
            ]

            for img_file in image_files:
                all_images.append(osp.join(class_dir, img_file))
                all_labels.append(self.class_to_idx[class_name])

        if not all_images:
            raise ValueError(f"No images found in DomainNet domain: {domain_dir}")

        if self.exclude_indices:
            kept_images = []
            kept_labels = []
            for idx in range(len(all_images)):
                if idx not in self.exclude_indices:
                    kept_images.append(all_images[idx])
                    kept_labels.append(all_labels[idx])
            all_images = kept_images
            all_labels = kept_labels

        np.random.seed(self.seed)
        indices = np.random.permutation(len(all_images))
        train_size = int(len(all_images) * self.train_ratio)
        val_size = int(len(all_images) * self.val_ratio)

        if self.split == 'train':
            split_indices = indices[:train_size]
        elif self.split == 'val':
            split_indices = indices[train_size:train_size + val_size]
        else:
            split_indices = indices[train_size + val_size:]

        images = [all_images[i] for i in split_indices]
        labels = [all_labels[i] for i in split_indices]
        return images, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.targets[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as error:
            logger.error(f"Error loading image {img_path}: {error}")
            image = Image.new('RGB', (224, 224), color='white')

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_domainnet_domain_data(root,
                               domain,
                               classes,
                               splits=(0.7, 0.0, 0.3),
                               transform=None,
                               val_transform=None,
                               test_transform=None,
                               seed=123,
                               exclude_indices=None):
    """Load a single DomainNet domain into train/val/test datasets."""
    train_ratio, val_ratio, _ = splits

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    if val_transform is None:
        val_transform = transform
    if test_transform is None:
        test_transform = transform

    train_dataset = DomainNet(
        root=root,
        domain=domain,
        classes=classes,
        split='train',
        transform=transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        exclude_indices=exclude_indices)

    val_dataset = DomainNet(
        root=root,
        domain=domain,
        classes=classes,
        split='val',
        transform=val_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        exclude_indices=exclude_indices)

    test_dataset = DomainNet(
        root=root,
        domain=domain,
        classes=classes,
        split='test',
        transform=test_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        exclude_indices=exclude_indices)

    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
