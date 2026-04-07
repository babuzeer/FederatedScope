"""
Office-Home Dataset for Domain Adaptation in Federated Learning

This dataset contains images from 4 domains:
- Art: artistic depictions of objects
- Clipart: clipart images
- Product: images without background
- Real World: regular images captured with a camera

Reference:
    Venkateswara, H., Eusebio, J., Chakraborty, S., & Panchanathan, S. (2017).
    Deep hashing network for unsupervised domain adaptation.
    CVPR 2017.
"""

import os
import os.path as osp
import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


def _candidate_roots(root):
    root = osp.expanduser(str(root))
    candidates = [root]

    if not osp.isabs(root):
        cwd = os.getcwd()
        candidates.append(osp.join(cwd, root))
        root_parts = osp.normpath(root).split(os.sep)
        if not root_parts or root_parts[0].lower() != 'data':
            candidates.append(osp.join(cwd, 'data', root))

    ordered = []
    seen = set()
    for candidate in candidates:
        candidate = osp.normpath(candidate)
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)

        nested = osp.join(candidate, osp.basename(candidate))
        nested = osp.normpath(nested)
        if nested not in seen:
            ordered.append(nested)
            seen.add(nested)

    return ordered


class OfficeHome(Dataset):
    """
    Office-Home Dataset

    Args:
        root: root directory of the dataset
        domain: which domain to load ('Art', 'Clipart', 'Product', 'Real_World')
        split: 'train', 'val', or 'test'
        transform: transformation to apply to images
        train_ratio: ratio of training data (default: 0.8)
        val_ratio: ratio of validation data (default: 0.1)
        seed: random seed for splitting
    """

    # 65 common classes across all domains
    CLASSES = [
        'Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike',
        'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles',
        'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains',
        'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan',
        'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork',
        'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard',
        'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor',
        'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven',
        'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes',
        'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler',
        'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers',
        'Soda', 'Speaker', 'Spoon', 'Table', 'Telephone',
        'ToothBrush', 'Toys', 'Trash_Can', 'TV', 'Webcam'
    ]

    DOMAINS = ['Art', 'Clipart', 'Product', 'Real_World']

    def __init__(self,
                 root,
                 domain='Art',
                 split='train',
                 transform=None,
                 train_ratio=0.8,
                 val_ratio=0.1,
                 seed=123,
                 exclude_indices=None):

        assert domain in self.DOMAINS, f"Domain must be one of {self.DOMAINS}"
        assert split in ['train', 'val', 'test'], "Split must be train, val, or test"

        self.root = root
        self.domain = domain
        self.split = split
        self.transform = transform
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.seed = seed
        self.exclude_indices = exclude_indices or set()  # Indices to exclude (e.g., held-out for server)

        # Load data
        self.data, self.targets = self._load_data()

        logger.info(f"Loaded {len(self.data)} images from {domain} ({split} split)")

    def _load_data(self):
        """Load images and labels from the specified domain"""
        domain_aliases = []
        domain_name = self.domain.replace('_', ' ') if 'Real' in self.domain \
            else self.domain
        domain_aliases.append(domain_name)
        underscore_name = self.domain.replace(' ', '_')
        if underscore_name not in domain_aliases:
            domain_aliases.append(underscore_name)

        tried_dirs = []
        domain_dir = None
        resolved_root = None
        for root_candidate in _candidate_roots(self.root):
            for candidate_domain in domain_aliases:
                candidate_dir = osp.join(root_candidate, candidate_domain)
                tried_dirs.append(candidate_dir)
                if osp.exists(candidate_dir):
                    domain_dir = candidate_dir
                    resolved_root = root_candidate
                    break
            if domain_dir is not None:
                break

        if domain_dir is None:
            tried_str = '\n'.join([f'  - {item}' for item in tried_dirs])
            raise FileNotFoundError(
                f"Domain directory not found for `{self.domain}`.\n"
                f"Configured root: {self.root}\n"
                f"Tried:\n{tried_str}\n"
                f"Please download Office-Home dataset first.\n"
                f"Expected domains: {self.DOMAINS}"
            )
        self.root = resolved_root

        all_images = []
        all_labels = []

        # Load images from each class
        for class_idx, class_name in enumerate(self.CLASSES):
            class_dir = osp.join(domain_dir, class_name)

            if not osp.exists(class_dir):
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            # Get all image files
            image_files = [
                f for f in os.listdir(class_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            for img_file in image_files:
                img_path = osp.join(class_dir, img_file)
                all_images.append(img_path)
                all_labels.append(class_idx)

        if len(all_images) == 0:
            raise ValueError(f"No images found in {domain_dir}")

        # Filter out excluded indices (e.g., held-out for server test set)
        if self.exclude_indices:
            filtered_images = []
            filtered_labels = []
            for idx in range(len(all_images)):
                if idx not in self.exclude_indices:
                    filtered_images.append(all_images[idx])
                    filtered_labels.append(all_labels[idx])
            all_images = filtered_images
            all_labels = filtered_labels

            if len(all_images) == 0:
                raise ValueError(f"No images remaining after excluding {len(self.exclude_indices)} held-out indices")

        # Split data
        np.random.seed(self.seed)
        indices = np.random.permutation(len(all_images))

        train_size = int(len(all_images) * self.train_ratio)
        val_size = int(len(all_images) * self.val_ratio)

        if self.split == 'train':
            indices = indices[:train_size]
        elif self.split == 'val':
            indices = indices[train_size:train_size + val_size]
        else:  # test
            indices = indices[train_size + val_size:]

        images = [all_images[i] for i in indices]
        labels = [all_labels[i] for i in indices]

        return images, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.targets[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='white')

        # Apply transform
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_office_home_domain_data(root,
                                   domain,
                                   splits=(0.8, 0.1, 0.1),
                                   transform=None,
                                   val_transform=None,
                                   test_transform=None,
                                   seed=123,
                                   exclude_indices=None):
    """
    Load Office-Home dataset for a single domain

    Args:
        root: root directory of the dataset
        domain: which domain to load
        splits: (train_ratio, val_ratio, test_ratio)
        transform: transform for training data
        val_transform: transform for validation data
        test_transform: transform for test data
        seed: random seed
        exclude_indices: set of indices to exclude (e.g., held-out for server test)

    Returns:
        Dictionary with 'train', 'val', 'test' datasets
    """
    train_ratio, val_ratio, test_ratio = splits

    # Use default transforms if not provided
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

    train_dataset = OfficeHome(
        root=root,
        domain=domain,
        split='train',
        transform=transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        exclude_indices=exclude_indices
    )

    val_dataset = OfficeHome(
        root=root,
        domain=domain,
        split='val',
        transform=val_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        exclude_indices=exclude_indices
    )

    test_dataset = OfficeHome(
        root=root,
        domain=domain,
        split='test',
        transform=test_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        exclude_indices=exclude_indices
    )

    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }


def load_balanced_office_home_data(root,
                                     samples_per_class=None,
                                     test_ratio=None,
                                     transform=None,
                                     seed=123):
    """
    Load balanced test datasets for server-side evaluation WITHOUT data leakage.

    Creates balanced test sets by:
    1. First sampling held-out test samples from the ENTIRE dataset
    2. These indices are then excluded from client data loading
    3. Ensures ZERO overlap between server test set and client train/val/test data

    Args:
        root: root directory of the dataset
        samples_per_class: number of samples per class to sample (fixed number)
                          If None, will use test_ratio instead
        test_ratio: ratio of samples to use for server test (e.g., 0.1 for 10%)
                   Only used if samples_per_class is None
        transform: transform to apply to images
        seed: random seed for sampling

    Returns:
        Dictionary with two keys:
        - 'datasets': dict mapping domain names to balanced test datasets
        - 'excluded_indices': dict mapping domain names to sets of excluded indices
                             (to be passed to OfficeHome when loading client data)

    Note:
        - If both samples_per_class and test_ratio are None, defaults to test_ratio=0.1
        - If samples_per_class is specified, it takes precedence over test_ratio
    """
    np.random.seed(seed)

    # Determine sampling strategy
    if samples_per_class is None and test_ratio is None:
        test_ratio = 0.1  # Default to 10%
        logger.info(f"Using default test_ratio={test_ratio}")
    elif samples_per_class is not None:
        logger.info(f"Using fixed samples_per_class={samples_per_class}")
    else:
        logger.info(f"Using test_ratio={test_ratio}")

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    balanced_datasets = {}
    excluded_indices_per_domain = {}

    logger.info("="*80)
    logger.info("Creating server-side held-out test sets (NO DATA LEAKAGE)")
    logger.info("="*80)

    for domain in OfficeHome.DOMAINS:
        try:
            # Handle both 'Real_World' and 'Real World' directory names
            domain_name = domain.replace('_', ' ') if 'Real' in domain else domain
            domain_dir = osp.join(root, domain_name)

            # Try with underscore version if space version doesn't exist
            if not osp.exists(domain_dir):
                domain_name = domain.replace(' ', '_')
                domain_dir = osp.join(root, domain_name)

            if not osp.exists(domain_dir):
                logger.warning(f"Domain directory not found: {domain_dir}")
                continue

            # Step 1: Load ALL images from this domain to get global indices
            all_images = []
            all_labels = []
            image_to_global_idx = {}  # Map image path to global index
            global_idx = 0

            for class_idx, class_name in enumerate(OfficeHome.CLASSES):
                class_dir = osp.join(domain_dir, class_name)

                if not osp.exists(class_dir):
                    logger.warning(f"Class directory not found: {class_dir}")
                    continue

                image_files = [
                    osp.join(class_dir, f) for f in os.listdir(class_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]

                for img_path in image_files:
                    all_images.append(img_path)
                    all_labels.append(class_idx)
                    image_to_global_idx[img_path] = global_idx
                    global_idx += 1

            if len(all_images) == 0:
                logger.warning(f"No images found in {domain_dir}")
                continue

            # Step 2: Sample balanced held-out test set
            held_out_images = []
            held_out_labels = []
            held_out_indices = set()

            for class_idx in range(len(OfficeHome.CLASSES)):
                # Get all images for this class
                class_image_paths = [all_images[i] for i in range(len(all_images))
                                    if all_labels[i] == class_idx]

                if len(class_image_paths) == 0:
                    logger.warning(f"Domain {domain}, class {OfficeHome.CLASSES[class_idx]}: no samples")
                    continue

                # Determine how many samples to take
                if samples_per_class is not None:
                    # Fixed number per class
                    n_samples = min(samples_per_class, len(class_image_paths))
                    if len(class_image_paths) < samples_per_class:
                        logger.warning(f"Domain {domain}, class {OfficeHome.CLASSES[class_idx]}: "
                                     f"only {len(class_image_paths)} samples available (requested {samples_per_class})")
                else:
                    # Ratio-based sampling
                    n_samples = max(1, int(len(class_image_paths) * test_ratio))

                # Random sample
                sampled_paths = list(np.random.choice(class_image_paths, size=n_samples, replace=False))

                for img_path in sampled_paths:
                    held_out_images.append(img_path)
                    held_out_labels.append(class_idx)
                    held_out_indices.add(image_to_global_idx[img_path])

            # Step 3: Create dataset
            class BalancedDataset(Dataset):
                def __init__(self, images, labels, transform):
                    self.data = images
                    self.targets = labels
                    self.transform = transform

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    img_path = self.data[idx]
                    label = self.targets[idx]

                    try:
                        image = Image.open(img_path).convert('RGB')
                    except Exception as e:
                        logger.error(f"Error loading image {img_path}: {e}")
                        image = Image.new('RGB', (224, 224), color='white')

                    if self.transform is not None:
                        image = self.transform(image)

                    return image, label

            balanced_dataset = BalancedDataset(held_out_images, held_out_labels, transform)
            balanced_datasets[domain] = balanced_dataset
            excluded_indices_per_domain[domain] = held_out_indices

            # Log statistics
            class_counts = {}
            for label in held_out_labels:
                class_counts[label] = class_counts.get(label, 0) + 1

            logger.info(f"✓ Domain '{domain}': Created held-out test set")
            logger.info(f"  Total: {len(balanced_dataset)} / {len(all_images)} images ({100*len(balanced_dataset)/len(all_images):.1f}%)")
            logger.info(f"  Per-class: {dict(sorted(class_counts.items()))}")
            logger.info(f"  Excluded {len(held_out_indices)} indices from client data")

        except Exception as e:
            logger.error(f"Failed to create balanced dataset for domain {domain}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("="*80)
    logger.info(f"✓ Server test sets created for {len(balanced_datasets)} domains")
    logger.info(f"✓ NO DATA LEAKAGE: These samples are excluded from client data")
    logger.info("="*80)

    return {
        'datasets': balanced_datasets,
        'excluded_indices': excluded_indices_per_domain
    }

