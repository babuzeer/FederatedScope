"""
GGEUR_Clip Data Loader

Provides data loading for GGEUR_Clip multi-domain federated learning.
Supports:
- Multiple clients per domain (e.g., 12 clients = 3 per domain × 4 domains)
- LDS (Label Distribution Skew) using Dirichlet distribution for non-IID data
"""

import logging
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from federatedscope.register import register_data

logger = logging.getLogger(__name__)


def load_ggeur_data(config, client_cfgs=None):
    """
    Load data for GGEUR_Clip multi-domain federated learning.

    Supports:
    - PACS: 4 domains (photo, art_painting, cartoon, sketch)
    - Office-Home: 4 domains (Art, Clipart, Product, Real_World)

    Each domain can have multiple clients (data split among them).
    If LDS is enabled, uses Dirichlet distribution for non-IID data split.

    Args:
        config: FederatedScope configuration object
        client_cfgs: Optional per-client configurations

    Returns:
        Tuple of (data_dict, modified_config)
        data_dict: {client_id: {'train': dataloader, 'val': dataloader, 'test': dataloader}}
    """
    data_type = config.data.type.lower()

    if data_type == 'pacs':
        return _load_pacs_ggeur_data(config, client_cfgs)
    elif data_type in ['office-home', 'officehome', 'office_home']:
        return _load_officehome_ggeur_data(config, client_cfgs)
    else:
        return None


def _generate_dirichlet_matrix(num_domains, num_classes, alpha, seed=42):
    """
    Generate Dirichlet distribution matrix for LDS data split.

    Args:
        num_domains: Number of domains (clients)
        num_classes: Number of classes
        alpha: Dirichlet distribution parameter (smaller = more non-IID)
        seed: Random seed

    Returns:
        Matrix of shape (num_domains, num_classes) where each column sums to 1
    """
    np.random.seed(seed)
    # Generate dirichlet distribution for each class
    # dirichlet([alpha] * num_domains, num_classes) returns (num_classes, num_domains)
    # We transpose to get (num_domains, num_classes)
    matrix = np.random.dirichlet([alpha] * num_domains, num_classes).T

    logger.info(f"Generated Dirichlet matrix with alpha={alpha}")
    logger.info(f"  Shape: {matrix.shape}")
    logger.info(f"  Min proportion: {matrix.min():.4f}, Max: {matrix.max():.4f}")

    return matrix


def _split_dataset_with_lds(dataset, dirichlet_proportions, seed=42):
    """
    Split dataset using Dirichlet proportions (LDS).

    Each class's samples are split according to the Dirichlet proportions.
    This creates a highly non-IID distribution where some classes may have
    very few samples for a given client.

    Args:
        dataset: Dataset with .data (paths) and .targets (labels)
        dirichlet_proportions: Array of proportions for this client (one per class)
        seed: Random seed

    Returns:
        Subset containing only the allocated samples
    """
    np.random.seed(seed)

    # Get all labels
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))

    # Collect indices for this client
    client_indices = []
    class_counts = {}

    for class_idx in range(num_classes):
        # Get all indices for this class
        class_mask = targets == class_idx
        class_indices = np.where(class_mask)[0]

        if len(class_indices) == 0:
            continue

        # Shuffle indices
        np.random.shuffle(class_indices)

        # Get proportion for this class
        proportion = dirichlet_proportions[class_idx] if class_idx < len(dirichlet_proportions) else 0

        # Calculate number of samples to allocate
        num_to_allocate = int(proportion * len(class_indices))

        # Allocate samples
        allocated = class_indices[:num_to_allocate].tolist()
        client_indices.extend(allocated)
        class_counts[class_idx] = len(allocated)

    # Log distribution
    total_allocated = len(client_indices)
    total_available = len(dataset)
    non_empty_classes = sum(1 for c in class_counts.values() if c > 0)

    logger.info(f"  LDS allocation: {total_allocated}/{total_available} samples "
                f"({100*total_allocated/total_available:.1f}%), "
                f"{non_empty_classes}/{num_classes} classes with data")

    return Subset(dataset, client_indices), class_counts


def _split_dataset_for_clients(dataset, num_clients, seed=123):
    """
    Split a dataset among multiple clients uniformly.

    Args:
        dataset: The dataset to split
        num_clients: Number of clients to split among
        seed: Random seed for reproducible splitting

    Returns:
        List of Subset objects, one per client
    """
    n_samples = len(dataset)
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)

    # Split indices among clients
    splits = np.array_split(indices, num_clients)

    subsets = []
    for client_indices in splits:
        subsets.append(Subset(dataset, client_indices.tolist()))

    return subsets


def _split_subset_for_clients(subset, num_clients, seed=123):
    """
    Split a Subset among multiple clients uniformly.

    This is similar to _split_dataset_for_clients but works on Subset objects,
    which is needed when we first apply LDS and then split among multiple clients.

    Args:
        subset: The Subset to split (from LDS allocation)
        num_clients: Number of clients to split among
        seed: Random seed for reproducible splitting

    Returns:
        List of Subset objects, one per client
    """
    n_samples = len(subset)
    np.random.seed(seed)
    local_indices = np.random.permutation(n_samples)

    # Split local indices among clients
    splits = np.array_split(local_indices, num_clients)

    subsets = []
    for client_local_indices in splits:
        # Map local indices back to base dataset indices
        original_indices = [subset.indices[i] for i in client_local_indices]
        subsets.append(Subset(subset.dataset, original_indices))

    return subsets


def _load_pacs_ggeur_data(config, client_cfgs=None):
    """Load PACS dataset for GGEUR_Clip"""
    from federatedscope.cv.dataset.pacs import PACS, load_pacs_domain_data

    root = config.data.root
    batch_size = config.dataloader.batch_size
    num_workers = config.dataloader.num_workers

    # Default splits
    splits = tuple(config.data.splits) if hasattr(config.data, 'splits') else (0.8, 0.1, 0.1)

    # Standard transforms for PACS
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    domains = PACS.DOMAINS  # ['photo', 'art_painting', 'cartoon', 'sketch']
    num_domains = len(domains)
    num_classes = len(PACS.CLASSES)  # 7 classes

    # Check if LDS is enabled
    use_lds = getattr(config.ggeur, 'use_lds', False) if hasattr(config, 'ggeur') else False

    # Get configured client number
    configured_client_num = config.federate.client_num

    # Calculate clients per domain
    if configured_client_num % num_domains != 0:
        logger.warning(f"client_num ({configured_client_num}) is not divisible by "
                      f"num_domains ({num_domains}). Adjusting to {num_domains} clients.")
        clients_per_domain = 1
        total_clients = num_domains
    else:
        clients_per_domain = configured_client_num // num_domains
        total_clients = configured_client_num

    if use_lds:
        # LDS mode: use Dirichlet distribution for non-IID data split
        lds_alpha = getattr(config.ggeur, 'lds_alpha', 0.1)
        lds_seed = getattr(config.ggeur, 'lds_seed', 42)

        logger.info(f"GGEUR_Clip PACS with LDS: alpha={lds_alpha}, "
                   f"{total_clients} clients ({clients_per_domain} per domain)")

        # Generate Dirichlet matrix for domain-level LDS
        dirichlet_matrix = _generate_dirichlet_matrix(num_domains, num_classes, lds_alpha, lds_seed)
    else:
        logger.info(f"GGEUR_Clip PACS: {total_clients} clients, {clients_per_domain} per domain")
        dirichlet_matrix = None

    data_dict = {}
    client_id = 1

    # Load data for each domain
    for domain_idx, domain in enumerate(domains):
        logger.info(f"Loading PACS domain '{domain}'")

        try:
            domain_data = load_pacs_domain_data(
                root=root,
                domain=domain,
                splits=splits,
                transform=transform,
                seed=config.seed
            )

            train_dataset = domain_data['train']
            val_dataset = domain_data['val']
            test_dataset = domain_data['test']

            if use_lds:
                # LDS mode: use Dirichlet distribution
                train_subset, class_counts = _split_dataset_with_lds(
                    train_dataset,
                    dirichlet_matrix[domain_idx],
                    seed=config.seed + domain_idx
                )
                train_subsets = [train_subset]
            else:
                # Standard mode: uniform split
                if clients_per_domain > 1:
                    train_subsets = _split_dataset_for_clients(
                        train_dataset, clients_per_domain, seed=config.seed
                    )
                else:
                    train_subsets = [train_dataset]

            # Create data loaders for each client in this domain
            for i, train_subset in enumerate(train_subsets):
                data_dict[client_id] = {
                    'train': DataLoader(
                        train_subset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        drop_last=False
                    ),
                    'val': DataLoader(
                        val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers
                    ) if len(val_dataset) > 0 else None,
                    'test': DataLoader(
                        test_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers
                    )
                }

                train_size = len(train_subset)
                logger.info(f"  Client {client_id} ({domain}): train={train_size}")
                client_id += 1

        except Exception as e:
            logger.error(f"Failed to load PACS domain {domain}: {e}")
            raise

    # Update config with actual client number
    config.federate.client_num = total_clients

    logger.info(f"GGEUR_Clip PACS data loaded: {len(data_dict)} clients, "
                f"domains={domains}, LDS={use_lds}")

    return data_dict, config


def _load_officehome_ggeur_data(config, client_cfgs=None):
    """Load Office-Home dataset for GGEUR_Clip"""
    from federatedscope.cv.dataset.office_home import OfficeHome, load_office_home_domain_data

    root = config.data.root
    batch_size = config.dataloader.batch_size
    num_workers = config.dataloader.num_workers

    # Default splits
    splits = tuple(config.data.splits) if hasattr(config.data, 'splits') else (0.7, 0.0, 0.3)

    # Standard transforms for Office-Home
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    domains = OfficeHome.DOMAINS  # ['Art', 'Clipart', 'Product', 'Real_World']
    num_domains = len(domains)
    num_classes = len(OfficeHome.CLASSES)  # 65 classes

    # Check if LDS is enabled
    use_lds = getattr(config.ggeur, 'use_lds', False) if hasattr(config, 'ggeur') else False

    # Get configured client number
    configured_client_num = config.federate.client_num

    # Calculate clients per domain
    if configured_client_num % num_domains != 0:
        logger.warning(f"client_num ({configured_client_num}) is not divisible by "
                      f"num_domains ({num_domains}). Adjusting to {num_domains} clients.")
        clients_per_domain = 1
        total_clients = num_domains
    else:
        clients_per_domain = configured_client_num // num_domains
        total_clients = configured_client_num

    if use_lds:
        # LDS mode: use Dirichlet distribution for non-IID data split
        lds_alpha = getattr(config.ggeur, 'lds_alpha', 0.1)
        lds_seed = getattr(config.ggeur, 'lds_seed', 42)

        logger.info(f"GGEUR_Clip Office-Home with LDS: alpha={lds_alpha}, "
                   f"{total_clients} clients ({clients_per_domain} per domain)")

        # Generate Dirichlet matrix for domain-level LDS
        dirichlet_matrix = _generate_dirichlet_matrix(num_domains, num_classes, lds_alpha, lds_seed)
    else:
        logger.info(f"GGEUR_Clip Office-Home: {total_clients} clients, {clients_per_domain} per domain")
        dirichlet_matrix = None

    data_dict = {}
    client_id = 1

    # Track total samples for summary
    total_train_samples = 0
    total_original_samples = 0

    # Load data for each domain
    for domain_idx, domain in enumerate(domains):
        logger.info(f"Loading Office-Home domain '{domain}'")

        try:
            domain_data = load_office_home_domain_data(
                root=root,
                domain=domain,
                splits=splits,
                transform=transform,
                seed=config.seed
            )

            train_dataset = domain_data['train']
            val_dataset = domain_data['val']
            test_dataset = domain_data['test']

            total_original_samples += len(train_dataset)

            if use_lds:
                # LDS mode: first apply Dirichlet distribution to get domain's portion
                lds_subset, class_counts = _split_dataset_with_lds(
                    train_dataset,
                    dirichlet_matrix[domain_idx],
                    seed=config.seed + domain_idx
                )

                # Then split among multiple clients if needed
                if clients_per_domain > 1:
                    train_subsets = _split_subset_for_clients(
                        lds_subset, clients_per_domain, seed=config.seed + domain_idx
                    )
                else:
                    train_subsets = [lds_subset]
            else:
                # Standard mode: uniform split
                if clients_per_domain > 1:
                    train_subsets = _split_dataset_for_clients(
                        train_dataset, clients_per_domain, seed=config.seed
                    )
                else:
                    train_subsets = [train_dataset]

            # Create data loaders for each client in this domain
            for i, train_subset in enumerate(train_subsets):
                train_size = len(train_subset)
                total_train_samples += train_size

                data_dict[client_id] = {
                    'train': DataLoader(
                        train_subset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        drop_last=False
                    ),
                    'val': DataLoader(
                        val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers
                    ) if len(val_dataset) > 0 else None,
                    'test': DataLoader(
                        test_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers
                    )
                }

                logger.info(f"  Client {client_id} ({domain}): train={train_size}")
                client_id += 1

        except Exception as e:
            logger.error(f"Failed to load Office-Home domain {domain}: {e}")
            raise

    # Update config with actual client number
    config.federate.client_num = total_clients

    if use_lds:
        logger.info(f"GGEUR_Clip Office-Home LDS Summary:")
        logger.info(f"  Total clients: {total_clients}")
        logger.info(f"  Original train samples: {total_original_samples}")
        logger.info(f"  LDS allocated samples: {total_train_samples} ({100*total_train_samples/total_original_samples:.1f}%)")
        logger.info(f"  Alpha: {lds_alpha}")
    else:
        logger.info(f"GGEUR_Clip Office-Home data loaded: {len(data_dict)} clients, "
                    f"domains={domains}, clients_per_domain={clients_per_domain}")

    return data_dict, config


register_data('ggeur', load_ggeur_data)
