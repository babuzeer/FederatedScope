"""
GGEUR Heterogeneity Handler for membership inference attacks.

This handler implements the GGEUR (Gaussian Guided Feature Expansion)
method for handling non-IID data distributions in federated learning.
It collects local statistics, aggregates them globally, and generates
augmented features to balance class distributions.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any

import numpy as np
from numpy.linalg import LinAlgError

from .base import HeterogeneityHandler

logger = logging.getLogger(__name__)


def _per_class_normalize(scores_dict: Dict[int, np.ndarray],
                         labels_dict: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """Per-class mean subtraction on per-round score arrays.

    For each sample, subtract the class-conditional mean of the score
    across all rounds. This removes the class-level bias introduced by
    Non-IID data distributions (i.e., classes with inherently high/low
    loss due to data scarcity will no longer dominate the MIA signal).

    Parameters
    ----------
    scores_dict : dict
        Maps round -> np.ndarray of scores (shape: N_samples,)
    labels_dict : dict
        Maps round -> np.ndarray of labels (shape: N_samples,)

    Returns
    -------
    dict
        Same structure as scores_dict, with per-class mean subtracted.
    """
    if not scores_dict or not labels_dict:
        return scores_dict

    # Collect all (score, label) pairs across rounds to compute class means
    all_scores = []
    all_labels = []
    for r in scores_dict:
        if r in labels_dict:
            s = np.asarray(scores_dict[r], dtype=np.float64)
            l = np.asarray(labels_dict[r], dtype=np.int64).flatten()
            if len(s) == len(l):
                all_scores.append(s)
                all_labels.append(l)

    if not all_scores:
        return scores_dict

    all_scores_cat = np.concatenate(all_scores)
    all_labels_cat = np.concatenate(all_labels)

    # Compute per-class mean
    unique_classes = np.unique(all_labels_cat)
    class_means: Dict[int, float] = {}
    for c in unique_classes:
        mask = all_labels_cat == c
        if mask.sum() > 0:
            class_means[int(c)] = float(all_scores_cat[mask].mean())

    if not class_means:
        return scores_dict

    # Subtract per-class mean from each round
    normalized = {}
    for r, scores in scores_dict.items():
        s = np.asarray(scores, dtype=np.float64)
        if r not in labels_dict:
            normalized[r] = s
            continue
        l = np.asarray(labels_dict[r], dtype=np.int64).flatten()
        if len(s) != len(l):
            normalized[r] = s
            continue
        adj = s.copy()
        for c, mu in class_means.items():
            mask = l == c
            adj[mask] -= mu
        normalized[r] = adj

    return normalized


class GGEURHandler(HeterogeneityHandler):
    """GGEUR-based heterogeneity handler for membership inference attacks.

    This handler implements:
    1. Collect local statistics (mean, covariance, counts) per class
    2. Aggregate statistics across clients using parallel axis theorem
    3. Augment features using Gaussian sampling from global distributions
    4. preprocess_data: per-class mean normalization on attack scores
       to remove Non-IID class-level biases before MIA evaluation.

    The goal is to generate synthetic features that balance class
    distributions and improve attack effectiveness under non-IID settings.

    Attributes
    ----------
    name : str
        Handler identifier: ``'ggeur'``.
    """

    @property
    def name(self) -> str:
        return "ggeur"

    def preprocess_data(
        self,
        target_data: Dict,
        shadow_data=None,
    ) -> tuple:
        """Apply per-class mean normalization to remove Non-IID bias.

        For Non-IID federated learning, different clients have different
        class distributions. A client with rare classes tends to have
        higher loss for those classes purely due to data scarcity, not
        because of membership. GGEUR removes this bias by subtracting
        the class-conditional mean from loss/cosine/gradient signals.

        Parameters
        ----------
        target_data : dict
            Must contain:
            - ``'train_losses'``: dict {round -> loss array}
            - ``'train_labels'``: dict {round -> label array}  (optional)
            - ``'train_cos'``: dict {round -> cosine array}    (optional)
            - ``'train_grad_diff'``: dict {round -> diff array} (optional)
            - ``'train_grad_norm'``: dict {round -> norm array} (optional)
        shadow_data : list of dict, optional
            List of shadow client data dicts with same structure.

        Returns
        -------
        tuple
            (normalized_target_data, normalized_shadow_data)
        """
        def normalize_client_data(data: Dict) -> Dict:
            normalized = dict(data)  # Shallow copy
            labels = data.get('train_labels', {})

            fields_to_normalize = ['train_losses', 'train_cos',
                                   'train_grad_diff', 'train_grad_norm']
            for field in fields_to_normalize:
                if field in data and data[field] and labels:
                    try:
                        normalized[field] = _per_class_normalize(
                            data[field], labels
                        )
                    except Exception as e:
                        logger.warning(f"[GGEURHandler] preprocess_data: "
                                       f"failed to normalize {field}: {e}")
            return normalized

        logger.info("[GGEURHandler] preprocess_data: applying per-class "
                    "mean normalization to remove Non-IID bias")


        norm_target = normalize_client_data(target_data)

        if shadow_data is not None:
            norm_shadow = [normalize_client_data(sd) for sd in shadow_data]
        else:
            norm_shadow = shadow_data

        return norm_target, norm_shadow

    def collect_local_statistics(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
    ) -> Dict[str, Dict]:
        """Collect per-class statistics from local data.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (N, D).
        labels : np.ndarray
            Label array of shape (N,).
        num_classes : int
            Total number of classes.

        Returns
        -------
        dict
            Contains:
            - ``'means'``: dict mapping class_id to mean vector (D,)
            - ``'covs'``: dict mapping class_id to covariance matrix (D, D)
            - ``'counts'``: dict mapping class_id to sample count
        """
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64).flatten()

        N, D = features.shape

        means: Dict[int, np.ndarray] = {}
        covs: Dict[int, np.ndarray] = {}
        counts: Dict[int, int] = {}

        for c in range(num_classes):
            mask = labels == c
            n_c = int(mask.sum())

            if n_c == 0:
                continue

            counts[c] = n_c
            class_features = features[mask]
            means[c] = class_features.mean(axis=0)

            if n_c > 1:
                # Compute covariance with regularization
                cov = np.cov(class_features.T)
                # Handle 1D case
                if cov.ndim == 0:
                    cov = np.array([[float(cov)]])
                covs[c] = cov + 1e-6 * np.eye(D)
            else:
                # Single sample: use regularized identity
                covs[c] = 1e-6 * np.eye(D)

        return {
            "means": means,
            "covs": covs,
            "counts": counts,
        }

    def aggregate_statistics(
        self,
        all_client_stats: List[Dict],
    ) -> Dict[str, Any]:
        """Aggregate statistics from all clients using parallel axis theorem.

        The parallel axis theorem allows computing global covariance from
        local covariances and means without accessing raw data:

        Σ_global = (1/N) Σ_k [ n_k * Σ_k + n_k * (μ_k - μ_global)(μ_k - μ_global)^T ]

        Parameters
        ----------
        all_client_stats : list of dict
            List of statistics dicts from ``collect_local_statistics``.

        Returns
        -------
        dict
            Contains:
            - ``'global_means'``: dict mapping class_id to global mean
            - ``'global_covs'``: dict mapping class_id to global covariance
            - ``'client_prototypes'``: list of per-client prototype dicts
        """
        # Collect all class IDs
        all_classes = set()
        for stats in all_client_stats:
            all_classes.update(stats["counts"].keys())

        global_means: Dict[int, np.ndarray] = {}
        global_covs: Dict[int, np.ndarray] = {}

        # For each class, aggregate across clients that have it
        for c in all_classes:
            means_list = []
            covs_list = []
            counts_list = []

            for stats in all_client_stats:
                if c in stats["counts"]:
                    means_list.append(stats["means"][c])
                    covs_list.append(stats["covs"][c])
                    counts_list.append(stats["counts"][c])

            if not means_list:
                continue

            # Total count for this class
            N_c = sum(counts_list)

            # Global mean: weighted average
            mu_global = np.zeros_like(means_list[0])
            for mu_k, n_k in zip(means_list, counts_list):
                mu_global += n_k * mu_k
            mu_global /= N_c

            # Global covariance using parallel axis theorem
            D = mu_global.shape[0]
            cov_sum = np.zeros((D, D))

            for mu_k, cov_k, n_k in zip(means_list, covs_list, counts_list):
                diff = (mu_k - mu_global).reshape(-1, 1)
                cov_sum += n_k * cov_k + n_k * (diff @ diff.T)

            global_covs[c] = cov_sum / N_c + 1e-6 * np.eye(D)
            global_means[c] = mu_global

        # Build client prototypes (for cross-client augmentation)
        client_prototypes = []
        for stats in all_client_stats:
            prototype = {
                "means": stats["means"],
                "counts": stats["counts"],
            }
            client_prototypes.append(prototype)

        return {
            "global_means": global_means,
            "global_covs": global_covs,
            "client_prototypes": client_prototypes,
        }

    def augment_features(
        self,
        local_features: np.ndarray,
        local_labels: np.ndarray,
        global_stats: Dict,
        client_id: int,
        config: Optional[Dict] = None,
    ) -> tuple:
        """Augment features using Gaussian sampling from global distributions.

        Parameters
        ----------
        local_features : np.ndarray
            Local feature matrix of shape (N, D).
        local_labels : np.ndarray
            Local label array of shape (N,).
        global_stats : dict
            Output from ``aggregate_statistics``.
        client_id : int
            ID of the current client (for skipping self in cross-client).
        config : dict, optional
            Augmentation configuration:
            - ``num_generated_per_sample``: samples to generate per original (default 50)
            - ``target_size_per_class``: target samples per class (default 50)
            - ``use_cross_client_prototypes``: whether to use other clients' prototypes (default True)

        Returns
        -------
        tuple
            (augmented_features, augmented_labels)
        """
        config = config or {}
        num_per_sample = config.get("num_generated_per_sample", 50)
        target_size = config.get("target_size_per_class", 50)
        use_cross = config.get("use_cross_client_prototypes", True)

        local_features = np.asarray(local_features, dtype=np.float64)
        local_labels = np.asarray(local_labels, dtype=np.int64).flatten()

        D = local_features.shape[1]

        aug_features = [local_features]
        aug_labels = [local_labels]

        global_means = global_stats["global_means"]
        global_covs = global_stats["global_covs"]
        client_prototypes = global_stats["client_prototypes"]

        # Get unique classes in local data
        local_classes = set(np.unique(local_labels).tolist())

        for c in local_classes:
            if c not in global_means:
                continue

            mu_c = global_means[c]
            cov_c = global_covs[c]

            # Count local samples for this class
            local_mask = local_labels == c
            local_count = int(local_mask.sum())

            # Determine how many to generate
            if local_count < target_size:
                num_generate = target_size - local_count
            else:
                num_generate = min(num_per_sample, target_size)

            if num_generate <= 0:
                continue

            # Generate from global distribution
            try:
                generated = np.random.multivariate_normal(mu_c, cov_c, num_generate)
            except LinAlgError:
                # Add regularization and retry
                cov_reg = cov_c + 1e-4 * np.eye(D)
                try:
                    generated = np.random.multivariate_normal(mu_c, cov_reg, num_generate)
                except LinAlgError:
                    logger.warning(f"Failed to generate for class {c}, skipping")
                    continue

            aug_features.append(generated)
            aug_labels.append(np.full(num_generate, c, dtype=np.int64))

            # Cross-client prototype augmentation
            if use_cross and len(client_prototypes) > 1:
                cross_per_client = max(1, num_per_sample // 5)
                cross_per_client = min(cross_per_client, 10)

                for k, proto in enumerate(client_prototypes):
                    if k == client_id:
                        continue  # Skip self

                    if c in proto["means"] and proto["counts"].get(c, 0) > 0:
                        other_mean = proto["means"][c]
                        try:
                            cross_gen = np.random.multivariate_normal(
                                other_mean, cov_c, cross_per_client
                            )
                        except LinAlgError:
                            cov_reg = cov_c + 1e-4 * np.eye(D)
                            try:
                                cross_gen = np.random.multivariate_normal(
                                    other_mean, cov_reg, cross_per_client
                                )
                            except LinAlgError:
                                continue

                        aug_features.append(cross_gen)
                        aug_labels.append(np.full(cross_per_client, c, dtype=np.int64))

        return np.concatenate(aug_features, axis=0), np.concatenate(aug_labels, axis=0)

    def __repr__(self) -> str:
        return f"GGEURHandler(name={self.name!r})"
