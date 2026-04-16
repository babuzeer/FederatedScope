"""
Base class for heterogeneity-aware attack extensions.

Heterogeneity handlers modify attack data or scores to account for
non-IID data distributions in federated learning scenarios.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class HeterogeneityHandler(ABC):
    """Abstract base class for heterogeneity-aware attack extensions.

    Subclasses implement methods to transform attack data or scores
    to better handle non-IID data distributions.

    The typical workflow is:
    1. ``preprocess_data()`` - Transform raw data before attack
    2. ``postprocess_scores()`` - Adjust scores after attack
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this handler."""

    def preprocess_data(
        self,
        target_data: Dict,
        shadow_data: Optional[Dict] = None,
    ) -> tuple:
        """Preprocess attack data before running attacks.

        Parameters
        ----------
        target_data : dict
            Target client's feature dictionary.
        shadow_data : dict, optional
            Shadow clients' feature dictionaries.

        Returns
        -------
        tuple
            (processed_target_data, processed_shadow_data)
        """
        # Default: no transformation
        return target_data, shadow_data

    def postprocess_scores(
        self,
        scores_member: np.ndarray,
        scores_nonmember: np.ndarray,
        attack_name: str,
    ) -> tuple:
        """Postprocess attack scores after computing them.

        Parameters
        ----------
        scores_member : np.ndarray
            Member sample scores.
        scores_nonmember : np.ndarray
            Non-member sample scores.
        attack_name : str
            Name of the attack that produced these scores.

        Returns
        -------
        tuple
            (processed_scores_member, processed_scores_nonmember)
        """
        # Default: no transformation
        return scores_member, scores_nonmember

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
