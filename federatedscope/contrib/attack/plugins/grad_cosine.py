"""
Gradient Cosine Similarity Attack Plugin.

A membership inference attack that uses gradient cosine similarity
between the current gradient and the global model update direction.

Attack Logic
------------
- Member score: cosine similarity (higher = more likely member)
- Non-member score: cosine similarity

Member samples contribute to the global model update, so their gradients
tend to have higher cosine similarity with the update direction.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from ..base import AttackPlugin, AttackResult
from ..registry import AttackRegistry

logger = logging.getLogger(__name__)


@AttackRegistry.register
class GradCosinePlugin(AttackPlugin):
    """Gradient cosine similarity-based membership inference attack.

    This attack requires per-sample gradient features. It exploits the
    observation that gradients on member samples have higher cosine
    similarity with the global update direction.

    Attributes
    ----------
    name : str
        Plugin identifier: ``'grad_cosine'``.
    needs_gradient : bool
        ``True`` - gradient features are required.
    """

    @property
    def name(self) -> str:
        return "grad_cosine"

    @property
    def needs_gradient(self) -> bool:
        return True

    def _get_last_valid_round(self, data_dict: Dict) -> int:
        """Find the last round with non-None, non-empty data.

        Parameters
        ----------
        data_dict : dict
            Mapping from round number to data array.

        Returns
        -------
        int
            The highest round number with valid data.

        Raises
        ------
        ValueError
            If no valid rounds are found.
        """
        valid_rounds = []
        for round_num, data in data_dict.items():
            if data is not None:
                arr = np.asarray(data)
                if arr.size > 0 and not np.all(np.isnan(arr)):
                    valid_rounds.append(round_num)

        if not valid_rounds:
            raise ValueError("No valid rounds found in data_dict")

        return max(valid_rounds)

    def compute_scores(
        self,
        target_data: Dict,
        shadow_data: Optional[List[Dict]] = None,
        global_model=None,
        config=None,
    ) -> AttackResult:
        """Compute membership scores based on gradient cosine similarity.

        Parameters
        ----------
        target_data : dict
            Must contain:
            - ``'train_cos'``: dict[int, np.ndarray] - per-round member cosine similarities
            - ``'test_cos'``: np.ndarray - non-member cosine similarities

        shadow_data : list of dict, optional
            Not used by this plugin.

        global_model : torch.nn.Module, optional
            Not used by this plugin.

        config : CfgNode, optional
            Not used by this plugin.

        Returns
        -------
        AttackResult
            Contains cosine similarity scores for members and non-members.
        """
        # Get train cosine dict
        train_cos_dict = target_data.get('train_cos', {})

        if not train_cos_dict:
            logger.warning(f"{self.name}: No train_cos available")
            return AttackResult(
                name=self.name,
                scores_member=np.array([]),
                scores_nonmember=np.array([]),
                metadata={"error": "No train_cos data"},
            )

        # Find last valid round
        try:
            last_round = self._get_last_valid_round(train_cos_dict)
        except ValueError as e:
            logger.warning(f"{self.name}: {e}")
            return AttackResult(
                name=self.name,
                scores_member=np.array([]),
                scores_nonmember=np.array([]),
                metadata={"error": str(e)},
            )

        # Get member scores: cosine similarity (higher = more likely member)
        member_cos = np.asarray(train_cos_dict[last_round], dtype=np.float64)
        member_scores = member_cos

        # Get non-member scores (from same round)
        test_cos_dict = target_data.get('test_cos', {})
        if isinstance(test_cos_dict, dict) and last_round in test_cos_dict:
            nonmember_scores = np.asarray(test_cos_dict[last_round], dtype=np.float64)
        elif test_cos_dict is not None and not isinstance(test_cos_dict, dict):
            # Legacy: direct array
            nonmember_scores = np.asarray(test_cos_dict, dtype=np.float64)
        else:
            logger.warning(f"{self.name}: No test_cos available for round {last_round}")
            nonmember_scores = np.array([])

        return AttackResult(
            name=self.name,
            scores_member=member_scores,
            scores_nonmember=nonmember_scores,
            metadata={
                "last_round": last_round,
                "num_members": len(member_scores),
                "num_nonmembers": len(nonmember_scores),
            },
        )
