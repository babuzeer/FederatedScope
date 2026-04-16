"""
Average Cosine Similarity Attack Plugin.

A cross-round membership inference attack that aggregates gradient cosine
similarity across multiple training rounds. This captures the consistent
alignment between member sample gradients and the global update direction.

Attack Logic
------------
- Collect train_cos from all valid rounds
- For each sample, compute mean cosine similarity across rounds
- Member score: mean_cosine (higher = more likely member)
- Non-member score: test_cos (single test set, not per-round)

Member samples' gradients consistently align with global updates across
training rounds, resulting in higher average cosine similarity.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from ..base import AttackPlugin, AttackResult
from ..registry import AttackRegistry

logger = logging.getLogger(__name__)


@AttackRegistry.register
class AvgCosinePlugin(AttackPlugin):
    """Cross-round average cosine similarity-based membership inference attack.

    This attack aggregates gradient cosine similarity across multiple rounds
    to capture consistent alignment patterns that distinguish members.

    Attributes
    ----------
    name : str
        Plugin identifier: ``'avg_cosine'``.
    needs_gradient : bool
        ``True`` - gradient features are required.
    is_cross_round : bool
        ``True`` - uses data from multiple training rounds.
    """

    @property
    def name(self) -> str:
        return "avg_cosine"

    @property
    def needs_gradient(self) -> bool:
        return True

    @property
    def is_cross_round(self) -> bool:
        return True

    def _get_valid_rounds(self, data_dict: Dict) -> List[int]:
        """Find all rounds with non-None, non-empty data.

        Parameters
        ----------
        data_dict : dict
            Mapping from round number to data array.

        Returns
        -------
        list[int]
            Sorted list of valid round numbers.
        """
        valid_rounds = []
        for round_num, data in data_dict.items():
            if data is not None:
                arr = np.asarray(data)
                if arr.size > 0 and not np.all(np.isnan(arr)):
                    valid_rounds.append(round_num)

        return sorted(valid_rounds)

    def compute_scores(
        self,
        target_data: Dict,
        shadow_data: Optional[List[Dict]] = None,
        global_model=None,
        config=None,
    ) -> AttackResult:
        """Compute membership scores by aggregating cosine similarities.

        Parameters
        ----------
        target_data : dict
            Must contain:
            - ``'train_cos'``: dict[int, np.ndarray] - per-round member cosine similarities
            - ``'test_cos'``: np.ndarray - non-member cosine similarities (single array)

        shadow_data : list of dict, optional
            Not used by this plugin.

        global_model : torch.nn.Module, optional
            Not used by this plugin.

        config : CfgNode, optional
            Not used by this plugin.

        Returns
        -------
        AttackResult
            Contains mean cosine similarity scores for members and non-members.
        """
        # Get configuration
        mode = 'mix'
        mix_length = 1000
        if config is not None:
            if hasattr(config, 'attack'):
                mode = getattr(config.attack, 'mode', 'mix')
                mix_length = getattr(config.attack, 'mix_length', 1000)
        
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

        # Get all valid rounds
        valid_rounds = self._get_valid_rounds(train_cos_dict)

        if not valid_rounds:
            logger.warning(f"{self.name}: No valid rounds found")
            return AttackResult(
                name=self.name,
                scores_member=np.array([]),
                scores_nonmember=np.array([]),
                metadata={"error": "No valid rounds"},
            )

        # Collect scores per round (aligned approach)
        all_member_scores = []
        all_nonmember_scores = []
        
        for round_num in valid_rounds:
            # Member scores for this round
            cos_sim = np.asarray(train_cos_dict[round_num], dtype=np.float64)
            all_member_scores.append(cos_sim)
            
            # Non-member scores for this round (fresh sample for mix mode)
            # Now uses per-round test_cos/mix_cos
            nonmember_scores = self._get_nonmember_scores(target_data, 'cos', mode, mix_length, round_num=round_num)
            all_nonmember_scores.append(nonmember_scores)

        # Average across rounds
        min_member = min(len(s) for s in all_member_scores)
        min_nonmember = min(len(s) for s in all_nonmember_scores)
        
        member_scores = np.mean([s[:min_member] for s in all_member_scores], axis=0)
        nonmember_scores = np.mean([s[:min_nonmember] for s in all_nonmember_scores], axis=0)

        return AttackResult(
            name=self.name,
            scores_member=member_scores,
            scores_nonmember=nonmember_scores,
            metadata={
                "num_rounds": len(valid_rounds),
                "rounds_used": valid_rounds,
                "num_members": len(member_scores),
                "num_nonmembers": len(nonmember_scores),
            },
        )
