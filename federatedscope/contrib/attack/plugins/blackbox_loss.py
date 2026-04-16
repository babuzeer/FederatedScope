"""
Blackbox Loss Attack Plugin.

A simple membership inference attack that uses the model's loss values.
Member samples typically have lower loss than non-member samples because
the model has seen them during training.

Attack Logic
------------
- Member score: -loss (negative loss, higher = more likely member)
- Non-member score: -loss (negative loss)

Lower loss → Higher negative score → More likely to be a member.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from ..base import AttackPlugin, AttackResult
from ..registry import AttackRegistry

logger = logging.getLogger(__name__)


@AttackRegistry.register
class BlackboxLossPlugin(AttackPlugin):
    """Blackbox loss-based membership inference attack.

    This attack only requires access to the model's predictions (logits)
    and does not need gradient information. It exploits the observation
    that member samples tend to have lower loss values.

    Attributes
    ----------
    name : str
        Plugin identifier: ``'blackbox_loss'``.
    needs_gradient : bool
        ``False`` - only loss values are needed.
    """

    @property
    def name(self) -> str:
        return "blackbox_loss"

    @property
    def needs_gradient(self) -> bool:
        return False

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
        """Compute membership scores based on loss values.

        Parameters
        ----------
        target_data : dict
            Must contain:
            - ``'train_losses'``: dict[int, np.ndarray] - per-round member losses
            - ``'test_losses'``: np.ndarray - non-member losses

        shadow_data : list of dict, optional
            Not used by this plugin.

        global_model : torch.nn.Module, optional
            Not used by this plugin.

        config : CfgNode, optional
            Not used by this plugin.

        Returns
        -------
        AttackResult
            Contains negative loss scores for members and non-members.
        """
        # Get train losses dict
        train_losses_dict = target_data.get('train_losses', {})

        if not train_losses_dict:
            logger.warning(f"{self.name}: No train_losses available")
            return AttackResult(
                name=self.name,
                scores_member=np.array([]),
                scores_nonmember=np.array([]),
                metadata={"error": "No train_losses data"},
            )

        # Find last valid round
        try:
            last_round = self._get_last_valid_round(train_losses_dict)
        except ValueError as e:
            logger.warning(f"{self.name}: {e}")
            return AttackResult(
                name=self.name,
                scores_member=np.array([]),
                scores_nonmember=np.array([]),
                metadata={"error": str(e)},
            )

        # Get member scores: negative loss (lower loss = higher score)
        member_losses = np.asarray(train_losses_dict[last_round], dtype=np.float64)
        member_scores = -member_losses

        # Get non-member scores: negative loss (from same round)
        test_losses_dict = target_data.get('test_losses', {})
        if isinstance(test_losses_dict, dict) and last_round in test_losses_dict:
            nonmember_losses = np.asarray(test_losses_dict[last_round], dtype=np.float64)
            nonmember_scores = -nonmember_losses
        elif test_losses_dict is not None and not isinstance(test_losses_dict, dict):
            # Legacy: direct array
            nonmember_losses = np.asarray(test_losses_dict, dtype=np.float64)
            nonmember_scores = -nonmember_losses
        else:
            logger.warning(f"{self.name}: No test_losses available for round {last_round}")
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

