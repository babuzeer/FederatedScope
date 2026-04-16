"""
Gradient Difference Attack Plugin.

A membership inference attack that uses gradient difference (distance)
between per-sample gradient and the average gradient direction.

Attack Logic
------------
- Member score: -gradient_difference (negative, smaller diff = higher score)
- Non-member score: -gradient_difference

Member samples' gradients tend to be more aligned with the global update,
so their gradient differences (distances) from the average are smaller.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from ..base import AttackPlugin, AttackResult
from ..registry import AttackRegistry

logger = logging.getLogger(__name__)


@AttackRegistry.register
class GradDiffPlugin(AttackPlugin):
    """Gradient difference-based membership inference attack.

    This attack requires per-sample gradient features. It exploits the
    observation that gradients on member samples have smaller differences
    from the global update direction.

    Attributes
    ----------
    name : str
        Plugin identifier: ``'grad_diff'``.
    needs_gradient : bool
        ``True`` - gradient features are required.
    """

    @property
    def name(self) -> str:
        return "grad_diff"

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
        """Compute membership scores based on gradient differences.

        Parameters
        ----------
        target_data : dict
            Must contain:
            - ``'train_grad_diff'``: dict[int, np.ndarray] - per-round member gradient differences
            - ``'test_grad_diff'``: dict[int, np.ndarray] - per-round non-member gradient differences
            - ``'mix_grad_diff'``: dict[int, np.ndarray] - per-round mix gradient differences (for mix mode)

        shadow_data : list of dict, optional
            Not used by this plugin.

        global_model : torch.nn.Module, optional
            Not used by this plugin.

        config : CfgNode, optional
            Contains attack.mode ('test' or 'mix') and attack.mix_length.

        Returns
        -------
        AttackResult
            Contains gradient difference scores for members and non-members.
            Higher scores indicate higher membership likelihood.
        """
        # Get config
        mode = 'mix'
        mix_length = 1000
        if config is not None and hasattr(config, 'attack'):
            mode = getattr(config.attack, 'mode', 'mix')
            mix_length = getattr(config.attack, 'mix_length', 1000)

        # Get train gradient difference dict
        train_diff_dict = target_data.get('train_grad_diff', {})

        if not train_diff_dict:
            logger.warning(f"{self.name}: No train_grad_diff available")
            return AttackResult(
                name=self.name,
                scores_member=np.array([]),
                scores_nonmember=np.array([]),
                metadata={"error": "No train_grad_diff data"},
            )

        # Find last valid round
        try:
            last_round = self._get_last_valid_round(train_diff_dict)
        except ValueError as e:
            logger.warning(f"{self.name}: {e}")
            return AttackResult(
                name=self.name,
                scores_member=np.array([]),
                scores_nonmember=np.array([]),
                metadata={"error": str(e)},
            )

        # Get member scores: use raw gradient diff (higher = more likely member)
        # Note: Original FedMIA does NOT negate for grad_diff
        member_scores = np.asarray(train_diff_dict[last_round], dtype=np.float64)

        # Get non-member scores with mix mode support
        test_diff_dict = target_data.get('test_grad_diff', {})
        mix_diff_dict = target_data.get('mix_grad_diff', {})

        if mode == 'test':
            if isinstance(test_diff_dict, dict) and last_round in test_diff_dict:
                nonmember_scores = np.asarray(test_diff_dict[last_round], dtype=np.float64)
            elif test_diff_dict is not None and not isinstance(test_diff_dict, dict):
                nonmember_scores = np.asarray(test_diff_dict, dtype=np.float64)
            else:
                logger.warning(f"{self.name}: No test_grad_diff available for round {last_round}")
                nonmember_scores = np.array([])
        else:  # mix mode
            test_scores = np.array([])
            mix_scores = np.array([])

            if isinstance(test_diff_dict, dict) and last_round in test_diff_dict:
                test_data = np.asarray(test_diff_dict[last_round], dtype=np.float64)
                # Shuffle and sample
                indices = np.random.permutation(len(test_data))
                if len(test_data) > mix_length:
                    test_scores = test_data[indices[:mix_length]]
                else:
                    test_scores = test_data[indices]

            if isinstance(mix_diff_dict, dict) and last_round in mix_diff_dict:
                mix_scores = np.asarray(mix_diff_dict[last_round], dtype=np.float64)

            if len(test_scores) > 0 and len(mix_scores) > 0:
                nonmember_scores = np.concatenate([test_scores, mix_scores])
            elif len(test_scores) > 0:
                nonmember_scores = test_scores
            elif len(mix_scores) > 0:
                nonmember_scores = mix_scores
            else:
                logger.warning(f"{self.name}: No non-member data for round {last_round}")
                nonmember_scores = np.array([])

        return AttackResult(
            name=self.name,
            scores_member=member_scores,
            scores_nonmember=nonmember_scores,
            metadata={
                "last_round": last_round,
                "mode": mode,
                "num_members": len(member_scores),
                "num_nonmembers": len(nonmember_scores),
            },
        )
