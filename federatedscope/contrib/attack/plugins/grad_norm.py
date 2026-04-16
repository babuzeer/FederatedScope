"""
Gradient Norm Attack Plugin.

A membership inference attack that uses gradient norm (magnitude)
of per-sample gradients.

Attack Logic
------------
- Member score: gradient_norm (higher = more likely member)
- Non-member score: gradient_norm

Member samples tend to have larger gradient norms because they contribute
more to the model update. The model fits to member samples, so their
gradients have larger magnitude during training.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from ..base import AttackPlugin, AttackResult
from ..registry import AttackRegistry

logger = logging.getLogger(__name__)


@AttackRegistry.register
class GradNormPlugin(AttackPlugin):
    """Gradient norm-based membership inference attack.

    This attack requires per-sample gradient features. It exploits the
    observation that gradients on member samples have larger norms because
    they contribute more significantly to model updates.

    Attributes
    ----------
    name : str
        Plugin identifier: ``'grad_norm'``.
    needs_gradient : bool
        ``True`` - gradient features are required.
    """

    @property
    def name(self) -> str:
        return "grad_norm"

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
        """Compute membership scores based on gradient norms.

        Parameters
        ----------
        target_data : dict
            Must contain:
            - ``'train_grad_norm'``: dict[int, np.ndarray] - per-round member gradient norms
            - ``'test_grad_norm'``: dict[int, np.ndarray] - per-round non-member gradient norms
            - ``'mix_grad_norm'``: dict[int, np.ndarray] - per-round mix gradient norms (for mix mode)

        shadow_data : list of dict, optional
            Not used by this plugin.

        global_model : torch.nn.Module, optional
            Not used by this plugin.

        config : CfgNode, optional
            Contains attack.mode ('test' or 'mix') and attack.mix_length.

        Returns
        -------
        AttackResult
            Contains negative gradient norm scores for members and non-members.
            Higher scores indicate higher membership likelihood.
        """
        # Get config
        mode = 'mix'
        mix_length = 1000
        if config is not None and hasattr(config, 'attack'):
            mode = getattr(config.attack, 'mode', 'mix')
            mix_length = getattr(config.attack, 'mix_length', 1000)

        # Get train gradient norm dict
        train_norm_dict = target_data.get('train_grad_norm', {})

        if not train_norm_dict:
            logger.warning(f"{self.name}: No train_grad_norm available")
            return AttackResult(
                name=self.name,
                scores_member=np.array([]),
                scores_nonmember=np.array([]),
                metadata={"error": "No train_grad_norm data"},
            )

        # Find last valid round
        try:
            last_round = self._get_last_valid_round(train_norm_dict)
        except ValueError as e:
            logger.warning(f"{self.name}: {e}")
            return AttackResult(
                name=self.name,
                scores_member=np.array([]),
                scores_nonmember=np.array([]),
                metadata={"error": str(e)},
            )

        # Get member scores: negative gradient norm (smaller norm = higher score)
        # Note: Original FedMIA negates for grad_norm
        member_scores = -np.asarray(train_norm_dict[last_round], dtype=np.float64)

        # Get non-member scores with mix mode support
        test_norm_dict = target_data.get('test_grad_norm', {})
        mix_norm_dict = target_data.get('mix_grad_norm', {})

        if mode == 'test':
            if isinstance(test_norm_dict, dict) and last_round in test_norm_dict:
                nonmember_scores = -np.asarray(test_norm_dict[last_round], dtype=np.float64)
            elif test_norm_dict is not None and not isinstance(test_norm_dict, dict):
                nonmember_scores = -np.asarray(test_norm_dict, dtype=np.float64)
            else:
                logger.warning(f"{self.name}: No test_grad_norm available for round {last_round}")
                nonmember_scores = np.array([])
        else:  # mix mode
            test_scores = np.array([])
            mix_scores = np.array([])

            if isinstance(test_norm_dict, dict) and last_round in test_norm_dict:
                test_data = np.asarray(test_norm_dict[last_round], dtype=np.float64)
                # Shuffle and sample
                indices = np.random.permutation(len(test_data))
                if len(test_data) > mix_length:
                    test_scores = -test_data[indices[:mix_length]]
                else:
                    test_scores = -test_data[indices]

            if isinstance(mix_norm_dict, dict) and last_round in mix_norm_dict:
                mix_scores = -np.asarray(mix_norm_dict[last_round], dtype=np.float64)

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
