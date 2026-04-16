"""
Loss Series Attack Plugin.

A cross-round membership inference attack that aggregates loss values
across multiple training rounds. This leverages the temporal pattern
of losses to improve attack effectiveness.

Attack Logic
------------
- Collect train_losses from all valid rounds
- For each sample, compute mean loss across rounds
- Member score: -mean_loss (lower average loss = higher score)
- Non-member score: -test_losses (single test set, not per-round)

Member samples tend to have consistently lower losses across rounds
because the model learns to fit them over time.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from ..base import AttackPlugin, AttackResult
from ..registry import AttackRegistry

logger = logging.getLogger(__name__)


@AttackRegistry.register
class LossSeriesPlugin(AttackPlugin):
    """Cross-round loss series-based membership inference attack.

    This attack aggregates loss values across multiple rounds to capture
    temporal patterns that distinguish member from non-member samples.

    Attributes
    ----------
    name : str
        Plugin identifier: ``'loss_series'``.
    is_cross_round : bool
        ``True`` - uses data from multiple training rounds.
    """

    @property
    def name(self) -> str:
        return "loss_series"

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
        """Compute membership scores by aggregating losses across rounds.

        Aligned with original FedMIA: computes loss from logit/labels if available,
        otherwise uses pre-computed losses.
        """
        # Get configuration
        mode = 'mix'
        mix_length = 1000
        if config is not None:
            if hasattr(config, 'attack'):
                mode = getattr(config.attack, 'mode', 'mix')
                mix_length = getattr(config.attack, 'mix_length', 1000)
        
        # Try to use logit/labels for computing loss
        train_logit_dict = target_data.get('train_logit', {})
        train_labels_dict = target_data.get('train_labels', {})
        
        if train_logit_dict and train_labels_dict:
            # Compute loss from logit/labels - aligned with original FedMIA
            valid_rounds = []
            for round_num in sorted(train_logit_dict.keys()):
                if round_num in train_labels_dict:
                    logit = train_logit_dict[round_num]
                    labels = train_labels_dict[round_num]
                    if logit is not None and len(logit) > 0:
                        valid_rounds.append(round_num)
            
            if valid_rounds:
                # Collect member and non-member scores per round (aligned approach)
                all_member_scores = []
                all_nonmember_scores = []
                
                for round_num in valid_rounds:
                    # Member scores for this round
                    loss = self._compute_ce_loss(train_logit_dict[round_num], train_labels_dict[round_num])
                    member_scores = -loss
                    all_member_scores.append(member_scores)
                    
                    # Non-member scores for this round (fresh sample for mix mode)
                    nonmember_scores = self._get_nonmember_scores(target_data, 'loss', mode, mix_length, round_num=round_num)
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
                        "source": "logit_labels",
                    },
                )
        
        
        # Fallback to pre-computed losses
        train_losses_dict = target_data.get('train_losses', {})

        if not train_losses_dict:
            logger.warning(f"{self.name}: No train_losses available")
            return AttackResult(
                name=self.name,
                scores_member=np.array([]),
                scores_nonmember=np.array([]),
                metadata={"error": "No train_losses data"},
            )

        # Get all valid rounds
        valid_rounds = self._get_valid_rounds(train_losses_dict)

        if not valid_rounds:
            logger.warning(f"{self.name}: No valid rounds found")
            return AttackResult(
                name=self.name,
                scores_member=np.array([]),
                scores_nonmember=np.array([]),
                metadata={"error": "No valid rounds"},
            )

        # Stack losses from all rounds: shape (num_rounds, num_samples)
        loss_stack = []
        for round_num in valid_rounds:
            losses = np.asarray(train_losses_dict[round_num], dtype=np.float64)
            loss_stack.append(losses)

        # Find minimum sample count across rounds (in case of varying sizes)
        min_samples = min(len(l) for l in loss_stack)

        # Truncate all arrays to same length and stack
        loss_stack = np.array([l[:min_samples] for l in loss_stack])

        # Member scores: negative mean loss across rounds
        # Shape: (num_samples,)
        member_scores = -np.mean(loss_stack, axis=0)

        # Get non-member scores with mix mode support
        nonmember_scores = self._get_nonmember_scores(target_data, 'loss', mode, mix_length)
        
        if len(nonmember_scores) == 0:
            logger.warning(f"{self.name}: No non-member losses available")

        return AttackResult(
            name=self.name,
            scores_member=member_scores,
            scores_nonmember=nonmember_scores,
            metadata={
                "num_rounds": len(valid_rounds),
                "rounds_used": valid_rounds,
                "num_members": len(member_scores),
                "num_nonmembers": len(nonmember_scores),
                "source": "precomputed_losses",
            },
        )
