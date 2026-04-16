"""
FedMIA-II Attack Plugin.

A membership inference attack that uses CDF normalization with shadow clients
to estimate the Q_out distribution. This is the gradient cosine-based variant.

Attack Logic (Aligned with Original FedMIA)
-------------------------------------------
1. For each round where target client participated:
   - Get member scores (cosine similarity) and non-member scores
   - Collect shadow clients' train_cos
   - Estimate per-sample Q_out distribution: mu_out, var_out
   - Compute CDF normalization for member and non-member scores
2. Average CDF scores across all rounds
3. Final score = average CDF across rounds

The key insight is that member samples' gradients have higher cosine
similarity with the global update direction compared to the Q_out
distribution estimated from shadow clients.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

from ..base import AttackPlugin, AttackResult
from ..registry import AttackRegistry

logger = logging.getLogger(__name__)


@AttackRegistry.register
class FedMIAIIPlugin(AttackPlugin):
    """FedMIA-II: CDF-normalized cosine-based membership inference attack.

    This attack uses shadow clients to estimate the Q_out distribution
    (distribution of scores for non-member samples). It computes CDF scores
    per round and averages them for final membership scores.

    Attributes
    ----------
    name : str
        Plugin identifier: ``'fedmia_ii'``.
    needs_gradient : bool
        ``True`` - requires gradient features.
    needs_shadow : bool
        ``True`` - requires shadow client data for Q_out estimation.
    is_cross_round : bool
        ``True`` - aggregates CDF scores across multiple rounds.
    """

    @property
    def name(self) -> str:
        return "fedmia_ii"

    @property
    def needs_gradient(self) -> bool:
        return True

    @property
    def needs_shadow(self) -> bool:
        return True

    @property
    def is_cross_round(self) -> bool:
        return True

    def _get_valid_rounds(self, data_dict: Dict) -> List[int]:
        """Find all rounds with non-None, non-empty data."""
        valid_rounds = []
        for round_num, data in data_dict.items():
            if data is not None:
                arr = np.asarray(data)
                if arr.size > 0 and not np.all(np.isnan(arr)):
                    valid_rounds.append(round_num)
        return sorted(valid_rounds)

    def _get_member_scores_round(
        self,
        target_data: Dict,
        round_num: int,
    ) -> np.ndarray:
        """Get member scores (cosine similarity) for a specific round."""
        train_cos_dict = target_data.get('train_cos', {})
        
        if round_num in train_cos_dict:
            cos_sim = train_cos_dict[round_num]
            if cos_sim is not None and len(cos_sim) > 0:
                return np.asarray([x.item() if hasattr(x, 'item') else x for x in cos_sim], dtype=np.float64)
        
        return np.array([])

    def _get_nonmember_scores_round(
        self,
        target_data: Dict,
        round_num: int,
        mode: str,
        mix_length: int,
    ) -> np.ndarray:
        """Get non-member scores for a specific round with mix mode support."""
        # Use per-round cosine dicts
        test_cos_dict = target_data.get('test_cos', {})
        mix_cos_dict = target_data.get('mix_cos', {})
        
        # Get cosine for this round
        test_cos = test_cos_dict.get(round_num)
        mix_cos = mix_cos_dict.get(round_num)
        
        if test_cos is not None and len(test_cos) > 0:
            test_scores = np.asarray([x.item() if hasattr(x, 'item') else x for x in test_cos], dtype=np.float64)
            
            if mode == 'mix':
                # Always shuffle test samples (like original implementation)
                indices = torch.randperm(len(test_scores))
                if len(test_scores) > mix_length:
                    test_scores = test_scores[indices.numpy()[:mix_length]]
                else:
                    test_scores = test_scores[indices.numpy()]
                
                if mix_cos is not None and len(mix_cos) > 0:
                    mix_scores = np.asarray([x.item() if hasattr(x, 'item') else x for x in mix_cos], dtype=np.float64)
                    if len(mix_scores) > 0:
                        return np.concatenate([test_scores, mix_scores])
            
            return test_scores
        
        return np.array([])

    def _get_shadow_scores_round(
        self,
        shadow_data: List[Dict],
        round_num: int,
    ) -> List[np.ndarray]:
        """Get shadow clients' train scores for a specific round."""
        shadow_scores_list = []
        
        for shadow in shadow_data:
            shadow_train_dict = shadow.get('train_cos', {})
            if round_num in shadow_train_dict:
                cos_sim = shadow_train_dict[round_num]
                if cos_sim is not None and len(cos_sim) > 0:
                    scores = np.asarray([x.item() if hasattr(x, 'item') else x for x in cos_sim], dtype=np.float64)
                    shadow_scores_list.append(scores)
        
        return shadow_scores_list

    def compute_scores(
        self,
        target_data: Dict,
        shadow_data: Optional[List[Dict]] = None,
        global_model=None,
        config=None,
    ) -> AttackResult:
        """Compute CDF-normalized membership scores based on cosine similarity.

        Aligned with original FedMIA implementation:
        1. Compute CDF scores per round
        2. Average CDF scores across rounds
        """
        from scipy.stats import norm
        
        # Get configuration
        mode = 'mix'
        mix_length = 1000
        if config is not None:
            if hasattr(config, 'attack'):
                mode = getattr(config.attack, 'mode', 'mix')
                mix_length = getattr(config.attack, 'mix_length', 1000)
        
        # Get train cosine dict to find valid rounds
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
        
        # Collect CDF scores across rounds
        all_member_cdf = []
        all_nonmember_cdf = []
        
        # Get shadow data
        if shadow_data is None or len(shadow_data) == 0:
            logger.warning(f"{self.name}: No shadow_data available")
            return AttackResult(
                name=self.name,
                scores_member=np.array([]),
                scores_nonmember=np.array([]),
                metadata={"error": "No shadow data"},
            )
        
        for round_num in valid_rounds:
            # Get member scores for this round
            member_scores = self._get_member_scores_round(target_data, round_num)
            if len(member_scores) == 0:
                continue
            
            # Get non-member scores for this round
            nonmember_scores = self._get_nonmember_scores_round(
                target_data, round_num, mode, mix_length)
            if len(nonmember_scores) == 0:
                continue
            
            # Get shadow scores for this round
            shadow_scores_list = self._get_shadow_scores_round(shadow_data, round_num)
            
            if shadow_scores_list:
                # Stack shadow scores for Q_out estimation
                shadow_stack = np.vstack(shadow_scores_list)
                train_mu_out = np.mean(shadow_stack, axis=0) if shadow_stack.shape[0] > 1 else shadow_stack[0]
                train_var_out = np.var(shadow_stack, axis=0) + 1e-8 if shadow_stack.shape[0] > 1 else np.ones_like(shadow_stack[0]) * 0.1
            else:
                # Fallback: use nonmember scores
                train_mu_out = np.mean(nonmember_scores)
                train_var_out = np.var(nonmember_scores) + 1e-8
            
            # Compute CDF scores
            if isinstance(train_mu_out, np.ndarray) and len(train_mu_out) > 1:
                min_len = min(len(member_scores), len(train_mu_out))
                member_cdf = norm.cdf(
                    member_scores[:min_len],
                    train_mu_out[:min_len],
                    np.sqrt(train_var_out[:min_len])
                )
                nonmember_min = min(len(nonmember_scores), len(train_mu_out))
                nonmember_cdf = norm.cdf(
                    nonmember_scores[:nonmember_min],
                    train_mu_out[:nonmember_min],
                    np.sqrt(train_var_out[:nonmember_min])
                )
            else:
                member_cdf = norm.cdf(member_scores, train_mu_out, np.sqrt(train_var_out))
                nonmember_cdf = norm.cdf(nonmember_scores, train_mu_out, np.sqrt(train_var_out))
            
            all_member_cdf.append(member_cdf)
            all_nonmember_cdf.append(nonmember_cdf)
        
        if not all_member_cdf or not all_nonmember_cdf:
            logger.warning(f"{self.name}: No valid CDF scores computed")
            return AttackResult(
                name=self.name,
                scores_member=np.array([]),
                scores_nonmember=np.array([]),
                metadata={"error": "No valid CDF scores"},
            )
        
        # Average CDF across rounds
        min_member_len = min(len(s) for s in all_member_cdf)
        min_nonmember_len = min(len(s) for s in all_nonmember_cdf)
        
        member_avg = np.mean([s[:min_member_len] for s in all_member_cdf], axis=0)
        nonmember_avg = np.mean([s[:min_nonmember_len] for s in all_nonmember_cdf], axis=0)
        
        return AttackResult(
            name=self.name,
            scores_member=member_avg,
            scores_nonmember=nonmember_avg,
            metadata={
                "num_rounds": len(all_member_cdf),
                "rounds_used": valid_rounds[:len(all_member_cdf)],
                "num_members": len(member_avg),
                "num_nonmembers": len(nonmember_avg),
            },
        )
