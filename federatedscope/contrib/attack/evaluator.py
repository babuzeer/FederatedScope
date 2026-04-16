"""
Attack evaluation utilities for membership inference.

Provides metrics such as AUC and TPR@FPR that quantify how well an attack
distinguishes member samples from non-member samples.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class AttackEvaluator:
    """Evaluates membership inference attack results.

    The primary metric is AUC (Area Under the ROC Curve).  In privacy
    contexts, TPR at low FPR thresholds (e.g., 1%, 0.1%) is often more
    informative than overall AUC, because a successful privacy breach
    can occur even if the attacker achieves a modest true positive rate
    while keeping false positives extremely low.

    Example
    -------
    >>> from federatedscope.contrib.attack.base import AttackResult
    >>> from federatedscope.contrib.attack.evaluator import AttackEvaluator
    >>> evaluator = AttackEvaluator()
    >>> result = AttackResult(
    ...     name='test',
    ...     scores_member=np.random.randn(100) + 0.5,
    ...     scores_nonmember=np.random.randn(100) - 0.5,
    ... )
    >>> metrics = evaluator.evaluate(result)
    >>> print(f"AUC = {metrics['auc']:.3f}")
    """

    # FPR thresholds at which to report TPR
    FPR_THRESHOLDS = [0.1, 0.01, 0.001]

    def evaluate(self, result) -> Dict:
        """Compute evaluation metrics for an ``AttackResult``.

        Parameters
        ----------
        result : AttackResult
            Attack output containing member and non-member scores.

        Returns
        -------
        dict
            Contains:
            - ``auc``: Area Under ROC Curve (float, may be NaN)
            - ``tpr_at_fpr``: Dict mapping each FPR threshold to TPR
            - ``num_members``: Number of member samples after NaN filtering
            - ``num_nonmembers``: Number of non-member samples after filtering
            - ``metadata``: Copy of ``result.metadata``
        """
        # Import sklearn locally to avoid hard dependency if not needed
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for AttackEvaluator.evaluate(). "
                "Install with: pip install scikit-learn"
            ) from exc

        # Concatenate scores and build labels
        scores_member = np.asarray(result.scores_member, dtype=np.float64)
        scores_nonmember = np.asarray(result.scores_nonmember, dtype=np.float64)

        scores = np.concatenate([scores_member, scores_nonmember])
        labels = np.concatenate([
            np.ones(len(scores_member), dtype=np.int32),
            np.zeros(len(scores_nonmember), dtype=np.int32),
        ])

        # Filter out NaN values
        valid_mask = ~np.isnan(scores)
        n_valid_members = int(np.sum(valid_mask[:len(scores_member)]))
        n_valid_nonmembers = int(np.sum(valid_mask[len(scores_member):]))

        if not np.any(valid_mask):
            logger.warning(
                f"AttackEvaluator: all scores are NaN for attack '{result.name}'"
            )
            return {
                "auc": float("nan"),
                "tpr_at_fpr": {fpr: float("nan") for fpr in self.FPR_THRESHOLDS},
                "num_members": 0,
                "num_nonmembers": 0,
                "metadata": result.metadata,
            }

        scores_valid = scores[valid_mask]
        labels_valid = labels[valid_mask]

        # Compute AUC
        try:
            auc = float(roc_auc_score(labels_valid, scores_valid))
        except ValueError as exc:
            logger.warning(f"AttackEvaluator: cannot compute AUC: {exc}")
            auc = float("nan")

        # Compute ROC curve and interpolate TPR at specified FPR thresholds
        try:
            fpr_arr, tpr_arr, _ = roc_curve(labels_valid, scores_valid)
            tpr_at_fpr = {}
            for fpr_thresh in self.FPR_THRESHOLDS:
                tpr = self._interpolate_tpr_at_fpr(fpr_arr, tpr_arr, fpr_thresh)
                tpr_at_fpr[fpr_thresh] = float(tpr)
        except Exception as exc:
            logger.warning(f"AttackEvaluator: failed to compute ROC curve: {exc}")
            tpr_at_fpr = {fpr: float("nan") for fpr in self.FPR_THRESHOLDS}

        return {
            "auc": auc,
            "tpr_at_fpr": tpr_at_fpr,
            "num_members": n_valid_members,
            "num_nonmembers": n_valid_nonmembers,
            "metadata": result.metadata,
        }

    @staticmethod
    def _interpolate_tpr_at_fpr(
        fpr_arr: np.ndarray,
        tpr_arr: np.ndarray,
        target_fpr: float,
    ) -> float:
        """Interpolate TPR at a specific FPR threshold.

        Parameters
        ----------
        fpr_arr : np.ndarray
            FPR values from ``roc_curve`` (monotonically increasing).
        tpr_arr : np.ndarray
            Corresponding TPR values.
        target_fpr : float
            Desired FPR threshold (e.g., 0.01).

        Returns
        -------
        float
            Interpolated TPR at ``target_fpr``.  Returns 0.0 if
            ``target_fpr`` is below the smallest FPR in the curve.
        """
        if target_fpr <= 0:
            return 0.0
        if target_fpr >= 1.0:
            return 1.0

        # fpr_arr is monotonically increasing; find insertion point
        idx = np.searchsorted(fpr_arr, target_fpr, side="right")

        if idx == 0:
            # target_fpr is smaller than all FPR values
            return 0.0
        if idx >= len(fpr_arr):
            # target_fpr is larger than all FPR values
            return float(tpr_arr[-1])

        # Linear interpolation between adjacent points
        fpr_low = fpr_arr[idx - 1]
        fpr_high = fpr_arr[idx]
        tpr_low = tpr_arr[idx - 1]
        tpr_high = tpr_arr[idx]

        if fpr_high == fpr_low:
            return float(tpr_low)

        ratio = (target_fpr - fpr_low) / (fpr_high - fpr_low)
        return float(tpr_low + ratio * (tpr_high - tpr_low))

    def __repr__(self) -> str:
        return f"AttackEvaluator(fpr_thresholds={self.FPR_THRESHOLDS})"
