"""
Base classes for the modular membership inference attack framework.

Attack Plugin Interface
-----------------------
Each attack is implemented as an ``AttackPlugin`` subclass and registered
via ``AttackRegistry.register``.  At evaluation time the framework collects
all registered plugins, filters them based on data availability, calls
``compute_scores`` on each one, and packages the results.

``target_data`` / ``shadow_data`` structure
-------------------------------------------
Both ``target_data`` and every element of ``shadow_data`` share the same
structure (a plain ``dict``)::

    {
        # ---- per-round training-set values ----
        'train_losses':    dict[int, np.ndarray],  # round_id -> per-sample loss
        'train_logit':     dict[int, torch.Tensor], # round_id -> logits for CE loss
        'train_labels':    dict[int, torch.Tensor], # round_id -> labels for CE loss
        'train_cos':       dict[int, np.ndarray],  # round_id -> gradient cosine similarity
        'train_grad_diff': dict[int, np.ndarray],  # round_id -> gradient difference norm
        'train_grad_norm': dict[int, np.ndarray],  # round_id -> gradient norm

        # ---- test-set (non-member) values ----
        'test_losses':    np.ndarray,              # per-sample loss on test set
        'test_logit':     dict[int, torch.Tensor], # round_id -> test logits for CE loss
        'test_labels':    dict[int, torch.Tensor], # round_id -> test labels for CE loss
        'test_cos':       dict[int, np.ndarray],   # round_id -> test cosine similarity
        'test_grad_diff': dict[int, np.ndarray],   # round_id -> test gradient diff
        'test_grad_norm': dict[int, np.ndarray],   # round_id -> test gradient norm

        # ---- mix-set (additional non-member for mix mode) ----
        'mix_losses':     np.ndarray,              # per-sample loss on mix set
        'mix_logit':      dict[int, torch.Tensor], # round_id -> mix logits for CE loss
        'mix_labels':     dict[int, torch.Tensor], # round_id -> mix labels for CE loss
        'mix_cos':        dict[int, np.ndarray],   # round_id -> mix cosine similarity
        'mix_grad_diff':  dict[int, np.ndarray],   # round_id -> mix gradient diff
        'mix_grad_norm':  dict[int, np.ndarray],   # round_id -> mix gradient norm
    }

Keys may be absent when the corresponding data is unavailable (e.g. gradient
features are skipped when the model does not retain gradients).  Each plugin
must handle missing keys gracefully.

Attack Configuration
--------------------
The ``config`` parameter passed to ``compute_scores`` should contain:

    attack.mode: 'test' or 'mix' (default: 'mix')
    attack.mix_length: number of test samples to use in mix mode (default: 1000)
```

# Original docstring after this
``shadow_data`` is a ``List[dict]`` with one entry per shadow client.
It is ``None`` when shadow-model-based attacks are not configured.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AttackResult
# ---------------------------------------------------------------------------

@dataclass
class AttackResult:
    """Container for the output of a single attack plugin.

    Attributes
    ----------
    name : str
        Human-readable attack name (matches ``AttackPlugin.name``).
    scores_member : np.ndarray
        Attack scores for *member* samples (shape ``(N_member,)``).
        Higher scores should indicate higher membership likelihood.
    scores_nonmember : np.ndarray
        Attack scores for *non-member* samples (shape ``(N_nonmember,)``).
    metadata : dict, optional
        Any extra diagnostic information produced by the plugin
        (e.g. fitted distribution parameters, per-round intermediates).
    """

    name: str
    scores_member: np.ndarray
    scores_nonmember: np.ndarray
    metadata: Optional[Dict] = field(default=None)

    def __post_init__(self):
        self.scores_member = np.asarray(self.scores_member, dtype=np.float64)
        self.scores_nonmember = np.asarray(self.scores_nonmember, dtype=np.float64)

    @property
    def all_scores(self) -> np.ndarray:
        """Concatenated scores: members first, then non-members."""
        return np.concatenate([self.scores_member, self.scores_nonmember])

    @property
    def all_labels(self) -> np.ndarray:
        """Binary ground-truth labels (1 = member, 0 = non-member)."""
        return np.concatenate([
            np.ones(len(self.scores_member), dtype=np.int32),
            np.zeros(len(self.scores_nonmember), dtype=np.int32),
        ])

    def __repr__(self) -> str:
        return (
            f"AttackResult(name={self.name!r}, "
            f"n_member={len(self.scores_member)}, "
            f"n_nonmember={len(self.scores_nonmember)})"
        )


# ---------------------------------------------------------------------------
# AttackPlugin (abstract base class)
# ---------------------------------------------------------------------------

class AttackPlugin(ABC):
    """Abstract base class for membership inference attack plugins.

    Subclasses must:
    1. Override the ``name`` property to return a unique string identifier.
    2. Implement ``compute_scores``.
    3. Optionally override ``needs_gradient``, ``is_cross_round``,
       and ``needs_shadow`` to declare data requirements.

    The framework uses the requirement flags to skip plugins whose required
    data is not available in the current experiment configuration.

    Example
    -------
    >>> from federatedscope.contrib.attack.base import AttackPlugin, AttackResult
    >>> from federatedscope.contrib.attack.registry import AttackRegistry
    >>>
    >>> @AttackRegistry.register
    ... class MyAttack(AttackPlugin):
    ...     @property
    ...     def name(self) -> str:
    ...         return "my_attack"
    ...
    ...     def compute_scores(self, target_data, shadow_data=None,
    ...                        global_model=None, config=None):
    ...         member_scores = -target_data['train_losses'][
    ...             max(target_data['train_losses'])]
    ...         nonmember_scores = -target_data['test_losses']
    ...         return AttackResult(self.name, member_scores, nonmember_scores)
    """

    # ------------------------------------------------------------------
    # Properties that describe data requirements (override as needed)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique string identifier for this attack (e.g. ``'blackbox_loss'``)."""

    @property
    def needs_gradient(self) -> bool:
        """Return ``True`` if the plugin requires per-sample gradient features.

        When ``False`` (default) the plugin only uses loss values.
        """
        return False

    @property
    def is_cross_round(self) -> bool:
        """Return ``True`` if the plugin uses features from *multiple* rounds.

        When ``False`` (default) only the last (or a single) round is needed.
        """
        return False

    @property
    def needs_shadow(self) -> bool:
        """Return ``True`` if the plugin requires shadow-client data.

        When ``False`` (default) only ``target_data`` is consumed.
        """
        return False

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_scores(
        self,
        target_data: Dict,
        shadow_data: Optional[List[Dict]] = None,
        global_model=None,
        config=None,
    ) -> AttackResult:
        """Compute membership scores for the target client's samples.

        Parameters
        ----------
        target_data : dict
            Feature dictionary for the *target* client.
            See module-level docstring for the expected structure.
        shadow_data : list of dict, optional
            Feature dictionaries for shadow clients.
            Present only when ``needs_shadow`` is ``True``.
        global_model : torch.nn.Module, optional
            The global model at the end of training.  Available when the
            attacker has white-box access.
        config : yacs CfgNode, optional
            Full experiment configuration (``cfg``).

        Returns
        -------
        AttackResult
            Populated result object with per-sample membership scores.
        """

    # ------------------------------------------------------------------
    # Convenience helpers (available to all subclasses)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_last_round(round_dict: Dict) -> np.ndarray:
        """Return the value for the largest round key in *round_dict*."""
        if not round_dict:
            raise ValueError("round_dict is empty")
        last_key = max(round_dict.keys())
        return np.asarray(round_dict[last_key], dtype=np.float64)

    @staticmethod
    def _get_sorted_rounds(round_dict: Dict) -> List[int]:
        """Return round keys sorted in ascending order."""
        return sorted(round_dict.keys())

    @staticmethod
    def _compute_ce_loss(logit: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
        """Compute per-sample cross-entropy loss from logit and labels."""
        if logit is None or len(logit) == 0 or labels is None or len(labels) == 0:
            return np.array([])
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        return loss_fn(logit, labels).cpu().numpy()

    @staticmethod
    def _get_nonmember_scores(
        target_data: Dict,
        attack_mode: str,
        mode: str = 'mix',
        mix_length: int = 1000,
        round_num: Optional[int] = None,
    ) -> np.ndarray:
        """Get non-member scores with mix mode support.
        
        Args:
            target_data: Target client data dict
            attack_mode: 'loss', 'cos', 'diff', 'norm'
            mode: 'test' or 'mix'
            mix_length: number of test samples to use in mix mode
            round_num: specific round to get test/mix data for (required for loss)
            
        Returns:
            Non-member scores array
        """
        key_map = {
            'loss': ('test_losses', 'mix_losses'),
            'cos': ('test_cos', 'mix_cos'),
            'diff': ('test_grad_diff', 'mix_grad_diff'),
            'norm': ('test_grad_norm', 'mix_grad_norm'),
        }
        test_key, mix_key = key_map[attack_mode]
        
        if attack_mode == 'loss':
            # For loss, try computing from logit/labels per round
            test_logit_dict = target_data.get('test_logit', {})
            test_labels_dict = target_data.get('test_labels', {})
            mix_logit_dict = target_data.get('mix_logit', {})
            mix_labels_dict = target_data.get('mix_labels', {})
            
            # Use round_num if provided, otherwise use any available round
            if round_num is not None and round_num in test_logit_dict:
                test_logit = test_logit_dict[round_num]
                test_labels = test_labels_dict[round_num]
                mix_logit = mix_logit_dict.get(round_num)
                mix_labels = mix_labels_dict.get(round_num)
            elif test_logit_dict:
                # Fallback: use first available round
                first_round = min(test_logit_dict.keys())
                test_logit = test_logit_dict[first_round]
                test_labels = test_labels_dict[first_round]
                mix_logit = mix_logit_dict.get(first_round)
                mix_labels = mix_labels_dict.get(first_round)
            else:
                # No logit/labels available
                test_logit = None
                mix_logit = None
            
            if test_logit is not None and len(test_logit) > 0:
                test_loss = AttackPlugin._compute_ce_loss(test_logit, test_labels)
                test_scores = -test_loss
                
                if mode == 'mix':
                    # Always shuffle test samples (like original implementation)
                    indices = torch.randperm(len(test_scores))
                    if len(test_scores) > mix_length:
                        test_scores = test_scores[indices.numpy()[:mix_length]]
                    else:
                        test_scores = test_scores[indices.numpy()]
                    
                    if mix_logit is not None and len(mix_logit) > 0:
                        mix_loss = AttackPlugin._compute_ce_loss(mix_logit, mix_labels)
                        mix_scores = -mix_loss
                        if len(mix_scores) > 0:
                            return np.concatenate([test_scores, mix_scores])
                
                return test_scores
                
                
        # Fallback to pre-computed values (now per-round dicts for cos/diff/norm)
        test_dict = target_data.get(test_key, {})
        mix_dict = target_data.get(mix_key, {})
                
        # Get data for specific round
        if isinstance(test_dict, dict) and round_num is not None and round_num in test_dict:
            test_scores = np.asarray(test_dict[round_num], dtype=np.float64)
            mix_scores = np.asarray(mix_dict.get(round_num, []), dtype=np.float64)
        elif isinstance(test_dict, dict) and test_dict:
            # Fallback: use first available round
            first_round = min(test_dict.keys())
            test_scores = np.asarray(test_dict[first_round], dtype=np.float64)
            mix_scores = np.asarray(mix_dict.get(first_round, []), dtype=np.float64)
        elif isinstance(test_dict, np.ndarray):
            # Legacy: test_dict is actually an array (not per-round)
            test_scores = np.asarray(test_dict, dtype=np.float64)
            mix_scores = np.asarray(mix_dict, dtype=np.float64)
        else:
            return np.array([])
                
        # Apply negation for diff/norm (lower is better for these)
        if attack_mode in ['diff', 'norm']:
            test_scores = -test_scores if len(test_scores) > 0 else np.array([])
            mix_scores = -mix_scores if len(mix_scores) > 0 else np.array([])
                
        if mode == 'test':
            return test_scores
                
        elif mode == 'mix':
            # Always shuffle test samples (like original implementation)
            if len(test_scores) > 0:
                indices = torch.randperm(len(test_scores))
                if len(test_scores) > mix_length:
                    test_scores = test_scores[indices.numpy()[:mix_length]]
                else:
                    test_scores = test_scores[indices.numpy()]
                    
            if len(test_scores) > 0 and len(mix_scores) > 0:
                return np.concatenate([test_scores, mix_scores])
            elif len(test_scores) > 0:
                return test_scores
            elif len(mix_scores) > 0:
                return mix_scores
            else:
                return np.array([])
                
        return np.array([])

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"needs_gradient={self.needs_gradient}, "
            f"is_cross_round={self.is_cross_round}, "
            f"needs_shadow={self.needs_shadow})"
        )
