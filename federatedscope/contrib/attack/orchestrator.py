"""
Attack Orchestrator for running multiple membership inference attacks.

The orchestrator coordinates:
- Plugin selection and instantiation
- Attack execution with dependency checking
- Score evaluation and metric computation
- Ablation studies for comparing baseline vs. enhanced methods
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any

import numpy as np

from .base import AttackResult
from .registry import AttackRegistry
from .evaluator import AttackEvaluator
from .heterogeneity.base import HeterogeneityHandler

logger = logging.getLogger(__name__)


class AttackOrchestrator:
    """Orchestrates multiple membership inference attacks.

    This class manages the execution of multiple attack plugins,
    handles heterogeneity-aware preprocessing/postprocessing,
    and aggregates results for comparison.

    Parameters
    ----------
    het_handler : HeterogeneityHandler, optional
        Handler for heterogeneity-aware data/score transformations.
    attack_names : list of str, optional
        Names of specific attacks to run. If None, runs all registered
        plugins that satisfy data availability constraints.
    config : CfgNode, optional
        Configuration object passed to attack plugins.

    Attributes
    ----------
    plugins : list of AttackPlugin
        Instantiated attack plugins.
    evaluator : AttackEvaluator
        Evaluator for computing attack metrics.
    """

    def __init__(
        self,
        het_handler: Optional[HeterogeneityHandler] = None,
        attack_names: Optional[List[str]] = None,
        config: Optional[Any] = None,
    ):
        self.het_handler = het_handler
        self.config = config or {}

        # Instantiate plugins
        if attack_names is not None:
            self.plugins = []
            for name in attack_names:
                try:
                    plugin_class = AttackRegistry.get(name)
                    self.plugins.append(plugin_class())
                except KeyError as e:
                    logger.warning(f"Unknown attack plugin '{name}': {e}")
        else:
            # Get all plugins (default filters apply)
            self.plugins = AttackRegistry.get_all()

        # Create evaluator
        self.evaluator = AttackEvaluator()

        logger.info(
            f"AttackOrchestrator initialized with {len(self.plugins)} plugins: "
            f"{[p.name for p in self.plugins]}"
        )

    def run_all_attacks(
        self,
        target_data: Dict,
        shadow_data: Optional[List[Dict]] = None,
        global_model=None,
    ) -> Dict[str, Dict]:
        """Run all configured attacks and return metrics.

        Parameters
        ----------
        target_data : dict
            Target client's feature dictionary. See ``AttackPlugin`` docstring
            for expected structure.
        shadow_data : list of dict, optional
            Shadow clients' feature dictionaries. Required for attacks with
            ``needs_shadow=True``.
        global_model : torch.nn.Module, optional
            Global model (for white-box attacks).

        Returns
        -------
        dict
            Maps attack name to metrics dict containing:
            - ``auc``: Area Under ROC Curve
            - ``tpr_at_fpr``: Dict mapping FPR thresholds to TPR values
            - ``num_members``: Number of member samples
            - ``num_nonmembers``: Number of non-member samples
            - ``metadata``: Attack-specific metadata
            - ``error``: Error message (if attack failed)
        """
        results: Dict[str, Dict] = {}

        # Apply heterogeneity preprocessing if available
        if self.het_handler is not None:
            target_data, shadow_data = self.het_handler.preprocess_data(
                target_data, shadow_data
            )

        for plugin in self.plugins:
            name = plugin.name

            # Check dependencies
            if plugin.needs_shadow and shadow_data is None:
                logger.warning(
                    f"Skipping '{name}': requires shadow_data but none provided"
                )
                results[name] = {
                    "error": "requires shadow_data",
                    "auc": float("nan"),
                    "tpr_at_fpr": {},
                }
                continue

            try:
                # Run attack
                result = plugin.compute_scores(
                    target_data=target_data,
                    shadow_data=shadow_data,
                    global_model=global_model,
                    config=self.config,
                )

                # Apply heterogeneity postprocessing if available
                if self.het_handler is not None:
                    result.scores_member, result.scores_nonmember = \
                        self.het_handler.postprocess_scores(
                            result.scores_member,
                            result.scores_nonmember,
                            name,
                        )

                # Evaluate
                metrics = self.evaluator.evaluate(result)
                results[name] = metrics

                # Log results
                auc = metrics.get("auc", float("nan"))
                tpr_01 = metrics.get("tpr_at_fpr", {}).get(0.01, float("nan"))
                logger.info(
                    f"Attack '{name}': AUC={auc:.4f}, TPR@0.01={tpr_01:.4f}"
                )

            except Exception as e:
                logger.error(f"Attack '{name}' failed: {e}")
                results[name] = {
                    "error": str(e),
                    "auc": float("nan"),
                    "tpr_at_fpr": {},
                }

        return results

    def run_from_pkls(
        self,
        pkl_dir: str,
        num_clients: int = 10,
        target_client_id: int = 1,
    ) -> Dict[str, Dict]:
        """Run all attacks from PKL files directory.

        Convenience method that loads PKL files and runs all configured
        attacks in a single call.

        Parameters
        ----------
        pkl_dir : str
            Directory containing PKL files saved by FedMIA server.
        num_clients : int, default=10
            Total number of clients in the federation.
        target_client_id : int, default=1
            ID of the target client (attacker's target).

        Returns
        -------
        dict
            Maps attack name to metrics dict (same as run_all_attacks).
        """
        from .data_loader import PKLDataLoader

        loader = PKLDataLoader(
            pkl_dir=pkl_dir,
            num_clients=num_clients,
            target_client_id=target_client_id,
        )
        target_data, shadow_data = loader.load_target_and_shadow()

        return self.run_all_attacks(
            target_data=target_data,
            shadow_data=shadow_data,
        )

    def run_ablation(
        self,
        target_data_baseline: Dict,
        target_data_ggeur: Dict,
        shadow_data_baseline: Optional[List[Dict]] = None,
        shadow_data_ggeur: Optional[List[Dict]] = None,
        global_model_baseline=None,
        global_model_ggeur=None,
    ) -> Dict[str, Dict]:
        """Run ablation study comparing baseline vs. GGEUR-enhanced attacks.

        Parameters
        ----------
        target_data_baseline : dict
            Target client data from baseline (no heterogeneity handling).
        target_data_ggeur : dict
            Target client data from GGEUR-enhanced training.
        shadow_data_baseline : list of dict, optional
            Shadow data for baseline.
        shadow_data_ggeur : list of dict, optional
            Shadow data for GGEUR.
        global_model_baseline : torch.nn.Module, optional
            Baseline global model.
        global_model_ggeur : torch.nn.Module, optional
            GGEUR global model.

        Returns
        -------
        dict
            Maps attack name to ablation results containing:
            - ``baseline_auc``: AUC from baseline attack
            - ``ggeur_auc``: AUC from GGEUR attack
            - ``delta_auc``: Difference (ggeur_auc - baseline_auc)
            - ``baseline_tpr01``: TPR@0.01 from baseline
            - ``ggeur_tpr01``: TPR@0.01 from GGEUR
            - ``delta_tpr01``: Difference in TPR@0.01
        """
        logger.info("Running ablation study: baseline vs. GGEUR")

        # Run attacks on both conditions
        baseline_results = self.run_all_attacks(
            target_data=target_data_baseline,
            shadow_data=shadow_data_baseline,
            global_model=global_model_baseline,
        )

        ggeur_results = self.run_all_attacks(
            target_data=target_data_ggeur,
            shadow_data=shadow_data_ggeur,
            global_model=global_model_ggeur,
        )

        # Compare results
        ablation: Dict[str, Dict] = {}

        all_attack_names = set(baseline_results.keys()) | set(ggeur_results.keys())

        for name in all_attack_names:
            baseline = baseline_results.get(name, {})
            ggeur = ggeur_results.get(name, {})

            baseline_auc = baseline.get("auc", float("nan"))
            ggeur_auc = ggeur.get("auc", float("nan"))
            delta_auc = ggeur_auc - baseline_auc if not (
                np.isnan(baseline_auc) or np.isnan(ggeur_auc)
            ) else float("nan")

            baseline_tpr01 = baseline.get("tpr_at_fpr", {}).get(0.01, float("nan"))
            ggeur_tpr01 = ggeur.get("tpr_at_fpr", {}).get(0.01, float("nan"))
            delta_tpr01 = ggeur_tpr01 - baseline_tpr01 if not (
                np.isnan(baseline_tpr01) or np.isnan(ggeur_tpr01)
            ) else float("nan")

            ablation[name] = {
                "baseline_auc": baseline_auc,
                "ggeur_auc": ggeur_auc,
                "delta_auc": delta_auc,
                "baseline_tpr01": baseline_tpr01,
                "ggeur_tpr01": ggeur_tpr01,
                "delta_tpr01": delta_tpr01,
            }

            logger.info(
                f"Ablation '{name}': "
                f"AUC {baseline_auc:.4f} → {ggeur_auc:.4f} (Δ={delta_auc:+.4f}), "
                f"TPR@0.01 {baseline_tpr01:.4f} → {ggeur_tpr01:.4f} (Δ={delta_tpr01:+.4f})"
            )

        return ablation

    def __repr__(self) -> str:
        plugin_names = [p.name for p in self.plugins]
        het_name = self.het_handler.name if self.het_handler else None
        return (
            f"AttackOrchestrator("
            f"plugins={plugin_names}, "
            f"het_handler={het_name})"
        )
