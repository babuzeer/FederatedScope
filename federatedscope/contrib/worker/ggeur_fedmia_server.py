"""
GGEUR-enhanced FedMIA Server.

This server extends FedMIAServer with:
1. GGEUR heterogeneity handling (Round 0 statistics collection)
2. Modular attack framework integration

The GGEUR enhancement collects local statistics from clients at Round 0,
aggregates them globally, and uses them to augment features for improved
attack effectiveness under non-IID data distributions.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Dict, List, Optional, Any

import numpy as np
import torch

from federatedscope.attack.worker_as_attacker.fedmia_server import FedMIAServer
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


class GGEURFedMIAServer(FedMIAServer):
    """GGEUR-enhanced FedMIA Attack Server.

    This server extends FedMIAServer to support:
    1. GGEUR heterogeneity handling for non-IID data
    2. Modular attack framework with plugin-based attacks

    Attributes
    ----------
    het_handler : GGEURHandler, optional
        Handler for GGEUR heterogeneity processing.
    orchestrator : AttackOrchestrator, optional
        Orchestrator for running modular attacks.
    _global_stats : dict, optional
        Aggregated global statistics from Round 0.
    _round_zero_done : bool
        Whether Round 0 statistics collection is complete.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize GGEUR handler if enabled
        self.het_handler = None
        self._global_stats = None
        self._round_zero_done = False

        if hasattr(self._cfg, 'attack') and getattr(self._cfg.attack, 'use_ggeur', False):
            from federatedscope.contrib.attack.heterogeneity import GGEURHandler
            self.het_handler = GGEURHandler()
            logger.info("GGEURFedMIAServer: GGEUR handler initialized")


        # Initialize modular attack orchestrator if enabled
        self.orchestrator = None

        if hasattr(self._cfg, 'attack') and getattr(self._cfg.attack, 'modular_attacks', False):
            # Import all plugins to ensure registration
            import federatedscope.contrib.attack.plugins.blackbox_loss
            import federatedscope.contrib.attack.plugins.grad_cosine
            import federatedscope.contrib.attack.plugins.grad_diff
            import federatedscope.contrib.attack.plugins.grad_norm
            import federatedscope.contrib.attack.plugins.loss_series
            import federatedscope.contrib.attack.plugins.avg_cosine
            import federatedscope.contrib.attack.plugins.fedmia_i
            import federatedscope.contrib.attack.plugins.fedmia_ii

            from federatedscope.contrib.attack.orchestrator import AttackOrchestrator

            # Get attack plugin names from config
            attack_names = list(getattr(self._cfg.attack, 'attack_plugins', []))

            # Create orchestrator config
            orchestrator_config = {
                'target_client_id': self.target_client_id,
                'ggeur_num_per_sample': getattr(self._cfg.attack, 'ggeur_num_per_sample', 50),
                'ggeur_target_size': getattr(self._cfg.attack, 'ggeur_target_size', 50),
                'ggeur_use_cross_client': getattr(self._cfg.attack, 'ggeur_use_cross_client', True),
            }

            self.orchestrator = AttackOrchestrator(
                het_handler=self.het_handler,
                attack_names=attack_names if attack_names else None,
                config=orchestrator_config,
            )
            logger.info(f"GGEURFedMIAServer: Modular attack orchestrator initialized "
                       f"with {len(self.orchestrator.plugins)} plugins")

    def run_final_attack(self):
        """Execute attacks using either modular framework or legacy method."""
        if self.orchestrator is not None:
            self._run_modular_attacks()
        else:
            # Fall back to parent class implementation
            super().run_final_attack()

    def _run_modular_attacks(self):
        """Run attacks using the modular framework."""
        logger.info("=" * 20 + " GGEUR FedMIA Modular Attack " + "=" * 20)

        if self.pkl_dir is None or not os.path.exists(self.pkl_dir):
            logger.warning("No PKL files found, cannot execute attack")
            return

        # Load all PKL files
        pkl_data = self._load_all_pkl_files()
        if not pkl_data:
            logger.warning("Failed to load PKL data")
            return

        # Get all round numbers
        all_rounds = sorted(set(
            int(f.split('epoch')[1].split('.')[0])
            for f in os.listdir(self.pkl_dir)
            if f.endswith('.pkl')
        ))

        logger.info(f"Loaded PKL data for rounds: {all_rounds}")

        # Convert PKL data to target_data and shadow_data format
        target_data, shadow_data = self._convert_pkl_to_plugin_format(pkl_data, all_rounds)

        # Run attacks using orchestrator
        results = self.orchestrator.run_all_attacks(
            target_data=target_data,
            shadow_data=shadow_data,
            global_model=self.model,
        )

        # Print and save results
        self._print_and_save_modular_results(results, all_rounds)

    def _convert_pkl_to_plugin_format(
        self,
        pkl_data: Dict,
        rounds: List[int],
    ) -> tuple:
        """Convert PKL data to the format expected by AttackPlugin.

        PKL structure (from FedMIAServer):
            {
                'train_res': {'loss': array, 'logit': tensor, 'labels': tensor},
                'test_res': {'loss': array, ...},
                'tarin_cos': list,  # Note: typo in original
                'test_cos': list,
                'tarin_diffs': list,
                'test_diffs': list,
                'tarin_grad_norm': list,
                'test_grad_norm': list,
            }

        Plugin target_data structure:
            {
                'train_losses': {round: array},
                'test_losses': array,
                'train_cos': {round: array},
                'test_cos': array,
                'train_grad_diff': {round: array},
                'test_grad_diff': array,
                'train_grad_norm': {round: array},
                'test_grad_norm': array,
            }

        Parameters
        ----------
        pkl_data : dict
            {round -> {client_id -> data}}
        rounds : list
            List of round numbers.

        Returns
        -------
        tuple
            (target_data, shadow_data_list)
        """
        # Find target client data across all rounds
        target_data = {
            'train_losses': {},
            'test_losses': np.array([]),
            'train_logit': {},
            'train_labels': {},
            'test_logit': {},   # Per-round test logits
            'test_labels': {},  # Per-round test labels
            'mix_logit': {},    # Per-round mix logits
            'mix_labels': {},   # Per-round mix labels
            'train_cos': {},
            'test_cos': {},     # Per-round test cosine (dict[round -> array])
            'mix_cos': {},      # Per-round mix cosine (dict[round -> array])
            'train_grad_diff': {},
            'test_grad_diff': np.array([]),
            'train_grad_norm': {},
            'test_grad_norm': np.array([]),
        }

        # Extract per-round data for target client (test/mix also per-round)
        for round_num in rounds:
            if round_num in pkl_data and self.target_client_id in pkl_data[round_num]:
                client_pkl = pkl_data[round_num][self.target_client_id]

                # Train losses
                train_res = client_pkl.get('train_res', {})
                if isinstance(train_res, dict):
                    if 'loss' in train_res and len(train_res.get('loss', [])) > 0:
                        target_data['train_losses'][round_num] = np.asarray(
                            train_res['loss'], dtype=np.float64
                        )
                    if 'logit' in train_res and len(train_res.get('logit', [])) > 0:
                        target_data['train_logit'][round_num] = train_res['logit']
                        target_data['train_labels'][round_num] = train_res['labels']

                # Test/mix logit per-round
                test_res = client_pkl.get('test_res', {})
                mix_res = client_pkl.get('mix_res', {})
                if isinstance(test_res, dict) and 'logit' in test_res and len(test_res.get('logit', [])) > 0:
                    target_data['test_logit'][round_num] = test_res['logit']
                    target_data['test_labels'][round_num] = test_res['labels']
                if isinstance(mix_res, dict) and 'logit' in mix_res and len(mix_res.get('logit', [])) > 0:
                    target_data['mix_logit'][round_num] = mix_res['logit']
                    target_data['mix_labels'][round_num] = mix_res['labels']

                # Train cosine (note: typo 'tarin_cos' in original)
                train_cos = client_pkl.get('tarin_cos', [])
                if train_cos is not None and len(train_cos) > 0:
                    target_data['train_cos'][round_num] = np.asarray(
                        [x.item() if hasattr(x, 'item') else x for x in train_cos], dtype=np.float64
                    )

                # Test/mix cosine per-round (per-client, use target client's own data)
                test_cos = client_pkl.get('test_cos', [])
                mix_cos = client_pkl.get('mix_cos', [])
                if test_cos is not None and len(test_cos) > 0:
                    target_data['test_cos'][round_num] = np.asarray(
                        [x.item() if hasattr(x, 'item') else x for x in test_cos], dtype=np.float64
                    )
                if mix_cos is not None and len(mix_cos) > 0:
                    target_data['mix_cos'][round_num] = np.asarray(
                        [x.item() if hasattr(x, 'item') else x for x in mix_cos], dtype=np.float64
                    )

                # Train gradient diff
                train_diffs = client_pkl.get('tarin_diffs', [])
                if train_diffs is not None and len(train_diffs) > 0:
                    target_data['train_grad_diff'][round_num] = np.asarray(
                        train_diffs, dtype=np.float64
                    )

                # Train gradient norm
                train_norm = client_pkl.get('tarin_grad_norm', [])
                if train_norm is not None and len(train_norm) > 0:
                    target_data['train_grad_norm'][round_num] = np.asarray(
                        train_norm, dtype=np.float64
                    )

        # Build shadow data list
        shadow_data = []

        # Get all client IDs except target
        all_client_ids = set()
        for round_num in rounds:
            if round_num in pkl_data:
                all_client_ids.update(pkl_data[round_num].keys())
        all_client_ids.discard(self.target_client_id)

        for client_id in sorted(all_client_ids):
            shadow_client_data = {
                'train_losses': {},
                'train_logit': {},
                'train_labels': {},
                'test_logit': {},
                'test_labels': {},
                'mix_logit': {},
                'mix_labels': {},
                'train_cos': {},
                'test_cos': {},
                'mix_cos': {},
                'train_grad_diff': {},
                'train_grad_norm': {},
            }

            for round_num in rounds:
                if round_num in pkl_data and client_id in pkl_data[round_num]:
                    client_pkl = pkl_data[round_num][client_id]

                    # Train losses
                    train_res = client_pkl.get('train_res', {})
                    if isinstance(train_res, dict) and 'loss' in train_res and len(train_res.get('loss', [])) > 0:
                        shadow_client_data['train_losses'][round_num] = np.asarray(
                            train_res['loss'], dtype=np.float64
                        )
                    if isinstance(train_res, dict) and 'logit' in train_res and len(train_res.get('logit', [])) > 0:
                        shadow_client_data['train_logit'][round_num] = train_res['logit']
                        shadow_client_data['train_labels'][round_num] = train_res['labels']

                    # Test/mix logit per-round
                    test_res = client_pkl.get('test_res', {})
                    mix_res = client_pkl.get('mix_res', {})
                    if isinstance(test_res, dict) and 'logit' in test_res and len(test_res.get('logit', [])) > 0:
                        shadow_client_data['test_logit'][round_num] = test_res['logit']
                        shadow_client_data['test_labels'][round_num] = test_res['labels']
                    if isinstance(mix_res, dict) and 'logit' in mix_res and len(mix_res.get('logit', [])) > 0:
                        shadow_client_data['mix_logit'][round_num] = mix_res['logit']
                        shadow_client_data['mix_labels'][round_num] = mix_res['labels']

                    # Train cosine
                    train_cos = client_pkl.get('tarin_cos', [])
                    if train_cos is not None and len(train_cos) > 0:
                        shadow_client_data['train_cos'][round_num] = np.asarray(
                            [x.item() if hasattr(x, 'item') else x for x in train_cos], dtype=np.float64
                        )

                    # Test/mix cosine per-round
                    test_cos = client_pkl.get('test_cos', [])
                    mix_cos = client_pkl.get('mix_cos', [])
                    if test_cos is not None and len(test_cos) > 0:
                        shadow_client_data['test_cos'][round_num] = np.asarray(
                            [x.item() if hasattr(x, 'item') else x for x in test_cos], dtype=np.float64
                        )
                    if mix_cos is not None and len(mix_cos) > 0:
                        shadow_client_data['mix_cos'][round_num] = np.asarray(
                            [x.item() if hasattr(x, 'item') else x for x in mix_cos], dtype=np.float64
                        )

                    # Train gradient diff
                    train_diffs = client_pkl.get('tarin_diffs', [])
                    if train_diffs is not None and len(train_diffs) > 0:
                        shadow_client_data['train_grad_diff'][round_num] = np.asarray(
                            train_diffs, dtype=np.float64
                        )

                    # Train gradient norm
                    train_norm = client_pkl.get('tarin_grad_norm', [])
                    if train_norm is not None and len(train_norm) > 0:
                        shadow_client_data['train_grad_norm'][round_num] = np.asarray(
                            train_norm, dtype=np.float64
                        )

            shadow_data.append(shadow_client_data)

        logger.info(f"Converted PKL data: target_data has {len(target_data['train_losses'])} rounds, "
                   f"test_cos rounds: {len(target_data['test_cos'])}, "
                   f"{len(shadow_data)} shadow clients")

        return target_data, shadow_data

    def _print_and_save_modular_results(self, results: Dict, rounds: List[int]):
        """Print and save modular attack results."""
        logger.info("\n" + "=" * 60)
        logger.info("GGEUR FedMIA Modular Attack Results")
        logger.info("=" * 60)

        output = {
            'rounds': rounds,
            'attacks': {}
        }

        for attack_name, metrics in results.items():
            if 'error' in metrics:
                logger.warning(f"\n{attack_name}: Error - {metrics['error']}")
                output['attacks'][attack_name] = {'error': metrics['error']}
                continue

            auc = metrics.get('auc', float('nan'))
            tpr_at_fpr = metrics.get('tpr_at_fpr', {})
            tpr_01 = tpr_at_fpr.get(0.1, float('nan'))
            tpr_001 = tpr_at_fpr.get(0.01, float('nan'))
            tpr_0001 = tpr_at_fpr.get(0.001, float('nan'))

            logger.info(f"\n{attack_name}:")
            logger.info(f"  AUC: {auc:.4f}")
            logger.info(f"  TPR@FPR=0.1: {tpr_01:.4f}")
            logger.info(f"  TPR@FPR=0.01: {tpr_001:.4f}")
            logger.info(f"  TPR@FPR=0.001: {tpr_0001:.4f}")
            logger.info(f"  Members: {metrics.get('num_members', 'N/A')}, "
                       f"Non-members: {metrics.get('num_nonmembers', 'N/A')}")

            output['attacks'][attack_name] = {
                'auc': float(auc) if not np.isnan(auc) else None,
                'tpr@0.1': float(tpr_01) if not np.isnan(tpr_01) else None,
                'tpr@0.01': float(tpr_001) if not np.isnan(tpr_001) else None,
                'tpr@0.001': float(tpr_0001) if not np.isnan(tpr_0001) else None,
                'num_members': metrics.get('num_members'),
                'num_nonmembers': metrics.get('num_nonmembers'),
            }

        # Save to JSON file
        results_path = os.path.join(self.pkl_dir, "modular_attack_results.log")
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to: {results_path}")
