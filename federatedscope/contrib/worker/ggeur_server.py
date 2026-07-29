"""
GGEUR_Clip Server Implementation

Handles:
1. Collecting local statistics from all clients
2. Aggregating covariance matrices using parallel axis theorem
3. Broadcasting global covariance matrices to clients
4. Standard FedAvg aggregation for MLP training
5. Evaluating on test sets from all domains
6. (Optional) CNN model aggregation and evaluation for knowledge distillation
7. (Optional) CNN model aggregation and evaluation for feature alignment
"""

import os
import time
import logging
import copy
import re
import queue
import math
import base64
import io
import zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from federatedscope.attack.auxiliary.a3fl_utils import \
    get_a3fl_active_attacker_ids, parse_attacker_ids
from federatedscope.core.message import Message, b64serializer
from federatedscope.core.workers import Server
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.utils import param2tensor

logger = logging.getLogger(__name__)


class GGEURServer(Server):
    """
    GGEUR_Clip Server that:
    1. Collects local statistics (means, covariances, counts) from clients
    2. Aggregates covariance matrices using parallel axis theorem
    3. Broadcasts global covariances to all clients
    4. Performs standard FedAvg aggregation on MLP parameters
    5. Evaluates on test sets from all domains after each round
    6. (Optional) Aggregates and evaluates CNN model for knowledge distillation
    7. (Optional) Aggregates and evaluates CNN model for feature alignment
    """

    def __init__(self, ID=-1, state=0, config=None, data=None, model=None,
                 client_num=5, total_round_num=10, device='cpu',
                 strategy=None, unseen_clients_id=None, **kwargs):
        super(GGEURServer, self).__init__(ID, state, config, data, model,
                                          client_num, total_round_num, device,
                                          strategy, unseen_clients_id, **kwargs)

        if config is None:
            return

        self.ggeur_cfg = config.ggeur
        self.head_only_mode = getattr(self.ggeur_cfg, 'head_only_mode', False)
        if self.head_only_mode:
            logger.info(
                "Server: GGEUR HeadOnly system mode active. Round 0 keeps "
                "statistics/covariance/prototype/augmentation; later rounds "
                "aggregate MLP-head parameters only.")

        # Statistics collection buffers
        self.local_statistics_buffer = {}  # {client_id: statistics}
        self.statistics_collected = False

        # Aggregated covariance matrices
        self.global_cov_matrices = {}  # {class_idx: cov_matrix}

        # All client prototypes for cross-client augmentation
        self.all_prototypes = {}  # {client_id: {class_idx: prototype}}

        # Global prototypes (aggregated means) for feature alignment
        self.global_prototypes = {}  # {class_idx: mean_vector}

        # MLP model for aggregation
        self.global_mlp = None

        # ===== FedOpt (server-side) =====
        self.use_fedopt = getattr(config.fedopt, 'use', False) if hasattr(config, 'fedopt') else False
        self.fedopt_optimizer = None
        self.fedopt_scheduler = None
        self.fedopt_annealing = getattr(config.fedopt, 'annealing', False) if hasattr(config, 'fedopt') else False

        # Track which clients have completed augmentation
        self.augmentation_ready_clients = set()

        # Test data for evaluation
        self.test_features = {}  # {domain_name: features array}
        self.test_labels = {}    # {domain_name: labels array}
        self.test_data_loaded = False

        # CLIP model for test feature extraction
        self.clip_model = None
        self.clip_preprocess = None

        # Tracking best model
        self.best_avg_accuracy = 0.0
        self.best_model_state = None
        self.test_accuracies_history = {}  # {domain: [acc_per_round]}

        # ===== CNN Mode =====
        self.use_cnn_distillation = getattr(self.ggeur_cfg, 'use_cnn_distillation', False)
        self.use_feature_alignment = getattr(self.ggeur_cfg, 'use_feature_alignment', False)
        self.global_cnn = None  # Global CNN model
        self.cnn_test_accuracies_history = {}  # {domain: [acc_per_round]}
        self.best_cnn_avg_accuracy = 0.0
        self.best_cnn_model_state = None
        # Test images for CNN evaluation (original images, not CLIP features)
        self.test_images_loaded = False
        self.test_image_loaders = {}  # {domain_name: DataLoader}

        # ===== Feature Extractor Mode =====
        # 'clip': Use CLIP (ViT-based, original method)
        # 'cnn': Use pretrained CNN (ConvNeXt, ResNet, etc.)
        # 'timm': Use any timm vision backbone (e.g., GFNet / Mixer / etc.)
        self.feature_extractor_type = getattr(self.ggeur_cfg, 'feature_extractor', 'clip')
        self.cnn_extractor = None  # CNN feature extractor for evaluation
        self.timm_extractor = None  # timm feature extractor for evaluation

        # Embedding dim inferred from clients/statistics (preferred over cfg.ggeur.embedding_dim)
        self.inferred_embedding_dim = None

        # ===== Separated Training Mode =====
        self.use_separated_training = getattr(self.ggeur_cfg, 'use_separated_training', False)
        self.classifier_pretrain_rounds = getattr(self.ggeur_cfg, 'classifier_pretrain_rounds', 20)
        self.freeze_classifier = getattr(self.ggeur_cfg, 'freeze_classifier', True)
        # Phase tracking: 'classifier' or 'cnn_backbone'
        self.training_phase = 'classifier'
        # Store pretrained classifier for Phase 2
        self.pretrained_classifier = None

        # ===== PromptFL Mode =====
        self.use_promptfl = getattr(self.ggeur_cfg, 'use_promptfl', False)
        self.global_prompt_ctx = None          # Aggregated ctx tensor [n_ctx, ctx_dim]
        self.prompt_test_accuracies_history = {}
        self.best_prompt_avg_accuracy = 0.0

        # Training wall-clock timer
        self._train_start_time = time.time()

        # Per-round timing
        self._round_start_time = None
        self._round_durations = []   # seconds per round
        self._round_system_metrics = []

        # Communication volume (bytes)
        self._bytes_sent = 0      # server → clients
        self._bytes_recv = 0      # clients → server
        # CLIP model and prompt components for server-side evaluation
        self.prompt_learner_eval = None
        self.text_encoder_eval = None

        self.distributed_stage_timeout = int(
            getattr(self.ggeur_cfg, 'distributed_stage_timeout', 1800))
        self._stage_name = None
        self._stage_start_time = None
        self.current_round_clients = list(range(1, self._client_num + 1))
        self.a3fl_enabled = str(getattr(config.attack, 'attack_method', '')).lower() == 'a3fl'
        self.latest_a3fl_meta = None
        self.a3fl_shared_trigger = None
        self.a3fl_test_loaders = {}
        self.a3fl_test_loaded = False
        self.foolsgold_update_history = {}

        attack_method = str(getattr(config.attack, 'attack_method', '')).lower()
        self.cerberus_enabled = attack_method == 'cerberus'
        self.cerberus_cfg = getattr(config.attack, 'cerberus', None)
        self.latest_cerberus_meta = None
        self.cerberus_shared_trigger = None
        self.cerberus_peer_model_bank = {}
        self.sabre_enabled = attack_method == 'sabre'
        self.sabre_cfg = getattr(config.attack, 'sabre', None)
        self.latest_sabre_meta = None
        self.sabre_shared_trigger = None
        self.label_flip_enabled = attack_method in (
            'label_flip', 'label_flipping', 'data_poisoning')
        self.label_flip_cfg = getattr(config.attack, 'label_flip', None)
        self.latest_label_flip_meta = None
        self.lie_enabled = attack_method in (
            'little_is_enough', 'lie', 'alie')
        self.lie_cfg = getattr(config.attack, 'little_is_enough', None)
        self.lie_attacker_ids = set(parse_attacker_ids(config.attack.attacker_id))
        self.aggregate_benign_only = bool(getattr(
            config.attack, 'aggregate_benign_only', False))
        self.aggregate_benign_attacker_ids = set(parse_attacker_ids(
            config.attack.attacker_id))

    def _is_cerberus_active_round(self, round_idx):
        if not self.cerberus_enabled or self.cerberus_cfg is None:
            return False
        start_round = int(getattr(
            self.cerberus_cfg, 'start_round',
            getattr(self._cfg.attack, 'inject_round', 0)))
        if int(round_idx) < start_round:
            return False
        poison_epochs = int(getattr(self.cerberus_cfg, 'poison_epochs', 0))
        if poison_epochs <= 0:
            return True
        return int(round_idx) < start_round + poison_epochs

    def _get_cerberus_active_attacker_ids(self, round_idx):
        if not self._is_cerberus_active_round(round_idx):
            return []
        return parse_attacker_ids(self._cfg.attack.attacker_id)

    def _is_sabre_active_round(self, round_idx):
        if not self.sabre_enabled or self.sabre_cfg is None:
            return False
        start_round = int(getattr(
            self.sabre_cfg, 'start_round',
            getattr(self._cfg.attack, 'inject_round', 0)))
        if int(round_idx) < start_round:
            return False
        poison_epochs = int(getattr(self.sabre_cfg, 'poison_epochs', 0))
        if poison_epochs <= 0:
            return True
        return int(round_idx) < start_round + poison_epochs

    def _get_sabre_active_attacker_ids(self, round_idx):
        if not self._is_sabre_active_round(round_idx):
            return []
        return parse_attacker_ids(self._cfg.attack.attacker_id)

    def _is_label_flip_active_round(self, round_idx):
        if not self.label_flip_enabled or self.label_flip_cfg is None:
            return False
        start_round = int(getattr(
            self.label_flip_cfg, 'start_round',
            getattr(self._cfg.attack, 'inject_round', 0)))
        if start_round < 0:
            start_round = int(getattr(self._cfg.attack, 'inject_round', 0))
        if int(round_idx) < start_round:
            return False
        poison_epochs = int(getattr(self.label_flip_cfg, 'poison_epochs', 0))
        if poison_epochs <= 0:
            return True
        return int(round_idx) < start_round + poison_epochs

    def _get_label_flip_active_attacker_ids(self, round_idx):
        if not self._is_label_flip_active_round(round_idx):
            return []
        return parse_attacker_ids(self._cfg.attack.attacker_id)

    def _is_lie_active_round(self, round_idx):
        if not self.lie_enabled or self.lie_cfg is None:
            return False
        start_round = int(getattr(
            self.lie_cfg, 'start_round',
            getattr(self._cfg.attack, 'inject_round', 0)))
        if start_round < 0:
            start_round = int(getattr(self._cfg.attack, 'inject_round', 0))
        if int(round_idx) < start_round:
            return False
        poison_epochs = int(getattr(self.lie_cfg, 'poison_epochs', 0))
        if poison_epochs <= 0:
            return True
        return int(round_idx) < start_round + poison_epochs

    def _get_lie_active_attacker_ids(self, round_idx):
        if not self._is_lie_active_round(round_idx):
            return []
        return parse_attacker_ids(self._cfg.attack.attacker_id)

    def _filter_benign_updates_for_aggregation(self, valid_params, round_idx):
        """Optionally drop configured malicious clients before aggregation."""
        if not self.aggregate_benign_only:
            return valid_params

        attacker_ids = set(self.aggregate_benign_attacker_ids)
        if not attacker_ids:
            logger.warning(
                "Server: aggregate_benign_only is enabled but "
                "cfg.attack.attacker_id is empty; keeping all updates")
            return valid_params

        kept_params = []
        dropped_ids = []
        for sample_size, params, sender in valid_params:
            try:
                sender_id = int(sender)
            except (TypeError, ValueError):
                sender_id = None

            if sender_id is not None and sender_id in attacker_ids:
                dropped_ids.append(sender_id)
                continue
            kept_params.append((sample_size, params, sender))

        if not dropped_ids:
            logger.info(
                f"Server: aggregate_benign_only enabled for round {round_idx}, "
                "but no received update matched cfg.attack.attacker_id")
            return valid_params

        if not kept_params:
            logger.warning(
                f"Server: aggregate_benign_only would remove all valid updates "
                f"in round {round_idx}; falling back to all updates")
            return valid_params

        logger.info(
            f"Server: aggregate_benign_only dropped malicious client updates "
            f"{sorted(set(dropped_ids))} in round {round_idx}; "
            f"kept {len(kept_params)}/{len(valid_params)} updates")
        return kept_params

    def _filter_benign_statistics_for_aggregation(self):
        """Optionally exclude malicious clients from statistics aggregation."""
        if not self.aggregate_benign_only:
            return self.local_statistics_buffer

        attacker_ids = set(self.aggregate_benign_attacker_ids)
        if not attacker_ids:
            logger.warning(
                "Server: aggregate_benign_only is enabled but "
                "cfg.attack.attacker_id is empty; keeping all statistics")
            return self.local_statistics_buffer

        kept_stats = {}
        dropped_ids = []
        for client_id, client_stats in self.local_statistics_buffer.items():
            try:
                normalized_client_id = int(client_id)
            except (TypeError, ValueError):
                normalized_client_id = None

            if normalized_client_id is not None and \
                    normalized_client_id in attacker_ids:
                dropped_ids.append(normalized_client_id)
                continue
            kept_stats[client_id] = client_stats

        if not dropped_ids:
            logger.info(
                "Server: aggregate_benign_only enabled for statistics, "
                "but no received statistics matched cfg.attack.attacker_id")
            return self.local_statistics_buffer

        if not kept_stats:
            logger.warning(
                "Server: aggregate_benign_only would remove all local "
                "statistics; falling back to all statistics")
            return self.local_statistics_buffer

        logger.info(
            "Server: aggregate_benign_only dropped malicious client "
            f"statistics/prototypes {sorted(set(dropped_ids))}; "
            f"kept {len(kept_stats)}/{len(self.local_statistics_buffer)} "
            "statistics payloads")
        return kept_stats

    def _resolve_lie_z(self, n_workers, m_attackers):
        cfg_z = float(getattr(self.lie_cfg, 'z', 1.0))
        if not bool(getattr(self.lie_cfg, 'auto_z', False)):
            return cfg_z

        if n_workers <= 0 or m_attackers <= 0:
            return cfg_z

        seduced = math.floor(n_workers / 2 + 1) - m_attackers
        seduced = max(0, seduced)
        prob = (n_workers - seduced) / float(n_workers)
        prob = min(max(prob, 1e-6), 1.0 - 1e-6)
        try:
            z_value = torch.distributions.Normal(0.0, 1.0).icdf(
                torch.tensor(prob)).item()
        except Exception as exc:
            logger.debug(
                f"Server: LIE auto_z failed ({exc}); using configured z={cfg_z}")
            z_value = cfg_z

        max_z = float(getattr(self.lie_cfg, 'max_z', 1.5))
        if max_z > 0:
            z_value = min(z_value, max_z)
        return float(z_value)

    def _lie_should_target_model(self, aggregation_name):
        targets = getattr(self.lie_cfg, 'target_models', ['mlp', 'classifier'])
        if isinstance(targets, str):
            targets = [item.strip() for item in targets.split(',')]
        targets = {str(item).lower() for item in targets}
        return 'all' in targets or str(aggregation_name).lower() in targets

    def _apply_little_is_enough_attack_to_params(self, valid_params,
                                                 aggregation_name,
                                                 base_params=None):
        """Apply the A Little Is Enough parameter-poisoning transform.

        This simulates coordinated malicious GGEUR clients after local training:
        for each floating parameter dimension, replace every attacker update by
        mu +/- z * sigma, where mu/sigma are estimated from the configured
        source updates in the same round.
        """
        if not self.lie_enabled or self.lie_cfg is None:
            return valid_params
        if not self._is_lie_active_round(self.state):
            return valid_params
        if not self._lie_should_target_model(aggregation_name):
            return valid_params

        active_attackers = set(self._get_lie_active_attacker_ids(self.state))
        if not active_attackers:
            return valid_params

        attacker_indices = [
            idx for idx, (_, _, sender) in enumerate(valid_params)
            if sender is not None and int(sender) in active_attackers
        ]
        min_attackers = int(getattr(self.lie_cfg, 'min_attackers', 1))
        if len(attacker_indices) < max(1, min_attackers):
            if bool(getattr(self.lie_cfg, 'log_detail', True)):
                logger.info(
                    f"Server: LIE skipped for {aggregation_name} in round "
                    f"{self.state}; active attackers in valid updates="
                    f"{len(attacker_indices)} < min_attackers={min_attackers}")
            return valid_params

        stats_source = str(getattr(self.lie_cfg, 'stats_source',
                                   'attacker')).lower()
        if stats_source in ('all', 'all_clients', 'valid'):
            stats_indices = list(range(len(valid_params)))
        else:
            stats_indices = attacker_indices

        if len(stats_indices) < 1:
            return valid_params

        z_value = self._resolve_lie_z(len(valid_params), len(attacker_indices))
        direction = str(getattr(self.lie_cfg, 'direction', 'positive')).lower()
        sign = -1.0 if direction in ('negative', 'minus', '-', 'lower') else 1.0
        min_std = float(getattr(self.lie_cfg, 'min_std', 1e-6))

        first_params = valid_params[0][1]
        poisoned_state = {}
        poisoned_keys = 0
        sigma_sq_sum = 0.0
        perturb_sq_sum = 0.0
        poison_elem_count = 0

        for key, value in first_params.items():
            if isinstance(value, dict):
                continue
            first_tensor = self._to_param_tensor(value)
            if first_tensor is None or not torch.is_floating_point(first_tensor):
                continue

            source_tensors = []
            for idx in stats_indices:
                params = valid_params[idx][1]
                if key not in params or isinstance(params[key], dict):
                    continue
                tensor = self._to_param_tensor(params[key])
                if tensor is None or not torch.is_floating_point(tensor):
                    continue
                source_tensors.append(tensor.detach().float().to(self.device))

            if not source_tensors:
                continue

            stacked = torch.stack(source_tensors, dim=0)
            mean_tensor = stacked.mean(dim=0)
            if stacked.shape[0] > 1:
                std_tensor = stacked.std(dim=0, unbiased=False)
            else:
                std_tensor = torch.zeros_like(mean_tensor)
            if min_std > 0:
                std_tensor = torch.clamp(std_tensor, min=min_std)

            if direction in ('away_from_global', 'away', 'diverge') and \
                    base_params is not None and key in base_params:
                base_tensor = self._to_param_tensor(base_params[key])
                if base_tensor is not None and torch.is_floating_point(
                        base_tensor):
                    base_tensor = base_tensor.detach().float().to(self.device)
                    sign_tensor = torch.sign(mean_tensor - base_tensor)
                    sign_tensor = torch.where(
                        sign_tensor == 0,
                        torch.ones_like(sign_tensor),
                        sign_tensor)
                    poisoned = mean_tensor + z_value * std_tensor * sign_tensor
                else:
                    poisoned = mean_tensor + sign * z_value * std_tensor
            else:
                poisoned = mean_tensor + sign * z_value * std_tensor
            poisoned_state[key] = poisoned.to(
                dtype=first_tensor.dtype,
                device=first_tensor.device)
            poisoned_keys += 1
            if bool(getattr(self.lie_cfg, 'log_norms', True)):
                sigma_sq_sum += float(torch.sum(std_tensor ** 2).item())
                perturb_sq_sum += float(
                    torch.sum((poisoned - mean_tensor) ** 2).item())
                poison_elem_count += int(std_tensor.numel())

        if not poisoned_state:
            logger.warning(
                f"Server: LIE found no floating parameters to poison for "
                f"{aggregation_name} in round {self.state}")
            return valid_params

        poisoned_params = []
        for idx, (sample_size, params, sender) in enumerate(valid_params):
            if idx not in attacker_indices:
                poisoned_params.append((sample_size, params, sender))
                continue
            replaced = copy.deepcopy(params)
            for key, value in poisoned_state.items():
                replaced[key] = value.detach().clone()
            poisoned_params.append((sample_size, replaced, sender))

        if bool(getattr(self.lie_cfg, 'log_detail', True)):
            norm_msg = ''
            if poison_elem_count > 0:
                sigma_rms = math.sqrt(sigma_sq_sum / poison_elem_count)
                perturb_rms = math.sqrt(perturb_sq_sum / poison_elem_count)
                norm_msg = (
                    f", sigma_rms={sigma_rms:.6g}, "
                    f"perturb_rms={perturb_rms:.6g}")
            logger.info(
                f"Server: Applied LIE attack to {aggregation_name} in round "
                f"{self.state}: attackers={len(attacker_indices)}/"
                f"{len(valid_params)}, stats_source={stats_source}, "
                f"z={z_value:.4f}, direction={direction}, "
                f"poisoned_keys={poisoned_keys}{norm_msg}")

        return poisoned_params

    def _attach_shared_trigger_payload(self, model_para, attack_name,
                                       shared_trigger):
        if shared_trigger is None:
            return model_para
        if not isinstance(model_para, dict) or 'mlp' not in model_para:
            model_para = {'mlp': model_para}
        shared_trigger = copy.deepcopy(shared_trigger)
        payload = model_para.get(attack_name, {})
        if not isinstance(payload, dict):
            payload = {}
        payload['shared_trigger'] = shared_trigger
        model_para[attack_name] = payload
        model_para[f'{attack_name}_shared_trigger'] = shared_trigger
        return model_para

    def _attach_a3fl_payload(self, model_para):
        if not self.a3fl_enabled:
            return model_para
        return self._attach_shared_trigger_payload(
            model_para, 'a3fl', self.a3fl_shared_trigger)

    def _attach_cerberus_payload(self, model_para):
        if not self.cerberus_enabled:
            return model_para
        if not isinstance(model_para, dict) or 'mlp' not in model_para:
            model_para = {'mlp': model_para}
        cerberus_payload = {
            'peer_models': copy.deepcopy(self.cerberus_peer_model_bank)
        }
        if self.cerberus_shared_trigger is not None:
            shared_trigger = copy.deepcopy(self.cerberus_shared_trigger)
            cerberus_payload['shared_trigger'] = shared_trigger
            model_para['cerberus_shared_trigger'] = shared_trigger
        model_para['cerberus'] = cerberus_payload
        return model_para

    def _attach_sabre_payload(self, model_para):
        if not self.sabre_enabled:
            return model_para
        return self._attach_shared_trigger_payload(
            model_para, 'sabre', self.sabre_shared_trigger)

    def _update_shared_trigger(self, active_updates, attack_name, shared_attr):
        for meta in active_updates:
            if not isinstance(meta, dict):
                continue
            trigger = meta.get('trigger', None)
            mask = meta.get('mask', None)
            if trigger is None or mask is None:
                continue
            try:
                trigger = param2tensor(trigger)
                mask = param2tensor(mask)
            except Exception as exc:
                logger.debug(
                    f"Server: Could not restore {attack_name.upper()} "
                    f"shared trigger from client "
                    f"{meta.get('client_id', 'unknown')}: {exc}")
                continue
            if not isinstance(trigger, torch.Tensor) or \
                    not isinstance(mask, torch.Tensor):
                continue
            shared_trigger = {
                'trigger': trigger.detach().cpu(),
                'mask': mask.detach().cpu(),
                'source_client_id': int(meta.get('client_id', -1)),
                'source_round': int(meta.get('round', self.state)),
            }
            if 'target_label' in meta:
                shared_trigger['target_label'] = int(meta.get('target_label'))
            if 'trigger_mode' in meta:
                shared_trigger['trigger_mode'] = meta.get('trigger_mode')
            setattr(self, shared_attr, shared_trigger)
            nonzero = int(mask.detach().cpu().ne(0).sum().item())
            logger.info(
                f"Server: Updated {attack_name.upper()} shared trigger "
                f"from client {int(meta.get('client_id', -1))} "
                f"for round {int(meta.get('round', self.state))} "
                f"(mask_nonzero={nonzero})")
            return

    def _update_a3fl_shared_trigger(self, active_a3fl):
        if not self.a3fl_enabled:
            return
        self._update_shared_trigger(
            active_a3fl, 'a3fl', 'a3fl_shared_trigger')

    def _update_cerberus_shared_trigger(self, active_cerberus):
        if not self.cerberus_enabled:
            return
        for meta in active_cerberus:
            if not isinstance(meta, dict):
                continue
            trigger = meta.get('trigger', None)
            mask = meta.get('mask', None)
            if trigger is None or mask is None:
                continue
            try:
                trigger = param2tensor(trigger)
                mask = param2tensor(mask)
            except Exception as exc:
                logger.debug(
                    "Server: Could not restore CERBERUS shared trigger "
                    f"from client {meta.get('client_id', 'unknown')}: {exc}")
                continue
            if not isinstance(trigger, torch.Tensor) or \
                    not isinstance(mask, torch.Tensor):
                continue
            self.cerberus_shared_trigger = {
                'trigger': trigger.detach().cpu(),
                'mask': mask.detach().cpu(),
                'source_client_id': int(meta.get('client_id', -1)),
                'source_round': int(meta.get('round', self.state)),
            }
            nonzero = int(mask.detach().cpu().ne(0).sum().item())
            logger.info(
                "Server: Updated CERBERUS shared trigger "
                f"from client {int(meta.get('client_id', -1))} "
                f"for round {int(meta.get('round', self.state))} "
                f"(mask_nonzero={nonzero})")
            return

    def _update_sabre_shared_trigger(self, active_sabre):
        if not self.sabre_enabled:
            return
        self._update_shared_trigger(
            active_sabre, 'sabre', 'sabre_shared_trigger')

    def _mark_stage(self, stage_name):
        self._stage_name = stage_name
        self._stage_start_time = time.time()
        logger.info(
            f"Server: Entered distributed stage '{stage_name}' "
            f"(timeout={self.distributed_stage_timeout}s)")

    def _clear_stage(self):
        self._stage_name = None
        self._stage_start_time = None

    def _check_distributed_stage_timeout(self):
        if self.is_finish:
            return
        if self.distributed_stage_timeout <= 0:
            return
        if self._stage_name is None or self._stage_start_time is None:
            return

        elapsed = time.time() - self._stage_start_time
        if elapsed <= self.distributed_stage_timeout:
            return

        logger.error(
            f"Server: Timeout in distributed stage '{self._stage_name}' "
            f"after {elapsed:.1f}s. joined_clients="
            f"{list(getattr(self.comm_manager, 'neighbors', {}).keys())}, "
            f"augmentation_ready={sorted(self.augmentation_ready_clients)}, "
            f"train_buffer_rounds={list(self.msg_buffer.get('train', {}).keys())}")
        self._notify_joined_clients_to_finish()
        if hasattr(self.comm_manager, 'shutdown'):
            self.comm_manager.shutdown()
        self.is_finish = True
        raise TimeoutError(
            f"GGEUR distributed stage timeout: {self._stage_name}")

    def run(self):
        """Run GGEUR distributed server with GGEUR-specific stage timeouts."""
        while self.join_in_client_num < self.client_num:
            self._check_join_timeout()
            try:
                msg = self.comm_manager.receive(timeout=1.0)
            except queue.Empty:
                continue
            self.msg_handlers[msg.msg_type](msg)

        if self._stage_name is None:
            self._mark_stage('statistics')

        while self.state <= self.total_round_num and not self.is_finish:
            try:
                msg = self.comm_manager.receive(timeout=1.0)
            except queue.Empty:
                self._check_distributed_stage_timeout()
                continue
            self.msg_handlers[msg.msg_type](msg)

        if not self.is_finish:
            self.terminate(msg_type='finish')

    def _get_domainnet_eval_metadata(self):
        """Resolve DomainNet evaluation domains and classes."""
        from federatedscope.cv.dataset.domainnet import discover_domainnet_metadata

        selected_domains = list(getattr(self.ggeur_cfg, 'domainnet_domains', []))
        shared_classes_only = getattr(self.ggeur_cfg,
                                      'domainnet_shared_classes_only', False)
        return discover_domainnet_metadata(self._cfg.data.root, selected_domains,
                                           shared_classes_only)

    def _get_embedding_dim(self):
        """Get embedding dim, preferring inferred value from client statistics."""
        if isinstance(self.inferred_embedding_dim, int) and self.inferred_embedding_dim > 0:
            return int(self.inferred_embedding_dim)

        # Infer from any received mean vector
        for client_stats in self.local_statistics_buffer.values():
            means = client_stats.get('means', {})
            for _, mean in means.items():
                if hasattr(mean, 'shape') and len(mean.shape) == 1:
                    self.inferred_embedding_dim = int(mean.shape[0])
                    return int(self.inferred_embedding_dim)
                try:
                    # Fallback for list-like
                    self.inferred_embedding_dim = int(len(mean))
                    return int(self.inferred_embedding_dim)
                except Exception:
                    continue

        return int(getattr(self.ggeur_cfg, 'embedding_dim', 512))

    def _register_default_handlers(self):
        """Register message handlers"""
        super()._register_default_handlers()

        # Register handler for local statistics
        self.register_handlers('local_statistics',
                               self.callback_for_local_statistics)

        # Register handler for augmentation ready signal
        self.register_handlers('augmentation_ready',
                               self.callback_for_augmentation_ready)

    def _build_global_mlp(self, num_classes):
        """Build global MLP classifier"""
        input_dim = self._get_embedding_dim()
        hidden_dim = self.ggeur_cfg.mlp_hidden_dim

        if hidden_dim > 0:
            self.global_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.ggeur_cfg.mlp_dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.global_mlp = nn.Linear(input_dim, num_classes)

        self.global_mlp = self.global_mlp.to(self.device)
        logger.info(f"Server: Built global MLP classifier with {num_classes} classes")

        # Initialize FedOpt optimizer if enabled (needs a built global model)
        self._init_fedopt_if_needed()

    def _init_fedopt_if_needed(self):
        """Initialize FedOpt optimizer/scheduler for global MLP if enabled."""
        if not self.use_fedopt:
            return
        if self.global_mlp is None:
            return
        if self.fedopt_optimizer is not None:
            return

        try:
            self.fedopt_optimizer = get_optimizer(model=self.global_mlp,
                                                  **self._cfg.fedopt.optimizer)
        except Exception as e:
            logger.error(f"Server: Failed to build FedOpt optimizer: {e}")
            self.fedopt_optimizer = None
            return

        if self.fedopt_annealing:
            try:
                self.fedopt_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.fedopt_optimizer,
                    step_size=self._cfg.fedopt.annealing_step_size,
                    gamma=self._cfg.fedopt.annealing_gamma
                )
            except Exception as e:
                logger.warning(f"Server: Failed to build FedOpt scheduler, disabling annealing: {e}")
                self.fedopt_scheduler = None
                self.fedopt_annealing = False

        opt_type = getattr(self._cfg.fedopt.optimizer, 'type', 'Unknown')
        opt_lr = getattr(self._cfg.fedopt.optimizer, 'lr', 'Unknown')
        logger.info(f"Server: FedOpt enabled - optimizer={opt_type}, lr={opt_lr}, annealing={self.fedopt_annealing}")

    def _build_global_cnn(self, num_classes):
        """Build global CNN model for knowledge distillation or feature alignment"""
        from federatedscope.contrib.model.ggeur_cnn import GGEUR_CNN_FeatureAlign

        cnn_model_name = getattr(self.ggeur_cfg, 'cnn_model', 'resnet18')
        clip_dim = getattr(self.ggeur_cfg, 'embedding_dim', 512)

        # For feature alignment: no pretrained weights (from scratch)
        # For distillation: use pretrained weights
        if self.use_feature_alignment:
            cnn_pretrained = False
            mode_str = "feature alignment (from scratch)"
        else:
            cnn_pretrained = getattr(self.ggeur_cfg, 'cnn_pretrained', True)
            mode_str = f"distillation (pretrained={cnn_pretrained})"

        self.global_cnn = GGEUR_CNN_FeatureAlign(
            model_name=cnn_model_name,
            num_classes=num_classes,
            clip_dim=clip_dim,
            pretrained=cnn_pretrained
        )
        self.global_cnn = self.global_cnn.to(self.device)
        logger.info(f"Server: Built global CNN ({cnn_model_name}) for {mode_str} with {num_classes} classes")

    def _load_clip_model(self):
        """Load CLIP model for test feature extraction"""
        if self.clip_model is not None:
            return

        try:
            import open_clip

            model_name = self.ggeur_cfg.clip_model
            pretrained = self.ggeur_cfg.clip_pretrained
            local_path = self.ggeur_cfg.clip_model_path

            if local_path and os.path.exists(local_path):
                logger.info(f"Server: Loading CLIP from local path: {local_path}")
                self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=local_path
                )
            else:
                logger.info(f"Server: Loading CLIP from pretrained: {pretrained}")
                self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained
                )

            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            logger.info(f"Server: Loaded CLIP model {model_name}")

        except ImportError:
            logger.error("open_clip not installed. Please install: pip install open_clip_torch")
            raise

    def _load_cnn_extractor(self):
        """Load CNN feature extractor for test feature extraction"""
        if self.cnn_extractor is not None:
            return

        try:
            from federatedscope.contrib.model.ggeur_cnn_extractor import CNNFeatureExtractor

            model_name = getattr(self.ggeur_cfg, 'cnn_backbone', 'convnext_base')
            pretrained = getattr(self.ggeur_cfg, 'cnn_pretrained', True)
            checkpoint_path = getattr(self.ggeur_cfg, 'cnn_checkpoint_path', '')
            freeze = True  # Always freeze for feature extraction

            self.cnn_extractor = CNNFeatureExtractor(
                model_name=model_name,
                pretrained=pretrained,
                freeze=freeze,
                checkpoint_path=checkpoint_path,
            )
            self.cnn_extractor = self.cnn_extractor.to(self.device)

            logger.info(f"Server: Loaded CNN extractor {model_name}, "
                       f"feature_dim={self.cnn_extractor.get_feature_dim()}")

        except Exception as e:
            logger.error(f"Server: Failed to load CNN extractor: {e}")
            raise

    def _load_timm_extractor(self):
        """Load timm feature extractor for test feature extraction"""
        if self.timm_extractor is not None:
            return

        try:
            from federatedscope.contrib.model.ggeur_timm_extractor import TimmFeatureExtractor

            model_name = getattr(self.ggeur_cfg, 'timm_model', 'gfnet_tiny')
            pretrained = getattr(self.ggeur_cfg, 'timm_pretrained', True)
            checkpoint_path = getattr(self.ggeur_cfg, 'timm_checkpoint_path', '')
            freeze = True  # Always freeze for feature extraction
            in_chans = getattr(self.ggeur_cfg, 'timm_in_chans', 3)
            global_pool = getattr(self.ggeur_cfg, 'timm_global_pool', 'avg')

            self.timm_extractor = TimmFeatureExtractor(
                model_name=model_name,
                pretrained=pretrained,
                freeze=freeze,
                checkpoint_path=checkpoint_path,
                in_chans=in_chans,
                global_pool=global_pool,
            )
            self.timm_extractor = self.timm_extractor.to(self.device)

            logger.info(f"Server: Loaded timm extractor {model_name}, "
                        f"feature_dim={self.timm_extractor.get_feature_dim()}")

        except Exception as e:
            logger.error(f"Server: Failed to load timm extractor: {e}")
            raise

    def _load_feature_extractor(self):
        """Load the appropriate feature extractor (CLIP or CNN)"""
        if self.feature_extractor_type == 'cnn':
            self._load_cnn_extractor()
        elif self.feature_extractor_type == 'timm':
            self._load_timm_extractor()
        else:
            self._load_clip_model()

    def _get_test_cache_path(self, domain):
        """Get cache path for test features"""
        cache_dir = getattr(self.ggeur_cfg, 'feature_cache_dir', '')
        if not cache_dir:
            cache_dir = os.path.join(os.path.dirname(self._cfg.data.root), 'clip_feature_cache')

        os.makedirs(cache_dir, exist_ok=True)

        # Get split parameters to include in cache filename
        data_type = self._cfg.data.type.lower()
        if hasattr(self._cfg.data, 'splits'):
            splits = tuple(self._cfg.data.splits)
        else:
            if 'office' in data_type and 'home' in data_type:
                splits = (0.7, 0.0, 0.3)
            else:
                splits = (0.8, 0.1, 0.1)
        seed = self._cfg.seed if hasattr(self._cfg, 'seed') else 123

        dataset_name = data_type
        cache_suffix = ''
        if 'domainnet' in data_type or 'domain-net' in data_type or 'domain_net' in data_type:
            selected_domains = list(getattr(self.ggeur_cfg, 'domainnet_domains', []))
            shared_classes_only = getattr(self.ggeur_cfg,
                                          'domainnet_shared_classes_only', False)
            domain_token = '-'.join(selected_domains) if selected_domains else 'auto'
            shared_token = 'shared' if shared_classes_only else 'union'
            cache_suffix = f"_{shared_token}_{domain_token}"

        # Build model string based on feature extractor type
        if self.feature_extractor_type == 'cnn':
            model_name = getattr(self.ggeur_cfg, 'cnn_backbone', 'convnext_base')
            model_str = model_name.replace('/', '_').replace('-', '_')
            prefix = 'cnn'
        elif self.feature_extractor_type == 'timm':
            model_name = getattr(self.ggeur_cfg, 'timm_model', 'gfnet_tiny')
            model_str = str(model_name).replace('/', '_').replace('-', '_')
            prefix = 'timm'
        else:
            clip_model = self.ggeur_cfg.clip_model.replace('/', '_').replace('-', '_')
            pretrained = self.ggeur_cfg.clip_pretrained.replace('/', '_').replace('-', '_')
            model_str = f"{clip_model}_{pretrained}"
            prefix = 'clip'

        # Include split params in filename to ensure cache invalidation when params change
        split_str = f"split{int(splits[0]*100)}_{int(splits[1]*100)}_{int(100-splits[0]*100-splits[1]*100)}_seed{seed}"
        cache_filename = f"{dataset_name}_{domain}_test_{prefix}_{model_str}_{split_str}{cache_suffix}.npz"

        return os.path.join(cache_dir, cache_filename)

    def _load_test_data_and_features(self):
        """Load test data from all domains and extract CLIP features"""
        if self.test_data_loaded:
            return

        logger.info("Server: Loading test data from all domains...")

        data_type = self._cfg.data.type.lower()
        data_root = self._cfg.data.root

        # Get the same split ratios and seed as client data loading
        if hasattr(self._cfg.data, 'splits'):
            splits = tuple(self._cfg.data.splits)
        else:
            # Default splits based on dataset type
            if 'office' in data_type and 'home' in data_type:
                splits = (0.7, 0.0, 0.3)  # Same as ggeur_data.py
            else:
                splits = (0.8, 0.1, 0.1)

        train_ratio, val_ratio = splits[0], splits[1]
        seed = self._cfg.seed if hasattr(self._cfg, 'seed') else 123

        logger.info(f"Server: Using splits={splits}, seed={seed} (same as client data)")

        # Determine domains based on dataset type
        if 'pacs' in data_type:
            domains = ['photo', 'art_painting', 'cartoon', 'sketch']
            from federatedscope.cv.dataset.pacs import PACS
            dataset_class = PACS
            dataset_kwargs = {}
        elif 'office' in data_type and 'home' in data_type:
            domains = ['Art', 'Clipart', 'Product', 'Real_World']
            from federatedscope.cv.dataset.office_home import OfficeHome
            dataset_class = OfficeHome
            dataset_kwargs = {}
        elif 'domainnet' in data_type or 'domain-net' in data_type or \
                'domain_net' in data_type:
            from federatedscope.cv.dataset.domainnet import DomainNet
            domains, classes = self._get_domainnet_eval_metadata()
            dataset_class = DomainNet
            dataset_kwargs = {'classes': classes}
            cache_meta = {
                'class_count': len(classes),
                'shared_classes_only': int(getattr(
                    self.ggeur_cfg, 'domainnet_shared_classes_only', False)),
                'selected_domains': '|'.join(
                    list(getattr(self.ggeur_cfg, 'domainnet_domains', [])))
            }
        elif 'office' in data_type and 'caltech' in data_type:
            domains = ['amazon', 'caltech', 'dslr', 'webcam']
            from federatedscope.cv.dataset.office_caltech import OfficeCaltech10
            dataset_class = OfficeCaltech10
            dataset_kwargs = {}
            cache_meta = {}
        else:
            logger.warning(f"Server: Unknown dataset type {data_type}, skipping test evaluation")
            return
        if 'domainnet' not in data_type and 'domain-net' not in data_type and \
                'domain_net' not in data_type:
            cache_meta = {}

        # Load test data for each domain
        for domain in domains:
            cache_path = self._get_test_cache_path(domain)

            # Try to load from cache first
            if os.path.exists(cache_path):
                try:
                    data = np.load(cache_path)
                    if cache_meta:
                        cached_class_count = int(data['class_count']) if 'class_count' in data.files else -1
                        cached_shared = int(data['shared_classes_only']) if 'shared_classes_only' in data.files else -1
                        cached_domains = str(data['selected_domains']) if 'selected_domains' in data.files else ''
                        if cached_class_count != cache_meta['class_count'] or \
                                cached_shared != cache_meta['shared_classes_only'] or \
                                cached_domains != cache_meta['selected_domains']:
                            raise ValueError('DomainNet test cache metadata mismatch')
                    self.test_features[domain] = data['features']
                    self.test_labels[domain] = data['labels']
                    logger.info(f"Server: Loaded {len(self.test_labels[domain])} cached test features for {domain}")
                    continue
                except Exception as e:
                    logger.warning(f"Server: Failed to load cache for {domain}: {e}")

            # Load test dataset with CLIP normalization (CRITICAL for PromptFL)
            try:
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                       std=[0.26862954, 0.26130258, 0.27577711])
                ])

                test_dataset = dataset_class(
                    root=data_root,
                    domain=domain,
                    split='test',
                    transform=transform,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    seed=seed,
                    **dataset_kwargs
                )

                if len(test_dataset) == 0:
                    logger.warning(f"Server: No test data for domain {domain}")
                    continue

                # Log the split info for verification
                logger.info(f"Server: {domain} test set has {len(test_dataset)} samples")

                # Extract features using appropriate extractor (CLIP or CNN)
                self._load_feature_extractor()
                features_list = []
                labels_list = []

                dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                if self.feature_extractor_type == 'cnn':
                    extractor_name = 'CNN'
                elif self.feature_extractor_type == 'timm':
                    extractor_name = 'timm'
                else:
                    extractor_name = 'CLIP'

                with torch.no_grad():
                    for images, labels in dataloader:
                        images = images.to(self.device)
                        if self.feature_extractor_type == 'cnn':
                            features = self.cnn_extractor(images)
                        elif self.feature_extractor_type == 'timm':
                            features = self.timm_extractor(images)
                        else:
                            features = self.clip_model.encode_image(images)
                        features_list.append(features.cpu().numpy())
                        labels_list.append(labels.numpy())

                self.test_features[domain] = np.vstack(features_list)
                self.test_labels[domain] = np.concatenate(labels_list)

                # Save to cache
                np.savez(cache_path,
                        features=self.test_features[domain],
                        labels=self.test_labels[domain],
                        **cache_meta)

                logger.info(f"Server: Extracted and cached {len(self.test_labels[domain])} test features for {domain}")

            except Exception as e:
                logger.warning(f"Server: Failed to load test data for {domain}: {e}")
                import traceback
                traceback.print_exc()

        self.test_data_loaded = True
        logger.info(f"Server: Loaded test data for {len(self.test_features)} domains")

    def _evaluate_on_test_sets(self):
        """Evaluate global model on all test sets"""
        if self.global_mlp is None:
            return {}

        # Load test data if not already loaded
        if not self.test_data_loaded:
            self._load_test_data_and_features()

        if not self.test_features:
            return {}

        self.global_mlp.eval()
        results = {}

        with torch.no_grad():
            for domain, features in self.test_features.items():
                labels = self.test_labels[domain]

                features_tensor = torch.from_numpy(features).float().to(self.device)
                labels_tensor = torch.from_numpy(labels).long().to(self.device)

                outputs = self.global_mlp(features_tensor)
                _, predicted = torch.max(outputs, 1)

                correct = (predicted == labels_tensor).sum().item()
                total = labels_tensor.size(0)
                accuracy = correct / total if total > 0 else 0

                results[domain] = accuracy

                # Track history
                if domain not in self.test_accuracies_history:
                    self.test_accuracies_history[domain] = []
                self.test_accuracies_history[domain].append(accuracy)

        # Compute average
        if results:
            avg_accuracy = sum(results.values()) / len(results)
            results['average'] = avg_accuracy

            if 'average' not in self.test_accuracies_history:
                self.test_accuracies_history['average'] = []
            self.test_accuracies_history['average'].append(avg_accuracy)

            # Track best model
            if avg_accuracy > self.best_avg_accuracy:
                self.best_avg_accuracy = avg_accuracy
                self.best_model_state = copy.deepcopy(self.global_mlp.state_dict())

        return results

    def callback_for_local_statistics(self, message: Message):
        """Handle receiving local statistics from a client"""
        client_id = message.sender
        content = message.content
        self._bytes_recv += self._sizeof_content(content)

        logger.info(f"Server: Received local statistics from client {client_id}")

        # Infer embedding dimension from client-reported value (preferred) or from means later
        client_embedding_dim = content.get('embedding_dim', None) if isinstance(content, dict) else None
        if client_embedding_dim is not None:
            try:
                client_embedding_dim = int(client_embedding_dim)
            except Exception:
                client_embedding_dim = None

        if client_embedding_dim:
            if self.inferred_embedding_dim is None:
                self.inferred_embedding_dim = client_embedding_dim
                logger.info(f"Server: Inferred embedding_dim={self.inferred_embedding_dim} from client {client_id}")
            elif int(self.inferred_embedding_dim) != int(client_embedding_dim):
                raise ValueError(
                    f"Embedding dim mismatch across clients: "
                    f"server_inferred={self.inferred_embedding_dim}, client_{client_id}={client_embedding_dim}"
                )

        self._validate_received_statistics(client_id, content)
        content = self._normalize_received_statistics(client_id, content)

        # Store statistics
        self.local_statistics_buffer[client_id] = {
            'means': content['means'],
            'covs': content['covs'],
            'counts': content['counts'],
            'prototypes': content['prototypes']
        }

        # Store prototypes for cross-client sharing
        self.all_prototypes[client_id] = content['prototypes']

        # Check if all clients have uploaded statistics
        if len(self.local_statistics_buffer) >= self._client_num:
            logger.info(f"Server: Received statistics from all {self._client_num} clients")
            statistics_for_aggregation = \
                self._filter_benign_statistics_for_aggregation()

            # Aggregate covariance matrices
            self._aggregate_covariances(statistics_for_aggregation)

            # Compute global prototypes (aggregated means for each class)
            self._compute_global_prototypes(statistics_for_aggregation)

            # Prepare other client prototypes for each client
            other_prototypes = self._prepare_other_prototypes(
                statistics_for_aggregation)

            # Build global MLP
            # IMPORTANT: Use config's num_classes, not the number of classes in covariance matrices
            # In LDS mode, some classes may have no data across all clients
            num_classes = self._cfg.model.num_classes
            if num_classes > 0:
                self._build_global_mlp(num_classes)

                # Build global CNN if using CNN mode
                if self.use_cnn_distillation or self.use_feature_alignment:
                    self._build_global_cnn(num_classes)

                # Initialize global prompt ctx if PromptFL enabled
                if self.use_promptfl:
                    self._init_global_prompt(num_classes)

            # Broadcast global covariances to all clients
            self._broadcast_global_covariances(other_prototypes)

            self.statistics_collected = True

    def _validate_received_statistics(self, client_id, content):
        """Fail early when a client uploads statistics with wrong dimensions."""
        if not isinstance(content, dict):
            raise ValueError(
                f"Server: local_statistics from client {client_id} must be "
                f"a dict, got {type(content)}.")

        expected_dim = int(self._get_embedding_dim())
        invalid = []
        means = content.get('means', {})
        covs = content.get('covs', {})
        counts = content.get('counts', {})

        for class_idx, mean in means.items():
            mean_arr = self._payload_to_ndarray(mean)
            cov = self._get_class_value(covs, class_idx)
            count = self._get_class_value(counts, class_idx)
            cov_arr = self._payload_to_ndarray(cov)
            if mean_arr is None:
                invalid.append(f"class {class_idx}: mean decode failed")
                continue
            if cov_arr is None:
                invalid.append(f"class {class_idx}: cov decode failed")
                continue
            mean_arr = np.asarray(mean_arr)
            cov_arr = np.asarray(cov_arr)
            if mean_arr.shape != (expected_dim, ):
                invalid.append(
                    f"class {class_idx}: mean shape {mean_arr.shape}")
                continue
            if cov_arr.shape != (expected_dim, expected_dim):
                invalid.append(f"class {class_idx}: cov shape {cov_arr.shape}")
            if count is None:
                invalid.append(f"class {class_idx}: missing count")

        if invalid:
            detail = '; '.join(invalid[:5])
            raise ValueError(
                f"Server: Invalid local_statistics from client {client_id}. "
                f"Expected embedding_dim={expected_dim}. {detail}. "
                f"Please clear stale feature cache or align all clients to "
                f"the same feature_extractor/embedding_dim.")

    def _normalize_received_statistics(self, client_id, content):
        """Decode compact local-statistics payloads from a client."""
        normalized = dict(content)
        expected_dim = int(self._get_embedding_dim())

        means = {}
        for class_idx, mean in content.get('means', {}).items():
            key = int(class_idx)
            means[key] = np.asarray(
                self._payload_to_ndarray(mean), dtype=np.float32)

        covs = {}
        for class_idx, cov in content.get('covs', {}).items():
            key = int(class_idx)
            covs[key] = np.asarray(
                self._payload_to_ndarray(cov), dtype=np.float32)

        counts = {}
        for class_idx, count in content.get('counts', {}).items():
            counts[int(class_idx)] = int(count)

        prototypes = {}
        for class_idx, proto in content.get('prototypes', {}).items():
            key = int(class_idx)
            proto_arr = np.asarray(
                self._payload_to_ndarray(proto), dtype=np.float32)
            if proto_arr.shape != (expected_dim, ):
                raise ValueError(
                    f"Server: Invalid prototype from client {client_id}, "
                    f"class {class_idx}: shape {proto_arr.shape}, "
                    f"expected ({expected_dim},).")
            prototypes[key] = proto_arr

        normalized['means'] = means
        normalized['covs'] = covs
        normalized['counts'] = counts
        normalized['prototypes'] = prototypes
        return normalized

    @staticmethod
    def _get_class_value(mapping, class_idx):
        """Read class-keyed stats regardless of int/string key serialization."""
        if not isinstance(mapping, dict):
            return None
        if class_idx in mapping:
            return mapping[class_idx]
        str_key = str(class_idx)
        if str_key in mapping:
            return mapping[str_key]
        try:
            int_key = int(class_idx)
        except (TypeError, ValueError):
            return None
        return mapping.get(int_key)

    @staticmethod
    def _payload_to_ndarray(payload):
        """Decode compact gRPC payloads back to ndarray-compatible values."""
        if isinstance(payload, bytes):
            payload = payload.decode('utf-8')
        if isinstance(payload, str):
            if payload.startswith('znp:'):
                try:
                    raw = zlib.decompress(base64.b64decode(payload[4:]))
                    return np.load(io.BytesIO(raw), allow_pickle=False)
                except Exception:
                    return None
            try:
                payload = param2tensor(payload)
            except Exception:
                return None
        if isinstance(payload, torch.Tensor):
            return payload.detach().cpu().numpy()
        return payload

    def _ensure_stat_shapes(self, client_id, class_idx, mean, cov, count,
                            embedding_dim):
        mean_arr = np.asarray(mean)
        cov_arr = np.asarray(cov)
        if mean_arr.shape != (embedding_dim, ):
            raise ValueError(
                f"Server: Invalid statistic during aggregation from client "
                f"{client_id}, class {class_idx}: mean shape "
                f"{mean_arr.shape}, expected ({embedding_dim},). "
                f"Clear stale feature cache or align all clients to the same "
                f"feature_extractor/embedding_dim.")
        if cov_arr.shape != (embedding_dim, embedding_dim):
            raise ValueError(
                f"Server: Invalid statistic during aggregation from client "
                f"{client_id}, class {class_idx}: cov shape {cov_arr.shape}, "
                f"expected ({embedding_dim}, {embedding_dim}).")
        if count is None:
            raise ValueError(
                f"Server: Missing count during aggregation from client "
                f"{client_id}, class {class_idx}.")
        return mean_arr, cov_arr, int(count)

    def callback_for_augmentation_ready(self, message: Message):
        """Handle client signaling augmentation is complete"""
        client_id = message.sender
        self.augmentation_ready_clients.add(client_id)

        logger.info(f"Server: Client {client_id} augmentation ready ({len(self.augmentation_ready_clients)}/{self._client_num})")

        # When all clients are ready, start training
        if len(self.augmentation_ready_clients) >= self._client_num:
            if self.global_mlp is None:
                num_classes = self._cfg.model.num_classes
                if num_classes > 0:
                    self._build_global_mlp(num_classes)
            if self.use_fedopt:
                logger.info("Server: All clients ready, starting FedOpt training...")
            else:
                logger.info("Server: All clients ready, starting FedAvg training...")
            self.state = 1  # Move to round 1
            self._start_training_round()

    def _start_training_round(self):
        """Start a new training round by broadcasting model"""
        logger.info(f"Server: Starting training round {self.state}")

        # Check for phase transition in separated training mode
        if self.use_separated_training:
            if self.state == self.classifier_pretrain_rounds + 1 and self.training_phase == 'classifier':
                # Transition from Phase 1 to Phase 2
                self._transition_to_cnn_phase()

        # Prepare model parameters based on current mode and phase
        if self.use_separated_training:
            model_para = self._prepare_separated_training_params()
        elif self.use_cnn_distillation or self.use_feature_alignment:
            # Send both MLP and CNN parameters (for distillation or feature alignment)
            model_para = {
                'mlp': copy.deepcopy(self.global_mlp.state_dict()) if self.global_mlp else None,
                'cnn': copy.deepcopy(self.global_cnn.state_dict()) if self.global_cnn else None
            }
        else:
            # Standard mode: only MLP
            if self.global_mlp is not None:
                model_para = copy.deepcopy(self.global_mlp.state_dict())
            else:
                model_para = None

        # Attach global prompt ctx if PromptFL enabled
        if self.use_promptfl and self.global_prompt_ctx is not None:
            if not isinstance(model_para, dict):
                model_para = {'mlp': model_para}
            model_para['prompt'] = {'ctx': self.global_prompt_ctx.cpu()}

        model_para = self._attach_a3fl_payload(model_para)
        model_para = self._attach_cerberus_payload(model_para)
        model_para = self._attach_sabre_payload(model_para)

        # Broadcast to selected clients
        receiver = self._select_round_receivers()
        self.current_round_clients = list(receiver)
        send_bytes = self._sizeof_content(model_para)
        self._bytes_sent += send_bytes * len(receiver)
        self._round_start_time = time.time()
        for client_id in receiver:
            self.comm_manager.send(
                Message(
                    msg_type='model_para',
                    sender=self.ID,
                    receiver=[client_id],
                    state=self.state,
                    content=model_para
                )
            )
        logger.info(f"Server: Round {self.state} receivers={receiver}")
        self._mark_stage(f'round_{self.state}_model_updates')

    def _select_round_receivers(self):
        sample_num = int(self.sample_client_num)
        all_clients = list(range(1, self._client_num + 1))
        if sample_num <= 0 or sample_num >= self._client_num:
            return all_clients

        if self.a3fl_enabled:
            active_attackers = get_a3fl_active_attacker_ids(
                self._cfg, self.state, sample_num)
            active_attackers = [cid for cid in active_attackers if cid in all_clients]
            benign_pool = [cid for cid in all_clients if cid not in active_attackers]
            benign_needed = max(0, sample_num - len(active_attackers))
            if benign_needed > 0:
                benign_selected = np.random.choice(benign_pool,
                                                   size=benign_needed,
                                                   replace=False).tolist()
            else:
                benign_selected = []
            receiver = active_attackers + benign_selected
            if len(receiver) == sample_num:
                return receiver

        if self.cerberus_enabled:
            active_attackers = self._get_cerberus_active_attacker_ids(self.state)
            active_attackers = [cid for cid in active_attackers if cid in all_clients]
            benign_pool = [cid for cid in all_clients if cid not in active_attackers]
            benign_needed = max(0, sample_num - len(active_attackers))
            if benign_needed > 0:
                benign_selected = np.random.choice(benign_pool,
                                                   size=benign_needed,
                                                   replace=False).tolist()
            else:
                benign_selected = []
            receiver = active_attackers + benign_selected
            if len(receiver) == sample_num:
                return receiver

        if self.sabre_enabled:
            active_attackers = self._get_sabre_active_attacker_ids(self.state)
            active_attackers = [cid for cid in active_attackers if cid in all_clients]
            benign_pool = [cid for cid in all_clients if cid not in active_attackers]
            benign_needed = max(0, sample_num - len(active_attackers))
            if benign_needed > 0:
                benign_selected = np.random.choice(benign_pool,
                                                   size=benign_needed,
                                                   replace=False).tolist()
            else:
                benign_selected = []
            receiver = active_attackers + benign_selected
            if len(receiver) == sample_num:
                return receiver

        if self.label_flip_enabled:
            active_attackers = self._get_label_flip_active_attacker_ids(
                self.state)
            active_attackers = [
                cid for cid in active_attackers if cid in all_clients
            ]
            if len(active_attackers) > sample_num:
                active_attackers = np.random.choice(
                    active_attackers, size=sample_num,
                    replace=False).tolist()
            benign_pool = [
                cid for cid in all_clients if cid not in active_attackers
            ]
            benign_needed = max(0, sample_num - len(active_attackers))
            if benign_needed > 0:
                benign_selected = np.random.choice(
                    benign_pool, size=benign_needed,
                    replace=False).tolist()
            else:
                benign_selected = []
            receiver = active_attackers + benign_selected
            if len(receiver) == sample_num:
                return receiver

        if self.lie_enabled:
            active_attackers = self._get_lie_active_attacker_ids(self.state)
            active_attackers = [
                cid for cid in active_attackers if cid in all_clients
            ]
            if len(active_attackers) > sample_num:
                active_attackers = np.random.choice(
                    active_attackers, size=sample_num,
                    replace=False).tolist()
            benign_pool = [
                cid for cid in all_clients if cid not in active_attackers
            ]
            benign_needed = max(0, sample_num - len(active_attackers))
            if benign_needed > 0:
                benign_selected = np.random.choice(
                    benign_pool, size=benign_needed,
                    replace=False).tolist()
            else:
                benign_selected = []
            receiver = active_attackers + benign_selected
            if len(receiver) == sample_num:
                return receiver

        return self.sampler.sample(size=sample_num)

    def _prepare_separated_training_params(self):
        """Prepare model parameters for separated training mode"""
        if self.training_phase == 'classifier':
            # Phase 1: Only send classifier (MLP) parameters
            logger.info(f"Server: Phase 1 (Classifier Training) - Round {self.state}/{self.classifier_pretrain_rounds}")
            return {
                'phase': 'classifier',
                'classifier': copy.deepcopy(self.global_mlp.state_dict()) if self.global_mlp else None
            }
        else:
            # Phase 2: Send frozen classifier + CNN backbone
            logger.info(f"Server: Phase 2 (CNN Backbone Training) - Round {self.state}")
            return {
                'phase': 'cnn_backbone',
                'classifier': copy.deepcopy(self.pretrained_classifier) if self.pretrained_classifier else None,
                'cnn_backbone': copy.deepcopy(self.global_cnn.state_dict()) if self.global_cnn else None,
                'freeze_classifier': self.freeze_classifier
            }

    def _transition_to_cnn_phase(self):
        """Transition from classifier training to CNN backbone training"""
        logger.info("=" * 60)
        logger.info("Server: Transitioning to Phase 2 - CNN Backbone Training")
        logger.info("=" * 60)

        # Save the pretrained classifier
        if self.global_mlp is not None:
            self.pretrained_classifier = copy.deepcopy(self.global_mlp.state_dict())
            logger.info(f"Server: Saved pretrained classifier (Best MLP accuracy: {self.best_avg_accuracy:.4f})")
            logger.info(f"Server: Classifier state_dict keys: {list(self.pretrained_classifier.keys())}")
        else:
            logger.error("Server: global_mlp is None! Cannot save pretrained classifier!")
            return

        # Build CNN backbone (without classifier - will use pretrained one)
        self._build_cnn_backbone_only()

        # Update phase
        self.training_phase = 'cnn_backbone'
        logger.info(f"Server: Phase transition complete. Now in '{self.training_phase}' phase")

    def _build_cnn_backbone_only(self):
        """Build CNN backbone for separated training (Phase 2)"""
        from federatedscope.contrib.model.ggeur_cnn import GGEUR_CNN_Backbone

        num_classes = self._cfg.model.num_classes
        cnn_model_name = getattr(self.ggeur_cfg, 'cnn_model', 'resnet18')

        # Always train from scratch in separated training mode
        self.global_cnn = GGEUR_CNN_Backbone(
            model_name=cnn_model_name,
            num_classes=num_classes,
            pretrained=False  # From scratch
        )
        self.global_cnn = self.global_cnn.to(self.device)
        logger.info(f"Server: Built CNN backbone ({cnn_model_name}) for Phase 2 - FROM SCRATCH")

    def _build_classifier_from_state_dict(self, state_dict):
        """Build classifier model from saved state dict for evaluation"""
        num_classes = self._cfg.model.num_classes
        input_dim = None
        hidden_dim = self.ggeur_cfg.mlp_hidden_dim

        # Infer input dim from state dict when possible (more robust than cfg.ggeur.embedding_dim)
        try:
            # Linear classifier
            if 'weight' in state_dict and hasattr(state_dict['weight'], 'shape'):
                input_dim = int(state_dict['weight'].shape[1])
            else:
                # Sequential: pick the smallest numeric prefix layer's weight
                candidates = []
                for k, v in state_dict.items():
                    if not hasattr(v, 'shape'):
                        continue
                    if len(v.shape) != 2:
                        continue
                    m = re.match(r'^(\\d+)\\.weight$', str(k))
                    if m:
                        candidates.append((int(m.group(1)), int(v.shape[1])))
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    input_dim = candidates[0][1]
        except Exception:
            input_dim = None

        if input_dim is None:
            input_dim = self._get_embedding_dim()

        if hidden_dim > 0:
            classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.ggeur_cfg.mlp_dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            classifier = nn.Linear(input_dim, num_classes)

        classifier = classifier.to(self.device)

        try:
            classifier.load_state_dict(state_dict)
            logger.debug(f"Server: Classifier loaded successfully, keys: {list(state_dict.keys())}")
        except Exception as e:
            logger.error(f"Server: Failed to load classifier state dict: {e}")
            logger.error(f"Server: Expected keys: {list(classifier.state_dict().keys())}")
            logger.error(f"Server: Received keys: {list(state_dict.keys())}")

        return classifier

    def _aggregate_covariances(self, statistics_buffer=None):
        """Aggregate covariance matrices using parallel axis theorem"""
        logger.info("Server: Aggregating covariance matrices...")
        statistics_buffer = statistics_buffer or self.local_statistics_buffer

        # Collect all class indices
        all_classes = set()
        for client_stats in statistics_buffer.values():
            all_classes.update(client_stats['means'].keys())

        embedding_dim = self._get_embedding_dim()

        for class_idx in all_classes:
            class_idx = int(class_idx)
            means = []
            covs = []
            counts = []

            # Collect statistics for this class from all clients
            for client_id, client_stats in statistics_buffer.items():
                mean = self._get_class_value(client_stats['means'], class_idx)
                if mean is not None:
                    cov = self._get_class_value(client_stats['covs'],
                                                class_idx)
                    count = self._get_class_value(client_stats['counts'],
                                                  class_idx)
                    mean, cov, count = self._ensure_stat_shapes(
                        client_id, class_idx, mean, cov, count, embedding_dim)
                    means.append(mean)
                    covs.append(cov)
                    counts.append(count)

            if len(counts) == 0:
                self.global_cov_matrices[class_idx] = np.eye(embedding_dim) * 0.01
                continue

            # Compute aggregated mean
            total_count = sum(counts)
            aggregated_mean = np.zeros(embedding_dim)
            for i, (mean, count) in enumerate(zip(means, counts)):
                aggregated_mean += count * mean
            aggregated_mean /= total_count

            # Compute aggregated covariance using parallel axis theorem
            aggregated_cov = np.zeros((embedding_dim, embedding_dim))

            # First term: weighted average of local covariances
            for i, (cov, count) in enumerate(zip(covs, counts)):
                aggregated_cov += count * cov

            # Second term: between-client variance
            for i, (mean, count) in enumerate(zip(means, counts)):
                diff = mean - aggregated_mean
                aggregated_cov += count * np.outer(diff, diff)

            aggregated_cov /= total_count

            self.global_cov_matrices[class_idx] = aggregated_cov

        logger.info(f"Server: Aggregated covariances for {len(self.global_cov_matrices)} classes")

    def _compute_global_prototypes(self, statistics_buffer=None):
        """Compute global prototypes (weighted average of local means) for each class"""
        logger.info("Server: Computing global prototypes...")
        statistics_buffer = statistics_buffer or self.local_statistics_buffer

        embedding_dim = self._get_embedding_dim()

        # Collect all class indices
        all_classes = set()
        for client_stats in statistics_buffer.values():
            all_classes.update(client_stats['means'].keys())

        for class_idx in all_classes:
            class_idx = int(class_idx)
            means = []
            counts = []

            # Collect means for this class from all clients
            for client_id, client_stats in statistics_buffer.items():
                mean = self._get_class_value(client_stats['means'], class_idx)
                if mean is not None:
                    count = self._get_class_value(client_stats['counts'],
                                                  class_idx)
                    mean_arr = np.asarray(mean)
                    if mean_arr.shape != (embedding_dim, ):
                        raise ValueError(
                            f"Server: Invalid prototype statistic from client "
                            f"{client_id}, class {class_idx}: mean shape "
                            f"{mean_arr.shape}, expected ({embedding_dim},).")
                    if count is None:
                        raise ValueError(
                            f"Server: Missing prototype count from client "
                            f"{client_id}, class {class_idx}.")
                    means.append(mean_arr)
                    counts.append(int(count))

            if len(counts) == 0:
                continue

            # Compute weighted average mean (global prototype)
            total_count = sum(counts)
            global_mean = np.zeros(embedding_dim)
            for mean, count in zip(means, counts):
                global_mean += count * mean
            global_mean /= total_count

            self.global_prototypes[class_idx] = global_mean

        logger.info(f"Server: Computed global prototypes for {len(self.global_prototypes)} classes")

    def _prepare_other_prototypes(self, statistics_buffer=None):
        """Prepare prototypes from other clients for each client"""
        other_prototypes = {}
        prototype_sources = statistics_buffer or self.local_statistics_buffer

        for client_id in self.all_prototypes.keys():
            other_prototypes[client_id] = {}

            for other_client_id, client_stats in prototype_sources.items():
                if other_client_id == client_id:
                    continue
                prototypes = client_stats.get('prototypes', {})

                for class_idx, prototype in prototypes.items():
                    class_idx = int(class_idx)
                    if class_idx not in other_prototypes[client_id]:
                        other_prototypes[client_id][class_idx] = []
                    other_prototypes[client_id][class_idx].append(prototype)

        return other_prototypes

    def _broadcast_global_covariances(self, other_prototypes):
        """Broadcast global covariance matrices and prototypes to all clients"""
        logger.info("Server: Broadcasting global covariances to clients...")

        for client_id in self.local_statistics_buffer.keys():
            content = {
                'cov_matrices': self._serialize_array_payload(
                    self.global_cov_matrices),
                'other_prototypes': {
                    client_id: self._serialize_array_payload(
                        other_prototypes.get(client_id, {}))
                },
                'global_prototypes': self._serialize_array_payload(
                    self.global_prototypes)  # For feature alignment
            }
            self._bytes_sent += self._sizeof_content(content)
            logger.info(
                f"Server: Sending global covariances to client {client_id}")
            self.comm_manager.send(
                Message(
                    msg_type='global_covariances',
                    sender=self.ID,
                    receiver=[client_id],
                    state=self.state,
                    content=content
                )
            )
            logger.info(
                f"Server: Sent global covariances to client {client_id}")

        logger.info(f"Server: Broadcasted global covariances to {len(self.local_statistics_buffer)} clients")
        self._mark_stage('augmentation_ready')

    @staticmethod
    def _serialize_array_payload(payload):
        """Encode array/tensor leaves to compact strings for gRPC transfer."""
        if isinstance(payload, dict):
            return {
                key: GGEURServer._serialize_array_payload(value)
                for key, value in payload.items()
            }
        if isinstance(payload, list):
            return [
                GGEURServer._serialize_array_payload(value)
                for value in payload
            ]
        if isinstance(payload, tuple):
            return [
                GGEURServer._serialize_array_payload(value)
                for value in payload
            ]
        if isinstance(payload, (np.ndarray, torch.Tensor)):
            return GGEURServer._serialize_ndarray_payload(payload)
        return payload

    @staticmethod
    def _serialize_ndarray_payload(payload):
        """Serialize ndarray/tensor as compressed base64 to keep gRPC payloads small."""
        if isinstance(payload, torch.Tensor):
            payload = payload.detach().cpu().numpy()
        array = np.asarray(payload, dtype=np.float16)
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        compressed = zlib.compress(buffer.getvalue(), level=3)
        return 'znp:' + base64.b64encode(compressed).decode('ascii')

    def callback_funcs_model_para(self, message: Message):
        """
        Handle model parameter messages from clients.
        Aggregate using FedAvg.
        """
        round_idx = message.state
        sender = message.sender
        content = message.content
        self._bytes_recv += self._sizeof_content(content)

        if isinstance(content, (tuple, list)) and len(content) == 2:
            sample_size, model_para = content
        else:
            sample_size, model_para = 0, content

        # Store in message buffer
        if round_idx not in self.msg_buffer['train']:
            self.msg_buffer['train'][round_idx] = []

        self.msg_buffer['train'][round_idx].append((sample_size, model_para, sender))

        expected_num = len(getattr(self, 'current_round_clients', [])) or self._client_num
        logger.info(f"Server: Received model from client {sender} for round {round_idx} "
                    f"({len(self.msg_buffer['train'][round_idx])}/{expected_num})")

        # Check if all clients have responded
        if len(self.msg_buffer['train'][round_idx]) >= expected_num:
            self._perform_fedavg(round_idx)

    def _perform_fedavg(self, round_idx):
        """Perform aggregation for MLP (and optionally CNN).

        - Default: FedAvg weighted averaging
        - If cfg.fedopt.use: FedOpt server-side update for the global MLP
        """
        logger.info(
            f"Server: Performing {'FedOpt' if self.use_fedopt else 'FedAvg'} aggregation for round {round_idx}"
        )
        self._clear_stage()

        # Collect all model parameters
        all_params = self.msg_buffer['train'][round_idx]

        # Filter out empty updates
        valid_params = [
            (s, p, sender) for s, p, sender in all_params
            if s > 0 and p is not None
        ]

        if not valid_params:
            logger.warning("Server: No valid model parameters received")
            self.state = round_idx + 1
            if self.state < self._total_round_num:
                self._start_training_round()
            else:
                self._finish()
            return

        a3fl_updates = []
        cerberus_updates = []
        sabre_updates = []
        label_flip_updates = []
        cleaned_valid_params = []
        for sample_size, params, sender in valid_params:
            cleaned_params = params
            if isinstance(params, dict) and 'a3fl' in params:
                a3fl_updates.append(params.get('a3fl'))
                if 'mlp' not in params:
                    cleaned_params = copy.deepcopy(params)
                    cleaned_params.pop('a3fl', None)
            if isinstance(params, dict) and 'cerberus' in params:
                cerberus_meta = params.get('cerberus')
                cerberus_updates.append(cerberus_meta)
                if isinstance(cerberus_meta, dict) and \
                        cerberus_meta.get('active', False):
                    if 'mlp' in params and params.get('mlp') is not None:
                        self.cerberus_peer_model_bank[int(sender)] = \
                            copy.deepcopy(params.get('mlp'))
                    elif params is not None:
                        clean_for_bank = copy.deepcopy(params)
                        clean_for_bank.pop('cerberus', None)
                        clean_for_bank.pop('a3fl', None)
                        self.cerberus_peer_model_bank[int(sender)] = \
                            clean_for_bank
                if 'mlp' not in params:
                    cleaned_params = copy.deepcopy(cleaned_params)
                    cleaned_params.pop('cerberus', None)
            if isinstance(params, dict) and 'sabre' in params:
                sabre_updates.append(params.get('sabre'))
                if 'mlp' not in params:
                    cleaned_params = copy.deepcopy(cleaned_params)
                    cleaned_params.pop('sabre', None)
            if isinstance(params, dict) and 'label_flip' in params:
                label_flip_updates.append(params.get('label_flip'))
                if 'mlp' not in params:
                    cleaned_params = copy.deepcopy(cleaned_params)
                    cleaned_params.pop('label_flip', None)
            cleaned_valid_params.append((sample_size, cleaned_params, sender))
        valid_params = cleaned_valid_params

        active_a3fl = [
            meta for meta in a3fl_updates
            if isinstance(meta, dict) and meta.get('active', False)
        ]
        self.latest_a3fl_meta = active_a3fl[0] if active_a3fl else None
        self._update_a3fl_shared_trigger(active_a3fl)
        if self.a3fl_enabled:
            logger.info(
                f"Server: Round {round_idx} received {len(a3fl_updates)} A3FL metadata payloads, "
                f"active={len(active_a3fl)}, "
                f"shared_trigger={'yes' if self.a3fl_shared_trigger is not None else 'no'}")

        active_cerberus = [
            meta for meta in cerberus_updates
            if isinstance(meta, dict) and meta.get('active', False)
        ]
        self.latest_cerberus_meta = (
            active_cerberus[0] if active_cerberus else None)
        self._update_cerberus_shared_trigger(active_cerberus)
        if self.cerberus_enabled:
            logger.info(
                f"Server: Round {round_idx} received "
                f"{len(cerberus_updates)} CERBERUS metadata payloads, "
                f"active={len(active_cerberus)}, "
                f"peer_bank={len(self.cerberus_peer_model_bank)}")

        active_sabre = [
            meta for meta in sabre_updates
            if isinstance(meta, dict) and meta.get('active', False)
        ]
        self.latest_sabre_meta = active_sabre[0] if active_sabre else None
        self._update_sabre_shared_trigger(active_sabre)
        if self.sabre_enabled:
            logger.info(
                f"Server: Round {round_idx} received "
                f"{len(sabre_updates)} SABRE metadata payloads, "
                f"active={len(active_sabre)}, "
                f"shared_trigger={'yes' if self.sabre_shared_trigger is not None else 'no'}")

        active_label_flip = [
            meta for meta in label_flip_updates
            if isinstance(meta, dict) and meta.get('active', False)
        ]
        self.latest_label_flip_meta = (
            active_label_flip[0] if active_label_flip else None)
        if self.label_flip_enabled:
            total_flipped = sum(
                int(meta.get('flipped_samples', 0))
                for meta in active_label_flip
                if isinstance(meta, dict))
            logger.info(
                f"Server: Round {round_idx} received "
                f"{len(label_flip_updates)} label-flip metadata payloads, "
                f"active={len(active_label_flip)}, "
                f"flipped_samples={total_flipped}")

        valid_params = self._filter_benign_updates_for_aggregation(
            valid_params, round_idx)
        total_samples = sum(s for s, _, _ in valid_params)
        if total_samples <= 0:
            logger.warning(
                "Server: No positive sample weights available for aggregation")
            self.state = round_idx + 1
            if self.state < self._total_round_num:
                self._start_training_round()
            else:
                self._finish()
            return

        # Handle separated training mode
        if self.use_separated_training:
            self._perform_separated_fedavg(valid_params, total_samples, round_idx)
        else:
            # Check if we're in CNN distillation mode
            first_params = valid_params[0][1]
            is_combined_params = isinstance(first_params, dict) and 'mlp' in first_params

            if is_combined_params:
                # Aggregate MLP and CNN separately
                mlp_aggregated = self._aggregate_model_params(
                    [(s, p['mlp'], sender) for s, p, sender in valid_params
                     if p.get('mlp') is not None],
                    total_samples,
                    base_params=self.global_mlp.state_dict()
                    if self.global_mlp is not None else None,
                    aggregation_name='mlp'
                )
                cnn_aggregated = self._aggregate_model_params(
                    [(s, p['cnn'], sender) for s, p, sender in valid_params
                     if p.get('cnn') is not None],
                    total_samples,
                    base_params=self.global_cnn.state_dict()
                    if self.global_cnn is not None else None,
                    aggregation_name='cnn'
                )

                # Update global MLP (FedAvg or FedOpt)
                if mlp_aggregated and self.global_mlp is not None:
                    try:
                        if self.use_fedopt:
                            self._apply_fedopt_update(mlp_aggregated)
                        else:
                            self.global_mlp.load_state_dict(mlp_aggregated)
                    except Exception as e:
                        logger.debug(f"Server: Could not update MLP params: {e}")

                # Update global CNN
                if cnn_aggregated and self.global_cnn is not None:
                    try:
                        self.global_cnn.load_state_dict(cnn_aggregated)
                    except Exception as e:
                        logger.debug(f"Server: Could not load CNN params: {e}")

            else:
                # Standard mode: only MLP
                mlp_aggregated = self._aggregate_model_params(
                    valid_params,
                    total_samples,
                    base_params=self.global_mlp.state_dict()
                    if self.global_mlp is not None else None,
                    aggregation_name='mlp')

                if mlp_aggregated and self.global_mlp is not None:
                    try:
                        if self.use_fedopt:
                            self._apply_fedopt_update(mlp_aggregated)
                        else:
                            self.global_mlp.load_state_dict(mlp_aggregated)
                    except Exception as e:
                        logger.debug(f"Server: Could not load MLP params: {e}")

        should_eval = self._should_run_server_eval()

        # Evaluate MLP on test sets (using CLIP features)
        if should_eval:
            test_results = self._evaluate_on_test_sets()
            if test_results:
                acc_str = ', '.join([f"{k}: {v:.4f}" for k, v in test_results.items()])
                logger.info(f"Server: Round {round_idx} MLP Test Accuracy - {acc_str}")
            if self.a3fl_enabled and self.latest_a3fl_meta is not None:
                poison_results = self._evaluate_a3fl_on_test_sets()
                if poison_results:
                    poison_str = ', '.join([f"{k}: {v:.4f}" for k, v in poison_results.items()])
                    logger.info(f"Server: Round {round_idx} A3FL Poison Accuracy - {poison_str}")
                else:
                    logger.info(
                        f"Server: Round {round_idx} skipped A3FL Poison Accuracy logging "
                        f"because evaluation returned no results")
            if self.cerberus_enabled and self.latest_cerberus_meta is not None:
                cerberus_eval = self._evaluate_cerberus_on_test_sets()
                poison_results = cerberus_eval.get('asr', {})
                non_target_results = cerberus_eval.get('non_target_asr', {})
                clean_target_results = cerberus_eval.get(
                    'clean_target_rate', {})
                attacker_id = int(
                    self.latest_cerberus_meta.get('client_id', -1))
                if poison_results:
                    poison_str = ', '.join([f"{k}: {v:.4f}" for k, v in poison_results.items()])
                    logger.info(
                        f"Server: Round {round_idx} CERBERUS ASR "
                        f"(attacker {attacker_id}) "
                        f"- {poison_str}")
                    if non_target_results:
                        non_target_str = ', '.join([
                            f"{k}: {v:.4f}"
                            for k, v in non_target_results.items()
                        ])
                        logger.info(
                            f"Server: Round {round_idx} CERBERUS non-target ASR "
                            f"(attacker {attacker_id}) "
                            f"- {non_target_str}")
                    if clean_target_results:
                        clean_target_str = ', '.join([
                            f"{k}: {v:.4f}"
                            for k, v in clean_target_results.items()
                        ])
                        logger.info(
                            f"Server: Round {round_idx} CERBERUS clean target rate "
                            f"(target {int(self.latest_cerberus_meta.get('target_label', self._cfg.attack.target_label_ind))}) "
                            f"- {clean_target_str}")
                else:
                    logger.info(
                        f"Server: Round {round_idx} skipped CERBERUS ASR logging "
                        f"because evaluation returned no results")
            if self.sabre_enabled and self.latest_sabre_meta is not None:
                sabre_eval = self._evaluate_sabre_on_test_sets()
                poison_results = sabre_eval.get('asr', {})
                non_target_results = sabre_eval.get('non_target_asr', {})
                clean_target_results = sabre_eval.get(
                    'clean_target_rate', {})
                attacker_id = int(
                    self.latest_sabre_meta.get('client_id', -1))
                if poison_results:
                    poison_str = ', '.join([f"{k}: {v:.4f}" for k, v in poison_results.items()])
                    logger.info(
                        f"Server: Round {round_idx} SABRE ASR "
                        f"(attacker {attacker_id}) "
                        f"- {poison_str}")
                    if non_target_results:
                        non_target_str = ', '.join([
                            f"{k}: {v:.4f}"
                            for k, v in non_target_results.items()
                        ])
                        logger.info(
                            f"Server: Round {round_idx} SABRE non-target ASR "
                            f"(attacker {attacker_id}) "
                            f"- {non_target_str}")
                    if clean_target_results:
                        clean_target_str = ', '.join([
                            f"{k}: {v:.4f}"
                            for k, v in clean_target_results.items()
                        ])
                        logger.info(
                            f"Server: Round {round_idx} SABRE clean target rate "
                            f"(target {int(self.latest_sabre_meta.get('target_label', self._cfg.attack.target_label_ind))}) "
                            f"- {clean_target_str}")
                else:
                    logger.info(
                        f"Server: Round {round_idx} skipped SABRE ASR logging "
                        f"because evaluation returned no results")
        # Aggregate and evaluate PromptFL if enabled
        if self.use_promptfl:
            self._aggregate_prompt(valid_params, total_samples)
            if should_eval:
                prompt_results = self._evaluate_prompt_on_test_sets()
                if prompt_results:
                    acc_str = ', '.join([f"{k}: {v:.4f}" for k, v in prompt_results.items()])
                    logger.info(f"Server: Round {round_idx} Prompt Test Accuracy - {acc_str}")

        # Evaluate CNN on test sets (using original images) if enabled
        # Include separated training Phase 2
        should_eval_cnn = (
            (self.use_cnn_distillation or self.use_feature_alignment) or
            (self.use_separated_training and self.training_phase == 'cnn_backbone')
        )
        if should_eval and should_eval_cnn and self.global_cnn is not None:
            cnn_test_results = self._evaluate_cnn_on_test_sets()
            if cnn_test_results:
                acc_str = ', '.join([f"{k}: {v:.4f}" for k, v in cnn_test_results.items()])
                logger.info(f"Server: Round {round_idx} CNN Test Accuracy - {acc_str}")

        # Log progress
        if self._round_start_time is not None:
            round_elapsed = time.time() - self._round_start_time
            self._round_durations.append(round_elapsed)
            train_qps = total_samples / round_elapsed if round_elapsed > 0 else 0.0
            self._round_system_metrics.append({
                'round': int(round_idx),
                'total_samples': int(total_samples),
                'round_time_sec': float(round_elapsed),
                'train_qps_samples_per_sec': float(train_qps),
                'valid_client_updates': len(valid_params),
                'received_client_updates': len(all_params),
                'bytes_sent_total': int(self._bytes_sent),
                'bytes_recv_total': int(self._bytes_recv),
            })
            logger.info(f"Server: Round {round_idx} aggregation complete, "
                        f"total samples: {total_samples}, "
                        f"round time: {round_elapsed:.1f}s, "
                        f"train_qps={train_qps:.2f} samples/s, "
                        f"valid_updates={len(valid_params)}/{len(self.current_round_clients)}")

        # Move to next round
        self.state = round_idx + 1

        if self.state < self._total_round_num:
            self._start_training_round()
        else:
            self._finish()

    def _should_run_server_eval(self):
        """Whether this GGEUR server should run server-side evaluation."""
        eval_freq = getattr(getattr(self._cfg, 'eval', None), 'freq', 1)
        try:
            return int(eval_freq) > 0
        except Exception:
            return True

    def _apply_fedopt_update(self, averaged_state_dict):
        """
        Apply FedOpt server-side update based on the averaged client model.

        FedOpt uses the pseudo-gradient:
            g = w_t - w_bar
        and performs an optimizer step on the server model parameters.
        """
        if self.global_mlp is None:
            return
        self._init_fedopt_if_needed()
        if self.fedopt_optimizer is None:
            # Fallback to FedAvg if optimizer is not available
            self.global_mlp.load_state_dict(averaged_state_dict)
            return

        with torch.no_grad():
            current_state = self.global_mlp.state_dict()
            grads = {}
            for key, cur in current_state.items():
                if key not in averaged_state_dict:
                    continue
                avg = averaged_state_dict[key]
                if not isinstance(avg, torch.Tensor):
                    avg = torch.tensor(avg)
                if avg.device != cur.device:
                    avg = avg.to(cur.device)
                grads[key] = (cur.detach() - avg.detach()).type_as(cur)

        self.fedopt_optimizer.zero_grad()
        for name, p in self.global_mlp.named_parameters():
            if name in grads:
                p.grad = grads[name]
        self.fedopt_optimizer.step()

        if self.fedopt_annealing and self.fedopt_scheduler is not None:
            self.fedopt_scheduler.step()

    def _perform_separated_fedavg(self, valid_params, total_samples, round_idx):
        """Perform FedAvg for separated training mode"""
        first_params = valid_params[0][1]

        if self.training_phase == 'classifier':
            # Phase 1: Aggregate classifier parameters
            if isinstance(first_params, dict) and 'classifier' in first_params:
                classifier_aggregated = self._aggregate_model_params(
                    [(s, p['classifier'], sender)
                     for s, p, sender in valid_params
                     if p.get('classifier') is not None],
                    total_samples,
                    base_params=self.global_mlp.state_dict()
                    if self.global_mlp is not None else None,
                    aggregation_name='classifier'
                )
            else:
                classifier_aggregated = self._aggregate_model_params(
                    valid_params,
                    total_samples,
                    base_params=self.global_mlp.state_dict()
                    if self.global_mlp is not None else None,
                    aggregation_name='classifier')

            if classifier_aggregated and self.global_mlp is not None:
                try:
                    self.global_mlp.load_state_dict(classifier_aggregated)
                    logger.info(f"Server: Phase 1 - Aggregated classifier from {len(valid_params)} clients")
                except Exception as e:
                    logger.debug(f"Server: Could not load classifier params: {e}")

        else:
            # Phase 2: Aggregate only CNN backbone parameters (classifier is frozen)
            if isinstance(first_params, dict) and 'cnn_backbone' in first_params:
                cnn_aggregated = self._aggregate_model_params(
                    [(s, p['cnn_backbone'], sender)
                     for s, p, sender in valid_params
                     if p.get('cnn_backbone') is not None],
                    total_samples,
                    base_params=self.global_cnn.state_dict()
                    if self.global_cnn is not None else None,
                    aggregation_name='cnn_backbone'
                )

                if cnn_aggregated and self.global_cnn is not None:
                    try:
                        self.global_cnn.load_state_dict(cnn_aggregated)
                        logger.info(f"Server: Phase 2 - Aggregated CNN backbone from {len(valid_params)} clients")
                    except Exception as e:
                        logger.debug(f"Server: Could not load CNN backbone params: {e}")

    def _is_flame_enabled(self):
        """Whether GGEUR should use FLAME for server-side aggregation."""
        ggeur_method = str(getattr(self.ggeur_cfg, 'defense_method', '')).lower()
        agg_method = str(getattr(self._cfg.aggregator, 'robust_rule',
                                 '')).lower() if hasattr(self._cfg, 'aggregator') else ''
        return ggeur_method == 'flame' or agg_method == 'flame'

    def _is_foolsgold_enabled(self):
        """Whether GGEUR should use FoolsGold for server-side aggregation."""
        ggeur_method = str(getattr(self.ggeur_cfg, 'defense_method', '')).lower()
        agg_method = str(getattr(self._cfg.aggregator, 'robust_rule',
                                 '')).lower() if hasattr(self._cfg, 'aggregator') else ''
        return ggeur_method in ('foolsgold', 'fools_gold') or \
            agg_method in ('foolsgold', 'fools_gold')

    def _is_multi_krum_enabled(self):
        """Whether GGEUR should use Multi-Krum for server-side aggregation."""
        ggeur_method = str(getattr(self.ggeur_cfg, 'defense_method', '')).lower()
        agg_method = str(getattr(self._cfg.aggregator, 'robust_rule',
                                 '')).lower() if hasattr(self._cfg, 'aggregator') else ''
        return ggeur_method in ('multi_krum', 'multikrum') or \
            agg_method in ('multi_krum', 'multikrum')

    def _is_trimmed_mean_enabled(self):
        """Whether GGEUR should use Trimmed Mean for server-side aggregation."""
        ggeur_method = str(getattr(self.ggeur_cfg, 'defense_method', '')).lower()
        agg_method = str(getattr(self._cfg.aggregator, 'robust_rule',
                                 '')).lower() if hasattr(self._cfg, 'aggregator') else ''
        return ggeur_method in ('trimmed_mean', 'trimmedmean', 'trim_mean') or \
            agg_method in ('trimmed_mean', 'trimmedmean', 'trim_mean')

    def _is_align_ins_enabled(self):
        """Whether GGEUR should use AlignIns for server-side aggregation."""
        ggeur_method = str(getattr(self.ggeur_cfg, 'defense_method', '')).lower()
        agg_method = str(getattr(self._cfg.aggregator, 'robust_rule',
                                 '')).lower() if hasattr(self._cfg, 'aggregator') else ''
        return ggeur_method in ('align_ins', 'alignins') or \
            agg_method in ('align_ins', 'alignins')

    def _is_mars_enabled(self):
        """Whether GGEUR should use MARS for server-side aggregation."""
        ggeur_method = str(getattr(self.ggeur_cfg, 'defense_method', '')).lower()
        agg_method = str(getattr(self._cfg.aggregator, 'robust_rule',
                                 '')).lower() if hasattr(self._cfg, 'aggregator') else ''
        return ggeur_method in ('mars', 'mars_defense') or \
            agg_method in ('mars', 'mars_defense')

    def _mars_should_target_model(self, aggregation_name):
        targets = getattr(self.ggeur_cfg, 'mars_target_models',
                          ['mlp', 'classifier', 'model'])
        if isinstance(targets, str):
            targets = [item.strip() for item in targets.split(',')]
        targets = {str(item).lower() for item in targets}
        return 'all' in targets or str(aggregation_name).lower() in targets

    def _is_multi_metrics_enabled(self):
        """Whether GGEUR should use multi-metrics adaptive defense."""
        ggeur_method = str(getattr(self.ggeur_cfg, 'defense_method', '')).lower()
        agg_method = str(getattr(self._cfg.aggregator, 'robust_rule',
                                 '')).lower() if hasattr(self._cfg, 'aggregator') else ''
        aliases = ('multi_metrics', 'multi-metrics', 'multimetrics',
                   'multi_metric',
                   'multi_metrics_adaptive')
        return ggeur_method in aliases or agg_method in aliases

    def _multi_metrics_should_target_model(self, aggregation_name):
        targets = getattr(self.ggeur_cfg, 'multi_metrics_target_models',
                          ['mlp', 'classifier', 'model'])
        if isinstance(targets, str):
            targets = [item.strip() for item in targets.split(',')]
        targets = {str(item).lower() for item in targets}
        return 'all' in targets or str(aggregation_name).lower() in targets

    @staticmethod
    def _unpack_param_entry(entry):
        if len(entry) >= 3:
            return entry[0], entry[1], entry[2]
        return entry[0], entry[1], None

    @staticmethod
    def _to_param_tensor(param):
        if isinstance(param, torch.Tensor):
            return param
        try:
            restored = param2tensor(param)
            if isinstance(restored, torch.Tensor):
                return restored
            return torch.tensor(restored)
        except Exception as e:
            logger.debug(f"Server: Cannot convert model parameter to tensor ({e})")
            return None

    def _aggregate_model_params(self,
                                params_list,
                                total_samples,
                                base_params=None,
                                aggregation_name='model'):
        """Aggregate model parameters with FedAvg or a configured defense."""
        if not params_list:
            return None

        # Filter valid params
        valid_params = [
            (s, p, sender)
            for s, p, sender in
            (self._unpack_param_entry(entry) for entry in params_list)
            if s > 0 and p is not None
        ]
        if not valid_params:
            return None

        first_params = valid_params[0][1]
        if not isinstance(first_params, dict):
            logger.warning(f"Server: Expected dict for model params, got {type(first_params)}, skipping aggregation")
            return None

        valid_params = self._apply_little_is_enough_attack_to_params(
            valid_params, aggregation_name, base_params=base_params)
        first_params = valid_params[0][1]

        if self._is_mars_enabled() and \
                self._mars_should_target_model(aggregation_name):
            mars_result = self._mars_aggregate_model_params(
                valid_params,
                aggregation_name=aggregation_name)
            if mars_result is not None:
                return mars_result
            logger.warning(
                f"Server: MARS requested for {aggregation_name}, but no "
                "usable CBE features were available; falling back to FedAvg")

        if self._is_multi_metrics_enabled() and \
                self._multi_metrics_should_target_model(aggregation_name):
            if base_params is not None:
                multi_metrics_result = self._multi_metrics_aggregate_model_params(
                    valid_params,
                    base_params,
                    aggregation_name=aggregation_name)
                if multi_metrics_result is not None:
                    return multi_metrics_result
            logger.warning(
                f"Server: Multi-metrics requested for {aggregation_name}, "
                "but no usable base parameters were available; falling back "
                "to FedAvg")

        if self._is_flame_enabled():
            if base_params is not None:
                flame_result = self._flame_aggregate_model_params(
                    valid_params,
                    base_params,
                    aggregation_name=aggregation_name)
                if flame_result is not None:
                    return flame_result
            logger.warning(
                f"Server: FLAME requested for {aggregation_name}, but no "
                "usable base parameters were available; falling back to FedAvg")

        if self._is_foolsgold_enabled():
            if base_params is not None:
                foolsgold_result = self._foolsgold_aggregate_model_params(
                    valid_params,
                    base_params,
                    aggregation_name=aggregation_name)
                if foolsgold_result is not None:
                    return foolsgold_result
            logger.warning(
                f"Server: FoolsGold requested for {aggregation_name}, but no "
                "usable base parameters were available; falling back to FedAvg")

        if self._is_multi_krum_enabled():
            if base_params is not None:
                multi_krum_result = self._multi_krum_aggregate_model_params(
                    valid_params,
                    base_params,
                    aggregation_name=aggregation_name)
                if multi_krum_result is not None:
                    return multi_krum_result
            logger.warning(
                f"Server: Multi-Krum requested for {aggregation_name}, but no "
                "usable base parameters were available; falling back to FedAvg")

        if self._is_trimmed_mean_enabled():
            if base_params is not None:
                trimmed_mean_result = self._trimmed_mean_aggregate_model_params(
                    valid_params,
                    base_params,
                    aggregation_name=aggregation_name)
                if trimmed_mean_result is not None:
                    return trimmed_mean_result
            logger.warning(
                f"Server: Trimmed Mean requested for {aggregation_name}, but no "
                "usable base parameters were available; falling back to FedAvg")

        if self._is_align_ins_enabled():
            if base_params is not None:
                align_ins_result = self._align_ins_aggregate_model_params(
                    valid_params,
                    base_params,
                    aggregation_name=aggregation_name)
                if align_ins_result is not None:
                    return align_ins_result
            logger.warning(
                f"Server: AlignIns requested for {aggregation_name}, but no "
                "usable base parameters were available; falling back to FedAvg")

        aggregated_params = {}

        for key in first_params.keys():
            param_tensor = first_params[key]
            if isinstance(param_tensor, dict):
                # Nested dict (e.g. combined params accidentally passed in); skip
                logger.debug(f"Server: Skipping nested dict value for key '{key}' during aggregation setup")
                continue
            param_tensor = self._to_param_tensor(param_tensor)
            if param_tensor is None:
                logger.debug(f"Server: Cannot convert key '{key}' to tensor, skipping")
                continue
            aggregated_params[key] = torch.zeros_like(param_tensor).float()

        if not aggregated_params:
            return None

        # Weighted average — only iterate over keys validated from first_params
        for sample_size, params, _ in valid_params:
            weight = sample_size / total_samples
            for key in aggregated_params.keys():
                if key not in params:
                    continue
                param_tensor = params[key]
                if isinstance(param_tensor, dict):
                    continue
                param_tensor = self._to_param_tensor(param_tensor)
                if param_tensor is None:
                    continue
                aggregated_params[key] += weight * param_tensor.float()

        return aggregated_params

    def _collect_update_vectors(self, valid_params, base_params,
                                aggregation_name):
        first_params = valid_params[0][1]
        vector_keys = []
        base_tensors = {}
        shapes = {}
        dtypes = {}
        devices = {}

        for key, value in first_params.items():
            if isinstance(value, dict) or key not in base_params:
                continue
            base_tensor = self._to_param_tensor(base_params[key])
            value_tensor = self._to_param_tensor(value)
            if base_tensor is None or value_tensor is None:
                continue
            if not torch.is_floating_point(base_tensor) or \
                    not torch.is_floating_point(value_tensor):
                continue
            vector_keys.append(key)
            base_tensors[key] = base_tensor.detach().to(self.device).float()
            shapes[key] = base_tensor.shape
            dtypes[key] = base_tensor.dtype
            devices[key] = base_tensor.device

        if not vector_keys:
            logger.warning(
                f"Server: {aggregation_name} found no floating parameters "
                "for robust aggregation")
            return None

        update_vectors = []
        retained_valid_params = []
        for sample_size, params, sender in valid_params:
            pieces = []
            usable = True
            for key in vector_keys:
                if key not in params:
                    usable = False
                    break
                tensor = self._to_param_tensor(params[key])
                if tensor is None:
                    usable = False
                    break
                tensor = tensor.detach().to(self.device).float()
                pieces.append((tensor - base_tensors[key]).reshape(-1))
            if usable:
                update_vectors.append(torch.cat(pieces))
                retained_valid_params.append((sample_size, params, sender))

        if not update_vectors:
            return None

        return {
            'vector_keys': vector_keys,
            'base_tensors': base_tensors,
            'shapes': shapes,
            'dtypes': dtypes,
            'devices': devices,
            'update_tensor': torch.stack(update_vectors, dim=0),
            'retained_valid_params': retained_valid_params,
        }

    def _state_dict_from_update_vector(self, aggregated_update, vector_info,
                                       valid_params):
        aggregated_params = {}
        offset = 0
        for key in vector_info['vector_keys']:
            param_size = int(np.prod(vector_info['shapes'][key]))
            update = aggregated_update[offset:offset + param_size].view(
                vector_info['shapes'][key])
            new_value = vector_info['base_tensors'][key] + update
            aggregated_params[key] = new_value.to(
                device=vector_info['devices'][key],
                dtype=vector_info['dtypes'][key])
            offset += param_size

        fedavg_params = self._fedavg_model_params_no_defense(
            valid_params,
            sum(s for s, _, _ in valid_params))
        if fedavg_params:
            for key, value in fedavg_params.items():
                if key not in aggregated_params:
                    aggregated_params[key] = value
        return aggregated_params

    def _multi_metrics_aggregate_model_params(self,
                                              valid_params,
                                              base_params,
                                              aggregation_name='model'):
        """Multi-metrics adaptive aggregation for GGEUR state_dict payloads.

        This follows Huang et al. (ICCV 2023): build each client feature from
        Manhattan norm, Euclidean norm and Cosine similarity, whiten the
        round-wise feature-distance indicators, discard high-divergence
        clients, and FedAvg the retained updates.
        """
        min_clients = int(getattr(self.ggeur_cfg,
                                  'multi_metrics_min_clients', 4))
        min_clients = max(4, min_clients)
        if len(valid_params) < min_clients:
            logger.info(
                f"Server: Multi-metrics for {aggregation_name} needs at "
                f"least {min_clients} updates for whitening; using FedAvg "
                "fallback")
            return None

        vector_info = self._collect_update_vectors(
            valid_params,
            base_params,
            f"Multi-metrics {aggregation_name}")
        if vector_info is None:
            return None

        update_tensor = vector_info['update_tensor'].float()
        retained_valid_params = vector_info['retained_valid_params']
        n_users = len(retained_valid_params)
        if n_users < min_clients:
            logger.info(
                f"Server: Multi-metrics for {aggregation_name} has fewer "
                f"than {min_clients} usable updates; using FedAvg fallback")
            return None

        eps = float(getattr(self.ggeur_cfg,
                            'multi_metrics_cov_eps', 1e-6))
        eps = max(eps, 1e-12)
        base_vector = torch.cat([
            vector_info['base_tensors'][key].reshape(-1)
            for key in vector_info['vector_keys']
        ]).to(self.device).float()
        client_vectors = update_tensor + base_vector.view(1, -1)

        manhattan = torch.norm(update_tensor, p=1, dim=1)
        euclidean = torch.norm(update_tensor, p=2, dim=1)
        base_norm = torch.norm(base_vector, p=2) + eps
        client_norms = torch.norm(client_vectors, p=2, dim=1) + eps
        cosine = torch.sum(client_vectors * base_vector.view(1, -1),
                           dim=1) / (base_norm * client_norms)
        features = torch.stack([manhattan, euclidean, cosine], dim=1)

        indicators = torch.sum(
            torch.abs(features.unsqueeze(1) - features.unsqueeze(0)),
            dim=1)
        cov = self._multi_metrics_covariance(indicators, eps)
        inv_cov = torch.pinverse(cov)
        raw_scores = torch.sum((indicators @ inv_cov) * indicators, dim=1)
        scores = torch.sqrt(torch.clamp(raw_scores, min=0.0))
        scores = torch.where(torch.isfinite(scores), scores,
                             torch.full_like(scores, float('inf')))

        keep_ratio = float(getattr(self.ggeur_cfg,
                                   'multi_metrics_keep_ratio', 0.5))
        keep_ratio = max(0.0, min(1.0, keep_ratio))
        keep_num = int(math.ceil(n_users * keep_ratio))
        keep_num = max(1, min(keep_num, n_users))
        keep_indices = torch.topk(scores,
                                  keep_num,
                                  largest=False).indices
        keep_indices_list = keep_indices.detach().cpu().tolist()
        selected_params = [
            retained_valid_params[idx] for idx in keep_indices_list
        ]
        selected_total_samples = sum(s for s, _, _ in selected_params)
        if selected_total_samples <= 0:
            return None

        client_ids = [
            int(sender) if sender is not None else None
            for _, _, sender in retained_valid_params
        ]
        kept_client_ids = [client_ids[idx] for idx in keep_indices_list]
        logger.info(
            f"Server: Multi-metrics {aggregation_name} client_ids={client_ids}, "
            f"kept_client_ids={kept_client_ids}, keep_ratio={keep_ratio:.4f}, "
            f"keep_num={keep_num}/{n_users}")

        if bool(getattr(self.ggeur_cfg, 'multi_metrics_debug', False)):
            self._log_multi_metrics_debug(
                aggregation_name=aggregation_name,
                client_ids=client_ids,
                features=features,
                indicators=indicators,
                scores=scores,
                kept_client_ids=kept_client_ids)

        return self._fedavg_model_params_no_defense(
            selected_params,
            selected_total_samples)

    @staticmethod
    def _multi_metrics_covariance(features, eps):
        centered = features - features.mean(dim=0, keepdim=True)
        denom = max(int(features.shape[0]) - 1, 1)
        cov = centered.T @ centered / float(denom)
        eye = torch.eye(cov.shape[0], device=features.device, dtype=cov.dtype)
        return cov + eps * eye

    def _log_multi_metrics_debug(self,
                                 aggregation_name,
                                 client_ids,
                                 features,
                                 indicators,
                                 scores,
                                 kept_client_ids):
        def _round_list(tensor):
            return np.round(tensor.detach().cpu().float().numpy(),
                            6).tolist()

        logger.info(
            f"Server: Multi-metrics DEBUG {aggregation_name} "
            f"client_ids={client_ids}, features={_round_list(features)}, "
            f"indicators={_round_list(indicators)}, "
            f"scores={_round_list(scores)}, "
            f"kept_client_ids={kept_client_ids}")

    def _foolsgold_aggregate_model_params(self,
                                          valid_params,
                                          base_params,
                                          aggregation_name='model'):
        """FoolsGold aggregation adapted to GGEUR state_dict payloads."""
        if len(valid_params) < 2:
            logger.info(
                f"Server: FoolsGold for {aggregation_name} needs at least two "
                "updates; using FedAvg fallback")
            return None
        if any(sender is None for _, _, sender in valid_params):
            logger.warning(
                f"Server: FoolsGold for {aggregation_name} needs client ids; "
                "using FedAvg fallback")
            return None

        vector_info = self._collect_update_vectors(valid_params, base_params,
                                                   f"FoolsGold {aggregation_name}")
        if vector_info is None:
            return None

        update_tensor = vector_info['update_tensor']
        retained_valid_params = vector_info['retained_valid_params']
        if len(retained_valid_params) < 2:
            logger.info(
                f"Server: FoolsGold for {aggregation_name} has fewer than two "
                "usable updates; using FedAvg fallback")
            return None

        history_key = aggregation_name
        if history_key not in self.foolsgold_update_history:
            self.foolsgold_update_history[history_key] = {}
        history = self.foolsgold_update_history[history_key]

        selected_history = []
        for idx, (_, _, sender) in enumerate(retained_valid_params):
            sender = int(sender)
            current_update = update_tensor[idx].detach().clone()
            if sender not in history:
                history[sender] = current_update
            else:
                history[sender] = history[sender].to(self.device) + current_update
            selected_history.append(history[sender])

        eps = float(getattr(self.ggeur_cfg, 'foolsgold_eps', 1e-5))
        client_ids = [int(sender) for _, _, sender in retained_valid_params]
        update_norms = torch.norm(update_tensor.float(), dim=1)
        history_tensor = torch.stack(selected_history, dim=0).reshape(
            len(selected_history), -1)
        history_norms = torch.norm(history_tensor.float(), dim=1)
        normalized_history = F.normalize(history_tensor.float(),
                                         p=2,
                                         dim=1,
                                         eps=eps)
        cosine_similarity = torch.matmul(normalized_history,
                                         normalized_history.T)
        cosine_similarity = cosine_similarity - torch.eye(
            cosine_similarity.shape[0], device=self.device)
        maxcs = torch.max(cosine_similarity, dim=1)[0] + eps

        pardoned = cosine_similarity.clone()
        for i in range(pardoned.shape[0]):
            for j in range(pardoned.shape[1]):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    pardoned[i][j] = pardoned[i][j] * (maxcs[i] / maxcs[j])

        weights = 1.0 - torch.max(pardoned, dim=1)[0]
        trust_before_clip = weights.detach().clone()
        weights = torch.clamp(weights, min=0.0, max=1.0)
        weights = weights / (torch.max(weights) + eps)
        trust_after_rescale = weights.detach().clone()
        weights = torch.where(weights == 1.0,
                              torch.full_like(weights, 0.99),
                              weights)
        weights = torch.log((weights / (1.0 - weights + eps)) + eps) + 0.5
        weights = torch.where(torch.isinf(weights) | (weights > 1.0),
                              torch.ones_like(weights),
                              weights)
        weights = torch.clamp(weights, min=0.0, max=1.0)
        trust_after_logit = weights.detach().clone()

        if bool(getattr(self.ggeur_cfg, 'foolsgold_use_sample_weight',
                        False)):
            sample_weights = torch.tensor(
                [max(0, s) for s, _, _ in retained_valid_params],
                device=self.device,
                dtype=weights.dtype)
            weights = weights * sample_weights

        weight_sum = torch.sum(weights)
        if float(weight_sum.item()) <= eps:
            logger.warning(
                f"Server: FoolsGold {aggregation_name} produced zero trust "
                "mass; aggregated update will be near zero")
        normalized_weights = weights / (weight_sum + eps)

        aggregated_update = torch.sum(
            update_tensor * normalized_weights.view(-1, 1), dim=0)
        logger.info(
            f"Server: FoolsGold {aggregation_name} client_ids={client_ids}, "
            f"weights={normalized_weights.detach().cpu().numpy().tolist()}")
        if bool(getattr(self.ggeur_cfg, 'foolsgold_debug', False)):
            self._log_foolsgold_debug(
                aggregation_name=aggregation_name,
                client_ids=client_ids,
                update_norms=update_norms,
                history_norms=history_norms,
                cosine_similarity=cosine_similarity,
                maxcs=maxcs,
                pardoned=pardoned,
                trust_before_clip=trust_before_clip,
                trust_after_rescale=trust_after_rescale,
                trust_after_logit=trust_after_logit,
                final_weights=normalized_weights)

        return self._state_dict_from_update_vector(
            aggregated_update,
            vector_info,
            retained_valid_params)

    def _log_foolsgold_debug(self,
                             aggregation_name,
                             client_ids,
                             update_norms,
                             history_norms,
                             cosine_similarity,
                             maxcs,
                             pardoned,
                             trust_before_clip,
                             trust_after_rescale,
                             trust_after_logit,
                             final_weights):
        """Log FoolsGold internals without affecting other aggregators."""
        max_clients = int(getattr(self.ggeur_cfg,
                                  'foolsgold_debug_max_clients', 20))

        def _round_list(tensor):
            return np.round(tensor.detach().cpu().float().numpy(),
                            6).tolist()

        logger.info(
            f"Server: FoolsGold DEBUG {aggregation_name} client_ids={client_ids}, "
            f"update_norms={_round_list(update_norms)}, "
            f"history_norms={_round_list(history_norms)}, "
            f"maxcs={_round_list(maxcs)}, "
            f"trust_before_clip={_round_list(trust_before_clip)}, "
            f"trust_after_rescale={_round_list(trust_after_rescale)}, "
            f"trust_after_logit={_round_list(trust_after_logit)}, "
            f"final_weights={_round_list(final_weights)}")

        if len(client_ids) <= max_clients:
            logger.info(
                f"Server: FoolsGold DEBUG {aggregation_name} "
                f"cosine_similarity={_round_list(cosine_similarity)}")
            logger.info(
                f"Server: FoolsGold DEBUG {aggregation_name} "
                f"pardoned_similarity={_round_list(pardoned)}")
        else:
            logger.info(
                f"Server: FoolsGold DEBUG {aggregation_name} skipped matrix "
                f"dump because clients={len(client_ids)} exceeds "
                f"foolsgold_debug_max_clients={max_clients}")

    def _multi_krum_aggregate_model_params(self,
                                           valid_params,
                                           base_params,
                                           aggregation_name='model'):
        """Multi-Krum aggregation adapted to GGEUR state_dict payloads."""
        if len(valid_params) < 2:
            logger.info(
                f"Server: Multi-Krum for {aggregation_name} needs at least two "
                "updates; using FedAvg fallback")
            return None

        vector_info = self._collect_update_vectors(valid_params, base_params,
                                                   f"Multi-Krum {aggregation_name}")
        if vector_info is None:
            return None

        update_tensor = vector_info['update_tensor']
        retained_valid_params = vector_info['retained_valid_params']
        n_users = len(retained_valid_params)
        if n_users < 2:
            logger.info(
                f"Server: Multi-Krum for {aggregation_name} has fewer than two "
                "usable updates; using FedAvg fallback")
            return None

        num_malicious = int(getattr(self.ggeur_cfg,
                                    'multi_krum_num_malicious', 2))
        num_malicious = max(0, min(num_malicious, n_users - 1))
        if n_users < 2 * num_malicious + 3:
            logger.warning(
                f"Server: Multi-Krum {aggregation_name} received n={n_users}, "
                f"f={num_malicious}; formal condition n >= 2f + 3 is not met")

        distances = torch.cdist(update_tensor.float(), update_tensor.float(),
                                p=2)
        k = min(max(n_users - num_malicious - 2, 1), n_users - 1)
        nearest_distances = torch.sort(distances, dim=1)[0][:, 1:k + 1]
        krum_scores = torch.sum(nearest_distances, dim=1)
        select_num = min(max(n_users - 2 * num_malicious, 1), n_users)
        selected_indices = torch.topk(krum_scores,
                                      select_num,
                                      largest=False).indices
        selected_indices_list = selected_indices.detach().cpu().tolist()
        client_ids = [
            int(sender) if sender is not None else None
            for _, _, sender in retained_valid_params
        ]
        selected_client_ids = [client_ids[i] for i in selected_indices_list]
        aggregated_update = update_tensor[selected_indices].mean(dim=0)

        logger.info(
            f"Server: Multi-Krum {aggregation_name} client_ids={client_ids}, "
            f"num_malicious={num_malicious}, k={k}, "
            f"selected_client_ids={selected_client_ids}, "
            f"selected_indices={selected_indices_list}")

        if bool(getattr(self.ggeur_cfg, 'multi_krum_debug', False)):
            self._log_multi_krum_debug(
                aggregation_name=aggregation_name,
                client_ids=client_ids,
                distances=distances,
                krum_scores=krum_scores,
                selected_indices=selected_indices,
                selected_client_ids=selected_client_ids,
                k=k,
                select_num=select_num)

        return self._state_dict_from_update_vector(
            aggregated_update,
            vector_info,
            retained_valid_params)

    def _log_multi_krum_debug(self,
                              aggregation_name,
                              client_ids,
                              distances,
                              krum_scores,
                              selected_indices,
                              selected_client_ids,
                              k,
                              select_num):
        """Log Multi-Krum internals without affecting other aggregators."""
        max_clients = int(getattr(self.ggeur_cfg,
                                  'multi_krum_debug_max_clients', 20))

        def _round_list(tensor):
            return np.round(tensor.detach().cpu().float().numpy(),
                            6).tolist()

        logger.info(
            f"Server: Multi-Krum DEBUG {aggregation_name} client_ids={client_ids}, "
            f"k={k}, select_num={select_num}, "
            f"krum_scores={_round_list(krum_scores)}, "
            f"selected_indices={selected_indices.detach().cpu().tolist()}, "
            f"selected_client_ids={selected_client_ids}")

        if len(client_ids) <= max_clients:
            logger.info(
                f"Server: Multi-Krum DEBUG {aggregation_name} "
                f"distances={_round_list(distances)}")
        else:
            logger.info(
                f"Server: Multi-Krum DEBUG {aggregation_name} skipped matrix "
                f"dump because clients={len(client_ids)} exceeds "
                f"multi_krum_debug_max_clients={max_clients}")

    def _trimmed_mean_aggregate_model_params(self,
                                             valid_params,
                                             base_params,
                                             aggregation_name='model'):
        """Coordinate-wise Trimmed Mean aggregation for GGEUR state_dicts."""
        if len(valid_params) < 2:
            logger.info(
                f"Server: Trimmed Mean for {aggregation_name} needs at least two "
                "updates; using FedAvg fallback")
            return None

        vector_info = self._collect_update_vectors(
            valid_params,
            base_params,
            f"Trimmed Mean {aggregation_name}")
        if vector_info is None:
            return None

        update_tensor = vector_info['update_tensor']
        retained_valid_params = vector_info['retained_valid_params']
        n_users = len(retained_valid_params)
        if n_users < 2:
            logger.info(
                f"Server: Trimmed Mean for {aggregation_name} has fewer than two "
                "usable updates; using FedAvg fallback")
            return None

        trim_ratio = float(getattr(self.ggeur_cfg,
                                   'trimmed_mean_trim_ratio', 0.2))
        trim_ratio = max(0.0, min(trim_ratio, 0.5))
        trim_count = int(n_users * trim_ratio)
        if trim_ratio > 0.0:
            trim_count = max(1, trim_count)
        trim_count = min(trim_count, max((n_users - 1) // 2, 0))

        sorted_updates = torch.sort(update_tensor.float(), dim=0)[0]
        if trim_count > 0:
            trimmed_updates = sorted_updates[trim_count:n_users - trim_count]
        else:
            trimmed_updates = sorted_updates
        if trimmed_updates.shape[0] <= 0:
            logger.warning(
                f"Server: Trimmed Mean {aggregation_name} removed all updates; "
                "using FedAvg fallback")
            return None

        aggregated_update = trimmed_updates.mean(dim=0)
        client_ids = [
            int(sender) if sender is not None else None
            for _, _, sender in retained_valid_params
        ]

        logger.info(
            f"Server: Trimmed Mean {aggregation_name} client_ids={client_ids}, "
            f"trim_ratio={trim_ratio}, trim_count={trim_count}, "
            f"retained_per_coordinate={trimmed_updates.shape[0]}")

        if bool(getattr(self.ggeur_cfg, 'trimmed_mean_debug', False)):
            self._log_trimmed_mean_debug(
                aggregation_name=aggregation_name,
                client_ids=client_ids,
                update_tensor=update_tensor,
                trim_ratio=trim_ratio,
                trim_count=trim_count,
                retained_count=trimmed_updates.shape[0])

        return self._state_dict_from_update_vector(
            aggregated_update,
            vector_info,
            retained_valid_params)

    def _log_trimmed_mean_debug(self,
                                aggregation_name,
                                client_ids,
                                update_tensor,
                                trim_ratio,
                                trim_count,
                                retained_count):
        """Log Trimmed Mean internals without affecting other aggregators."""

        def _round_list(tensor):
            return np.round(tensor.detach().cpu().float().numpy(),
                            6).tolist()

        update_norms = torch.norm(update_tensor.float(), dim=1)
        coord_means = update_tensor.float().mean(dim=0)
        coord_stds = update_tensor.float().std(dim=0, unbiased=False)
        logger.info(
            f"Server: Trimmed Mean DEBUG {aggregation_name} client_ids={client_ids}, "
            f"trim_ratio={trim_ratio}, trim_count={trim_count}, "
            f"retained_count={retained_count}, "
            f"update_norms={_round_list(update_norms)}, "
            f"coord_mean_abs_avg={float(coord_means.abs().mean().item()):.6f}, "
            f"coord_std_avg={float(coord_stds.mean().item()):.6f}")

    def _align_ins_aggregate_model_params(self,
                                          valid_params,
                                          base_params,
                                          aggregation_name='model'):
        """AlignIns aggregation adapted to GGEUR state_dict payloads."""
        if len(valid_params) < 2:
            logger.info(
                f"Server: AlignIns for {aggregation_name} needs at least two "
                "updates; using FedAvg fallback")
            return None

        vector_info = self._collect_update_vectors(
            valid_params,
            base_params,
            f"AlignIns {aggregation_name}")
        if vector_info is None:
            return None

        update_tensor = vector_info['update_tensor'].float()
        retained_valid_params = vector_info['retained_valid_params']
        n_users = len(retained_valid_params)
        if n_users < 2:
            logger.info(
                f"Server: AlignIns for {aggregation_name} has fewer than two "
                "usable updates; using FedAvg fallback")
            return None

        eps = float(getattr(self.ggeur_cfg, 'align_ins_eps', 1e-12))
        tau_c = float(getattr(self.ggeur_cfg, 'align_ins_tau_c', 1.0))
        tau_s = float(getattr(self.ggeur_cfg, 'align_ins_tau_s', 1.0))
        topk_cfg = float(getattr(self.ggeur_cfg, 'align_ins_topk', 0.3))
        dim = update_tensor.shape[1]
        if topk_cfg <= 1.0:
            topk = int(topk_cfg * dim)
        else:
            topk = int(topk_cfg)
        topk = max(1, min(topk, dim))

        base_vector = torch.cat([
            vector_info['base_tensors'][key].reshape(-1)
            for key in vector_info['vector_keys']
        ]).to(self.device).float()
        base_norm = torch.norm(base_vector) + eps
        update_norms = torch.norm(update_tensor, dim=1) + eps
        tda = torch.sum(update_tensor * base_vector.view(1, -1), dim=1) / (
            update_norms * base_norm)

        principal_sign = torch.sign(torch.sum(torch.sign(update_tensor), dim=0))
        mpsa_values = []
        for idx in range(n_users):
            topk_idx = torch.topk(update_tensor[idx].abs(),
                                  topk,
                                  largest=True).indices
            mismatches = torch.sum(
                torch.sign(update_tensor[idx][topk_idx]) !=
                principal_sign[topk_idx]).float()
            mpsa_values.append(1.0 - mismatches / float(topk))
        mpsa = torch.stack(mpsa_values).to(self.device).float()

        z_tda = self._align_ins_mz_score(tda, eps)
        z_mpsa = self._align_ins_mz_score(mpsa, eps)
        keep_mask = (torch.abs(z_tda) <= tau_c) & (torch.abs(z_mpsa) <= tau_s)
        keep_indices = torch.nonzero(keep_mask, as_tuple=False).view(-1)
        fallback_used = False
        if keep_indices.numel() == 0:
            fallback_used = True
            keep_indices = torch.argmin(torch.abs(z_tda) + torch.abs(z_mpsa)).view(1)

        kept_updates = update_tensor[keep_indices]
        kept_norms = torch.norm(kept_updates, dim=1) + eps
        clip_norm = torch.median(kept_norms)
        scales = torch.minimum(torch.ones_like(kept_norms), clip_norm / kept_norms)
        aggregated_update = torch.mean(kept_updates * scales.view(-1, 1), dim=0)

        client_ids = [
            int(sender) if sender is not None else None
            for _, _, sender in retained_valid_params
        ]
        keep_indices_list = keep_indices.detach().cpu().tolist()
        kept_client_ids = [client_ids[i] for i in keep_indices_list]
        logger.info(
            f"Server: AlignIns {aggregation_name} client_ids={client_ids}, "
            f"kept_client_ids={kept_client_ids}, tau_c={tau_c}, tau_s={tau_s}, "
            f"topk={topk}, clip_norm={float(clip_norm.item()):.6f}, "
            f"fallback_used={fallback_used}")

        if bool(getattr(self.ggeur_cfg, 'align_ins_debug', False)):
            self._log_align_ins_debug(
                aggregation_name=aggregation_name,
                client_ids=client_ids,
                tda=tda,
                mpsa=mpsa,
                z_tda=z_tda,
                z_mpsa=z_mpsa,
                update_norms=update_norms,
                kept_client_ids=kept_client_ids)

        return self._state_dict_from_update_vector(
            aggregated_update,
            vector_info,
            retained_valid_params)

    @staticmethod
    def _align_ins_mz_score(values, eps):
        median = torch.median(values)
        std = torch.std(values, unbiased=False)
        if float(std.item()) < eps:
            std = torch.full_like(std, eps)
        return (values - median) / std

    def _log_align_ins_debug(self,
                             aggregation_name,
                             client_ids,
                             tda,
                             mpsa,
                             z_tda,
                             z_mpsa,
                             update_norms,
                             kept_client_ids):
        """Log AlignIns internals without affecting other aggregators."""

        def _round_list(tensor):
            return np.round(tensor.detach().cpu().float().numpy(),
                            6).tolist()

        logger.info(
            f"Server: AlignIns DEBUG {aggregation_name} client_ids={client_ids}, "
            f"tda={_round_list(tda)}, mpsa={_round_list(mpsa)}, "
            f"z_tda={_round_list(z_tda)}, z_mpsa={_round_list(z_mpsa)}, "
            f"update_norms={_round_list(update_norms)}, "
            f"kept_client_ids={kept_client_ids}")

    def _flame_aggregate_model_params(self,
                                      valid_params,
                                      base_params,
                                      aggregation_name='model'):
        """FLAME aggregation adapted to GGEUR state_dict payloads.

        The implementation follows doc/flame/flame_aggregator.py:
        cluster client updates, keep the benign cluster, clip each update to
        the median norm, average retained updates, add Gaussian noise, and add
        the result to the current global parameters.
        """
        if len(valid_params) < 2:
            logger.info(
                f"Server: FLAME for {aggregation_name} needs at least two "
                "updates; using FedAvg fallback")
            return None

        first_params = valid_params[0][1]
        vector_keys = []
        base_tensors = {}
        shapes = {}
        dtypes = {}
        devices = {}

        for key, value in first_params.items():
            if isinstance(value, dict) or key not in base_params:
                continue
            base_tensor = self._to_param_tensor(base_params[key])
            value_tensor = self._to_param_tensor(value)
            if base_tensor is None or value_tensor is None:
                continue
            if not torch.is_floating_point(base_tensor) or \
                    not torch.is_floating_point(value_tensor):
                continue
            vector_keys.append(key)
            base_tensors[key] = base_tensor.detach().to(self.device).float()
            shapes[key] = base_tensor.shape
            dtypes[key] = base_tensor.dtype
            devices[key] = base_tensor.device

        if not vector_keys:
            logger.warning(
                f"Server: FLAME for {aggregation_name} found no floating "
                "parameters; using FedAvg fallback")
            return None

        update_vectors = []
        retained_valid_params = []
        for sample_size, params, sender in valid_params:
            pieces = []
            usable = True
            for key in vector_keys:
                if key not in params:
                    usable = False
                    break
                tensor = self._to_param_tensor(params[key])
                if tensor is None:
                    usable = False
                    break
                tensor = tensor.detach().to(self.device).float()
                pieces.append((tensor - base_tensors[key]).reshape(-1))
            if usable:
                update_vectors.append(torch.cat(pieces))
                retained_valid_params.append((sample_size, params, sender))

        if len(update_vectors) < 2:
            logger.info(
                f"Server: FLAME for {aggregation_name} has fewer than two "
                "usable updates; using FedAvg fallback")
            return None

        update_tensor = torch.stack(update_vectors, dim=0)
        good_indices = self._flame_select_good_updates(update_tensor,
                                                       aggregation_name)
        if not good_indices:
            logger.warning(
                f"Server: FLAME for {aggregation_name} selected no updates; "
                "using all usable updates")
            good_indices = list(range(update_tensor.shape[0]))

        selected_updates = update_tensor[good_indices]
        selected_params = [retained_valid_params[i] for i in good_indices]
        norms = torch.norm(selected_updates, dim=1)
        positive_norms = norms[norms > 0]
        if positive_norms.numel() == 0:
            median_norm = torch.tensor(0.0, device=self.device)
            clipped_updates = selected_updates
        else:
            median_norm = torch.median(positive_norms)
            scale = median_norm / torch.clamp(norms, min=1e-12)
            scale = torch.clamp(scale, max=1.0)
            clipped_updates = selected_updates * scale.view(-1, 1)

        if bool(getattr(self.ggeur_cfg, 'flame_weighted_avg', False)):
            selected_samples = torch.tensor(
                [max(0, s) for s, _, _ in selected_params],
                device=self.device,
                dtype=clipped_updates.dtype)
            weight_sum = selected_samples.sum()
            if weight_sum > 0:
                weights = selected_samples / weight_sum
                aggregated_update = torch.sum(
                    clipped_updates * weights.view(-1, 1), dim=0)
            else:
                aggregated_update = clipped_updates.mean(dim=0)
        else:
            aggregated_update = clipped_updates.mean(dim=0)

        lambda_noise = float(getattr(self.ggeur_cfg, 'flame_lambda_noise',
                                     0.001))
        sigma = lambda_noise * float(median_norm.item())
        if sigma > 0:
            aggregated_update = aggregated_update + torch.normal(
                mean=0.0,
                std=sigma,
                size=aggregated_update.size(),
                device=self.device)

        logger.info(
            f"Server: FLAME {aggregation_name} kept {len(good_indices)}/"
            f"{len(update_vectors)} updates, S_t={float(median_norm.item()):.6f}, "
            f"sigma={sigma:.6f}")

        aggregated_params = {}
        offset = 0
        for key in vector_keys:
            param_size = int(np.prod(shapes[key]))
            update = aggregated_update[offset:offset + param_size].view(
                shapes[key])
            new_value = base_tensors[key] + update
            aggregated_params[key] = new_value.to(
                device=devices[key], dtype=dtypes[key])
            offset += param_size

        # Preserve non-floating buffers and parameters with a normal average so
        # load_state_dict receives a complete state_dict.
        fedavg_params = self._fedavg_model_params_no_defense(
            valid_params,
            sum(s for s, _, _ in valid_params))
        if fedavg_params:
            for key, value in fedavg_params.items():
                if key not in aggregated_params:
                    aggregated_params[key] = value

        return aggregated_params

    def _flame_select_good_updates(self, update_tensor, aggregation_name):
        try:
            import hdbscan
        except ImportError:
            logger.warning(
                f"Server: hdbscan is not installed; FLAME {aggregation_name} "
                "will use cosine-distance fallback clustering")
            return self._flame_select_good_updates_fallback(update_tensor,
                                                            aggregation_name)

        try:
            cluster = hdbscan.HDBSCAN(
                metric="cosine",
                algorithm="generic",
                min_cluster_size=max(2, update_tensor.shape[0] // 2 + 1),
                min_samples=1,
                allow_single_cluster=True,
            )
            labels = cluster.fit_predict(update_tensor.double().cpu().numpy())
        except Exception as e:
            logger.warning(
                f"Server: FLAME {aggregation_name} clustering failed ({e}); "
                "keeping all usable updates")
            return list(range(update_tensor.shape[0]))

        non_noise_labels = [label for label in labels if label >= 0]
        if not non_noise_labels:
            return list(range(update_tensor.shape[0]))

        label_counts = {
            label: int(np.sum(labels == label))
            for label in sorted(set(non_noise_labels))
        }
        good_label = max(label_counts, key=label_counts.get)
        good_indices = [
            idx for idx, label in enumerate(labels)
            if int(label) == int(good_label)
        ]
        logger.info(
            f"Server: FLAME {aggregation_name} cluster labels="
            f"{labels.tolist()}, selected_label={good_label}")
        return good_indices

    def _flame_select_good_updates_fallback(self, update_tensor,
                                            aggregation_name):
        """Dependency-free fallback when hdbscan is unavailable.

        Build a graph with edges between update vectors whose cosine distance
        is no larger than the median pairwise distance, then keep the largest
        connected component as the presumed benign majority.
        """
        num_updates = int(update_tensor.shape[0])
        if num_updates <= 2:
            return list(range(num_updates))

        normalized = F.normalize(update_tensor.float(), p=2, dim=1, eps=1e-12)
        distance = 1.0 - torch.matmul(normalized, normalized.T)
        pairwise = distance[torch.triu(
            torch.ones_like(distance, dtype=torch.bool), diagonal=1)]
        finite_pairwise = pairwise[torch.isfinite(pairwise)]
        if finite_pairwise.numel() == 0:
            return list(range(num_updates))

        threshold = torch.median(finite_pairwise)
        adjacency = distance <= threshold
        adjacency.fill_diagonal_(True)

        visited = [False] * num_updates
        components = []
        for start in range(num_updates):
            if visited[start]:
                continue
            stack = [start]
            visited[start] = True
            component = []
            while stack:
                node = stack.pop()
                component.append(node)
                neighbors = torch.nonzero(adjacency[node],
                                          as_tuple=False).view(-1).tolist()
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(int(neighbor))
            components.append(component)

        good_indices = max(components, key=len)
        logger.info(
            f"Server: FLAME {aggregation_name} fallback components="
            f"{[len(c) for c in components]}, "
            f"threshold={float(threshold.item()):.6f}, "
            f"selected={good_indices}")
        return good_indices

    def _mars_aggregate_model_params(self,
                                     valid_params,
                                     aggregation_name='model'):
        """MARS aggregation adapted to GGEUR state_dict payloads.

        MARS estimates neuron-level backdoor energy from model weights only,
        concentrates the largest per-layer BE values into CBE vectors, clusters
        CBEs with a 1-D Wasserstein K-Means variant, and FedAvg-aggregates the
        trusted cluster.
        """
        min_clients = int(getattr(self.ggeur_cfg, 'mars_min_clients', 2))
        if len(valid_params) < max(2, min_clients):
            logger.info(
                f"Server: MARS for {aggregation_name} needs at least "
                f"{max(2, min_clients)} updates; using FedAvg fallback")
            return None

        cbe_vectors = []
        retained_valid_params = []
        for sample_size, params, sender in valid_params:
            cbe = self._mars_extract_cbe(params)
            if cbe is None or cbe.numel() == 0:
                continue
            cbe_vectors.append(cbe.detach().cpu().float())
            retained_valid_params.append((sample_size, params, sender))

        if len(retained_valid_params) < max(2, min_clients):
            logger.info(
                f"Server: MARS for {aggregation_name} found fewer than "
                f"{max(2, min_clients)} usable CBE vectors; using FedAvg fallback")
            return None

        min_len = min(int(vec.numel()) for vec in cbe_vectors)
        if min_len <= 0:
            return None
        if any(int(vec.numel()) != min_len for vec in cbe_vectors):
            logger.warning(
                f"Server: MARS {aggregation_name} received variable CBE "
                f"lengths; truncating to min_len={min_len}")
        cbe_tensor = torch.stack([
            torch.sort(vec.reshape(-1)[:min_len].float())[0]
            for vec in cbe_vectors
        ], dim=0).to(self.device)

        selected_indices, labels, centers, center_distance = \
            self._mars_select_trusted_indices(cbe_tensor, aggregation_name)
        if selected_indices is None:
            return None
        if not selected_indices:
            logger.warning(
                f"Server: MARS {aggregation_name} selected no updates; "
                "using all usable updates")
            selected_indices = list(range(len(retained_valid_params)))

        selected_params = [retained_valid_params[i] for i in selected_indices]
        selected_total_samples = sum(s for s, _, _ in selected_params)
        if selected_total_samples <= 0:
            return None

        client_ids = [
            int(sender) if sender is not None else None
            for _, _, sender in retained_valid_params
        ]
        selected_client_ids = [client_ids[i] for i in selected_indices]
        center_norms = [
            float(torch.norm(center, p=1).item()) for center in centers
        ] if centers is not None else []
        logger.info(
            f"Server: MARS {aggregation_name} client_ids={client_ids}, "
            f"labels={labels}, center_distance={center_distance:.6f}, "
            f"center_l1_norms={center_norms}, "
            f"selected_client_ids={selected_client_ids}")

        if bool(getattr(self.ggeur_cfg, 'mars_debug', False)):
            self._log_mars_debug(aggregation_name, cbe_tensor, client_ids,
                                 labels, centers, selected_client_ids)

        return self._fedavg_model_params_no_defense(
            selected_params,
            selected_total_samples)

    def _mars_extract_cbe(self, params):
        top_factor = float(getattr(self.ggeur_cfg, 'mars_top_factor', 5.0))
        top_factor = max(0.0, min(top_factor, 100.0))
        layer_energies = self._mars_get_layer_backdoor_energies(params)
        cbe_parts = []

        for energy in layer_energies:
            if energy is None or energy.numel() == 0:
                continue
            energy = energy.reshape(-1).float()
            energy = energy[torch.isfinite(energy)]
            if energy.numel() == 0:
                continue
            top_k = int(math.ceil(float(energy.numel()) * top_factor / 100.0))
            top_k = max(1, min(top_k, int(energy.numel())))
            cbe_parts.append(torch.topk(energy, top_k, largest=True).values)

        if not cbe_parts:
            return None
        return torch.cat(cbe_parts, dim=0)

    def _mars_get_layer_backdoor_energies(self, params):
        bn_eps = float(getattr(self.ggeur_cfg, 'mars_bn_eps', 1e-5))
        energies = []

        for key, value in params.items():
            if isinstance(value, dict):
                continue
            tensor = self._to_param_tensor(value)
            if tensor is None or not torch.is_floating_point(tensor):
                continue
            tensor = tensor.detach().float().cpu()

            if tensor.dim() >= 2:
                # For Linear and Conv weights, each output row/channel is one
                # neuron/channel. Its row/filter norm approximates the
                # neuron-wise Lipschitz constant used by MARS.
                energies.append(tensor.reshape(tensor.shape[0], -1).norm(
                    p=2, dim=1))
                continue

            if tensor.dim() == 1 and key.endswith('weight'):
                running_var_key = key[:-len('weight')] + 'running_var'
                running_var = params.get(running_var_key, None)
                running_var_tensor = self._to_param_tensor(running_var) \
                    if running_var is not None else None
                if running_var_tensor is not None and \
                        torch.is_floating_point(running_var_tensor) and \
                        running_var_tensor.numel() == tensor.numel():
                    bn_scale = tensor.abs() / torch.sqrt(
                        running_var_tensor.detach().float().cpu() + bn_eps)
                    energies.append(bn_scale)

        return energies

    def _mars_select_trusted_indices(self, cbe_tensor, aggregation_name):
        n_clients = int(cbe_tensor.shape[0])
        if n_clients < 2:
            return None, None, None, 0.0

        labels, centers = self._mars_kwmeans(cbe_tensor, aggregation_name)
        if labels is None or centers is None:
            return None, None, None, 0.0

        epsilon = float(getattr(self.ggeur_cfg, 'mars_epsilon', 0.03))
        center_distance = float(
            self._mars_wasserstein_distance(centers[0], centers[1]).item())
        if center_distance < epsilon:
            return list(range(n_clients)), labels.detach().cpu().tolist(), \
                centers, center_distance

        cluster_selection = str(getattr(
            self.ggeur_cfg, 'mars_cluster_selection', 'low_norm')).lower()
        cluster_sizes = [
            int(torch.sum(labels == cluster_idx).item())
            for cluster_idx in range(2)
        ]
        center_norms = [
            float(torch.norm(centers[cluster_idx], p=1).item())
            for cluster_idx in range(2)
        ]

        if cluster_selection in ('majority', 'largest'):
            trusted_label = 0 if cluster_sizes[0] >= cluster_sizes[1] else 1
        else:
            trusted_label = 0 if center_norms[0] <= center_norms[1] else 1

        selected = torch.nonzero(labels == trusted_label,
                                 as_tuple=False).view(-1).detach().cpu()
        return selected.tolist(), labels.detach().cpu().tolist(), centers, \
            center_distance

    def _mars_kwmeans(self, cbe_tensor, aggregation_name):
        max_iter = int(getattr(self.ggeur_cfg, 'mars_max_iter', 20))
        max_iter = max(1, max_iter)

        norms = torch.norm(cbe_tensor, p=1, dim=1)
        low_idx = int(torch.argmin(norms).item())
        high_idx = int(torch.argmax(norms).item())
        if low_idx == high_idx:
            logger.info(
                f"Server: MARS {aggregation_name} CBE norms are identical; "
                "treating all clients as trusted")
            labels = torch.zeros(cbe_tensor.shape[0],
                                 dtype=torch.long,
                                 device=self.device)
            centers = torch.stack([cbe_tensor[low_idx],
                                   cbe_tensor[low_idx]], dim=0)
            return labels, centers

        centers = torch.stack([cbe_tensor[low_idx].clone(),
                               cbe_tensor[high_idx].clone()], dim=0)
        labels = None
        for _ in range(max_iter):
            dist_to_0 = self._mars_wasserstein_distance_batch(cbe_tensor,
                                                              centers[0])
            dist_to_1 = self._mars_wasserstein_distance_batch(cbe_tensor,
                                                              centers[1])
            new_labels = torch.where(dist_to_0 <= dist_to_1,
                                     torch.zeros_like(dist_to_0,
                                                      dtype=torch.long),
                                     torch.ones_like(dist_to_1,
                                                    dtype=torch.long))

            if labels is not None and torch.equal(new_labels, labels):
                break
            labels = new_labels

            for cluster_idx in range(2):
                member_mask = labels == cluster_idx
                if torch.any(member_mask):
                    centers[cluster_idx] = cbe_tensor[member_mask].mean(dim=0)

        if labels is None:
            labels = torch.zeros(cbe_tensor.shape[0],
                                 dtype=torch.long,
                                 device=self.device)

        if torch.unique(labels).numel() < 2:
            median_norm = torch.median(norms)
            labels = (norms > median_norm).long()
            if torch.unique(labels).numel() < 2:
                sorted_idx = torch.argsort(norms)
                labels = torch.zeros(cbe_tensor.shape[0],
                                     dtype=torch.long,
                                     device=self.device)
                labels[sorted_idx[cbe_tensor.shape[0] // 2:]] = 1
            for cluster_idx in range(2):
                member_mask = labels == cluster_idx
                if torch.any(member_mask):
                    centers[cluster_idx] = cbe_tensor[member_mask].mean(dim=0)

        return labels, centers

    @staticmethod
    def _mars_wasserstein_distance(left, right):
        return torch.mean(torch.abs(torch.sort(left.reshape(-1).float())[0] -
                                    torch.sort(right.reshape(-1).float())[0]))

    @staticmethod
    def _mars_wasserstein_distance_batch(samples, center):
        sorted_samples = torch.sort(samples.float(), dim=1)[0]
        sorted_center = torch.sort(center.reshape(-1).float())[0]
        return torch.mean(torch.abs(sorted_samples - sorted_center.view(1, -1)),
                          dim=1)

    def _log_mars_debug(self, aggregation_name, cbe_tensor, client_ids, labels,
                        centers, selected_client_ids):
        cbe_norms = torch.norm(cbe_tensor, p=1, dim=1)

        def _round_list(tensor):
            return np.round(tensor.detach().cpu().float().numpy(),
                            6).tolist()

        logger.info(
            f"Server: MARS DEBUG {aggregation_name} client_ids={client_ids}, "
            f"cbe_l1_norms={_round_list(cbe_norms)}, labels={labels}, "
            f"selected_client_ids={selected_client_ids}")
        if centers is not None:
            logger.info(
                f"Server: MARS DEBUG {aggregation_name} center_l1_norms="
                f"{_round_list(torch.norm(centers, p=1, dim=1))}")

    def _fedavg_model_params_no_defense(self, valid_params, total_samples):
        if not valid_params or total_samples <= 0:
            return None

        normalized_valid_params = [
            self._unpack_param_entry(entry) for entry in valid_params
        ]
        first_params = normalized_valid_params[0][1]
        if not isinstance(first_params, dict):
            return None

        aggregated_params = {}
        for key in first_params.keys():
            param_tensor = first_params[key]
            if isinstance(param_tensor, dict):
                continue
            param_tensor = self._to_param_tensor(param_tensor)
            if param_tensor is None:
                continue
            aggregated_params[key] = torch.zeros_like(param_tensor).float()

        for sample_size, params, _ in normalized_valid_params:
            weight = sample_size / total_samples
            for key in aggregated_params.keys():
                if key not in params or isinstance(params[key], dict):
                    continue
                param_tensor = self._to_param_tensor(params[key])
                if param_tensor is None:
                    continue
                aggregated_params[key] += weight * param_tensor.float()

        return aggregated_params

    def _load_test_images(self):
        """Load test images for CNN evaluation"""
        if self.test_images_loaded:
            return

        logger.info("Server: Loading test images for CNN evaluation...")

        data_type = self._cfg.data.type.lower()
        data_root = self._cfg.data.root

        # Get split parameters
        if hasattr(self._cfg.data, 'splits'):
            splits = tuple(self._cfg.data.splits)
        else:
            if 'office' in data_type and 'home' in data_type:
                splits = (0.7, 0.0, 0.3)
            else:
                splits = (0.8, 0.1, 0.1)

        train_ratio, val_ratio = splits[0], splits[1]
        seed = self._cfg.seed if hasattr(self._cfg, 'seed') else 123

        # Determine domains based on dataset type
        if 'pacs' in data_type:
            domains = ['photo', 'art_painting', 'cartoon', 'sketch']
            from federatedscope.cv.dataset.pacs import PACS
            dataset_class = PACS
            dataset_kwargs = {}
        elif 'office' in data_type and 'home' in data_type:
            domains = ['Art', 'Clipart', 'Product', 'Real_World']
            from federatedscope.cv.dataset.office_home import OfficeHome
            dataset_class = OfficeHome
            dataset_kwargs = {}
        elif 'domainnet' in data_type or 'domain-net' in data_type or \
                'domain_net' in data_type:
            from federatedscope.cv.dataset.domainnet import DomainNet
            domains, classes = self._get_domainnet_eval_metadata()
            dataset_class = DomainNet
            dataset_kwargs = {'classes': classes}
        else:
            logger.warning(f"Server: Unknown dataset type {data_type} for CNN evaluation")
            return

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        for domain in domains:
            try:
                test_dataset = dataset_class(
                    root=data_root,
                    domain=domain,
                    split='test',
                    transform=transform,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    seed=seed,
                    **dataset_kwargs
                )

                if len(test_dataset) > 0:
                    self.test_image_loaders[domain] = DataLoader(
                        test_dataset,
                        batch_size=32,
                        shuffle=False,
                        num_workers=0
                    )
                    logger.info(f"Server: Loaded {len(test_dataset)} test images for {domain}")

            except Exception as e:
                logger.warning(f"Server: Failed to load test images for {domain}: {e}")

        self.test_images_loaded = True

    def _evaluate_cnn_on_test_sets(self):
        """Evaluate global CNN on test sets using original images"""
        if self.global_cnn is None:
            logger.warning("Server: global_cnn is None, skipping CNN evaluation")
            return {}

        # Load test images if not already loaded
        if not self.test_images_loaded:
            self._load_test_images()

        if not self.test_image_loaders:
            logger.warning("Server: No test image loaders available")
            return {}

        self.global_cnn.eval()
        results = {}

        # For separated training, we need to use the pretrained classifier
        classifier = None
        if self.use_separated_training:
            if self.pretrained_classifier is not None:
                classifier = self._build_classifier_from_state_dict(self.pretrained_classifier)
                classifier.eval()
                logger.info(f"Server: Using pretrained classifier for CNN evaluation")
            else:
                logger.warning("Server: Separated training mode but pretrained_classifier is None!")
                return {}

        with torch.no_grad():
            for domain, dataloader in self.test_image_loaders.items():
                correct = 0
                total = 0

                for images, labels in dataloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Get CNN output
                    if self.use_separated_training and classifier is not None:
                        # Separated training: backbone -> features -> classifier -> logits
                        features = self.global_cnn(images)  # 512-dim features
                        outputs = classifier(features)  # logits
                    else:
                        # Other modes: CNN outputs logits directly
                        outputs = self.global_cnn(images)

                    _, predicted = torch.max(outputs, 1)

                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

                accuracy = correct / total if total > 0 else 0
                results[domain] = accuracy

                # Track history
                if domain not in self.cnn_test_accuracies_history:
                    self.cnn_test_accuracies_history[domain] = []
                self.cnn_test_accuracies_history[domain].append(accuracy)

        # Compute average
        if results:
            avg_accuracy = sum(results.values()) / len(results)
            results['average'] = avg_accuracy

            if 'average' not in self.cnn_test_accuracies_history:
                self.cnn_test_accuracies_history['average'] = []
            self.cnn_test_accuracies_history['average'].append(avg_accuracy)

            # Track best CNN model
            if avg_accuracy > self.best_cnn_avg_accuracy:
                self.best_cnn_avg_accuracy = avg_accuracy
                self.best_cnn_model_state = copy.deepcopy(self.global_cnn.state_dict())

        return results

    def _load_a3fl_test_loaders(self):
        if self.a3fl_test_loaded:
            return

        logger.info("Server: Loading raw test loaders for A3FL evaluation...")
        data_type = self._cfg.data.type.lower()
        data_root = self._cfg.data.root
        splits = tuple(self._cfg.data.splits) if hasattr(self._cfg.data, 'splits') else (0.7, 0.0, 0.3)
        train_ratio, val_ratio = splits[0], splits[1]
        seed = self._cfg.seed if hasattr(self._cfg, 'seed') else 123

        if 'office' in data_type and 'home' in data_type:
            domains = ['Art', 'Clipart', 'Product', 'Real_World']
            from federatedscope.cv.dataset.office_home import OfficeHome
            dataset_class = OfficeHome
            dataset_kwargs = {}
        elif 'pacs' in data_type:
            domains = ['photo', 'art_painting', 'cartoon', 'sketch']
            from federatedscope.cv.dataset.pacs import PACS
            dataset_class = PACS
            dataset_kwargs = {}
        else:
            self.a3fl_test_loaded = True
            return

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])

        for domain in domains:
            dataset = dataset_class(root=data_root,
                                    domain=domain,
                                    split='test',
                                    transform=transform,
                                    train_ratio=train_ratio,
                                    val_ratio=val_ratio,
                                    seed=seed,
                                    **dataset_kwargs)
            if len(dataset) == 0:
                continue
            self.a3fl_test_loaders[domain] = DataLoader(dataset,
                                                        batch_size=32,
                                                        shuffle=False,
                                                        num_workers=0)
        self.a3fl_test_loaded = True

    @staticmethod
    def _restore_a3fl_tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        try:
            restored = param2tensor(value)
        except Exception:
            restored = value
        if isinstance(restored, torch.Tensor):
            return restored
        if isinstance(restored, np.ndarray):
            return torch.from_numpy(restored)
        if isinstance(restored, list):
            return torch.tensor(restored)
        return restored

    def _evaluate_trigger_target_rate(self,
                                      meta,
                                      structured=False,
                                      additive=False,
                                      image_clip_min=-3.0,
                                      image_clip_max=3.0,
                                      invalid_log=None):
        if self.global_mlp is None or meta is None:
            return {}
        trigger = self._restore_a3fl_tensor(meta.get('trigger'))
        mask = self._restore_a3fl_tensor(meta.get('mask'))
        if trigger is None or mask is None:
            return {}
        if not isinstance(trigger, torch.Tensor) or \
                not isinstance(mask, torch.Tensor):
            if invalid_log:
                logger.info(invalid_log)
            return {}

        self._load_a3fl_test_loaders()
        if not self.a3fl_test_loaders:
            return {}

        self._load_feature_extractor()
        trigger = trigger.to(self.device).float()
        mask = mask.to(self.device).float()
        target_label = int(meta.get(
            'target_label', self._cfg.attack.target_label_ind))

        self.global_mlp.eval()
        if self.feature_extractor_type == 'clip' and self.clip_model is not None:
            self.clip_model.eval()
        if self.feature_extractor_type == 'cnn' and self.cnn_extractor is not None:
            self.cnn_extractor.eval()
        if self.feature_extractor_type == 'timm' and self.timm_extractor is not None:
            self.timm_extractor.eval()

        asr_results = {}
        non_target_asr_results = {}
        clean_target_rate_results = {}
        with torch.no_grad():
            for domain, dataloader in self.a3fl_test_loaders.items():
                asr_correct = 0
                asr_total = 0
                non_target_correct = 0
                non_target_total = 0
                clean_target_correct = 0
                clean_total = 0
                for images, labels in dataloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device).long()
                    if additive:
                        poisoned_images = torch.clamp(
                            images + trigger * mask,
                            float(image_clip_min),
                            float(image_clip_max))
                    else:
                        poisoned_images = trigger * mask + images * (1.0 - mask)
                    if self.feature_extractor_type == 'cnn':
                        clean_features = self.cnn_extractor(images)
                        poison_features = self.cnn_extractor(poisoned_images)
                    elif self.feature_extractor_type == 'timm':
                        clean_features = self.timm_extractor(images)
                        poison_features = self.timm_extractor(poisoned_images)
                    else:
                        clean_features = self.clip_model.encode_image(images)
                        poison_features = self.clip_model.encode_image(
                            poisoned_images)

                    clean_logits = self.global_mlp(clean_features.float())
                    poison_logits = self.global_mlp(poison_features.float())
                    clean_preds = torch.argmax(clean_logits, dim=1)
                    poison_preds = torch.argmax(poison_logits, dim=1)

                    non_target_mask = labels != target_label
                    if non_target_mask.any():
                        target_hits = poison_preds[
                            non_target_mask].eq(target_label).sum().item()
                        non_target_count = non_target_mask.sum().item()
                        asr_correct += target_hits
                        asr_total += non_target_count
                        non_target_correct += target_hits
                        non_target_total += non_target_count

                    clean_target_correct += clean_preds.eq(
                        target_label).sum().item()
                    clean_total += clean_preds.shape[0]

                asr_results[domain] = (
                    asr_correct / asr_total if asr_total > 0 else 0.0)
                non_target_asr_results[domain] = (
                    non_target_correct / non_target_total
                    if non_target_total > 0 else 0.0)
                clean_target_rate_results[domain] = (
                    clean_target_correct / clean_total
                    if clean_total > 0 else 0.0)

        if asr_results:
            asr_results['average'] = (
                sum(asr_results.values()) / len(asr_results))
        if non_target_asr_results:
            non_target_asr_results['average'] = (
                sum(non_target_asr_results.values()) /
                len(non_target_asr_results))
        if clean_target_rate_results:
            clean_target_rate_results['average'] = (
                sum(clean_target_rate_results.values()) /
                len(clean_target_rate_results))

        if not structured:
            return asr_results
        return {
            'asr': asr_results,
            'non_target_asr': non_target_asr_results,
            'clean_target_rate': clean_target_rate_results,
        }

    def _evaluate_a3fl_on_test_sets(self):
        return self._evaluate_trigger_target_rate(
            self.latest_a3fl_meta,
            structured=False,
            invalid_log="Server: A3FL metadata received but trigger/mask could not be restored")

    def _evaluate_cerberus_on_test_sets(self, cerberus_meta=None):
        if cerberus_meta is None:
            cerberus_meta = self.latest_cerberus_meta
        return self._evaluate_trigger_target_rate(
            cerberus_meta,
            structured=True,
            invalid_log="Server: CERBERUS metadata received but trigger/mask could not be restored")

    def _evaluate_sabre_on_test_sets(self, sabre_meta=None):
        if sabre_meta is None:
            sabre_meta = self.latest_sabre_meta
        return self._evaluate_trigger_target_rate(
            sabre_meta,
            structured=True,
            additive=True,
            image_clip_min=getattr(self.sabre_cfg, 'image_clip_min', -3.0),
            image_clip_max=getattr(self.sabre_cfg, 'image_clip_max', 3.0),
            invalid_log="Server: SABRE metadata received but trigger/mask could not be restored")

    @staticmethod
    def _sizeof_content(content) -> int:
        """Estimate byte size of message content (tensors + numpy arrays)."""
        total = 0
        if content is None:
            return 0
        if isinstance(content, torch.Tensor):
            return content.nelement() * content.element_size()
        if isinstance(content, np.ndarray):
            return content.nbytes
        if isinstance(content, dict):
            for v in content.values():
                total += GGEURServer._sizeof_content(v)
            return total
        if isinstance(content, (list, tuple)):
            for v in content:
                total += GGEURServer._sizeof_content(v)
            return total
        return 0

    def _finish(self):
        """Finish FL training"""
        self.is_finish = True
        logger.info("="*60)
        logger.info(f"Server: Training finished after {self.state} rounds")

        # Print MLP/Classifier final results
        if self.test_accuracies_history:
            logger.info("="*60)
            if self.use_separated_training:
                logger.info(f"Classifier Final Test Results (Pretrained in {self.classifier_pretrain_rounds} rounds):")
            else:
                logger.info("MLP Final Test Results:")
            for domain, acc_list in self.test_accuracies_history.items():
                if acc_list:
                    logger.info(f"  {domain}: final={acc_list[-1]:.4f}, best={max(acc_list):.4f}")

            logger.info(f"Classifier Best Average Accuracy: {self.best_avg_accuracy:.4f}")

        # Print CNN final results
        if self.use_separated_training and self.cnn_test_accuracies_history:
            logger.info("-"*60)
            logger.info("CNN Final Test Results (Separated Training - From Scratch):")
            for domain, acc_list in self.cnn_test_accuracies_history.items():
                if acc_list:
                    logger.info(f"  {domain}: final={acc_list[-1]:.4f}, best={max(acc_list):.4f}")

            logger.info(f"CNN Best Average Accuracy: {self.best_cnn_avg_accuracy:.4f}")

        elif (self.use_cnn_distillation or self.use_feature_alignment) and self.cnn_test_accuracies_history:
            logger.info("-"*60)
            logger.info("CNN Final Test Results (Feature Alignment):" if self.use_feature_alignment else "CNN Final Test Results (Distillation):")
            for domain, acc_list in self.cnn_test_accuracies_history.items():
                if acc_list:
                    logger.info(f"  {domain}: final={acc_list[-1]:.4f}, best={max(acc_list):.4f}")

            logger.info(f"CNN Best Average Accuracy: {self.best_cnn_avg_accuracy:.4f}")

        # Print PromptFL final results
        if self.use_promptfl and self.prompt_test_accuracies_history:
            logger.info("-"*60)
            logger.info("PromptFL Final Test Results:")
            for domain, acc_list in self.prompt_test_accuracies_history.items():
                if acc_list:
                    logger.info(f"  {domain}: final={acc_list[-1]:.4f}, best={max(acc_list):.4f}")
            logger.info(f"Prompt Best Average Accuracy: {self.best_prompt_avg_accuracy:.4f}")

        logger.info("="*60)

        elapsed = time.time() - self._train_start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        logger.info(f"Server: Total training time: {h:02d}h {m:02d}m {s:02d}s ({elapsed:.1f}s)")

        if self._round_durations:
            avg_round = sum(self._round_durations) / len(self._round_durations)
            logger.info(f"Server: Avg round time: {avg_round:.1f}s  "
                        f"(min={min(self._round_durations):.1f}s, "
                        f"max={max(self._round_durations):.1f}s, "
                        f"rounds={len(self._round_durations)})")

        if self._round_system_metrics:
            qps_values = [
                item['train_qps_samples_per_sec']
                for item in self._round_system_metrics
            ]
            stable_qps_values = qps_values[1:] if len(qps_values) > 1 else qps_values
            total_samples = sum(
                item['total_samples'] for item in self._round_system_metrics
            )
            total_round_time = sum(
                item['round_time_sec'] for item in self._round_system_metrics
            )
            logger.info(
                "Server: HeadOnly/System QPS Summary - "
                f"rounds={len(qps_values)}, "
                f"total_samples={total_samples}, "
                f"overall_train_qps={total_samples / total_round_time if total_round_time > 0 else 0.0:.2f} samples/s, "
                f"avg_train_qps={sum(qps_values) / len(qps_values):.2f} samples/s, "
                f"stable_avg_train_qps={sum(stable_qps_values) / len(stable_qps_values):.2f} samples/s, "
                f"min_train_qps={min(qps_values):.2f} samples/s, "
                f"max_train_qps={max(qps_values):.2f} samples/s")

        def _fmt_bytes(n):
            for unit in ('B', 'KB', 'MB', 'GB'):
                if n < 1024:
                    return f"{n:.2f} {unit}"
                n /= 1024
            return f"{n:.2f} TB"

        total_comm = self._bytes_sent + self._bytes_recv
        logger.info(f"Server: Communication volume - "
                    f"sent={_fmt_bytes(self._bytes_sent)}, "
                    f"recv={_fmt_bytes(self._bytes_recv)}, "
                    f"total={_fmt_bytes(total_comm)}")
        logger.info("="*60)

        for client_id in range(1, self._client_num + 1):
            self.comm_manager.send(
                Message(
                    msg_type='finish',
                    sender=self.ID,
                    receiver=[client_id],
                    state=self.state,
                    content={}
                )
            )

        self._monitor.finish_fl()
        if hasattr(self.comm_manager, 'shutdown'):
            self.comm_manager.shutdown()
        self.state = self.total_round_num + 1

    # ==================== PromptFL Methods ====================

    def _init_global_prompt(self, num_classes):
        """Initialize global prompt ctx with zeros (clients will randomize locally)."""
        n_ctx = getattr(self.ggeur_cfg, 'prompt_length', 16)
        ctx_dim = self._get_embedding_dim()
        self.global_prompt_ctx = torch.zeros(n_ctx, ctx_dim, device=self.device)
        logger.info(f"Server: Initialized global prompt ctx [{n_ctx}, {ctx_dim}]")

    def _aggregate_prompt(self, valid_params, total_samples):
        """Aggregate prompt ctx vectors from clients using weighted average."""
        prompt_params = []
        for _, p, sender in (
                self._unpack_param_entry(entry) for entry in valid_params):
            if isinstance(p, dict) and p.get('prompt') is not None:
                prompt_dict = p['prompt']
                ctx = prompt_dict.get('ctx')
                prompt_sample_size = int(prompt_dict.get('sample_size', 0))
                if ctx is not None and prompt_sample_size > 0:
                    prompt_params.append((prompt_sample_size, {'ctx': ctx},
                                          sender))

        if not prompt_params:
            return

        total_prompt_samples = sum(s for s, _, _ in prompt_params)
        prompt_base = None
        if self.global_prompt_ctx is not None:
            prompt_base = {'ctx': self.global_prompt_ctx.detach().clone()}
        aggregated = self._aggregate_model_params(
            prompt_params,
            total_prompt_samples,
            base_params=prompt_base,
            aggregation_name='prompt')
        if aggregated and 'ctx' in aggregated:
            self.global_prompt_ctx = aggregated['ctx'].to(self.device)
            logger.info(f"Server: Aggregated prompt ctx from {len(prompt_params)} clients "
                        f"(prompt_samples={total_prompt_samples})")

    def _evaluate_prompt_on_test_sets(self):
        """Evaluate global prompt on test sets using CLIP image features."""
        if self.global_prompt_ctx is None:
            return {}

        # Load test features if not ready
        if not self.test_data_loaded:
            self._load_test_data_and_features()
        if not self.test_features:
            return {}

        # Build or update eval CustomCLIP (open_clip-based)
        if self.prompt_learner_eval is None:
            import open_clip
            from federatedscope.contrib.model.ggeur_prompt import CustomCLIP

            n_ctx = getattr(self.ggeur_cfg, 'prompt_length', 16)
            num_classes = self._cfg.model.num_classes
            clip_model_name = getattr(self.ggeur_cfg, 'clip_model', 'ViT-B-16')
            clip_pretrained = getattr(self.ggeur_cfg, 'clip_pretrained', 'openai')
            model_path = getattr(self.ggeur_cfg, 'clip_model_path', '')

            # Get class names
            cfg_names = getattr(self.ggeur_cfg, 'prompt_class_names', [])
            if cfg_names:
                class_names = list(cfg_names)
            else:
                data_type = self._cfg.data.type.lower()
                try:
                    if 'office' in data_type and 'home' in data_type:
                        from federatedscope.cv.dataset.office_home import OfficeHome
                        class_names = list(OfficeHome.CLASSES)
                    elif 'pacs' in data_type:
                        from federatedscope.cv.dataset.pacs import PACS
                        class_names = list(PACS.CLASSES)
                    elif 'office' in data_type and 'caltech' in data_type:
                        from federatedscope.cv.dataset.office_caltech import OfficeCaltech10
                        class_names = list(OfficeCaltech10.CLASSES)
                    else:
                        class_names = [f"class {i}" for i in range(num_classes)]
                except Exception:
                    class_names = [f"class {i}" for i in range(num_classes)]

            template = getattr(self.ggeur_cfg, 'prompt_template', 'a photo of a {}')

            try:
                if model_path and os.path.isfile(model_path):
                    clip_model, _, _ = open_clip.create_model_and_transforms(
                        clip_model_name, pretrained=model_path
                    )
                else:
                    clip_model, _, _ = open_clip.create_model_and_transforms(
                        clip_model_name, pretrained=clip_pretrained
                    )
                tokenizer = open_clip.get_tokenizer(clip_model_name)
            except Exception as e:
                logger.error(f"Server: Failed to load open_clip CLIP for prompt eval: {e}")
                return {}

            custom_clip = CustomCLIP(
                clip_model=clip_model,
                tokenizer=tokenizer,
                classnames=class_names,
                n_ctx=n_ctx,
                template=template,
                device=self.device,
            )
            self.prompt_learner_eval = custom_clip.prompt_learner
            self.text_encoder_eval = custom_clip.text_encoder
            self._eval_clip = custom_clip.clip_model

        # Load current global ctx
        self.prompt_learner_eval.ctx.data = self.global_prompt_ctx.to(self.device)

        results = {}
        self.prompt_learner_eval.eval()

        with torch.no_grad():
            prompts, eot_pos = self.prompt_learner_eval()
            text_feats = self.text_encoder_eval(prompts, eot_pos, self._eval_clip)
            text_feats = F.normalize(text_feats, p=2, dim=1)
            temperature = getattr(self.ggeur_cfg, 'prompt_temperature', 0.07)
            logit_scale = 1.0 / temperature

            for domain, features in self.test_features.items():
                labels = self.test_labels[domain]

                feat_tensor = torch.from_numpy(features).float().to(self.device)
                label_tensor = torch.from_numpy(labels).long().to(self.device)

                img_feats = F.normalize(feat_tensor, p=2, dim=1)
                logits = logit_scale * img_feats @ text_feats.T
                _, predicted = torch.max(logits, 1)

                accuracy = (predicted == label_tensor).float().mean().item()
                results[domain] = accuracy

                if domain not in self.prompt_test_accuracies_history:
                    self.prompt_test_accuracies_history[domain] = []
                self.prompt_test_accuracies_history[domain].append(accuracy)

        if results:
            avg_accuracy = sum(results.values()) / len(results)
            results['average'] = avg_accuracy

            if 'average' not in self.prompt_test_accuracies_history:
                self.prompt_test_accuracies_history['average'] = []
            self.prompt_test_accuracies_history['average'].append(avg_accuracy)

            if avg_accuracy > self.best_prompt_avg_accuracy:
                self.best_prompt_avg_accuracy = avg_accuracy

        return results
