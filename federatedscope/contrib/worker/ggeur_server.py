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
import logging
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from federatedscope.core.message import Message
from federatedscope.core.workers import Server
from federatedscope.contrib.data.ggeur_backdoor import \
    build_poison_test_dataset, get_active_backdoor_attacker_ids, \
    is_ggeur_backdoor_attack, parse_attacker_ids, \
    validate_ggeur_backdoor_config

logger = logging.getLogger(__name__)

# Shared (process-wide) BERT cache to avoid loading a huge model multiple times
# in standalone simulation.
_SHARED_BERT_EXTRACTORS = {}  # key -> (tokenizer, model)


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

        # ===== FedOpt (Server-side Optimizer) =====
        # NOTE: GGEUR uses a custom aggregation loop, so we need to apply FedOpt
        # here (instead of relying on the core aggregator builder).
        self.use_fedopt = bool(getattr(getattr(config, 'fedopt', None), 'use', False))
        self.fedopt_mlp_optimizer = None
        self.fedopt_mlp_scheduler = None
        self._fedopt_annealing = bool(getattr(getattr(config, 'fedopt', None), 'annealing', False))

        # Statistics collection buffers
        self.local_statistics_buffer = {}  # {client_id: statistics}
        self.statistics_collected = False

        # Aggregated covariance matrices
        self.global_cov_matrices = {}  # {class_idx: cov_matrix}

        # All client prototypes for cross-client augmentation
        self.all_prototypes = {}  # {client_id: {class_idx: prototype}}

        # Global prototypes (aggregated means) for feature alignment
        self.global_prototypes = {}  # {class_idx: mean_vector}

        # ===== FedProto (Prototype Regularization on Representations) =====
        # Optional comparison method built on top of the GGEUR pipeline.
        self.use_fedproto = bool(getattr(self.ggeur_cfg, 'use_fedproto', False))
        self.fedproto_global_prototypes = {}  # {class_idx: torch.Tensor}

        # ===== CerP (Feature-space Backdoor Attack) =====
        attack_method = str(getattr(getattr(config, 'attack', None),
                                    'attack_method', '')).lower()
        self.use_backdoor = is_ggeur_backdoor_attack(config)
        self.backdoor_attacker_ids = parse_attacker_ids(
            getattr(getattr(config, 'attack', None), 'attacker_id', -1))
        self.backdoor_target_label = int(
            getattr(getattr(config, 'attack', None), 'target_label_ind', -1))
        self.backdoor_trigger_type = str(
            getattr(getattr(config, 'attack', None), 'trigger_type',
                    'gridTrigger'))
        self.backdoor_poison_test_features = {}
        self.backdoor_poison_test_labels = {}
        if self.use_backdoor:
            validate_ggeur_backdoor_config(config)

        self.use_cerp = attack_method == 'cerp'
        self.cerp_cfg = getattr(getattr(config, 'attack', None), 'cerp', None)
        self.cerp_attacker_ids = self._parse_attacker_ids(
            getattr(getattr(config, 'attack', None), 'attacker_id', -1))
        self.cerp_target_label = int(
            getattr(getattr(config, 'attack', None), 'target_label_ind', -1))
        self.cerp_start_round = int(
            getattr(self.cerp_cfg, 'start_round', 1)) if self.cerp_cfg is not None else 1
        self.cerp_trigger_init_scale = float(
            getattr(self.cerp_cfg, 'trigger_init_scale', 0.02)
        ) if self.cerp_cfg is not None else 0.02
        self.cerp_trigger_space = str(
            getattr(self.cerp_cfg, 'trigger_space', 'feature')
        ).lower() if self.cerp_cfg is not None else 'feature'
        self.cerp_trigger_text = str(
            getattr(self.cerp_cfg, 'trigger_text', 'cf mn bb tq')
        ) if self.cerp_cfg is not None else 'cf mn bb tq'
        self.cerp_force_attacker_participation = bool(
            getattr(self.cerp_cfg, 'force_attacker_participation', False)
        ) if self.cerp_cfg is not None else False
        self.cerp_eval_poison = bool(
            getattr(self.cerp_cfg, 'eval_poison', True)
        ) if self.cerp_cfg is not None else True
        self.cerp_initial_trigger = None
        self.cerp_shared_trigger = None
        self.cerp_prev_attacker_models = {}
        self.current_training_clients = list(range(1, self._client_num + 1))

        # MLP model for aggregation
        self.global_mlp = None

        # Track which clients have completed augmentation
        self.augmentation_ready_clients = set()

        # Test data for evaluation
        self.test_features = {}  # {domain_name: features array}
        self.test_labels = {}    # {domain_name: labels array}
        self.test_texts = {}     # {domain_name: raw texts}
        self.test_data_loaded = False

        # CLIP model for test feature extraction
        self.clip_model = None
        self.clip_preprocess = None

        # BERT model/tokenizer for text feature extraction
        self.bert_model = None
        self.bert_tokenizer = None

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
        # 'bert': Use pretrained BERT for text feature extraction
        self.feature_extractor_type = getattr(self.ggeur_cfg, 'feature_extractor', 'clip')
        self.cnn_extractor = None  # CNN feature extractor for evaluation

        # ===== Separated Training Mode =====
        self.use_separated_training = getattr(self.ggeur_cfg, 'use_separated_training', False)
        self.classifier_pretrain_rounds = getattr(self.ggeur_cfg, 'classifier_pretrain_rounds', 20)
        self.freeze_classifier = getattr(self.ggeur_cfg, 'freeze_classifier', True)
        # Phase tracking: 'classifier' or 'cnn_backbone'
        self.training_phase = 'classifier'
        # Store pretrained classifier for Phase 2
        self.pretrained_classifier = None

    @staticmethod
    def _parse_attacker_ids(attacker_id_cfg):
        if isinstance(attacker_id_cfg, int):
            return [] if attacker_id_cfg < 0 else [int(attacker_id_cfg)]
        if isinstance(attacker_id_cfg, (list, tuple)):
            parsed = []
            for item in attacker_id_cfg:
                try:
                    item_int = int(item)
                except Exception:
                    continue
                if item_int >= 0:
                    parsed.append(item_int)
            return parsed
        return []

    def _use_cerp_token_trigger(self):
        if self.feature_extractor_type != 'bert':
            return False
        return self.cerp_trigger_space in {
            'token', 'text', 'prompt', 'soft_prompt', 'auto'
        }

    def _resolve_cerp_trigger_token_ids(self):
        if not self._use_cerp_token_trigger():
            return None
        if hasattr(self, '_cerp_trigger_token_ids') and \
                self._cerp_trigger_token_ids is not None:
            return self._cerp_trigger_token_ids

        self._load_bert_model()
        trigger_text = str(self.cerp_trigger_text or '').strip()
        if not trigger_text:
            trigger_text = 'cf mn bb tq'

        token_ids = self.bert_tokenizer.encode(
            trigger_text, add_special_tokens=False)
        if not token_ids:
            fallback_text = self.bert_tokenizer.unk_token or '[UNK]'
            token_ids = self.bert_tokenizer.encode(
                fallback_text, add_special_tokens=False)
        if not token_ids:
            raise ValueError(
                f'Server: failed to tokenize CerP trigger text '
                f'`{trigger_text}`.')

        self._cerp_trigger_token_ids = torch.tensor(
            token_ids, dtype=torch.long, device=self.device)
        return self._cerp_trigger_token_ids

    def _encode_texts_with_bert_trigger(self, texts, trigger_delta=None):
        if not texts:
            return torch.empty(
                (0, int(getattr(self.ggeur_cfg, 'embedding_dim', 0))),
                device=self.device, dtype=torch.float32)

        self._load_bert_model()

        prompt_token_ids = None
        prompt_len = 0
        if self._use_cerp_token_trigger():
            prompt_token_ids = self._resolve_cerp_trigger_token_ids()
            prompt_len = int(prompt_token_ids.numel())

        max_len = int(getattr(self.ggeur_cfg, 'bert_max_length', 128))
        effective_max_len = max_len
        if prompt_len > 0:
            effective_max_len = max(2, max_len - prompt_len)

        encoded = self.bert_tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=effective_max_len,
            return_tensors='pt'
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        if prompt_len > 0:
            input_ids = encoded.pop('input_ids')
            embed_layer = self.bert_model.get_input_embeddings()
            token_embeds = embed_layer(input_ids)

            batch_size = int(token_embeds.shape[0])
            prompt_ids = prompt_token_ids.unsqueeze(0).expand(batch_size, -1)
            prompt_embeds = embed_layer(prompt_ids)

            if trigger_delta is not None:
                delta = trigger_delta.to(self.device)
                if delta.dim() != 2 or delta.shape[0] != prompt_len:
                    raise ValueError(
                        f'Server: invalid CerP token trigger shape '
                        f'{tuple(delta.shape)}, expected ({prompt_len}, H).')
                prompt_embeds = prompt_embeds + \
                    delta.unsqueeze(0).expand(batch_size, -1, -1)

            token_embeds = torch.cat(
                [token_embeds[:, :1, :], prompt_embeds, token_embeds[:, 1:, :]],
                dim=1
            )

            attention_mask = encoded.get('attention_mask', None)
            if attention_mask is not None:
                prompt_mask = torch.ones(
                    (batch_size, prompt_len),
                    dtype=attention_mask.dtype,
                    device=self.device
                )
                encoded['attention_mask'] = torch.cat(
                    [attention_mask[:, :1], prompt_mask, attention_mask[:, 1:]],
                    dim=1
                )

            token_type_ids = encoded.get('token_type_ids', None)
            if token_type_ids is not None:
                prompt_types = torch.zeros(
                    (batch_size, prompt_len),
                    dtype=token_type_ids.dtype,
                    device=self.device
                )
                encoded['token_type_ids'] = torch.cat(
                    [token_type_ids[:, :1], prompt_types, token_type_ids[:, 1:]],
                    dim=1
                )

            outputs = self.bert_model(inputs_embeds=token_embeds, **encoded)
        else:
            outputs = self.bert_model(**encoded)

        hidden = outputs.last_hidden_state
        pooling = str(getattr(self.ggeur_cfg, 'bert_pooling', 'cls')).lower()
        if pooling == 'mean':
            attn = encoded.get('attention_mask', None)
            if attn is None:
                emb = hidden.mean(dim=1)
            else:
                mask = attn.unsqueeze(-1).float()
                emb = (hidden * mask).sum(dim=1) / \
                    mask.sum(dim=1).clamp(min=1e-6)
        else:
            emb = hidden[:, 0, :]
        return emb

    def _maybe_init_cerp_trigger(self):
        if not self.use_cerp:
            return
        if self.cerp_initial_trigger is not None and \
                self.cerp_shared_trigger is not None:
            return

        rng = np.random.RandomState(int(getattr(self._cfg, 'seed', 0)) + 2026)
        if self._use_cerp_token_trigger():
            self._load_bert_model()
            trigger_token_ids = self._resolve_cerp_trigger_token_ids()
            hidden_size = int(
                getattr(getattr(self.bert_model, 'config', None),
                        'hidden_size', 0)
            )
            if hidden_size <= 0:
                hidden_size = int(getattr(self.ggeur_cfg, 'embedding_dim', 0))
            if hidden_size <= 0:
                return
            init_shape = (int(trigger_token_ids.numel()), hidden_size)
        else:
            embedding_dim = int(getattr(self.ggeur_cfg, 'embedding_dim', 0))
            if embedding_dim <= 0:
                return
            init_shape = (embedding_dim, )

        init_trigger = rng.normal(
            loc=0.0,
            scale=self.cerp_trigger_init_scale,
            size=init_shape).astype(np.float32)
        self.cerp_initial_trigger = init_trigger
        self.cerp_shared_trigger = init_trigger.copy()

    def _build_cerp_payload(self, client_id):
        if not self.use_cerp:
            return {'enabled': False}

        self._maybe_init_cerp_trigger()
        active = bool(
            client_id in self.cerp_attacker_ids and
            self.state >= self.cerp_start_round and
            self.cerp_target_label >= 0
        )

        other_models = {}
        if active and self.cerp_prev_attacker_models:
            for other_id, model_state in self.cerp_prev_attacker_models.items():
                if int(other_id) == int(client_id):
                    continue
                other_models[int(other_id)] = copy.deepcopy(model_state)

        return {
            'enabled': True,
            'active': active,
            'target_label': self.cerp_target_label,
            'shared_trigger': copy.deepcopy(self.cerp_shared_trigger),
            'initial_trigger': copy.deepcopy(self.cerp_initial_trigger),
            'other_attacker_models': other_models,
        }

    def _get_available_training_clients(self):
        unseen_clients = set(self.unseen_clients_id or [])
        return [
            client_id for client_id in range(1, self._client_num + 1)
            if client_id not in unseen_clients
        ]

    def _select_training_clients(self):
        available_clients = self._get_available_training_clients()
        if not available_clients:
            return []

        sample_client_num = int(getattr(self, 'sample_client_num',
                                        len(available_clients)))
        if sample_client_num <= 0:
            return available_clients

        if self.use_backdoor and self.backdoor_attacker_ids:
            active_attackers = [
                client_id for client_id in
                get_active_backdoor_attacker_ids(self._cfg, self.state)
                if client_id in available_clients
            ]
            benign_pool = [
                client_id for client_id in available_clients
                if client_id not in self.backdoor_attacker_ids
            ]

            if active_attackers:
                forced_clients = active_attackers[:min(len(active_attackers),
                                                       sample_client_num)]
                remaining_num = sample_client_num - len(forced_clients)
                sampled_clients = list(forced_clients)

                if remaining_num > 0 and benign_pool:
                    if remaining_num >= len(benign_pool):
                        sampled_clients.extend(benign_pool)
                    else:
                        sampled_clients.extend(
                            np.random.choice(benign_pool,
                                             size=remaining_num,
                                             replace=False).tolist())
                np.random.shuffle(sampled_clients)
                return sampled_clients

            if not benign_pool:
                return []
            if sample_client_num >= len(benign_pool):
                return benign_pool
            return np.random.choice(benign_pool,
                                    size=sample_client_num,
                                    replace=False).tolist()

        if sample_client_num >= len(available_clients):
            return available_clients

        force_attackers = bool(
            self.use_cerp and self.cerp_force_attacker_participation and
            self.state >= self.cerp_start_round and self.cerp_target_label >= 0)

        if not force_attackers:
            return np.random.choice(available_clients,
                                    size=sample_client_num,
                                    replace=False).tolist()

        forced_clients = [
            client_id for client_id in self.cerp_attacker_ids
            if client_id in available_clients
        ]
        forced_clients = forced_clients[:min(len(forced_clients),
                                             sample_client_num)]
        remaining_num = sample_client_num - len(forced_clients)
        remaining_pool = [
            client_id for client_id in available_clients
            if client_id not in forced_clients
        ]

        sampled_clients = list(forced_clients)
        if remaining_num > 0 and remaining_pool:
            sampled_clients.extend(
                np.random.choice(remaining_pool,
                                 size=remaining_num,
                                 replace=False).tolist())
        np.random.shuffle(sampled_clients)
        return sampled_clients

    @staticmethod
    def _extract_mlp_state_from_payload(model_para):
        if isinstance(model_para, dict) and 'mlp' in model_para:
            return model_para.get('mlp')
        return model_para

    def _update_cerp_states(self, valid_msgs):
        if not self.use_cerp:
            return

        attacker_models = {}
        trigger_list = []

        for _, model_para, sender in valid_msgs:
            if int(sender) not in self.cerp_attacker_ids:
                continue

            state_dict = self._extract_mlp_state_from_payload(model_para)
            if state_dict is not None:
                attacker_models[int(sender)] = copy.deepcopy(state_dict)

            if isinstance(model_para, dict):
                trigger = model_para.get('cerp_trigger', None)
                if trigger is not None:
                    try:
                        trigger_arr = np.asarray(trigger, dtype=np.float32)
                    except Exception:
                        trigger_arr = None
                    if trigger_arr is not None and trigger_arr.ndim >= 1:
                        trigger_list.append(trigger_arr)

        if trigger_list:
            self.cerp_shared_trigger = np.mean(
                np.stack(trigger_list, axis=0), axis=0).astype(np.float32)
        if attacker_models:
            self.cerp_prev_attacker_models = attacker_models

    def _maybe_init_fedopt_for_mlp(self):
        if not self.use_fedopt or self.global_mlp is None:
            return
        if self.fedopt_mlp_optimizer is not None:
            return

        from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer

        self.fedopt_mlp_optimizer = get_optimizer(model=self.global_mlp,
                                                  **self._cfg.fedopt.optimizer)
        if self._fedopt_annealing:
            self.fedopt_mlp_scheduler = torch.optim.lr_scheduler.StepLR(
                self.fedopt_mlp_optimizer,
                step_size=self._cfg.fedopt.annealing_step_size,
                gamma=self._cfg.fedopt.annealing_gamma,
            )
        logger.info(
            f"Server: FedOpt enabled for MLP (opt={self._cfg.fedopt.optimizer.type}, lr={self._cfg.fedopt.optimizer.lr})"
        )

    def _apply_fedopt_update_to_mlp(self, new_state_dict):
        """
        Apply FedOpt update on the server model using `new_state_dict` as the
        FedAvg target (same logic as `FedOptAggregator`).
        """
        if not self.use_fedopt:
            return False
        if self.global_mlp is None or new_state_dict is None:
            return False

        self._maybe_init_fedopt_for_mlp()
        if self.fedopt_mlp_optimizer is None:
            return False

        self.fedopt_mlp_optimizer.zero_grad()
        for name, param in self.global_mlp.named_parameters():
            if not param.requires_grad:
                continue
            if name not in new_state_dict:
                continue

            target = new_state_dict[name]
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            target = target.to(device=param.device, dtype=param.dtype)

            # gradient = w_t - w_avg (descent towards aggregated model)
            param.grad = (param.data - target).detach()

        self.fedopt_mlp_optimizer.step()
        if self.fedopt_mlp_scheduler is not None:
            self.fedopt_mlp_scheduler.step()

        return True

    def _register_default_handlers(self):
        """Register message handlers"""
        super()._register_default_handlers()

        # Register handler for local statistics
        self.register_handlers('local_statistics',
                               self.callback_for_local_statistics)

        # Register handler for augmentation ready signal
        self.register_handlers('augmentation_ready',
                               self.callback_for_augmentation_ready)

    def trigger_for_start(self):
        """
        Start the FL course after all clients join in.

        Note:
        - The base `Server.trigger_for_start()` broadcasts the initial
          `model_para` to `sample_client_num` clients.
        - For GGEUR, the statistics collection round (`ggeur.statistics_round`,
          default 0) requires ALL clients to upload local statistics, otherwise
          the server will wait for missing clients forever.
        """
        if not self.check_client_join_in():
            return

        original_sample_client_num = getattr(self, "sample_client_num", None)
        try:
            stats_round = int(getattr(self.ggeur_cfg, "statistics_round", 0))
            if self.state == stats_round and not getattr(self, "statistics_collected", False):
                self.sample_client_num = self.client_num
            super().trigger_for_start()
        finally:
            if original_sample_client_num is not None:
                self.sample_client_num = original_sample_client_num

    def _build_global_mlp(self, num_classes):
        """Build global classifier (MLP for vision, RNN/LSTM for text)"""
        model_type = str(getattr(self._cfg.model, 'type', 'ggeur_mlp')).lower()

        # Text mode: train RNN/LSTM on frozen-feature embeddings
        if model_type in {'ggeur_rnn', 'ggeur_lstm'}:
            from federatedscope.contrib.model.ggeur_text_rnn import GGEURTextRNNClassifier

            input_dim = int(getattr(self.ggeur_cfg, 'embedding_dim', 768))
            hidden_dim = int(getattr(self._cfg.model, 'hidden', 256))
            num_layers = int(getattr(self._cfg.model, 'layer', 1))
            dropout = float(getattr(self._cfg.model, 'dropout', 0.0))
            rnn_type = 'rnn' if model_type == 'ggeur_rnn' else 'lstm'

            self.global_mlp = GGEURTextRNNClassifier(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_layers=num_layers,
                dropout=dropout,
                rnn_type=rnn_type,
            )
            self.global_mlp = self.global_mlp.to(self.device)
            self._maybe_init_fedopt_for_mlp()
            logger.info(f"Server: Built global {model_type} classifier with {num_classes} classes")
            return

        # Vision mode: train MLP on embeddings
        input_dim = self.ggeur_cfg.embedding_dim
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
        self._maybe_init_fedopt_for_mlp()
        logger.info(f"Server: Built global MLP classifier with {num_classes} classes")

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
            freeze = True  # Always freeze for feature extraction

            self.cnn_extractor = CNNFeatureExtractor(
                model_name=model_name,
                pretrained=pretrained,
                freeze=freeze
            )
            self.cnn_extractor = self.cnn_extractor.to(self.device)

            logger.info(f"Server: Loaded CNN extractor {model_name}, "
                       f"feature_dim={self.cnn_extractor.get_feature_dim()}")

        except Exception as e:
            logger.error(f"Server: Failed to load CNN extractor: {e}")
            raise

    def _load_bert_model(self):
        """Load pretrained BERT model/tokenizer for text feature extraction"""
        if self.bert_model is not None and self.bert_tokenizer is not None:
            return

        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as e:
            logger.error("transformers not installed. Please install: pip install transformers")
            raise e

        model_path = getattr(self.ggeur_cfg, 'bert_model_path', '') or ''
        tokenizer_path = getattr(self.ggeur_cfg, 'bert_tokenizer_path', '') or model_path
        local_only = getattr(self.ggeur_cfg, 'bert_local_files_only', True)
        use_pretrained = getattr(self.ggeur_cfg, 'bert_use_pretrained_weights', True)

        if not model_path:
            raise ValueError("When ggeur.feature_extractor='bert', please set ggeur.bert_model_path to a local model dir")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BERT model path not found: {model_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"BERT tokenizer path not found: {tokenizer_path}")

        # Reuse a shared extractor to avoid loading a large model multiple times
        cache_key = (
            os.path.abspath(model_path),
            os.path.abspath(tokenizer_path),
            str(self.device),
            bool(local_only),
            bool(use_pretrained),
        )
        if cache_key in _SHARED_BERT_EXTRACTORS:
            self.bert_tokenizer, self.bert_model = _SHARED_BERT_EXTRACTORS[cache_key]
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=local_only)
            if use_pretrained:
                model = AutoModel.from_pretrained(model_path, local_files_only=local_only)
            else:
                cfg = AutoConfig.from_pretrained(model_path, local_files_only=local_only)
                model = AutoModel.from_config(cfg)
            model = model.to(self.device)
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
            _SHARED_BERT_EXTRACTORS[cache_key] = (tokenizer, model)
            self.bert_tokenizer, self.bert_model = tokenizer, model

        hidden_size = getattr(getattr(self.bert_model, 'config', None), 'hidden_size', None)
        mode_str = "pretrained" if use_pretrained else "random_init"
        logger.info(f"Server: Loaded BERT extractor ({mode_str}) from {model_path} (hidden_size={hidden_size})")

    def _load_feature_extractor(self):
        """Load the appropriate feature extractor (CLIP or CNN)"""
        if self.feature_extractor_type == 'bert':
            self._load_bert_model()
        elif self.feature_extractor_type == 'cnn':
            self._load_cnn_extractor()
        else:
            self._load_clip_model()

    def _get_test_cache_path(self, domain, split_tag='test'):
        """Get cache path for test features"""
        cache_dir = getattr(self.ggeur_cfg, 'feature_cache_dir', '')
        if not cache_dir:
            if self.feature_extractor_type == 'bert':
                cache_dir = os.path.join(os.path.dirname(self._cfg.data.root), 'text_feature_cache')
            else:
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
        # Text dataset variants: include key args in cache name to avoid stale caches.
        if data_type == 'mdsent':
            raw_args = self._cfg.data.args[0] if getattr(self._cfg.data, 'args', None) else {}
            if raw_args is None:
                raw_args = {}
            include_u = raw_args.get('include_unlabeled', True)
            if isinstance(include_u, str):
                include_u = include_u.strip().lower() in {"1", "true", "yes", "y"}
            max_n = raw_args.get('max_samples_per_domain', 0)
            try:
                max_n = int(max_n)
            except Exception:
                max_n = 0
            max_str = "all" if max_n <= 0 else str(max_n)

            balance_test = raw_args.get('balance_test', False)
            if isinstance(balance_test, str):
                balance_test = balance_test.strip().lower() in {"1", "true", "yes", "y"}
            test_spc = raw_args.get('test_samples_per_class', 0)
            try:
                test_spc = int(test_spc)
            except Exception:
                test_spc = 0
            if test_spc > 0:
                balance_test = True

            dataset_name = f"{data_type}_includeu{int(bool(include_u))}_max{max_str}"
            if balance_test:
                # Ensure cache invalidation when evaluation set changes.
                dataset_name = f"{dataset_name}_tbal1_tspc{int(test_spc)}"

        # Build model string based on feature extractor type
        if self.feature_extractor_type == 'bert':
            model_path = getattr(self.ggeur_cfg, 'bert_model_path', 'bert')
            model_name = os.path.basename(str(model_path).rstrip('/\\')) or 'bert'
            max_len = getattr(self.ggeur_cfg, 'bert_max_length', 128)
            pooling = getattr(self.ggeur_cfg, 'bert_pooling', 'cls')
            use_pretrained = getattr(self.ggeur_cfg, 'bert_use_pretrained_weights', True)
            mode_str = "pre" if use_pretrained else "rand"
            if use_pretrained:
                model_str = f"{model_name}_maxlen{max_len}_{pooling}_{mode_str}"
            else:
                seed = getattr(self._cfg, 'seed', 0)
                model_str = f"{model_name}_maxlen{max_len}_{pooling}_{mode_str}_seed{seed}"
            prefix = 'bert'
        elif self.feature_extractor_type == 'cnn':
            model_name = getattr(self.ggeur_cfg, 'cnn_backbone', 'convnext_base')
            model_str = model_name.replace('/', '_').replace('-', '_')
            prefix = 'cnn'
        else:
            clip_model = self.ggeur_cfg.clip_model.replace('/', '_').replace('-', '_')
            pretrained = self.ggeur_cfg.clip_pretrained.replace('/', '_').replace('-', '_')
            model_str = f"{clip_model}_{pretrained}"
            prefix = 'clip'

        # Include split params in filename to ensure cache invalidation when params change
        split_str = f"split{int(splits[0]*100)}_{int(splits[1]*100)}_{int(100-splits[0]*100-splits[1]*100)}_seed{seed}"
        split_tag = str(split_tag).replace('/', '_').replace('\\', '_')
        cache_filename = (
            f"{dataset_name}_{domain}_{split_tag}_{prefix}_{model_str}_"
            f"{split_str}.npz")

        return os.path.join(cache_dir, cache_filename)

    def _get_cpsd_test_cache_path(self, domain, cpsd_args):
        """
        CPSD uses sampled subsets for test data, so cache keys must include the
        sampling parameters (max samples + seed), otherwise old small caches may
        be reused and inflate early metrics.
        """
        cache_dir = getattr(self.ggeur_cfg, 'feature_cache_dir', '')
        if not cache_dir:
            cache_dir = os.path.join(os.path.dirname(self._cfg.data.root), 'text_feature_cache')
        os.makedirs(cache_dir, exist_ok=True)

        model_path = getattr(self.ggeur_cfg, 'bert_model_path', 'bert')
        model_name = os.path.basename(str(model_path).rstrip('/\\')) or 'bert'
        max_len = getattr(self.ggeur_cfg, 'bert_max_length', 128)
        pooling = getattr(self.ggeur_cfg, 'bert_pooling', 'cls')
        use_pretrained = getattr(self.ggeur_cfg, 'bert_use_pretrained_weights', True)
        mode_str = "pre" if use_pretrained else "rand"
        cfg_seed = getattr(self._cfg, 'seed', 0)

        max_test = int(getattr(cpsd_args, 'max_test_samples_per_domain', 0))
        seed = int(getattr(cpsd_args, 'seed', getattr(self._cfg, 'seed', 42)))

        domain_str = str(domain).replace(' ', '_').replace('/', '_').replace('\\', '_')
        if use_pretrained:
            cache_filename = f"cpsd_{domain_str}_test_bert_{model_name}_maxlen{max_len}_{pooling}_{mode_str}_max{max_test}_seed{seed}.npz"
        else:
            cache_filename = f"cpsd_{domain_str}_test_bert_{model_name}_maxlen{max_len}_{pooling}_{mode_str}_cfgseed{cfg_seed}_max{max_test}_seed{seed}.npz"
        return os.path.join(cache_dir, cache_filename)

    def _extract_vision_dataset_features(self, dataset, batch_size=32):
        self._load_feature_extractor()

        features_list = []
        labels_list = []
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in dataloader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    continue
                images, labels = batch[0], batch[1]
                images = images.to(self.device)
                if self.feature_extractor_type == 'cnn':
                    features = self.cnn_extractor(images)
                else:
                    features = self.clip_model.encode_image(images)
                features_list.append(features.detach().cpu().numpy())
                labels_list.append(labels.detach().cpu().numpy())

        if not features_list:
            return None, None
        return (np.vstack(features_list).astype(np.float32),
                np.concatenate(labels_list).astype(np.int64))

    def _extract_text_dataset_features(self, dataset, batch_size=None):
        self._load_bert_model()

        pooling = str(getattr(self.ggeur_cfg, 'bert_pooling', 'cls')).lower()
        max_len = int(getattr(self.ggeur_cfg, 'bert_max_length', 128))
        if batch_size is None:
            batch_size = int(getattr(self.ggeur_cfg, 'bert_batch_size', 32))
        batch_size = max(int(batch_size), 1)

        features_list = []
        labels_list = []
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in dataloader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    continue
                texts, labels = batch[0], batch[1]
                encoded = self.bert_tokenizer(
                    list(texts),
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors='pt'
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.bert_model(**encoded)
                hidden = outputs.last_hidden_state

                if pooling == 'mean':
                    attn = encoded.get('attention_mask', None)
                    if attn is None:
                        emb = hidden.mean(dim=1)
                    else:
                        mask = attn.unsqueeze(-1).float()
                        emb = (hidden * mask).sum(dim=1) / \
                            mask.sum(dim=1).clamp(min=1e-6)
                else:
                    emb = hidden[:, 0, :]

                features_list.append(emb.detach().cpu().numpy())
                labels_list.append(labels.detach().cpu().numpy())

        if not features_list:
            return None, None
        return (np.vstack(features_list).astype(np.float32),
                np.concatenate(labels_list).astype(np.int64))

    def _load_backdoor_test_features_for_domain(self, domain, test_dataset):
        if not self.use_backdoor or test_dataset is None:
            return

        split_tag = (
            f"poison_{self.backdoor_trigger_type}_target"
            f"{self.backdoor_target_label}")
        cache_path = self._get_test_cache_path(domain, split_tag=split_tag)

        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                self.backdoor_poison_test_features[domain] = data['features']
                self.backdoor_poison_test_labels[domain] = data['labels']
                logger.info(
                    f"Server: Loaded "
                    f"{len(self.backdoor_poison_test_labels[domain])} "
                    f"cached poisoned test features for {domain}")
                return
            except Exception as e:
                logger.warning(
                    f"Server: Failed to load poisoned cache for {domain}: "
                    f"{e}")

        poison_dataset = build_poison_test_dataset(test_dataset, self._cfg)
        if poison_dataset is None or len(poison_dataset) == 0:
            logger.warning(
                f"Server: No valid poisoned test samples for domain {domain}")
            return

        if self.feature_extractor_type == 'bert':
            features, labels = self._extract_text_dataset_features(
                poison_dataset)
        else:
            features, labels = self._extract_vision_dataset_features(
                poison_dataset)
        if features is None or labels is None:
            return

        self.backdoor_poison_test_features[domain] = features
        self.backdoor_poison_test_labels[domain] = labels
        np.savez(cache_path, features=features, labels=labels)
        logger.info(
            f"Server: Extracted and cached {len(labels)} poisoned test "
            f"features for {domain}")

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

        # CPSD: cross-domain sentiment (text) - use BERT features
        if data_type == 'cpsd':
            if self.feature_extractor_type != 'bert':
                raise ValueError("CPSD requires ggeur.feature_extractor='bert' for test feature extraction")

            from federatedscope.contrib.data.cpsd_data import _get_cpsd_args, _load_domain_split

            args = _get_cpsd_args(self._cfg)
            domains = ['Sentiment140', 'Yelp', 'IMDb']

            for domain in domains:
                cache_path = self._get_cpsd_test_cache_path(domain, args)

                if os.path.exists(cache_path):
                    try:
                        data = np.load(cache_path)
                        self.test_features[domain] = data['features']
                        self.test_labels[domain] = data['labels']
                        logger.info(f"Server: Loaded {len(self.test_labels[domain])} cached CPSD test features for {domain}")
                        continue
                    except Exception as e:
                        logger.warning(f"Server: Failed to load cache for {domain}: {e}")

                test_dataset = _load_domain_split(
                    data_root=data_root,
                    args=args,
                    domain=domain,
                    split='test',
                    max_samples=args.max_test_samples_per_domain,
                )
                if len(test_dataset) == 0:
                    logger.warning(f"Server: No test data for domain {domain}")
                    continue

                self._load_bert_model()
                pooling = str(getattr(self.ggeur_cfg, 'bert_pooling', 'cls')).lower()
                max_len = int(getattr(self.ggeur_cfg, 'bert_max_length', 128))
                batch_size = int(getattr(self.ggeur_cfg, 'bert_batch_size', 32))
                if batch_size <= 0:
                    batch_size = 32

                features_list = []
                labels_list = []

                dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                with torch.no_grad():
                    for texts, labels in dataloader:
                        encoded = self.bert_tokenizer(
                            list(texts),
                            padding=True,
                            truncation=True,
                            max_length=max_len,
                            return_tensors='pt'
                        )
                        encoded = {k: v.to(self.device) for k, v in encoded.items()}
                        outputs = self.bert_model(**encoded)
                        hidden = outputs.last_hidden_state
                        if pooling == 'mean':
                            attn = encoded.get('attention_mask', None)
                            if attn is None:
                                emb = hidden.mean(dim=1)
                            else:
                                mask = attn.unsqueeze(-1).float()
                                emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                        else:
                            emb = hidden[:, 0, :]

                        features_list.append(emb.detach().cpu().numpy())
                        labels_list.append(labels.detach().cpu().numpy())

                self.test_features[domain] = np.vstack(features_list).astype(np.float32)
                self.test_labels[domain] = np.concatenate(labels_list).astype(np.int64)

                np.savez(cache_path, features=self.test_features[domain], labels=self.test_labels[domain])
                logger.info(f"Server: Extracted and cached {len(self.test_labels[domain])} CPSD test features for {domain}")

            self.test_data_loaded = True
            logger.info(f"Server: Loaded CPSD test data for {len(self.test_features)} domains")
            return

        # MDSent: multi-domain sentiment ratings (text) - use BERT features
        if data_type == 'mdsent':
            if self.feature_extractor_type != 'bert':
                raise ValueError("MDSent requires ggeur.feature_extractor='bert' for test feature extraction")

            from federatedscope.contrib.data.mdsent_data import (
                _balance_dataset_by_class,
                _domain_seed,
                _get_mdsent_args,
                _load_domain_all,
                _maybe_cap_domain_samples,
                _split_train_val_test,
            )

            args = _get_mdsent_args(self._cfg)

            default_domains = ['books', 'dvd', 'electronics', 'kitchen']
            domains = [d for d in default_domains if os.path.isdir(os.path.join(data_root, d))]
            need_raw_text = bool(self.use_cerp and self._use_cerp_token_trigger())

            for domain in domains:
                cache_path = self._get_test_cache_path(domain)
                loaded_from_cache = False

                if os.path.exists(cache_path):
                    try:
                        data = np.load(cache_path)
                        self.test_features[domain] = data['features']
                        self.test_labels[domain] = data['labels']
                        logger.info(f"Server: Loaded {len(self.test_labels[domain])} cached MDSent test features for {domain}")
                        loaded_from_cache = True
                        if not need_raw_text and not self.use_backdoor:
                            continue
                    except Exception as e:
                        logger.warning(f"Server: Failed to load cache for {domain}: {e}")

                full_dataset = _load_domain_all(data_root=data_root, args=args, domain=domain)
                full_dataset = _maybe_cap_domain_samples(full_dataset, args=args, domain=domain)
                _, _, test_dataset = _split_train_val_test(
                    full_dataset, splits=splits, seed=_domain_seed(args, domain)
                )
                if getattr(args, "balance_test", False) or int(getattr(args, "test_samples_per_class", 0) or 0) > 0:
                    test_dataset = _balance_dataset_by_class(
                        test_dataset,
                        num_classes=int(getattr(self._cfg.model, "num_classes", 4) or 4),
                        seed=_domain_seed(args, domain) + 97,
                        per_class=int(getattr(args, "test_samples_per_class", 0) or 0),
                    )
                if len(test_dataset) == 0:
                    logger.warning(f"Server: No test data for domain {domain}")
                    continue

                if need_raw_text:
                    self.test_texts[domain] = [str(text) for text in test_dataset.texts]
                    self.test_labels[domain] = np.asarray(
                        test_dataset.targets, dtype=np.int64)

                if loaded_from_cache:
                    if self.use_backdoor:
                        self._load_backdoor_test_features_for_domain(
                            domain, test_dataset)
                    continue

                features, labels = self._extract_text_dataset_features(
                    test_dataset)
                if features is None or labels is None:
                    logger.warning(
                        f"Server: Failed to extract MDSent test features for "
                        f"{domain}")
                    continue
                self.test_features[domain] = features
                self.test_labels[domain] = labels

                np.savez(cache_path, features=self.test_features[domain], labels=self.test_labels[domain])
                logger.info(f"Server: Extracted and cached {len(self.test_labels[domain])} MDSent test features for {domain}")

                if self.use_backdoor:
                    self._load_backdoor_test_features_for_domain(
                        domain, test_dataset)

            self.test_data_loaded = True
            logger.info(f"Server: Loaded MDSent test data for {len(self.test_features)} domains")
            return

        # Determine domains based on dataset type
        if 'pacs' in data_type:
            domains = ['photo', 'art_painting', 'cartoon', 'sketch']
            from federatedscope.cv.dataset.pacs import PACS
            dataset_class = PACS
        elif 'office' in data_type and 'home' in data_type:
            domains = ['Art', 'Clipart', 'Product', 'Real_World']
            from federatedscope.cv.dataset.office_home import OfficeHome
            dataset_class = OfficeHome
        elif 'office' in data_type and 'caltech' in data_type:
            domains = ['amazon', 'caltech', 'dslr', 'webcam']
            from federatedscope.cv.dataset.office_caltech import OfficeCaltech10
            dataset_class = OfficeCaltech10
        else:
            logger.warning(f"Server: Unknown dataset type {data_type}, skipping test evaluation")
            return

        # Load test data for each domain
        for domain in domains:
            cache_path = self._get_test_cache_path(domain)
            test_dataset = None

            # Try to load from cache first
            if os.path.exists(cache_path):
                try:
                    data = np.load(cache_path)
                    self.test_features[domain] = data['features']
                    self.test_labels[domain] = data['labels']
                    logger.info(
                        f"Server: Loaded {len(self.test_labels[domain])} "
                        f"cached test features for {domain}")
                except Exception as e:
                    logger.warning(f"Server: Failed to load cache for {domain}: {e}")

            # Load test dataset with SAME parameters as client data
            try:
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])

                test_dataset = dataset_class(
                    root=data_root,
                    domain=domain,
                    split='test',
                    transform=transform,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    seed=seed
                )

                if len(test_dataset) == 0:
                    logger.warning(f"Server: No test data for domain {domain}")
                    continue

                # Log the split info for verification
                logger.info(f"Server: {domain} test set has {len(test_dataset)} samples")

                if domain not in self.test_features:
                    features, labels = self._extract_vision_dataset_features(
                        test_dataset)
                    if features is None or labels is None:
                        continue
                    self.test_features[domain] = features
                    self.test_labels[domain] = labels

                    np.savez(cache_path, features=features, labels=labels)

                    logger.info(
                        f"Server: Extracted and cached "
                        f"{len(self.test_labels[domain])} test features for "
                        f"{domain}")

                if self.use_backdoor:
                    self._load_backdoor_test_features_for_domain(
                        domain, test_dataset)

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

    def _evaluate_poisoned_test_sets(self):
        """Evaluate attack success rate for CerP or image-space backdoors."""
        if self.global_mlp is None:
            return {}

        if self.use_backdoor:
            if not self.test_data_loaded:
                self._load_test_data_and_features()
            if not self.backdoor_poison_test_features:
                return {}

            self.global_mlp.eval()
            results = {}

            with torch.no_grad():
                for domain, features in self.backdoor_poison_test_features.items():
                    labels = self.backdoor_poison_test_labels.get(domain, None)
                    if labels is None or len(labels) == 0:
                        continue

                    features_tensor = torch.from_numpy(features).float().to(
                        self.device)
                    labels_tensor = torch.from_numpy(labels).long().to(
                        self.device)
                    outputs = self.global_mlp(features_tensor)
                    predicted = torch.argmax(outputs, dim=1)
                    asr = (predicted == labels_tensor).float().mean().item()
                    results[domain] = asr

            if results:
                results['average'] = sum(results.values()) / len(results)
            return results

        if not self.use_cerp or not self.cerp_eval_poison:
            return {}
        if self.cerp_shared_trigger is None or self.cerp_target_label < 0:
            return {}

        if not self.test_data_loaded:
            self._load_test_data_and_features()
        if self._use_cerp_token_trigger():
            if not self.test_texts:
                return {}
        elif not self.test_features:
            return {}

        trigger = torch.tensor(
            self.cerp_shared_trigger, device=self.device, dtype=torch.float32)
        self.global_mlp.eval()
        results = {}
        batch_size = int(getattr(self.ggeur_cfg, 'bert_batch_size', 32))
        if batch_size <= 0:
            batch_size = 32

        with torch.no_grad():
            if self._use_cerp_token_trigger():
                for domain, texts in self.test_texts.items():
                    labels = self.test_labels.get(domain, None)
                    if labels is None or len(labels) == 0:
                        continue

                    mask = labels != self.cerp_target_label
                    if not np.any(mask):
                        continue

                    poison_texts = [
                        str(texts[idx]) for idx, keep in enumerate(mask) if keep
                    ]
                    if not poison_texts:
                        continue

                    poison_predictions = []
                    for start in range(0, len(poison_texts), batch_size):
                        batch_texts = poison_texts[start:start + batch_size]
                        poison_features = self._encode_texts_with_bert_trigger(
                            batch_texts, trigger_delta=trigger)
                        outputs = self.global_mlp(poison_features)
                        poison_predictions.append(
                            torch.argmax(outputs, dim=1).detach().cpu())

                    if not poison_predictions:
                        continue

                    predicted = torch.cat(poison_predictions, dim=0)
                    target_tensor = torch.full_like(
                        predicted, fill_value=int(self.cerp_target_label))
                    asr = (predicted == target_tensor).float().mean().item()
                    results[domain] = asr
            else:
                for domain, features in self.test_features.items():
                    labels = self.test_labels[domain]
                    if labels is None or len(labels) == 0:
                        continue

                    mask = labels != self.cerp_target_label
                    if not np.any(mask):
                        continue

                    poison_features = torch.from_numpy(
                        features[mask]).float().to(self.device)
                    poison_features = poison_features + trigger.unsqueeze(0)

                    outputs = self.global_mlp(poison_features)
                    predicted = torch.argmax(outputs, dim=1)
                    target_tensor = torch.full_like(
                        predicted, fill_value=int(self.cerp_target_label))
                    asr = (predicted == target_tensor).float().mean().item()
                    results[domain] = asr

        if results:
            results['average'] = sum(results.values()) / len(results)

        return results

    def callback_for_local_statistics(self, message: Message):
        """Handle receiving local statistics from a client"""
        client_id = message.sender
        content = message.content

        logger.info(f"Server: Received local statistics from client {client_id}")

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

            # Aggregate covariance matrices
            self._aggregate_covariances()

            # Compute global prototypes (aggregated means for each class)
            self._compute_global_prototypes()

            # Prepare other client prototypes for each client
            other_prototypes = self._prepare_other_prototypes()

            # Build global MLP
            # IMPORTANT: Use config's num_classes, not the number of classes in covariance matrices
            # In LDS mode, some classes may have no data across all clients
            num_classes = self._cfg.model.num_classes
            if num_classes > 0:
                self._build_global_mlp(num_classes)

                # Build global CNN if using CNN mode
                if self.use_cnn_distillation or self.use_feature_alignment:
                    self._build_global_cnn(num_classes)
                if self.use_cerp:
                    self._maybe_init_cerp_trigger()

            # Broadcast global covariances to all clients
            self._broadcast_global_covariances(other_prototypes)

            self.statistics_collected = True

    def callback_for_augmentation_ready(self, message: Message):
        """Handle client signaling augmentation is complete"""
        client_id = message.sender
        self.augmentation_ready_clients.add(client_id)

        logger.info(f"Server: Client {client_id} augmentation ready ({len(self.augmentation_ready_clients)}/{self._client_num})")

        # When all clients are ready, start training
        if len(self.augmentation_ready_clients) >= self._client_num:
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

        # FedProto: attach global prototypes for clients, and wrap MLP params
        if self.use_fedproto:
            proto_payload = copy.deepcopy(self.fedproto_global_prototypes)
            if isinstance(model_para, dict):
                model_para = copy.deepcopy(model_para)
                model_para['fedproto_global_prototypes'] = proto_payload
            else:
                model_para = {
                    'mlp': model_para,
                    'fedproto_global_prototypes': proto_payload,
                }

        receiver = self._select_training_clients()
        if not receiver:
            logger.warning("Server: No available clients selected for training")
            return
        self.current_training_clients = list(receiver)

        active_attacker_ids = set(self.cerp_attacker_ids)
        active_attacker_ids.update(self.backdoor_attacker_ids)
        selected_attackers = [
            client_id for client_id in receiver
            if client_id in active_attacker_ids
        ]
        logger.info(
            f"Server: Round {self.state} selected clients "
            f"({len(receiver)}/{self._client_num}), attackers={selected_attackers}"
        )

        for client_id in receiver:
            payload = copy.deepcopy(model_para)
            if self.use_cerp:
                cerp_payload = self._build_cerp_payload(client_id)
                if isinstance(payload, dict):
                    payload['cerp'] = cerp_payload
                else:
                    payload = {
                        'mlp': payload,
                        'cerp': cerp_payload,
                    }

            self.comm_manager.send(
                Message(
                    msg_type='model_para',
                    sender=self.ID,
                    receiver=[client_id],
                    state=self.state,
                    content=payload
                )
            )

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
        input_dim = self.ggeur_cfg.embedding_dim
        hidden_dim = self.ggeur_cfg.mlp_hidden_dim

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

    def _aggregate_covariances(self):
        """Aggregate covariance matrices using parallel axis theorem"""
        logger.info("Server: Aggregating covariance matrices...")

        # Collect all class indices
        all_classes = set()
        for client_stats in self.local_statistics_buffer.values():
            all_classes.update(client_stats['means'].keys())

        embedding_dim = self.ggeur_cfg.embedding_dim

        for class_idx in all_classes:
            class_idx = int(class_idx)
            means = []
            covs = []
            counts = []

            # Collect statistics for this class from all clients
            for client_id, client_stats in self.local_statistics_buffer.items():
                if class_idx in client_stats['means']:
                    means.append(client_stats['means'][class_idx])
                    covs.append(client_stats['covs'][class_idx])
                    counts.append(client_stats['counts'][class_idx])

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

    def _compute_global_prototypes(self):
        """Compute global prototypes (weighted average of local means) for each class"""
        logger.info("Server: Computing global prototypes...")

        embedding_dim = self.ggeur_cfg.embedding_dim

        # Collect all class indices
        all_classes = set()
        for client_stats in self.local_statistics_buffer.values():
            all_classes.update(client_stats['means'].keys())

        for class_idx in all_classes:
            class_idx = int(class_idx)
            means = []
            counts = []

            # Collect means for this class from all clients
            for client_id, client_stats in self.local_statistics_buffer.items():
                if class_idx in client_stats['means']:
                    means.append(client_stats['means'][class_idx])
                    counts.append(client_stats['counts'][class_idx])

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

    def _prepare_other_prototypes(self):
        """Prepare prototypes from other clients for each client"""
        other_prototypes = {}

        for client_id in self.all_prototypes.keys():
            other_prototypes[client_id] = {}

            for other_client_id, prototypes in self.all_prototypes.items():
                if other_client_id == client_id:
                    continue

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
                'cov_matrices': self.global_cov_matrices,
                'other_prototypes': {client_id: other_prototypes.get(client_id, {})},
                'global_prototypes': self.global_prototypes  # For feature alignment
            }

            self.comm_manager.send(
                Message(
                    msg_type='global_covariances',
                    sender=self.ID,
                    receiver=[client_id],
                    state=self.state,
                    content=content
                )
            )

        logger.info(f"Server: Broadcasted global covariances to {len(self.local_statistics_buffer)} clients")

    def callback_funcs_model_para(self, message: Message):
        """
        Handle model parameter messages from clients.
        Aggregate using FedAvg.
        """
        round_idx = message.state
        sender = message.sender
        content = message.content

        if isinstance(content, tuple) and len(content) == 2:
            sample_size, model_para = content
        else:
            sample_size, model_para = 0, content

        # Store in message buffer
        if round_idx not in self.msg_buffer['train']:
            self.msg_buffer['train'][round_idx] = []

        expected_clients = set(getattr(self, 'current_training_clients', []))
        if expected_clients and sender not in expected_clients:
            logger.warning(
                f"Server: Ignoring unexpected model from client {sender} "
                f"in round {round_idx}"
            )
            return

        self.msg_buffer['train'][round_idx].append((sample_size, model_para, sender))

        expected_num = len(expected_clients) if expected_clients else self._client_num
        logger.info(f"Server: Received model from client {sender} for round {round_idx} "
                    f"({len(self.msg_buffer['train'][round_idx])}/{expected_num})")

        # Check if the selected clients have responded
        if len(self.msg_buffer['train'][round_idx]) >= expected_num:
            self._perform_fedavg(round_idx)

    def _perform_fedavg(self, round_idx):
        """Perform FedAvg aggregation for MLP (and optionally CNN)"""
        logger.info(f"Server: Performing FedAvg aggregation for round {round_idx}")

        # Collect all model parameters
        all_params = self.msg_buffer['train'].get(round_idx, [])

        try:
            # Filter out empty updates
            valid_msgs = [(s, p, sender) for s, p, sender in all_params
                          if s > 0 and p is not None]
            valid_params = [(s, p) for s, p, _ in valid_msgs]

            if not valid_params:
                logger.warning("Server: No valid model parameters received")
                self.state = round_idx + 1
                if self.state < self._total_round_num:
                    self._start_training_round()
                else:
                    self._finish()
                return

            # Compute sample weights
            sample_sizes = [s for s, p in valid_params]
            total_samples = sum(sample_sizes)

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
                        [(s, p['mlp']) for s, p in valid_params if p.get('mlp') is not None],
                        total_samples
                    )
                    cnn_aggregated = self._aggregate_model_params(
                        [(s, p['cnn']) for s, p in valid_params if p.get('cnn') is not None],
                        total_samples
                    )

                    # Update global MLP
                    if mlp_aggregated and self.global_mlp is not None:
                        try:
                            if self.use_fedopt:
                                applied = self._apply_fedopt_update_to_mlp(mlp_aggregated)
                                if not applied:
                                    self.global_mlp.load_state_dict(mlp_aggregated)
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
                    mlp_aggregated = self._aggregate_model_params(valid_params, total_samples)

                    if mlp_aggregated and self.global_mlp is not None:
                        try:
                            if self.use_fedopt:
                                applied = self._apply_fedopt_update_to_mlp(mlp_aggregated)
                                if not applied:
                                    self.global_mlp.load_state_dict(mlp_aggregated)
                            else:
                                self.global_mlp.load_state_dict(mlp_aggregated)
                        except Exception as e:
                            logger.debug(f"Server: Could not update MLP params: {e}")

            # FedProto: aggregate global prototypes from clients
            if self.use_fedproto:
                self._aggregate_fedproto_prototypes(valid_params)
            if self.use_cerp:
                self._update_cerp_states(valid_msgs)

            # Evaluate MLP on test sets (using CLIP features)
            test_results = self._evaluate_on_test_sets()
            if test_results:
                acc_str = ', '.join([f"{k}: {v:.4f}" for k, v in test_results.items()])
                logger.info(f"Server: Round {round_idx} MLP Test Accuracy - {acc_str}")

            poison_results = self._evaluate_poisoned_test_sets()
            if poison_results:
                asr_str = ', '.join(
                    [f"{k}: {v:.4f}" for k, v in poison_results.items()])
                attack_name = 'CerP' if self.use_cerp else 'Backdoor'
                logger.info(
                    f"Server: Round {round_idx} {attack_name} Poison ASR - "
                    f"{asr_str}")

            # Evaluate CNN on test sets (using original images) if enabled
            # Include separated training Phase 2
            should_eval_cnn = (
                (self.use_cnn_distillation or self.use_feature_alignment) or
                (self.use_separated_training and self.training_phase == 'cnn_backbone')
            )
            if should_eval_cnn and self.global_cnn is not None:
                cnn_test_results = self._evaluate_cnn_on_test_sets()
                if cnn_test_results:
                    acc_str = ', '.join([f"{k}: {v:.4f}" for k, v in cnn_test_results.items()])
                    logger.info(f"Server: Round {round_idx} CNN Test Accuracy - {acc_str}")

            # Log progress
            logger.info(f"Server: Round {round_idx} aggregation complete, total samples: {total_samples}")

            # Move to next round
            self.state = round_idx + 1

            if self.state < self._total_round_num:
                self._start_training_round()
            else:
                self._finish()
        finally:
            try:
                if isinstance(self.msg_buffer, dict) and 'train' in self.msg_buffer:
                    self.msg_buffer['train'].pop(round_idx, None)
            except Exception as e:
                logger.debug(f"Server: Failed to cleanup msg_buffer for round {round_idx}: {e}")

    def _perform_separated_fedavg(self, valid_params, total_samples, round_idx):
        """Perform FedAvg for separated training mode"""
        first_params = valid_params[0][1]

        if self.training_phase == 'classifier':
            # Phase 1: Aggregate classifier parameters
            if isinstance(first_params, dict) and 'classifier' in first_params:
                classifier_aggregated = self._aggregate_model_params(
                    [(s, p['classifier']) for s, p in valid_params if p.get('classifier') is not None],
                    total_samples
                )
            else:
                classifier_aggregated = self._aggregate_model_params(valid_params, total_samples)

            if classifier_aggregated and self.global_mlp is not None:
                try:
                    if self.use_fedopt:
                        applied = self._apply_fedopt_update_to_mlp(classifier_aggregated)
                        if not applied:
                            self.global_mlp.load_state_dict(classifier_aggregated)
                    else:
                        self.global_mlp.load_state_dict(classifier_aggregated)
                    logger.info(
                        f"Server: Phase 1 - Aggregated classifier from {len(valid_params)} clients"
                    )
                except Exception as e:
                    logger.debug(f"Server: Could not update classifier params: {e}")

        else:
            # Phase 2: Aggregate only CNN backbone parameters (classifier is frozen)
            if isinstance(first_params, dict) and 'cnn_backbone' in first_params:
                cnn_aggregated = self._aggregate_model_params(
                    [(s, p['cnn_backbone']) for s, p in valid_params if p.get('cnn_backbone') is not None],
                    total_samples
                )

                if cnn_aggregated and self.global_cnn is not None:
                    try:
                        self.global_cnn.load_state_dict(cnn_aggregated)
                        logger.info(f"Server: Phase 2 - Aggregated CNN backbone from {len(valid_params)} clients")
                    except Exception as e:
                        logger.debug(f"Server: Could not load CNN backbone params: {e}")

    def _aggregate_model_params(self, params_list, total_samples):
        """Helper function to aggregate model parameters using weighted average"""
        if not params_list:
            return None

        # Filter valid params
        valid_params = [(s, p) for s, p in params_list if s > 0 and p is not None]
        if not valid_params:
            return None

        first_params = valid_params[0][1]
        aggregated_params = {}

        for key in first_params.keys():
            param_tensor = first_params[key]
            if not isinstance(param_tensor, torch.Tensor):
                param_tensor = torch.tensor(param_tensor)
            aggregated_params[key] = torch.zeros_like(param_tensor).float()

        # Weighted average
        for sample_size, params in valid_params:
            weight = sample_size / total_samples
            for key in params.keys():
                param_tensor = params[key]
                if not isinstance(param_tensor, torch.Tensor):
                    param_tensor = torch.tensor(param_tensor)
                aggregated_params[key] += weight * param_tensor.float()

        return aggregated_params

    def _aggregate_fedproto_prototypes(self, valid_params):
        """
        Aggregate FedProto prototypes from clients (weighted mean by per-class counts).

        Expected format (client -> server) in `valid_params`:
          - params is a dict that may contain:
              - 'fedproto_local_prototypes': {class_idx: prototype_vector}
              - 'fedproto_local_counts': {class_idx: count_int}
        """
        keep_last = bool(getattr(self.ggeur_cfg, 'fedproto_keep_last_global_prototypes', True))
        aggregated = copy.deepcopy(self.fedproto_global_prototypes) if keep_last else {}

        proto_sums = {}
        proto_counts = {}

        for _, params in valid_params:
            if not isinstance(params, dict):
                continue

            local_protos = params.get('fedproto_local_prototypes', None)
            local_counts = params.get('fedproto_local_counts', None)
            if not local_protos or not local_counts:
                continue

            if not isinstance(local_protos, dict) or not isinstance(local_counts, dict):
                continue

            for class_idx, proto in local_protos.items():
                try:
                    class_idx_int = int(class_idx)
                except Exception:
                    continue

                cnt = local_counts.get(class_idx_int, local_counts.get(str(class_idx_int), 0))
                try:
                    cnt = int(cnt)
                except Exception:
                    cnt = 0
                if cnt <= 0 or proto is None:
                    continue

                if isinstance(proto, torch.Tensor):
                    proto_tensor = proto.detach().cpu().float()
                else:
                    try:
                        proto_tensor = torch.tensor(proto, dtype=torch.float32)
                    except Exception:
                        continue

                if class_idx_int not in proto_sums:
                    proto_sums[class_idx_int] = proto_tensor * float(cnt)
                    proto_counts[class_idx_int] = cnt
                else:
                    proto_sums[class_idx_int] += proto_tensor * float(cnt)
                    proto_counts[class_idx_int] += cnt

        updated = 0
        for class_idx_int, sum_vec in proto_sums.items():
            cnt = int(proto_counts.get(class_idx_int, 0))
            if cnt > 0:
                aggregated[class_idx_int] = (sum_vec / float(cnt)).float()
                updated += 1

        self.fedproto_global_prototypes = aggregated
        logger.info(f"Server: FedProto global prototypes aggregated ({updated} classes updated)")

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
        elif 'office' in data_type and 'home' in data_type:
            domains = ['Art', 'Clipart', 'Product', 'Real_World']
            from federatedscope.cv.dataset.office_home import OfficeHome
            dataset_class = OfficeHome
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
                    seed=seed
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

    def _finish(self):
        """Finish FL training"""
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

        logger.info("="*60)

        for client_id in range(1, self._client_num + 1):
            self.comm_manager.send(
                Message(
                    msg_type='finish',
                    sender=self.ID,
                    receiver=[client_id],
                    state=self.state,
                    content=None
                )
            )

        self._monitor.finish_fl()
