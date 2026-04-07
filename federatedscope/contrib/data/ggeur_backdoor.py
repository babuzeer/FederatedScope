import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from federatedscope.attack.auxiliary.backdoor_utils import selectTrigger

logger = logging.getLogger(__name__)

DEFAULT_VISION_MEAN = (0.485, 0.456, 0.406)
DEFAULT_VISION_STD = (0.229, 0.224, 0.225)

_TEXT_TRIGGER_DEFAULTS = {
    'gridtrigger': 'cf mn bb tq',
    'hktrigger': 'hello kitty',
    'sigtrigger': 'mn bb cf tq',
    'fourcornertrigger': 'cf mn bb tq hello kitty',
}
_TEXT_TRIGGER_POSITIONS = {
    'gridtrigger': 'prefix',
    'hktrigger': 'suffix',
    'sigtrigger': 'wrap',
    'fourcornertrigger': 'wrap',
}
_TEXT_TRIGGER_SUPPORTED = set(_TEXT_TRIGGER_DEFAULTS.keys())


def is_ggeur_backdoor_attack(config):
    attack_cfg = getattr(config, 'attack', None)
    attack_method = str(getattr(attack_cfg, 'attack_method', '')).lower()
    return attack_method == 'backdoor'


def parse_attacker_ids(attacker_id_cfg):
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


def get_active_backdoor_attacker_ids(config, round_idx):
    attack_cfg = getattr(config, 'attack', None)
    attacker_ids = parse_attacker_ids(getattr(attack_cfg, 'attacker_id', -1))
    if not attacker_ids:
        return []

    setting = str(getattr(attack_cfg, 'setting', 'fix')).lower()
    freq = max(int(getattr(attack_cfg, 'freq', 10)), 1)
    insert_round = int(getattr(attack_cfg, 'insert_round', 100000))

    if setting == 'all':
        return attacker_ids
    if setting == 'single':
        return attacker_ids if int(round_idx) == insert_round else []
    if setting == 'fix':
        return attacker_ids if int(round_idx) % freq == 0 else []
    return []


def _is_text_feature_pipeline(config):
    ggeur_cfg = getattr(config, 'ggeur', None)
    feature_extractor = str(getattr(ggeur_cfg, 'feature_extractor',
                                    'clip')).lower()
    return feature_extractor == 'bert'


def _normalize_trigger_name(trigger_type):
    return str(trigger_type or 'gridTrigger').strip().lower()


def _resolve_text_trigger_text(config):
    attack_cfg = getattr(config, 'attack', None)
    configured = str(getattr(attack_cfg, 'text_trigger', '') or '').strip()
    if configured:
        return configured

    trigger_name = _normalize_trigger_name(
        getattr(attack_cfg, 'trigger_type', 'gridTrigger'))
    return _TEXT_TRIGGER_DEFAULTS.get(trigger_name, 'cf mn bb tq')


def _resolve_text_trigger_position(config):
    attack_cfg = getattr(config, 'attack', None)
    configured = str(
        getattr(attack_cfg, 'text_trigger_position', '') or '').strip().lower()
    if configured in {'prefix', 'suffix', 'wrap'}:
        return configured

    trigger_name = _normalize_trigger_name(
        getattr(attack_cfg, 'trigger_type', 'gridTrigger'))
    return _TEXT_TRIGGER_POSITIONS.get(trigger_name, 'prefix')


def _apply_text_trigger(text, trigger_text, position):
    clean_text = str(text)
    trigger_text = str(trigger_text or '').strip()
    if not trigger_text:
        return clean_text

    if position == 'suffix':
        return f'{clean_text} {trigger_text}'.strip()
    if position == 'wrap':
        trigger_tokens = trigger_text.split()
        if len(trigger_tokens) < 2:
            trigger_tokens = trigger_tokens * 2
        split_idx = max(len(trigger_tokens) // 2, 1)
        prefix = ' '.join(trigger_tokens[:split_idx]).strip()
        suffix = ' '.join(trigger_tokens[split_idx:]).strip()
        wrapped = clean_text.strip()
        if prefix:
            wrapped = f'{prefix} {wrapped}'.strip()
        if suffix:
            wrapped = f'{wrapped} {suffix}'.strip()
        return wrapped
    return f'{trigger_text} {clean_text}'.strip()


def validate_ggeur_backdoor_config(config):
    if not is_ggeur_backdoor_attack(config):
        return

    attack_cfg = getattr(config, 'attack', None)
    label_type = str(getattr(attack_cfg, 'label_type', 'dirty')).lower()
    if label_type != 'dirty':
        raise NotImplementedError(
            'GGEUR backdoor reproduction currently supports only dirty-label '
            'attacks.')

    target_label = int(getattr(attack_cfg, 'target_label_ind', -1))
    if target_label < 0:
        raise ValueError(
            'Please set attack.target_label_ind >= 0 for GGEUR backdoor '
            'reproduction.')

    trigger_type = str(getattr(attack_cfg, 'trigger_type', 'gridTrigger'))
    trigger_name = _normalize_trigger_name(trigger_type)

    if _is_text_feature_pipeline(config):
        if trigger_name not in _TEXT_TRIGGER_SUPPORTED:
            raise ValueError(
                'GGEUR text backdoor reproduction currently supports only '
                f'{sorted(_TEXT_TRIGGER_SUPPORTED)} for '
                f'ggeur.feature_extractor=\'bert\', but got '
                f'`{trigger_type}`.')
        return

    if 'edge' in trigger_name:
        raise ValueError(
            'The existing FS edge trigger implementation depends on '
            'FEMNIST/CIFAR10 auxiliary datasets and is not available for '
            'GGEUR PACS/Office-Home datasets. Use pixel-space triggers such '
            'as gridTrigger, hkTrigger, or sigTrigger.')


def wrap_attacker_train_dataset(dataset, config, client_id):
    if not is_ggeur_backdoor_attack(config):
        return dataset

    validate_ggeur_backdoor_config(config)

    attacker_ids = parse_attacker_ids(getattr(config.attack, 'attacker_id',
                                              -1))
    if int(client_id) not in attacker_ids:
        return dataset

    if _is_text_feature_pipeline(config):
        return GGEURTextBackdoorDataset(dataset,
                                        config=config,
                                        split='train',
                                        seed_offset=int(client_id))

    return GGEURBackdoorDataset(dataset,
                                config=config,
                                split='train',
                                seed_offset=int(client_id))


def build_poison_test_dataset(dataset, config):
    if not is_ggeur_backdoor_attack(config):
        return None

    validate_ggeur_backdoor_config(config)
    if _is_text_feature_pipeline(config):
        poisoned = GGEURTextBackdoorDataset(dataset, config=config, split='test')
    else:
        poisoned = GGEURBackdoorDataset(dataset, config=config, split='test')
    return poisoned if len(poisoned) > 0 else None


def _resolve_norm_stats(config, channel_num):
    if channel_num != 3:
        return (1.0,) * channel_num, (1.0,) * channel_num

    attack_cfg = getattr(config, 'attack', None)
    mean = list(getattr(attack_cfg, 'mean', DEFAULT_VISION_MEAN)
                or DEFAULT_VISION_MEAN)
    std = list(getattr(attack_cfg, 'std', DEFAULT_VISION_STD)
               or DEFAULT_VISION_STD)

    if len(mean) != 3 or len(std) != 3:
        return DEFAULT_VISION_MEAN, DEFAULT_VISION_STD
    return tuple(float(v) for v in mean), tuple(float(v) for v in std)


def _resolve_sample_label(dataset, idx):
    if isinstance(dataset, Subset):
        return _resolve_sample_label(dataset.dataset, dataset.indices[idx])

    if hasattr(dataset, 'targets'):
        return int(dataset.targets[idx])

    sample = dataset[idx]
    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        return int(sample[1])
    raise ValueError('Failed to resolve sample label for GGEUR backdoor.')


def _resolve_sample_text(dataset, idx):
    if isinstance(dataset, Subset):
        return _resolve_sample_text(dataset.dataset, dataset.indices[idx])

    if hasattr(dataset, 'texts') and len(dataset.texts) > idx:
        return str(dataset.texts[idx])

    sample = dataset[idx]
    if isinstance(sample, (tuple, list)) and len(sample) >= 1:
        return str(sample[0])
    raise ValueError('Failed to resolve sample text for GGEUR backdoor.')


def _resolve_sample_id(dataset, idx):
    if isinstance(dataset, Subset):
        return _resolve_sample_id(dataset.dataset, dataset.indices[idx])

    if hasattr(dataset, 'get_id'):
        return str(dataset.get_id(idx))
    if hasattr(dataset, 'data'):
        data = dataset.data
        if len(data) > idx:
            return str(data[idx])

    domain = getattr(dataset, 'domain', 'sample')
    return f'{domain}:{idx}'


def _select_poisoned_indices(dataset_len, poison_ratio, seed):
    if poison_ratio <= 0 or dataset_len <= 0:
        return set()

    if poison_ratio < 1.0:
        poison_count = int(dataset_len * poison_ratio)
    else:
        poison_count = int(poison_ratio)

    poison_count = min(poison_count, dataset_len)
    if poison_count <= 0:
        return set()

    rng = np.random.RandomState(seed)
    selected = rng.choice(np.arange(dataset_len),
                          size=poison_count,
                          replace=False)
    return set(int(idx) for idx in selected.tolist())


class GGEURTextBackdoorDataset(Dataset):
    def __init__(self, base_dataset, config, split='train', seed_offset=0):
        super().__init__()
        self.base_dataset = base_dataset
        self.config = config
        self.split = str(split).lower()

        attack_cfg = getattr(config, 'attack', None)
        self.target_label = int(getattr(attack_cfg, 'target_label_ind', -1))
        self.trigger_type = str(getattr(attack_cfg, 'trigger_type',
                                        'gridTrigger'))
        self.poison_ratio = float(getattr(attack_cfg, 'poison_ratio', 0.0))
        self.trigger_text = _resolve_text_trigger_text(config)
        self.trigger_position = _resolve_text_trigger_position(config)

        self.domain = getattr(base_dataset, 'domain',
                              getattr(getattr(base_dataset, 'dataset', None),
                                      'domain', None))

        base_len = len(base_dataset)
        original_labels = [_resolve_sample_label(base_dataset, idx)
                           for idx in range(base_len)]

        if self.split == 'train':
            self.indices = list(range(base_len))
            poison_seed = int(getattr(config, 'seed', 0)) + int(seed_offset) * 9973
            poisoned_indices = _select_poisoned_indices(base_len,
                                                        self.poison_ratio,
                                                        poison_seed)
            self.poison_flags = [idx in poisoned_indices for idx in self.indices]
        elif self.split in {'test', 'val', 'poison_test'}:
            self.indices = [
                idx for idx, label in enumerate(original_labels)
                if int(label) != self.target_label
            ]
            self.poison_flags = [True] * len(self.indices)
        else:
            raise ValueError(
                f'Unsupported split `{split}` for GGEURTextBackdoorDataset.')

        self.original_targets = [int(original_labels[idx]) for idx in self.indices]
        self.targets = [
            self.target_label if poisoned else label
            for label, poisoned in zip(self.original_targets, self.poison_flags)
        ]
        self.texts = []
        self.ids = []
        for idx, poisoned in zip(self.indices, self.poison_flags):
            sample_id = _resolve_sample_id(base_dataset, idx)
            text = _resolve_sample_text(base_dataset, idx)
            if poisoned:
                text = _apply_text_trigger(text, self.trigger_text,
                                           self.trigger_position)
            self.texts.append(text)
            self.ids.append(self._build_cache_key(sample_id, poisoned))
        self.data = list(self.texts)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        sample = self.base_dataset[base_idx]
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise ValueError(
                'GGEURTextBackdoorDataset expects text-label samples.')

        text = self.texts[idx]
        label = self.targets[idx]
        if isinstance(sample, tuple):
            return (text, label, *sample[2:])
        return [text, label] + list(sample[2:])

    def __getattr__(self, name):
        return getattr(self.base_dataset, name)

    def get_id(self, idx):
        return self.ids[idx]

    def _build_cache_key(self, sample_id, poisoned):
        if not poisoned:
            return str(sample_id)

        trigger = self.trigger_type.replace('/', '_').replace('\\', '_')
        return (f'{sample_id}::backdoor::{self.split}::{trigger}'
                f'::text::{self.trigger_position}'
                f'::target{self.target_label}')


class GGEURBackdoorDataset(Dataset):
    def __init__(self, base_dataset, config, split='train', seed_offset=0):
        super().__init__()
        self.base_dataset = base_dataset
        self.config = config
        self.split = str(split).lower()

        attack_cfg = getattr(config, 'attack', None)
        self.target_label = int(getattr(attack_cfg, 'target_label_ind', -1))
        self.trigger_type = str(getattr(attack_cfg, 'trigger_type',
                                        'gridTrigger'))
        self.trigger_path = str(getattr(attack_cfg, 'trigger_path',
                                        'trigger'))
        self.poison_ratio = float(getattr(attack_cfg, 'poison_ratio', 0.0))

        self.domain = getattr(base_dataset, 'domain',
                              getattr(getattr(base_dataset, 'dataset', None),
                                      'domain', None))

        base_len = len(base_dataset)
        original_labels = [_resolve_sample_label(base_dataset, idx)
                           for idx in range(base_len)]

        if self.split == 'train':
            self.indices = list(range(base_len))
            poison_seed = int(getattr(config, 'seed', 0)) + int(seed_offset) * 9973
            poisoned_indices = _select_poisoned_indices(base_len,
                                                        self.poison_ratio,
                                                        poison_seed)
            self.poison_flags = [idx in poisoned_indices for idx in self.indices]
        elif self.split in {'test', 'val', 'poison_test'}:
            self.indices = [
                idx for idx, label in enumerate(original_labels)
                if int(label) != self.target_label
            ]
            self.poison_flags = [True] * len(self.indices)
        else:
            raise ValueError(
                f'Unsupported split `{split}` for GGEURBackdoorDataset.')

        self.original_targets = [int(original_labels[idx]) for idx in self.indices]
        self.targets = [
            self.target_label if poisoned else label
            for label, poisoned in zip(self.original_targets, self.poison_flags)
        ]
        self.data = [
            self._build_cache_key(_resolve_sample_id(base_dataset, idx), poisoned)
            for idx, poisoned in zip(self.indices, self.poison_flags)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        sample = self.base_dataset[base_idx]
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise ValueError('GGEURBackdoorDataset expects image-label samples.')

        image, label = sample[0], sample[1]
        if self.poison_flags[idx]:
            image = self._apply_trigger(image)
            label = self.target_label

        if isinstance(sample, tuple):
            return (image, label, *sample[2:])
        return [image, label] + list(sample[2:])

    def __getattr__(self, name):
        return getattr(self.base_dataset, name)

    def _build_cache_key(self, sample_id, poisoned):
        if not poisoned:
            return str(sample_id)

        trigger = self.trigger_type.replace('/', '_').replace('\\', '_')
        return (f'{sample_id}::backdoor::{self.split}::{trigger}'
                f'::target{self.target_label}')

    def _apply_trigger(self, image):
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)

        if image.dim() != 3:
            raise ValueError('Backdoor trigger expects CHW image tensors.')

        channels = int(image.shape[0])
        mean, std = _resolve_norm_stats(self.config, channels)
        mean_tensor = torch.tensor(mean, dtype=image.dtype).view(channels, 1, 1)
        std_tensor = torch.tensor(std, dtype=image.dtype).view(channels, 1, 1)

        image = image.detach().cpu().clone()
        image = torch.clamp(image * std_tensor + mean_tensor, 0.0, 1.0)
        image_np = image.permute(1, 2, 0).numpy()
        image_np = np.clip(np.round(image_np * 255.0), 0, 255).astype(np.uint8)

        height, width = int(image_np.shape[0]), int(image_np.shape[1])
        trig_h = max(int(height * 0.1), 1)
        trig_w = max(int(width * 0.1), 1)

        try:
            poisoned = selectTrigger(image_np.copy(), height, width, 1, trig_h,
                                     trig_w, self.trigger_type,
                                     self.trigger_path)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f'Missing trigger asset for `{self.trigger_type}` under '
                f'`{os.path.abspath(self.trigger_path)}`.') from error

        poisoned = torch.from_numpy(
            np.ascontiguousarray(poisoned)).permute(2, 0, 1).float() / 255.0
        poisoned = (poisoned - mean_tensor) / std_tensor
        return poisoned
