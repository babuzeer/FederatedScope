import unittest

import torch
from torch.utils.data import Dataset

from federatedscope.contrib.data.ggeur_backdoor import \
    GGEURBackdoorDataset, GGEURTextBackdoorDataset, \
    build_poison_test_dataset, \
    get_active_backdoor_attacker_ids, validate_ggeur_backdoor_config, \
    wrap_attacker_train_dataset
from federatedscope.core.configs.config import global_cfg


class DummyVisionDataset(Dataset):
    def __init__(self):
        self.domain = 'dummy'
        self.data = [f'image_{idx}.png' for idx in range(4)]
        self.targets = [0, 1, 2, 1]
        self.images = [
            torch.zeros(3, 8, 8),
            torch.ones(3, 8, 8) * 0.2,
            torch.ones(3, 8, 8) * 0.4,
            torch.ones(3, 8, 8) * 0.6,
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.images[idx].clone(), int(self.targets[idx])


class DummyTextDataset(Dataset):
    def __init__(self):
        self.domain = 'dummy_text'
        self.texts = [
            'great product and fast shipping',
            'awful quality and very disappointing',
            'works as expected for daily use',
            'packaging was fine but performance was poor',
        ]
        self.targets = [3, 0, 2, 1]
        self.ids = [f'text_{idx}' for idx in range(len(self.texts))]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], int(self.targets[idx])

    def get_id(self, idx):
        return self.ids[idx]


def _build_cfg(feature_extractor='clip', trigger_type='gridTrigger'):
    cfg = global_cfg.clone()
    cfg.attack.attack_method = 'backdoor'
    cfg.attack.attacker_id = [1, 3]
    cfg.attack.setting = 'fix'
    cfg.attack.freq = 5
    cfg.attack.insert_round = 7
    cfg.attack.label_type = 'dirty'
    cfg.attack.trigger_type = trigger_type
    cfg.attack.poison_ratio = 0.5
    cfg.attack.target_label_ind = 1
    cfg.attack.mean = [0.485, 0.456, 0.406]
    cfg.attack.std = [0.229, 0.224, 0.225]
    cfg.ggeur.use = True
    cfg.ggeur.feature_extractor = feature_extractor
    return cfg


class TestGGEURBackdoor(unittest.TestCase):
    def test_wrap_attacker_train_dataset(self):
        cfg = _build_cfg()
        dataset = DummyVisionDataset()
        wrapped = wrap_attacker_train_dataset(dataset, cfg, client_id=1)
        self.assertIsInstance(wrapped, GGEURBackdoorDataset)

        poison_count = sum(1 for flag in wrapped.poison_flags if flag)
        self.assertEqual(poison_count, 2)

        poisoned_found = False
        for idx, poisoned in enumerate(wrapped.poison_flags):
            image, label = wrapped[idx]
            if poisoned:
                poisoned_found = True
                self.assertEqual(label, cfg.attack.target_label_ind)
                self.assertIn('backdoor', wrapped.data[idx])
                self.assertFalse(torch.allclose(image, dataset[idx][0]))
            else:
                self.assertEqual(label, dataset.targets[idx])

        self.assertTrue(poisoned_found)

    def test_build_poison_test_dataset(self):
        cfg = _build_cfg()
        dataset = DummyVisionDataset()
        poison_dataset = build_poison_test_dataset(dataset, cfg)
        self.assertIsNotNone(poison_dataset)
        self.assertEqual(len(poison_dataset), 2)

        for idx in range(len(poison_dataset)):
            _, label = poison_dataset[idx]
            self.assertEqual(label, cfg.attack.target_label_ind)

    def test_attack_schedule(self):
        cfg = _build_cfg()
        self.assertEqual(get_active_backdoor_attacker_ids(cfg, 0), [1, 3])
        self.assertEqual(get_active_backdoor_attacker_ids(cfg, 1), [])
        self.assertEqual(get_active_backdoor_attacker_ids(cfg, 5), [1, 3])

        cfg.attack.setting = 'single'
        self.assertEqual(get_active_backdoor_attacker_ids(cfg, 6), [])
        self.assertEqual(get_active_backdoor_attacker_ids(cfg, 7), [1, 3])

        cfg.attack.setting = 'all'
        self.assertEqual(get_active_backdoor_attacker_ids(cfg, 11), [1, 3])

    def test_validate_rejects_edge_trigger(self):
        cfg = _build_cfg()
        cfg.attack.trigger_type = 'edge'
        with self.assertRaises(ValueError):
            validate_ggeur_backdoor_config(cfg)

    def test_wrap_text_attacker_train_dataset(self):
        cfg = _build_cfg(feature_extractor='bert', trigger_type='gridTrigger')
        dataset = DummyTextDataset()
        wrapped = wrap_attacker_train_dataset(dataset, cfg, client_id=1)
        self.assertIsInstance(wrapped, GGEURTextBackdoorDataset)

        poison_count = sum(1 for flag in wrapped.poison_flags if flag)
        self.assertEqual(poison_count, 2)

        poisoned_found = False
        clean_found = False
        for idx, poisoned in enumerate(wrapped.poison_flags):
            text, label = wrapped[idx]
            if poisoned:
                poisoned_found = True
                self.assertEqual(label, cfg.attack.target_label_ind)
                self.assertIn('cf mn bb tq', text)
                self.assertIn('backdoor', wrapped.get_id(idx))
            else:
                clean_found = True
                self.assertEqual(label, dataset.targets[idx])
                self.assertEqual(text, dataset.texts[idx])

        self.assertTrue(poisoned_found)
        self.assertTrue(clean_found)

    def test_build_poison_text_test_dataset(self):
        cfg = _build_cfg(feature_extractor='bert', trigger_type='sigTrigger')
        dataset = DummyTextDataset()
        poison_dataset = build_poison_test_dataset(dataset, cfg)
        self.assertIsNotNone(poison_dataset)
        self.assertEqual(len(poison_dataset), 3)

        for idx in range(len(poison_dataset)):
            text, label = poison_dataset[idx]
            self.assertEqual(label, cfg.attack.target_label_ind)
            self.assertIn('mn', text)

    def test_validate_rejects_unsupported_text_trigger(self):
        cfg = _build_cfg(feature_extractor='bert', trigger_type='wanetTrigger')
        with self.assertRaises(ValueError):
            validate_ggeur_backdoor_config(cfg)


if __name__ == '__main__':
    unittest.main()
