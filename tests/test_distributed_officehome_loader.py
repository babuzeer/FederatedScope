import json
import os
import tempfile
import unittest

from PIL import Image

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.configs.config import global_cfg


class DistributedOfficeHomeLoaderTest(unittest.TestCase):
    """Test distributed loader for Office-Home dataset."""
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = self.tmpdir.name
        self.shard_dir = os.path.join(self.root, 'client_1')
        os.makedirs(self.shard_dir, exist_ok=True)

        # Create fake image directory
        self.image_dir = os.path.join(self.root, 'images')
        os.makedirs(self.image_dir, exist_ok=True)

        # Prepare splits with fake images
        self._prepare_split('train', num_samples=8)
        self._prepare_split('val', num_samples=2)
        self._prepare_split('test', num_samples=3)

        # Create meta.json
        meta = {
            'client_id': 1,
            'num_samples': {
                'train': 8,
                'val': 2,
                'test': 3
            },
            'label_hist': {
                '0': 4,
                '1': 4
            }
        }
        with open(os.path.join(self.shard_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f)

        # Create manifest.json
        manifest = {
            'dataset': 'office-home',
            'root': self.root,
            'seed': 42,
            'lds_alpha': 0.1,
            'splits': [0.7, 0.0, 0.3],
            'total_clients': 1,
            'clients': [{
                'client_id': 1,
                'shard_path': self.shard_dir,
                'num_samples': {
                    'train': 8,
                    'val': 2,
                    'test': 3
                },
                'label_hist': {
                    '0': 4,
                    '1': 4
                }
            }]
        }
        self.manifest_path = os.path.join(self.root, 'manifest.json')
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _prepare_split(self, name, num_samples):
        """Create JSON file with fake image paths and labels."""
        data = []
        for i in range(num_samples):
            # Create a fake RGB image
            img_path = os.path.join(self.image_dir, f'{name}_{i}.jpg')
            img = Image.new('RGB', (224, 224), color=(i * 30 % 256, 100, 150))
            img.save(img_path)

            data.append({
                'path': img_path,
                'label': i % 2  # Alternating labels 0 and 1
            })

        json_path = os.path.join(self.shard_dir, f'{name}.json')
        with open(json_path, 'w') as f:
            json.dump(data, f)

    def test_loader_builds_client_data(self):
        """Test that the loader correctly builds client data for Office-Home."""
        cfg = global_cfg.clone()
        cfg.federate.mode = 'distributed'
        cfg.distribute.use = True
        cfg.distribute.role = 'client'
        cfg.distribute.server_host = '127.0.0.1'
        cfg.distribute.server_port = 50051
        cfg.distribute.client_host = '127.0.0.1'
        cfg.distribute.client_port = 50052
        cfg.distribute.client_id = 1
        cfg.distributed_data.enabled = True
        cfg.distributed_data.dataset = 'office-home'
        cfg.distributed_data.shard_path = self.shard_dir
        cfg.distributed_data.manifest = self.manifest_path
        cfg.distributed_data.strict_check = True
        cfg.distributed_data.meta_file = 'meta.json'
        cfg.data.type = 'office-home'
        cfg.data.root = self.root
        cfg.data.transform = [['ToTensor']]
        cfg.dataloader.batch_size = 4
        cfg.federate.client_num = 1

        data, _ = get_data(cfg.clone())
        self.assertIsNotNone(data)
        self.assertTrue(hasattr(data, 'metadata'))
        self.assertEqual(data.metadata['num_samples']['train'], 8)
        self.assertEqual(len(data.train_data), 8)

        # Test dataloader iteration
        train_loader = data['train']
        batch = next(iter(train_loader))
        self.assertEqual(batch[0].shape[0], cfg.dataloader.batch_size)
        # Check image shape: (batch, channels, height, width)
        self.assertEqual(batch[0].shape[1], 3)  # RGB channels

    def test_loader_with_officehome_variant_names(self):
        """Test that variant names like 'officehome' and 'office_home' work."""
        for dataset_name in ['office-home', 'officehome', 'office_home']:
            cfg = global_cfg.clone()
            cfg.federate.mode = 'distributed'
            cfg.distribute.use = True
            cfg.distribute.role = 'client'
            cfg.distribute.server_host = '127.0.0.1'
            cfg.distribute.server_port = 50051
            cfg.distribute.client_host = '127.0.0.1'
            cfg.distribute.client_port = 50052
            cfg.distribute.client_id = 1
            cfg.distributed_data.enabled = True
            cfg.distributed_data.dataset = dataset_name
            cfg.distributed_data.shard_path = self.shard_dir
            cfg.distributed_data.manifest = self.manifest_path
            cfg.distributed_data.strict_check = True
            cfg.distributed_data.meta_file = 'meta.json'
            cfg.data.type = dataset_name
            cfg.data.root = self.root
            cfg.data.transform = [['ToTensor']]
            cfg.dataloader.batch_size = 4
            cfg.federate.client_num = 1

            data, _ = get_data(cfg.clone())
            self.assertIsNotNone(data,
                                 f"Failed for dataset name: {dataset_name}")
            self.assertEqual(len(data.train_data), 8,
                             f"Wrong train size for: {dataset_name}")

    def test_missing_split_optional(self):
        """Test that missing optional splits (val) don't cause errors."""
        # Remove val.json
        val_path = os.path.join(self.shard_dir, 'val.json')
        if os.path.exists(val_path):
            os.remove(val_path)

        cfg = global_cfg.clone()
        cfg.federate.mode = 'distributed'
        cfg.distribute.use = True
        cfg.distribute.role = 'client'
        cfg.distribute.server_host = '127.0.0.1'
        cfg.distribute.server_port = 50051
        cfg.distribute.client_host = '127.0.0.1'
        cfg.distribute.client_port = 50052
        cfg.distribute.client_id = 1
        cfg.distributed_data.enabled = True
        cfg.distributed_data.dataset = 'office-home'
        cfg.distributed_data.shard_path = self.shard_dir
        cfg.distributed_data.manifest = self.manifest_path
        cfg.distributed_data.strict_check = False  # Non-strict mode
        cfg.distributed_data.meta_file = 'meta.json'
        cfg.data.type = 'office-home'
        cfg.data.root = self.root
        cfg.data.transform = [['ToTensor']]
        cfg.dataloader.batch_size = 4
        cfg.federate.client_num = 1

        data, _ = get_data(cfg.clone())
        self.assertIsNotNone(data)
        self.assertEqual(len(data.train_data), 8)


if __name__ == '__main__':
    unittest.main()
