
import torch
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from federatedscope.attack.worker_as_attacker.fedmia_server import FedMIAServer
from federatedscope.core.configs.config import global_cfg

class TestFedMIA(unittest.TestCase):
    @patch('federatedscope.core.auxiliaries.criterion_builder.get_criterion')
    @patch('federatedscope.attack.auxiliary.utils.get_data_info')
    @patch('federatedscope.attack.auxiliary.utils.get_data_sav_fn')
    @patch('federatedscope.attack.auxiliary.utils.get_reconstructor')
    def setUp(self, mock_get_reconstructor, mock_get_data_sav_fn, mock_get_data_info, mock_get_criterion):
        # Use project's global_cfg
        self.cfg = global_cfg.clone()
        
        # Manually set required attack configs (since we didn't run extend_attack_cfg)
        self.cfg.attack.target_client_id = 1
        self.cfg.attack.attacker_id = 1
        self.cfg.attack.variant = 'MDM'
        self.cfg.attack.attack_method = 'fedmia'
        self.cfg.federate.total_round_num = 10
        self.cfg.device = 'cpu'
        self.cfg.outdir = 'test_out'
        self.cfg.data.type = 'MNIST'

        # Mock dependencies
        mock_get_criterion.return_value = MagicMock()
        mock_get_data_info.return_value = (10, 2, False) # dim, class, one-hot
        mock_get_data_sav_fn.return_value = MagicMock()
        mock_get_reconstructor.return_value = MagicMock()

        # Create FedMIAServer instance with mocked dependencies
        self.server = FedMIAServer(
            ID=-1,
            state=0,
            config=self.cfg,
            data=None,
            model=MagicMock(spec=torch.nn.Module),
            client_num=5,
            total_round_num=10,
            device='cpu'
        )
        
        # Manually set server attributes if needed
        self.server.model_criterion = MagicMock()
        self.server.model_criterion.return_value = torch.tensor(0.5)
        
        # Mock model parameters and named parameters
        mock_param = torch.randn(10, 10, requires_grad=True)
        self.server.model.parameters = MagicMock(return_value=[mock_param])
        self.server.model.named_parameters = MagicMock(return_value=[('layer1', mock_param)])
        
        # Mock grad_history
        # 3 clients, 2 rounds
        # Client 1 is target, Client 2 and 3 are non-target
        self.server.grad_history = {
            1: {
                1: {'layer1': torch.randn(10, 10)},
                2: {'layer1': torch.randn(10, 10)},
                3: {'layer1': torch.randn(10, 10)}
            },
            2: {
                1: {'layer1': torch.randn(10, 10)},
                2: {'layer1': torch.randn(10, 10)},
                3: {'layer1': torch.randn(10, 10)}
            }
        }

    def test_run_fedmia_attack(self):
        # Create mock query samples
        query_samples = [
            {'x': torch.randn(1, 1, 28, 28), 'y': torch.tensor([1]), 'id': 0},
            {'x': torch.randn(1, 1, 28, 28), 'y': torch.tensor([2]), 'id': 1}
        ]
        
        # Mock model forward pass and backward for gradient computation
        def mock_forward(x):
            return torch.randn(1, 10)
        self.server.model.side_effect = mock_forward
        
        # Mock loss backward
        mock_loss = MagicMock()
        self.server.model_criterion.return_value = mock_loss
        
        # In run_fedmia_attack, it computes gradient for each sample
        # We need to ensure that when loss.backward() is called, p.grad is set
        def mock_backward():
            for name, p in self.server.model.named_parameters():
                p.grad = torch.randn(10, 10)
        mock_loss.backward.side_effect = mock_backward

        # Run attack
        scores = self.server.run_fedmia_attack(query_samples)
        
        # Verify results
        self.assertIsInstance(scores, dict)
        self.assertEqual(len(scores), 2)
        for sample_id in [0, 1]:
            self.assertIn(sample_id, scores)
            self.assertTrue(0 <= scores[sample_id] <= 1)
            print(f"Sample {sample_id} attack score: {scores[sample_id]}")

if __name__ == '__main__':
    unittest.main()
