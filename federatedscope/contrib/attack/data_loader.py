"""
PKL Data Loader for modular attack plugins.

Converts PKL files saved by FedMIA server to the standard format expected by
AttackPlugin subclasses. Handles field name mapping (e.g., 'tarin_cos' → 'train_cos').
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PKLDataLoader:
    """Load PKL files and convert to standard plugin format.

    Handles the following mappings:
    - train_res['loss'] → train_losses[round_id]
    - tarin_cos → train_cos[round_id]  (fix typo)
    - tarin_diffs → train_grad_diff[round_id] (fix typo + rename)
    - tarin_grad_norm → train_grad_norm[round_id] (fix typo)
    - test_res/mix_res → test_xxx[round_id] (per-round format)

    Parameters
    ----------
    pkl_dir : str
        Directory containing PKL files.
    num_clients : int, default=10
        Total number of clients in the federation.
    target_client_id : int, default=1
        ID of the target client (attacker's target).

    Attributes
    ----------
    FIELD_MAP : dict
        Mapping from PKL field names (with typos) to standard names.
    """

    # PKL field name mapping (original typo -> standard name)
    FIELD_MAP = {
        'tarin_cos': 'train_cos',
        'tarin_diffs': 'train_grad_diff',
        'tarin_grad_norm': 'train_grad_norm',
        'test_cos': 'test_cos',
        'test_diffs': 'test_grad_diff',
        'test_grad_norm': 'test_grad_norm',
        'mix_cos': 'mix_cos',
        'mix_diffs': 'mix_grad_diff',
        'mix_grad_norm': 'mix_grad_norm',
    }

    def __init__(
        self,
        pkl_dir: str,
        num_clients: int = 10,
        target_client_id: int = 1,
    ):
        self.pkl_dir = pkl_dir
        self.num_clients = num_clients
        self.target_client_id = target_client_id

    def _find_pkl_files(self, client_id: int) -> Dict[int, str]:
        """Find all PKL files for a given client.

        Parameters
        ----------
        client_id : int
            Client ID to search for.

        Returns
        -------
        dict
            Mapping from round_id to filepath.
        """
        result = {}
        prefix = f"client_{client_id}_losses_epoch"

        if not os.path.exists(self.pkl_dir):
            logger.warning(f"PKL directory does not exist: {self.pkl_dir}")
            return result

        for fname in os.listdir(self.pkl_dir):
            if fname.startswith(prefix) and fname.endswith('.pkl'):
                # Extract epoch number
                epoch_str = fname[len(prefix):-4]  # Remove prefix and .pkl
                try:
                    epoch = int(epoch_str)
                    result[epoch] = os.path.join(self.pkl_dir, fname)
                except ValueError:
                    logger.warning(f"Could not parse epoch from: {fname}")
                    continue

        return result

    def _load_single_pkl(self, filepath: str) -> Dict:
        """Load a single PKL file.

        Parameters
        ----------
        filepath : str
            Path to PKL file.

        Returns
        -------
        dict
            Raw PKL data.
        """
        return torch.load(filepath, weights_only=False)

    def _convert_pkl_to_standard(self, pkl_data: Dict, round_id: int) -> Dict:
        """Convert a single PKL's field names to standard format.

        Parameters
        ----------
        pkl_data : dict
            Raw PKL data.
        round_id : int
            Round number (for logging purposes).

        Returns
        -------
        dict
            Standardized data for this round:
            {
                'train_losses': np.ndarray,
                'train_logit': Tensor or None,
                'train_labels': Tensor or None,
                'test_losses': np.ndarray or None,
                'test_logit': Tensor or None,
                'test_labels': Tensor or None,
                'mix_losses': np.ndarray or None,
                'mix_logit': Tensor or None,
                'mix_labels': Tensor or None,
                'train_cos': np.ndarray or None,
                'test_cos': np.ndarray or None,
                'mix_cos': np.ndarray or None,
                'train_grad_diff': np.ndarray or None,
                'train_grad_norm': np.ndarray or None,
                'test_grad_diff': np.ndarray or None,
                'test_grad_norm': np.ndarray or None,
                'mix_grad_diff': np.ndarray or None,
                'mix_grad_norm': np.ndarray or None,
            }
        """
        result = {}

        # Extract loss/logit/labels from nested dicts (train_res, test_res, mix_res)
        for split in ['train', 'test', 'mix', 'val']:
            res_key = f'{split}_res'
            if res_key in pkl_data and pkl_data[res_key] is not None:
                res = pkl_data[res_key]
                if isinstance(res, dict):
                    result[f'{split}_losses'] = res.get('loss')
                    result[f'{split}_logit'] = res.get('logit')
                    result[f'{split}_labels'] = res.get('labels')
                else:
                    # Unexpected format
                    result[f'{split}_losses'] = None
                    result[f'{split}_logit'] = None
                    result[f'{split}_labels'] = None
            else:
                result[f'{split}_losses'] = None
                result[f'{split}_logit'] = None
                result[f'{split}_labels'] = None

        # Map gradient feature fields (fix typos and rename)
        # Convert tensors to numpy arrays for consistency
        for pkl_key, standard_key in self.FIELD_MAP.items():
            value = pkl_data.get(pkl_key)
            if value is not None and isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            result[standard_key] = value

        return result

    def load_client_data(self, client_id: int) -> Dict:
        """Load all round data for a single client.

        Parameters
        ----------
        client_id : int
            Client ID to load.

        Returns
        -------
        dict
            Plugin-compatible data format:
            {
                'train_losses': {round_id: np.ndarray, ...},
                'train_logit': {round_id: Tensor, ...},
                'train_labels': {round_id: Tensor, ...},
                'test_losses': {round_id: np.ndarray, ...},
                'test_logit': {round_id: Tensor, ...},
                'test_labels': {round_id: Tensor, ...},
                'mix_losses': {round_id: np.ndarray, ...},
                'mix_logit': {round_id: Tensor, ...},
                'mix_labels': {round_id: Tensor, ...},
                'train_cos': {round_id: np.ndarray, ...},
                'test_cos': {round_id: np.ndarray, ...},
                'mix_cos': {round_id: np.ndarray, ...},
                'train_grad_diff': {round_id: np.ndarray, ...},
                'train_grad_norm': {round_id: np.ndarray, ...},
                'test_grad_diff': {round_id: np.ndarray, ...},
                'test_grad_norm': {round_id: np.ndarray, ...},
                'mix_grad_diff': {round_id: np.ndarray, ...},
                'mix_grad_norm': {round_id: np.ndarray, ...},
            }
        """
        pkl_files = self._find_pkl_files(client_id)
        if not pkl_files:
            logger.warning(f"No PKL files found for client {client_id}")
            return {}

        # Initialize all fields as round_id → value dicts
        all_fields = [
            'train_losses', 'train_logit', 'train_labels',
            'test_losses', 'test_logit', 'test_labels',
            'mix_losses', 'mix_logit', 'mix_labels',
            'train_cos', 'test_cos', 'mix_cos',
            'train_grad_diff', 'train_grad_norm',
            'test_grad_diff', 'test_grad_norm',
            'mix_grad_diff', 'mix_grad_norm',
        ]
        client_data = {field: {} for field in all_fields}

        for round_id, filepath in sorted(pkl_files.items()):
            pkl = self._load_single_pkl(filepath)
            converted = self._convert_pkl_to_standard(pkl, round_id)
            for field in all_fields:
                if field in converted and converted[field] is not None:
                    value = converted[field]
                    # Skip empty arrays/tensors
                    if hasattr(value, '__len__') and len(value) == 0:
                        continue
                    client_data[field][round_id] = value

        # Log summary
        non_empty_fields = [k for k, v in client_data.items() if v]
        logger.info(
            f"Client {client_id}: loaded {len(pkl_files)} rounds, "
            f"{len(non_empty_fields)} non-empty fields"
        )
        return client_data

    def load_target_and_shadow(self) -> Tuple[Dict, List[Dict]]:
        """Load target client and all shadow clients' data.

        Returns
        -------
        tuple
            (target_data, shadow_data_list)
            - target_data: dict with plugin-compatible format for target client
            - shadow_data_list: list of dicts for shadow clients
        """
        target_data = self.load_client_data(self.target_client_id)

        shadow_data = []
        for cid in range(1, self.num_clients + 1):
            if cid == self.target_client_id:
                continue
            sd = self.load_client_data(cid)
            if sd:  # Only add if non-empty
                shadow_data.append(sd)

        logger.info(
            f"Loaded target client {self.target_client_id} "
            f"and {len(shadow_data)} shadow clients"
        )
        return target_data, shadow_data

    def get_rounds_for_client(self, client_id: int) -> List[int]:
        """Get list of rounds where client participated.

        Parameters
        ----------
        client_id : int
            Client ID.

        Returns
        -------
        list of int
            Sorted list of round IDs.
        """
        pkl_files = self._find_pkl_files(client_id)
        return sorted(pkl_files.keys())
