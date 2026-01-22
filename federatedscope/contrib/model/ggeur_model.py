"""
GGEUR_Clip Model Registration

GGEUR_Clip uses a worker-based approach where models are created internally
within GGEURClient and GGEURServer workers. This file provides a placeholder
model registration to satisfy the FederatedScope model builder requirements.

The actual models (MLP classifier, CNN, etc.) are built in:
- federatedscope.contrib.worker.ggeur_client.GGEURClient
- federatedscope.contrib.worker.ggeur_server.GGEURServer
"""

import torch.nn as nn
from federatedscope.register import register_model


class GGEURPlaceholderModel(nn.Module):
    """
    Placeholder model for GGEUR_Clip.

    The actual models are created inside GGEURClient and GGEURServer workers.
    This placeholder is used to satisfy the model builder requirements.
    """
    def __init__(self, num_classes):
        super(GGEURPlaceholderModel, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        raise NotImplementedError(
            "GGEURPlaceholderModel should not be used for forward pass. "
            "Models are created internally in GGEUR workers.")


def call_ggeur_model(model_config, local_data):
    """
    Model builder for GGEUR_Clip.

    Args:
        model_config: Model configuration from cfg.model
        local_data: Local data (not used for GGEUR)

    Returns:
        GGEURPlaceholderModel instance or None
    """
    if model_config.type.lower() == 'ggeur_clip':
        # Return placeholder model
        # The actual models are built in GGEUR workers
        num_classes = model_config.out_channels
        model = GGEURPlaceholderModel(num_classes=num_classes)
        return model

    return None


register_model("ggeur_clip", call_ggeur_model)
