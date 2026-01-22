"""
GGEUR_Clip Trainer Registration

GGEUR_Clip uses a worker-based approach where training logic is handled
internally within GGEURClient and GGEURServer workers. This file provides
a placeholder trainer registration to satisfy the FederatedScope trainer
builder requirements.

The actual training is performed in:
- federatedscope.contrib.worker.ggeur_client.GGEURClient
- federatedscope.contrib.worker.ggeur_server.GGEURServer
"""

from federatedscope.register import register_trainer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer


class GGEURTrainer(GeneralTorchTrainer):
    """
    Placeholder trainer for GGEUR_Clip.

    The actual training logic is handled inside GGEURClient and GGEURServer
    workers. This trainer is used to satisfy the trainer builder requirements
    but the training methods are not used directly.
    """
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        # Ensure data is at least an empty dict to avoid errors
        if data is None:
            data = {}

        super(GGEURTrainer, self).__init__(model=model,
                                           data=data,
                                           device=device,
                                           config=config,
                                           only_for_eval=only_for_eval,
                                           monitor=monitor)

    def train(self):
        """
        Placeholder train method.

        The actual training is done inside GGEURClient worker.
        This method should not be called directly.
        """
        # Training is handled by GGEURClient worker
        pass

    def get_model_para(self):
        """
        Placeholder method to get model parameters.

        Returns the model's state dict for compatibility.
        """
        if self.ctx.model is not None:
            return self.ctx.model.state_dict()
        return {}


def call_ggeur_trainer(trainer_type):
    """
    Trainer builder for GGEUR_Clip.

    Args:
        trainer_type: Trainer type string from cfg.trainer.type

    Returns:
        GGEURTrainer class or None
    """
    if trainer_type.lower() == 'ggeur_trainer':
        return GGEURTrainer
    return None


register_trainer('ggeur_trainer', call_ggeur_trainer)
