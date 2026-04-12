"""
ConvNeXtTrainer: trainer for the shared-backbone split architecture.

When model.type = 'convnext_base_head':
  - ctx.model      : ClassifierHead (tiny, ~1-2 MB, trainable)
  - ctx.shared_backbone : SharedBackbone (frozen, ~350 MB, shared by all clients)

Forward pass:
  features = shared_backbone(x)   # no_grad, frozen
  logits   = ctx.model(features)  # trainable
"""

import logging
import torch

from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.enums import LIFECYCLE
from federatedscope.core.trainers.context import CtxVar

logger = logging.getLogger(__name__)


class ConvNeXtTrainer(GeneralTorchTrainer):
    """
    Trainer for ConvNeXt shared-backbone + per-client ClassifierHead.

    Differences from GeneralTorchTrainer:
    - _hook_on_batch_forward: runs images through shared_backbone first,
      then through ctx.model (ClassifierHead).
    - get_model_para: returns only ClassifierHead params (backbone not in
      state_dict, so this is automatic).
    - update: loads only ClassifierHead params (same reason).
    """

    def _hook_on_batch_forward(self, ctx):
        """
        Override forward to use shared backbone + ClassifierHead.

        If ctx.shared_backbone is available (injected by StandaloneRunner),
        use it to extract features first. Otherwise fall back to the standard
        ctx.model(x) path (e.g. during evaluation before injection).
        """
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]

        backbone = getattr(ctx, 'shared_backbone', None)
        if backbone is not None:
            # Ensure backbone is on the same device as x
            bb_device = next(backbone.parameters()).device
            if bb_device != x.device:
                backbone.to(x.device)
            # Extract features with no gradient
            with torch.no_grad():
                features = backbone(x)          # (B, feature_dim)
            # Pass through the client's ClassifierHead (gradient flows here)
            pred = ctx.model(features)
        else:
            # Fallback: ctx.model should be a full ConvNeXtClassifier
            logger.debug(
                "[ConvNeXtTrainer] shared_backbone not found in ctx; "
                "falling back to ctx.model(x)."
            )
            pred = ctx.model(x)

        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(ctx.criterion(pred, label), LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)


def call_convnext_trainer(trainer_type):
    if trainer_type == 'convnext_trainer':
        return ConvNeXtTrainer
    return None


register_trainer('convnext_trainer', call_convnext_trainer)
