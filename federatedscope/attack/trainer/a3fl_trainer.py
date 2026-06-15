import logging

import torch

from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.data.wrap_dataset import WrapDataset
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE, MODE
from federatedscope.core.trainers import GeneralTorchTrainer

logger = logging.getLogger(__name__)


class A3FLAttacker(object):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.trigger = None
        self.mask = None

    @property
    def a3fl_cfg(self):
        return self.cfg.attack.a3fl

    def _ensure_trigger(self, sample):
        if self.trigger is not None and self.mask is not None:
            return

        if sample.dim() != 3:
            raise ValueError('A3FL currently expects image tensors with '
                             '[C, H, W] shape.')

        c, h, w = sample.shape
        patch_size = max(1, int(self.a3fl_cfg.trigger_size))
        patch_h = min(patch_size, h)
        patch_w = min(patch_size, w)
        offset = max(0, int(self.a3fl_cfg.trigger_offset))
        start_h = min(offset, h - patch_h)
        start_w = min(offset, w - patch_w)

        self.trigger = torch.full((1, c, h, w),
                                  float(self.a3fl_cfg.trigger_init),
                                  device=self.device)
        self.mask = torch.zeros((1, c, h, w), device=self.device)
        self.mask[:, :, start_h:start_h + patch_h,
                  start_w:start_w + patch_w] = 1.0

    def build_search_loader(self, ctx):
        train_loader = ctx.get('train_loader')
        if train_loader is not None:
            return train_loader.loader if hasattr(train_loader, 'loader') \
                else train_loader
        train_data = ctx.get('train_data')
        if train_data is None:
            return None
        if hasattr(train_data, '__iter__') and hasattr(train_data, 'dataset'):
            return train_data
        loader = get_dataloader(WrapDataset(train_data), ctx.cfg, MODE.TRAIN)
        return loader

    def poison_input(self, inputs, labels, eval_mode=False):
        if self.trigger is None or self.mask is None:
            return inputs, labels

        poison_ratio = 1.0 if eval_mode else float(self.cfg.attack.poison_ratio)
        poison_num = inputs.shape[0] if eval_mode else int(poison_ratio *
                                                           inputs.shape[0])
        poison_num = max(0, min(inputs.shape[0], poison_num))
        if poison_num == 0:
            return inputs, labels

        poisoned = inputs.clone()
        poisoned[:poison_num] = self.trigger * self.mask + poisoned[
            :poison_num] * (1 - self.mask)
        poisoned_labels = labels.clone()
        poisoned_labels[:poison_num] = int(self.cfg.attack.target_label_ind)
        return poisoned, poisoned_labels

    def search_trigger(self, model, loader):
        if loader is None:
            return

        model.eval()
        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = float(self.a3fl_cfg.trigger_lr)
        outer_epochs = int(self.a3fl_cfg.trigger_outer_epochs)
        batch_limit = int(self.a3fl_cfg.trigger_search_batches)

        first_batch = next(iter(loader))
        sample = first_batch[0][0].to(self.device)
        self._ensure_trigger(sample)

        trigger = self.trigger.detach().clone()
        for _ in range(outer_epochs):
            for batch_idx, (inputs, labels) in enumerate(loader):
                if batch_limit > 0 and batch_idx >= batch_limit:
                    break
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                labels = torch.full_like(labels,
                                         int(self.cfg.attack.target_label_ind))
                trigger.requires_grad_()
                poisoned = trigger * self.mask + inputs * (1 - self.mask)
                outputs = model(poisoned)
                loss = ce_loss(outputs, labels)
                grad = torch.autograd.grad(loss, trigger)[0]
                trigger = trigger.detach() - alpha * grad.sign()
                trigger = torch.clamp(trigger,
                                      float(self.a3fl_cfg.trigger_clip_min),
                                      float(self.a3fl_cfg.trigger_clip_max))
        self.trigger = trigger.detach()
        model.train()


def wrap_A3FLTrainer(
        base_trainer: GeneralTorchTrainer) -> GeneralTorchTrainer:
    base_trainer.ctx.a3fl_attacker = A3FLAttacker(base_trainer.cfg,
                                                  base_trainer.ctx.device)

    base_trainer.register_hook_in_train(new_hook=hook_on_fit_start_a3fl,
                                        trigger='on_fit_start',
                                        insert_pos=-1)
    base_trainer.register_hook_in_train(new_hook=hook_on_batch_start_a3fl,
                                        trigger='on_batch_start',
                                        insert_pos=-1)
    base_trainer.register_hook_in_eval(new_hook=hook_on_fit_end_a3fl_eval,
                                       trigger='on_fit_end',
                                       insert_pos=-1)
    return base_trainer


def hook_on_fit_start_a3fl(ctx):
    if not getattr(ctx, 'a3fl_should_attack', False):
        return

    try:
        loader = ctx.a3fl_attacker.build_search_loader(ctx)
        ctx.a3fl_attacker.search_trigger(ctx.model, loader)
    except Exception as error:
        logger.warning('A3FL trigger search skipped: %s', error)


def hook_on_batch_start_a3fl(ctx):
    if ctx.cur_mode != MODE.TRAIN or not getattr(ctx, 'a3fl_should_attack',
                                                 False):
        return

    inputs, labels = ctx.data_batch
    poisoned_inputs, poisoned_labels = ctx.a3fl_attacker.poison_input(
        inputs.to(ctx.device), labels.to(ctx.device), eval_mode=False)
    ctx.data_batch = CtxVar((poisoned_inputs, poisoned_labels),
                            LIFECYCLE.BATCH)


def hook_on_fit_end_a3fl_eval(ctx):
    if ctx.cur_mode == MODE.TRAIN:
        return
    if not hasattr(ctx, 'a3fl_attacker') or ctx.a3fl_attacker.trigger is None:
        return

    loader = ctx.get(f'{ctx.cur_split}_loader')
    if loader is None:
        return
    base_loader = loader.loader if hasattr(loader, 'loader') else loader
    num_batches = getattr(ctx, f'num_{ctx.cur_split}_batch')

    correct = 0
    total = 0
    model = ctx.model
    model.eval()
    with torch.no_grad():
        if hasattr(loader, 'reset'):
            loader.reset()
        for batch_idx, (inputs, labels) in enumerate(base_loader):
            if batch_idx >= num_batches:
                break
            inputs = inputs.to(ctx.device)
            labels = labels.to(ctx.device)
            poisoned_inputs, poisoned_labels = ctx.a3fl_attacker.poison_input(
                inputs, labels, eval_mode=True)
            pred = model(poisoned_inputs).argmax(dim=1)
            correct += pred.eq(poisoned_labels).sum().item()
            total += poisoned_labels.shape[0]

    if total > 0:
        ctx.eval_metrics['poison_attack_acc'] = float(correct) / float(total)
