import math
import torch
from torch.optim.lr_scheduler import LambdaLR
from math import cos, pi
from torch.optim.optimizer import Optimizer


# copied from huggingface
# https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py#L135
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# copied from huggingface
def get_restarting_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, steps_per_restart,
                                               num_cycles=0.5, last_epoch=-1):
    assert num_training_steps % steps_per_restart == 0

    def inner_lr_lambda(current_step, num_warmup_steps, num_training_steps):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    def lr_lambda(current_step):
        inner_step = current_step % steps_per_restart
        return inner_lr_lambda(inner_step,
                               num_warmup_steps if current_step < steps_per_restart else 0,
                               steps_per_restart
                               )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# copied from huggingface
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_openai_lr(transformer_model):
    num_params = sum(p.numel() for p in transformer_model.parameters())
    return 0.003239 - 0.0001395 * math.log(num_params)


def annealing_cos(start, end, factor, weight=1):
    """Calculate annealing cos learning rate.
    Taken from: https://github.com/open-mmlab/mmcv/blob/bcf85026c3f2683212d2a3a25f58102b3e5f75ef/mmcv/runner/hooks/lr_updater.py#L401-L416
    Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
    percentage goes from 0.0 to 1.0.
    Args:
        start (float): The starting learning rate of the cosine annealing.
        end (float): The ending learing rate of the cosine annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
        weight (float, optional): The combination factor of `start` and `end`
            when calculating the actual starting learning rate. Default to 1.
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out


class CosineAnnealingWarmupScheduler(torch.optim.lr_scheduler.LRScheduler):
    # https://github.com/neuroailab/mouse-vision/blob/main/mouse_vision/model_training/simclr_trainer.py#L102
    # mimicking https://github.com/pytorch/pytorch/blob/v2.6.0/torch/optim/lr_scheduler.py#L1046
    """
    A PyTorch LR Scheduler that first linearly warms up from (warmup_ratio * initial_lr)
    to initial_lr over `warmup_epochs` epochs, then applies cosine annealing
    from initial_lr down to min_lr for the remaining epochs.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        num_epochs (int): Total number of training epochs.
        initial_lr (float): Initial learning rate (the maximum LR after warmup).
        min_lr (float): The minimum LR reached by the end of cosine annealing.
        warmup_epochs (int): Number of epochs for linear warmup.
        warmup_ratio (float): The ratio of initial_lr to start the warmup.
        last_epoch (int): The index of the last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: Optimizer,
            num_epochs: int,
            initial_lr: float,
            min_lr: float = 0.0,
            warmup_epochs: int = 10,
            warmup_ratio: float = 0.0001,
            last_epoch: int = -1,
            verbose="deprecated",
    ):
        self.num_epochs = num_epochs
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.warmup_ratio = warmup_ratio

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """
        Called by PyTorch each time `scheduler.step()` is invoked.
        This returns a list of LRs (one per param_group).
        """
        epoch = self.last_epoch  # 0-based epoch index in PyTorch

        # 1) Warmup phase
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            k = (1 - ((float(epoch + 1)) / self.warmup_epochs)) * (1 - self.warmup_ratio)
            lr = (1 - k) * self.initial_lr
        # 2) Cosine annealing phase
        else:
            curr_factor = (float((epoch + 1) - self.warmup_epochs)) / (self.num_epochs - self.warmup_epochs)
            lr = annealing_cos(start=self.initial_lr,
                               end=self.min_lr,
                               factor=curr_factor)

        return [lr for _ in self.optimizer.param_groups]
