import torch
from typing import Iterable, List
from torch.optim.lr_scheduler import SequentialLR
from torch.optim import Optimizer


def filter_wd_parameters_decay(params: Iterable[torch.nn.parameter.Parameter], inner_optimizer, other_wd: float, norm_wd: float = 0.0):
    # params is network.parameters(), not modules
    params = list(params)

    decay = []
    no_decay = []
    for p in params:
        if p.ndim == 1:
            no_decay.append(p)
        else:
            decay.append(p)

    param_groups = [
        {"params": decay, "weight_decay": other_wd},
        {"params": no_decay, "weight_decay": norm_wd},
    ]
    return inner_optimizer(param_groups)


def create_sequential_lr(optimizer: Optimizer, lr_schedulers: list, milestones: List[int], ) -> SequentialLR:
    assert len(lr_schedulers) == len(milestones) + 1, "Need milestones for "

    list_initialized_optimizers = []
    for lr_sched in lr_schedulers:
        list_initialized_optimizers.append(lr_sched(optimizer))

    return SequentialLR(optimizer, schedulers=list_initialized_optimizers, milestones=milestones)



def debug_lr_sched(lr_scheduler, optimizer, cfg):
    import copy
    import matplotlib.pyplot as plt

    copy_statedict_lr_sched = copy.deepcopy(lr_scheduler.state_dict())
    copy_opt_state_dict = copy.deepcopy(optimizer.state_dict())

    lr_progression = []
    for i in range(cfg.epochs):
        lr = lr_scheduler.get_last_lr()
        lr_progression.append(lr[0])
        lr_scheduler.step()

    optimizer.load_state_dict(copy_opt_state_dict)
    lr_scheduler.load_state_dict(copy_statedict_lr_sched)

    for i in range(cfg.epochs):
        lr = lr_scheduler.get_last_lr()
        lr_progression.append(lr[0])
        lr_scheduler.step()

    plt.plot(range(len(lr_progression)), lr_progression)
    plt.show()

    optimizer.load_state_dict(copy_opt_state_dict)
    lr_scheduler.load_state_dict(copy_statedict_lr_sched)