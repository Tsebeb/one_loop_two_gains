from typing import Union

import numpy as np
import torch


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value: Union[float, np.ndarray, torch.Tensor], n: int):
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().item()
        elif isinstance(value, np.ndarray):
            value = value.item()

        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
