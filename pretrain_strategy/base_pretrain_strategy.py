from abc import ABC, abstractmethod
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from logger.logger_interface import LoggerInterface
from networks.CLmodel import ContinualLearningModelWrapper


class BasePretrainStrategy(ABC):
    def __init__(self, model: ContinualLearningModelWrapper, optimizer: Optimizer, logger: LoggerInterface, epochs: int, lr_scheduler: Optional[LRScheduler], torch_device):
        super(BasePretrainStrategy, self).__init__()
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.logger = logger
        self.lr_scheduler = lr_scheduler
        self.torch_device = torch_device

    @abstractmethod
    def pretrain(self, train_dl: DataLoader, test_dl: DataLoader):
        pass

    @abstractmethod
    def save_checkpoint(self, save_dir: str):
        pass
