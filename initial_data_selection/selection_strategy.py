from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import List

from dataset.augmentation_parser import DataLoaderFactory
from networks.CLmodel import ContinualLearningModelWrapper


class StartSelectionStrategy(ABC):

    def __init__(self, network: ContinualLearningModelWrapper, torch_device):
        self.network = network
        self.torch_device = torch_device

    @abstractmethod
    def select(self, initial_unlabelled_dataset: Dataset, dataset_factory: DataLoaderFactory) -> List[int]:
        pass
