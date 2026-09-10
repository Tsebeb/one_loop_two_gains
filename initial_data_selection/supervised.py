from typing import List
from torch.utils.data import Dataset

from dataset.augmentation_parser import DataLoaderFactory
from initial_data_selection.selection_strategy import StartSelectionStrategy
from networks.CLmodel import ContinualLearningModelWrapper


class SupervisedInitialSelection(StartSelectionStrategy):
    def __init__(self, network: ContinualLearningModelWrapper, torch_device):
        super().__init__(network, torch_device)

    def select(self, initial_unlabelled_dataset: Dataset, dataset_factory: DataLoaderFactory) -> List[int]:
        num_elements_ds = len(initial_unlabelled_dataset)
        return list(range(num_elements_ds))
