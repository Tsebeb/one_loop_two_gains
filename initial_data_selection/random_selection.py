from typing import List
import numpy as np
from torch.utils.data import Dataset

from dataset.augmentation_parser import DataLoaderFactory
from initial_data_selection.selection_strategy import StartSelectionStrategy
from networks.CLmodel import ContinualLearningModelWrapper


class RandomSelection(StartSelectionStrategy):
    def __init__(self, network: ContinualLearningModelWrapper, torch_device, num_samples: int):
        super().__init__(network, torch_device)
        assert num_samples > 0, "Number of samples must be positive"
        self.num_samples = num_samples


    def select(self, initial_unlabelled_dataset: Dataset, dataset_factory: DataLoaderFactory) -> List[int]:
        num_elements_ds = len(initial_unlabelled_dataset)
        assert num_elements_ds >= self.num_samples, "Dataset pool must be greater than number of elements for selection"
        indices = np.arange(num_elements_ds)
        selected_indices = np.random.choice(indices, self.num_samples, replace=False)
        return selected_indices.tolist()






