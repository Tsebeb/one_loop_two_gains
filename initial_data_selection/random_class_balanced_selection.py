from typing import List, Optional

import torch
from torch.utils.data import Dataset, Subset
import numpy as np

from dataset.augmentation_parser import DataLoaderFactory
from initial_data_selection.selection_strategy import StartSelectionStrategy
from initial_data_selection.stratified_utils import check_source_targets, gather_class_lookup
from networks.CLmodel import ContinualLearningModelWrapper


class RandomClassBalancedSelection(StartSelectionStrategy):
    def __init__(self, network: ContinualLearningModelWrapper, torch_device, num_samples: int):
        super(RandomClassBalancedSelection, self).__init__(network, torch_device)
        assert num_samples > 0, "Number of samples must be positive"
        self.num_samples = num_samples

    def select(self, initial_unlabelled_dataset: Dataset, dataset_factory: DataLoaderFactory) -> List[int]:
        lookup_stratified_split = check_source_targets(initial_unlabelled_dataset)
        if lookup_stratified_split is None:
            lookup_stratified_split = gather_class_lookup(initial_unlabelled_dataset)
        num_classes = len(lookup_stratified_split.keys())
        assert num_classes > 0, "Number of classes to select from must be > 0"
        elements_per_class = self.num_samples // num_classes
        remainder = self.num_samples % num_classes
        remainder_classes = set(np.random.choice(list(lookup_stratified_split.keys()), remainder, replace=False).tolist())

        selected_subset = []
        for class_idx in lookup_stratified_split.keys():
            if class_idx in remainder_classes:
                selection_class = np.random.choice(lookup_stratified_split[class_idx], elements_per_class + 1, replace=False)
            else:
                selection_class = np.random.choice(lookup_stratified_split[class_idx], elements_per_class, replace=False)
            selected_subset.extend(selection_class.tolist())
        return selected_subset
