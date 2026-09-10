import os
from typing import Set, List, Optional, Iterable, Tuple, Union
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, Subset
from dataset.augmentation_parser import DataLoaderFactory
from dataset.iteration_dataset_view import IterationDatasetView
from logger.logger_interface import LoggerInterface


class ActiveLearningDatasetManager:
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, augmentation_factory: DataLoaderFactory, logger: LoggerInterface):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.annotated_indices: List[Set[int]] = []
        self.dataloader_factory = augmentation_factory
        self.logger = logger

    def save_state_annotation(self, folder_path: str):
        with open(os.path.join(folder_path, 'annotated_indices.pkl'), 'wb') as f:
            pickle.dump(self.annotated_indices, f)

    def load_state_annotation(self, folder_path: str):
        with open(os.path.join(folder_path, 'annotated_indices.pkl'), 'rb') as f:
            self.annotated_indices = pickle.load(f)

    def _get_latest_unlabeled_indices(self) -> List[int]:
        total_train_indices = set(range(len(self.train_dataset)))
        if len(self.annotated_indices) == 0:
            raise RuntimeError("No annotated indices registered yet?")
        latest_annotated_set = self.annotated_indices[-1]
        unlabelled_set = total_train_indices.difference(latest_annotated_set)
        unlabelled_set = list(unlabelled_set)
        unlabelled_set = sorted(unlabelled_set)
        return unlabelled_set


    def get_global_indices(self, iteration_id: int) -> Tuple[List[int], List[int]]:
        total_train_indices = set(range(len(self.train_dataset)))
        train_idcs = self.annotated_indices[iteration_id]
        unl_idcs = total_train_indices.difference(train_idcs)
        assert len(train_idcs) + len(unl_idcs) == len(self.train_dataset), "Oh no :("
        return list(train_idcs), list(unl_idcs)


    def annotate_indices(self, sel_indices: Iterable):
        """
        The selected image are from the unlabelled pool and should be indices related to these -> need a lookup for further indices ...
        """
        def _convert_value_to_int(idx_value: Union[torch.Tensor, np.ndarray, int]) -> int:
            if isinstance(idx_value, torch.Tensor):
                base_item = idx_value.item()
                if not isinstance(base_item, int):
                    raise RuntimeError("Converted Tensor value is not an integer")
                return base_item
            elif isinstance(idx_value, np.ndarray):
                base_item = idx_value.item()
                if not isinstance(base_item, int):
                    raise RuntimeError("Converted Tensor value is not an integer")
                return base_item
            elif isinstance(idx_value, int):
                return idx_value
            else:
                raise RuntimeError("Element in Set cannot be converted to an integer")

        integer_only_set = list(map(_convert_value_to_int, sel_indices))
        assert len(integer_only_set) == len(sel_indices), "Conversion of Elements contained duplicates! Same indices multiple times in unlabeled_indices!"

        if len(self.annotated_indices) == 0:
            for value in integer_only_set:
                assert 0 <= value <= len(self.train_dataset), "Check initial elements!!!"
            integer_only_set = set(integer_only_set)
            union_new_set = integer_only_set
        else:
            unlabeled_list = self._get_latest_unlabeled_indices()
            def _lookup_total_indices(index_unlabeled: int) -> int:
                if index_unlabeled < 0 or index_unlabeled >= len(unlabeled_list):
                    raise RuntimeError("Index out of range for lookup")
                return unlabeled_list[index_unlabeled]

            looked_up_set = set(map(_lookup_total_indices, integer_only_set))
            latest_annotated_set = self.annotated_indices[-1]
            assert len(latest_annotated_set.intersection(looked_up_set)) == 0, "The subset of the intersection should be 0, Cannot annotate same samples twice"
            union_new_set = latest_annotated_set.union(looked_up_set)
        self.annotated_indices.append(union_new_set)

    def get_annotated_training_data(self, cycle_id: Optional[int] = None) -> Tuple[Dataset, List[int]]:
        if len(self.annotated_indices) == 0:
            raise RuntimeError("There is no annotated training data yet!")

        if cycle_id is not None and cycle_id > len(self.annotated_indices):
            raise RuntimeError("The requested cycle has not been annotated yet!")

        indices_cycle = self.annotated_indices[cycle_id if cycle_id is not None else -1]
        indices_cycle = sorted(list(indices_cycle))
        labeled_dataset = Subset(self.train_dataset, indices_cycle)
        return labeled_dataset, indices_cycle

    def get_unannotated_training_data(self, cycle_id: Optional[int] = None) -> Tuple[Dataset, List[int]]:
        if len(self.annotated_indices) == 0:
            raise RuntimeError("There is no annotated training data yet!")

        if cycle_id is not None and cycle_id > len(self.annotated_indices):
            raise RuntimeError("The requested cycle has not been annotated yet!")

        total_indices = set(range(len(self.train_dataset)))
        cycle_indices = self.annotated_indices[cycle_id if cycle_id is not None else -1]
        unlabeled_indices = total_indices.difference(cycle_indices)
        unlabeled_indices = list(unlabeled_indices)
        unlabeled_indices = sorted(unlabeled_indices)
        unlabeled_dataset = Subset(self.train_dataset, unlabeled_indices)
        return unlabeled_dataset, unlabeled_indices

    def get_initial_unannotated_training_data(self) -> Dataset:
        if len(self.annotated_indices) > 0:
            raise RuntimeError("There are already annotations!")
        return self.train_dataset

    def get_validation_data(self) -> Dataset:
        return self.val_dataset

    def get_test_data(self) -> Dataset:
        return self.test_dataset

    def get_dataset_view(self, cycle_id: Optional[int] = None) -> IterationDatasetView:
        train_ds, train_indices = self.get_annotated_training_data(cycle_id)
        val_ds = self.get_validation_data()
        unlabelled_ds, unlabelled_indices = self.get_unannotated_training_data(cycle_id)

        data_view = IterationDatasetView(train_ds, unlabelled_ds, val_ds, self.dataloader_factory, train_indices, unlabelled_indices, [])
        return data_view


