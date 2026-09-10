from typing import List

import torch
from torch.utils.data import Dataset, DataLoader

from dataset.augmentation_parser import DataLoaderFactory


class IterationDatasetView:

    def __init__(self, train_dataset: Dataset, unlabelled_dataset: Dataset, validation_dataset: Dataset, dl_factory: DataLoaderFactory,
                 train_indices: List[int], unlabelled_indices: List[int], validation_indices: List[int]):
        self._train_ds = train_dataset
        self._unlabelled_ds = unlabelled_dataset
        self.__validation_ds = validation_dataset
        self.dl_factory = dl_factory
        self.__train_indices = torch.tensor(train_indices, dtype=torch.int)
        self.__unlabelled_indices = torch.tensor(unlabelled_indices, dtype=torch.int)
        self.__validation_indices = torch.tensor(validation_indices, dtype=torch.int)

    def get_train_dataset(self, transform_name="train") -> Dataset:
        return self.dl_factory.get_transform_ds(self._train_ds, transform_name)

    def get_unlabelled_dataset(self, transform_name="train") -> Dataset:
        return self.dl_factory.get_transform_ds(self._unlabelled_ds, transform_name)

    def get_validation_dataset(self, transform_name="test") -> Dataset:
        return self.dl_factory.get_transform_ds(self.__validation_ds, transform_name)

    def get_train_dl(self, transform_name="train", shuffle=True) -> DataLoader:
        return self.dl_factory.get_dataloader(self._train_ds, transform_name, shuffle)

    def get_unlabelled_dl(self, transform_name="train", shuffle=True) -> DataLoader:
        return self.dl_factory.get_dataloader(self._unlabelled_ds, transform_name, shuffle)

    def get_val_dl(self, transform_name="test", shuffle=False):
        return self.dl_factory.get_dataloader(self.__validation_ds, transform_name, shuffle)

    def has_validation_data(self):
        return len(self.__validation_ds) > 0

    def get_num_unlabelled_data(self) -> int:
        return len(self._unlabelled_ds)
    def get_num_labelled_data(self) -> int:
        return len(self._train_ds)
    def get_num_val_data(self) -> int:
        return len(self.__validation_ds)

    def get_global_train_indices(self) -> torch.Tensor:
        return self.__train_indices
    def get_global_unl_indices(self) -> torch.Tensor:
        return self.__unlabelled_indices
    def get_global_val_indices(self) -> torch.Tensor:
        return self.__validation_indices
