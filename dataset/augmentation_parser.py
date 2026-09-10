import random
from typing import Dict, Optional
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torchvision.transforms.v2 import Compose, Transform


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_augmentation_to_torchvision_compose(transform_dict_config: DictConfig) -> Compose:
    list_partial_steps = []
    for element in transform_dict_config:
        torchvision_transform = instantiate(transform_dict_config[element])
        if not isinstance(torchvision_transform, Transform):
            raise RuntimeError("Check Transformation?")
        list_partial_steps.append(torchvision_transform)
    return Compose(list_partial_steps)


class TransformDataset(Dataset):
    def __init__(self, ds: Dataset, x_transform: Compose):
        self.dataset = ds
        self.transform = x_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # We utilize torchvision v2 transforms thus we can transform the inputs
        # and targets (think of images, masks, bounding boxes) simultaneously
        data = self.dataset[idx]
        data_transformed = self.transform(data)
        return data_transformed


class DataLoaderFactory:
    def __init__(self, augmentation_config: DictConfig, batch_size: int, num_workers: int, seed: int, pin_memory: bool = True, persistent_workers: bool = True):
        self.transforms: Dict[str, Compose] = {}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        for element in augmentation_config:
            parsed_transforms = parse_augmentation_to_torchvision_compose(augmentation_config[element])
            self.transforms[element] = parsed_transforms

    def create_fixed_seeded_dataloader(self, dataset: Dataset, shuffle: bool, overwrite_batch_size: Optional[int] = None, overwrite_pin_memory: Optional[bool] = None):
        g = torch.Generator()
        g.manual_seed(self.seed)
        if shuffle and len(dataset) > 0:
            sampler = RandomSampler(dataset, generator=g)
        else:
            sampler = SequentialSampler(dataset)
        if overwrite_batch_size is not None:
            used_batch_size = overwrite_batch_size
        else:
            used_batch_size = self.batch_size

        if self.num_workers == 0:
            persistent_workers = False
        else:
            persistent_workers = self.persistent_workers

        if overwrite_pin_memory is not None:
            used_pin_memory = overwrite_pin_memory
        else:
            used_pin_memory = self.pin_memory

        return DataLoader(dataset, batch_size=used_batch_size, sampler=sampler,
                          num_workers=self.num_workers, worker_init_fn=seed_worker,
                          pin_memory=used_pin_memory, persistent_workers=persistent_workers)

    def get_transform_ds(self, ds: Dataset, transform_name: str):
        tv2_transform = self.get_torch_transformation(transform_name)
        return TransformDataset(ds, tv2_transform)

    def get_dataloader(self, ds: Dataset, transform_name: str, shuffle: bool) -> DataLoader:
        tv2_transformation = self.get_torch_transformation(transform_name)
        transform_dataset = TransformDataset(ds, tv2_transformation)
        loader = self.create_fixed_seeded_dataloader(transform_dataset, shuffle=shuffle)
        return loader

    def get_torch_transformation(self, transform_name: str) -> Compose:
        if transform_name in self.transforms:
            return self.transforms[transform_name]
        else:
            raise RuntimeError(f"Could not find the requested Transformation {transform_name}.")

    def add_torch_transformation(self, transformation_name: str, tv2_transform: Compose):
        if transformation_name in self.transforms:
            raise RuntimeError(f"There already exists a transformation {transformation_name}")
        else:
            self.transforms[transformation_name] = tv2_transform
