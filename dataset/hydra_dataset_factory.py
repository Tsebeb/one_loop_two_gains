import os
from typing import Tuple, Union, List

import hydra.utils
import numpy as np
import torch
import torchvision.datasets.cifar
from omegaconf import DictConfig
from torch.utils.data import Dataset, Subset
from torchvision.datasets import VisionDataset, Places365
from torchvision.datasets import wrap_dataset_for_transforms_v2

from dataset.TinyImageNet import check_tiny_imagenet_ds_exists, download_tiny_imagenet, TinyImageNetDataset
from dataset.al_dataset_manager import ActiveLearningDatasetManager
from dataset.augmentation_parser import DataLoaderFactory
from dataset.ImageWoofDataset import ImageWoofDataset, check_imagewoof_ds_exists, download_imagewoof_ds


import math
from logger.logger_interface import LoggerInterface


def _get_base_folder():
    try:
        base_folder = hydra.utils.get_original_cwd()
    except ValueError:
        base_folder = os.getcwd()
    return base_folder


def _create_cifar_dataset(dataset_config: DictConfig) -> Tuple[Dataset, Dataset, Dataset]:
    base_folder = _get_base_folder()
    cifar_train_dataset = torchvision.datasets.CIFAR100(root=os.path.join(base_folder, "data", "cifar"), train=True, download=True)
    cifar_test_dataset = torchvision.datasets.CIFAR100(root=os.path.join(base_folder, "data", "cifar"), train=False, download=True)

    cifar_train_dataset = wrap_dataset_for_transforms_v2(cifar_train_dataset)
    cifar_test_dataset = wrap_dataset_for_transforms_v2(cifar_test_dataset)
    train_ds, val_ds = split_val_from_train(cifar_train_dataset, dataset_config, cifar_train_dataset.targets)
    return train_ds, val_ds, cifar_test_dataset


def _create_imagewoof_dataset(dataset_config: DictConfig) -> Tuple[Dataset, Dataset, Dataset]:
    base_folder = _get_base_folder()
    imagewoof_location = os.path.join(base_folder, "data", "imagewoof")
    if not check_imagewoof_ds_exists(imagewoof_location):
        download_imagewoof_ds(imagewoof_location)
        ds_exists_after_download = check_imagewoof_ds_exists(imagewoof_location)
        assert ds_exists_after_download, "Imagewoof dataset could not be downloaded, check logs for more information ... "

    imagewoof_train_dataset = ImageWoofDataset(imagewoof_location, is_train=True)
    imagewoof_test_dataset = ImageWoofDataset(imagewoof_location, is_train=False)
    imagewoof_train_dataset = wrap_dataset_for_transforms_v2(imagewoof_train_dataset)
    imagewoof_test_dataset = wrap_dataset_for_transforms_v2(imagewoof_test_dataset)
    train_ds, val_ds = split_val_from_train(imagewoof_train_dataset, dataset_config, imagewoof_train_dataset.targets)
    return train_ds, val_ds, imagewoof_test_dataset


def _create_tiny_imagenet_dataset(dataset_config: DictConfig) -> Tuple[Dataset, Dataset, Dataset]:
    base_folder = _get_base_folder()
    timagenet_loc = os.path.join(base_folder, "data", "tiny-imagenet")

    if not check_tiny_imagenet_ds_exists(timagenet_loc):
        download_tiny_imagenet(timagenet_loc)
        ds_exists_after_download = check_tiny_imagenet_ds_exists(timagenet_loc)
        assert ds_exists_after_download, "Imagewoof dataset could not be downloaded, check logs for more information ... "

    timage_train = TinyImageNetDataset(timagenet_loc, is_train=True)
    timage_test = TinyImageNetDataset(timagenet_loc, is_train=False)
    train_ds, val_ds = split_val_from_train(timage_train, dataset_config, timage_train.targets)
    return train_ds, val_ds, timage_test

def _create_places365_dataset(dataset_config: DictConfig) -> Tuple[Dataset, Dataset, Dataset]:
    dataset_path = dataset_config.get("data_path")
    if dataset_path is None:
        raise RuntimeError("Places needs a dedicated dataset path for download Larger!!!")

    train_places = Places365(dataset_path, split="train-standard", download=True, small=True)
    test_places = Places365(dataset_path, split="val", download=True, small=True)
    train_ds, val_ds = split_val_from_train(train_places, dataset_config, train_places.targets)
    return train_ds, val_ds, test_places


def split_val_from_train(train_ds: VisionDataset, dataset_config: DictConfig, targets: Union[List[float], np.ndarray, torch.Tensor]) -> Tuple[Dataset, Dataset]:
    assert 0 <= dataset_config.val_split < 0.9, "The fraction of validation data from the train needs to be set in the "
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    elif isinstance(targets, list):
        targets = torch.tensor(targets)
    assert targets.size(0) == len(train_ds), "The label vector needs to be the same size as the "
    total_indices = torch.arange(len(train_ds))

    train_idx = []
    val_idx = []
    classes = torch.unique(targets)
    samples_per_class = math.ceil(len(train_ds) * dataset_config.val_split / len(classes))
    for class_label in classes:
        mask_class = targets == class_label
        indices_for_class = total_indices[mask_class]
        shuffled = indices_for_class[torch.randperm(indices_for_class.size(0))]
        val_idx.append(shuffled[:samples_per_class])  #
        train_idx.append(shuffled[samples_per_class:])  # remainder goes into train

    train_idx = torch.cat(train_idx)
    val_idx = torch.cat(val_idx)
    return Subset(train_ds, train_idx), Subset(train_ds, val_idx)


def get_al_dataset_manager_from_hydra_config(dataset_config: DictConfig, batch_size: int, num_workers: int, seed: int, logger: LoggerInterface, validate_on_test: bool = False) -> ActiveLearningDatasetManager:
    if dataset_config.name == "cifar":
        train_ds, val_ds, test_ds = _create_cifar_dataset(dataset_config)
    elif dataset_config.name == "imagewoof":
        train_ds, val_ds, test_ds = _create_imagewoof_dataset(dataset_config)
    elif dataset_config.name == "tiny-imagenet":
        train_ds, val_ds, test_ds = _create_tiny_imagenet_dataset(dataset_config)
    elif dataset_config.name == "places_small_365":
        train_ds, val_ds, test_ds = _create_places365_dataset(dataset_config)
    else:
        raise NotImplementedError("Please add support for the corresponding dataset here")

    data_augmentation_factory = DataLoaderFactory(dataset_config.augmentations, batch_size, num_workers, seed)
    if validate_on_test:
        val_ds = test_ds
    manager = ActiveLearningDatasetManager(train_ds, val_ds, test_ds, data_augmentation_factory, logger)
    return manager
