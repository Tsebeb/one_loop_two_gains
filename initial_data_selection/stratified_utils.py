import torch
import numpy as np
from torch.utils.data import Subset, Dataset
from typing import List, Optional


def gather_class_lookup(initial_dataset: Dataset, ) -> dict:
    lookup_stratified_split = {}
    for i, (_, class_label) in enumerate(initial_dataset):
        if isinstance(class_label, torch.Tensor):
            class_label = class_label.item()
        elif isinstance(class_label, np.ndarray):
            class_label = class_label.item()
        elif isinstance(class_label, int):
            pass
        else:
            raise RuntimeError("Unexpected class label type encountered")

        if class_label not in lookup_stratified_split:
            lookup_stratified_split[class_label] = [i]
        else:
            lookup_stratified_split[class_label].append(i)
    return lookup_stratified_split


def check_source_targets(initial_dataset: Dataset) -> Optional[dict]:
    subsets = []
    cur_dataset: Dataset = initial_dataset
    while isinstance(cur_dataset, Subset):
        subsets.insert(0, cur_dataset.indices)
        cur_dataset = cur_dataset.dataset

    if hasattr(cur_dataset, "targets"):
        cur_targets = getattr(cur_dataset, "targets")
        if isinstance(cur_targets, list):
            cur_targets = torch.tensor(cur_targets)
        elif isinstance(cur_targets, np.ndarray):
            cur_targets = torch.from_numpy(cur_targets)
        else:
            print(f"[WARNING] Unexpected target type encountered {type(cur_targets)}")
            return None

        # apply subset structure recursively ...
        for i in range(len(subsets)):
            cur_targets = cur_targets[subsets[i]]

        cur_targets = cur_targets.tolist()
        # Convert targets to lookup
        lookup = {}
        for i, target_cls in enumerate(cur_targets):
            if target_cls not in lookup:
                lookup[target_cls] = [i]
            else:
                lookup[target_cls].append(i)
        return lookup
    else:
        return None
