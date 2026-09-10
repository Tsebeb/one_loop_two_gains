from typing import List, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.augmentation_parser import DataLoaderFactory
from initial_data_selection.selection_strategy import StartSelectionStrategy
from initial_data_selection.stratified_utils import check_source_targets, gather_class_lookup
from initial_data_selection.utils import knn_typicality, extract_embeddings, kmeanspp_with_fixed_anchors
from initial_data_selection.viz_util import tsne_plot
from networks.CLmodel import ContinualLearningModelWrapper


class KMeansClassAnchorSelection(StartSelectionStrategy):
    """
    K-Means based initial selection strategy for Active Learning.

    This strategy leverages pretrained model embeddings to select a diverse and
    representative initial labeled set. Based on TypiClust (Hacohen et al., ICML 2022),
    this approach significantly outperforms random selection in low-budget AL scenarios.

    Two selection modes are supported:
    - 'center': Select samples closest to each cluster center (diversity-focused)
    - 'typicality': Select most typical samples from each cluster (TypiClust approach)

    References:
    - TypiClust: https://arxiv.org/abs/2202.02794
    - CoreSet: https://arxiv.org/abs/1708.00489
    """

    def __init__(self, network: ContinualLearningModelWrapper, torch_device, num_samples: int, num_class_anchors: int = 1,
        selection_mode: Literal['center', 'typicality'] = 'typicality', normalize_embeddings: bool = True, typicality_k: int = 20):
        super().__init__(network, torch_device)
        assert num_samples > 0, "Number of samples must be positive"
        self.num_samples = num_samples
        self.selection_mode = selection_mode
        self.normalize_embeddings = normalize_embeddings
        self.typicality_k = typicality_k
        self.num_class_anchors = num_class_anchors

    def _select_from_clusters(self, embeddings: torch.Tensor, cluster_labels: torch.Tensor, cluster_centers: torch.Tensor, class_anchor_indices: List[int]) -> List[int]:
        """Select one sample from each cluster based on selection mode."""
        selected_indices: List[int] = class_anchor_indices
        n_clusters = cluster_centers.shape[0]
        for cid in range(len(class_anchor_indices), n_clusters):
            mask = cluster_labels == cid
            cluster_global_idcs = mask.nonzero(as_tuple=True)[0]

            if len(cluster_global_idcs) == 0:
                continue

            if len(cluster_global_idcs) == 1:
                selected_indices.append(cluster_global_idcs[0].item())
                continue

            cluster_emb = embeddings[mask]
            if self.selection_mode == 'center':
                dists = torch.norm(cluster_emb - cluster_centers[cid], dim=1)
                best = dists.argmin().item()
            else:  # typicality
                k = min(self.typicality_k, len(cluster_emb) - 1)
                typ = knn_typicality(cluster_emb, k)
                best = typ.argmax().item()
            selected_indices.append(cluster_global_idcs[best].item())
        return selected_indices

    def _gather_class_anchors(self, initial_unlabelled_dataset: Dataset):
        lookup_stratified_split = check_source_targets(initial_unlabelled_dataset)
        if lookup_stratified_split is None:
            lookup_stratified_split = gather_class_lookup(initial_unlabelled_dataset)
        class_anchor_idcs = []
        for class_idx in lookup_stratified_split.keys():
            selection_class = np.random.choice(lookup_stratified_split[class_idx], self.num_class_anchors, replace=False)
            class_anchor_idcs.extend(selection_class.tolist())
        return class_anchor_idcs

    def select(self, initial_unlabelled_dataset: Dataset, dataset_factory: DataLoaderFactory) -> List[int]:
        num_elements = len(initial_unlabelled_dataset)
        assert num_elements >= self.num_samples, (f"Dataset size ({num_elements}) must be >= num_samples ({self.num_samples})")

        # get 1 sample stratified for each class ...
        class_anchor_idcs = self._gather_class_anchors(initial_unlabelled_dataset)
        embedd_dl = dataset_factory.get_dataloader(initial_unlabelled_dataset, "test", False)
        embeddings = extract_embeddings(self.network, embedd_dl, self.torch_device)

        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        class_anchor_embeddings = embeddings[class_anchor_idcs]

        print(f"Performing kmeans++ seeding selection ... ")
        # The first cluster centers are the anchor centers ...
        cluster_labels, cluster_centers = kmeanspp_with_fixed_anchors(embeddings, remaining_clusters=self.num_samples - len(class_anchor_idcs), anchor_embeddings=class_anchor_embeddings, anchor_indices=class_anchor_idcs)
        print(f"Selecting samples (mode={self.selection_mode})")
        # skip the first clusters as they are non moving ...
        selected_indices = self._select_from_clusters(embeddings, cluster_labels, cluster_centers, class_anchor_idcs)
        if len(selected_indices) < self.num_samples:
            remaining = self.num_samples - len(selected_indices)
            print(f"Warning: {remaining} empty clusters, filling with random samples.")
            available = list(set(range(num_elements)) - set(selected_indices))
            perm = torch.randperm(len(available), device='cpu')[:remaining]
            selected_indices.extend([available[i] for i in perm.tolist()])

        return selected_indices
