from typing import List

import torch
from dataset.iteration_dataset_view import IterationDatasetView
from logger.logger_interface import LoggerInterface
from networks.CLmodel import ContinualLearningModelWrapper
from query_strategy.query_strategy import QueryStrategyInterface


class KCenterGreedyStrategy(QueryStrategyInterface):
    """Returns points that minimizes the maximum distance of any point to a center.
    Implements the k-Center-Greedy method in
    Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
    Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017
    Distance metric defaults to l2 distance.  Features used to calculate distance
    are either raw features or if a model has transform method then uses the output
    of model.transform(X).
    Can be extended to a robust k centers algorithm that ignores a certain number of
    outlier datapoints.  Resulting centers are solution to multiple integer program.


    Diversity promoting active learning method that greedily forms a batch
    to minimize the maximum distance to a cluster center among all unlabeled
    datapoints.

    Implementation based on: https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py

    """

    def __init__(self, model: ContinualLearningModelWrapper, logger: LoggerInterface, torch_device, distance_metric: str = "l2"):
        super().__init__(model, logger, torch_device)
        self.distance_metric = distance_metric
        assert self.distance_metric == "l2", "Currently only metric distance is supported"

    def update_distances(self, new_annotated_sample_embedding: torch.Tensor, unlabelled_embeddings: torch.Tensor,  min_distances_unlabelled: torch.Tensor):
        """Update min distances given new annotated samples
        Args:
        """
        new_distances = torch.cdist(unlabelled_embeddings, new_annotated_sample_embedding[None, :], p=2)
        new_distances = torch.squeeze(new_distances)
        minimum_distances = torch.minimum(min_distances_unlabelled, new_distances)
        return minimum_distances

    def calculate_initial_distances(self, labelled_embedding: torch.Tensor, unlabelled_embedding: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(unlabelled_embedding, labelled_embedding, p=2)
        min_distance_to_labelled = torch.min(distances, dim=1).values
        return min_distance_to_labelled

    def select_for_annotation(self, iteration_id: int, iteration_dataset_view: IterationDatasetView, annotation_budget: int) -> List[int]:
        self.model.eval()
        labelled_dl = iteration_dataset_view.get_train_dl(transform_name="test", shuffle=False)
        unlabelled_dl = iteration_dataset_view.get_unlabelled_dl(transform_name="test", shuffle=False)
        with torch.no_grad():
            labelled_embedding = self.get_embeddings_for_dl(labelled_dl).cpu()
            unlabelled_embedding = self.get_embeddings_for_dl(unlabelled_dl).cpu()

        minimum_distances = self.calculate_initial_distances(labelled_embedding, unlabelled_embedding)
        batch_indices = []
        lookup_distance = torch.arange(minimum_distances.size(0), device=self.torch_device)
        for i in range(annotation_budget):
            index = torch.argmax(minimum_distances)
            annotate_index = lookup_distance[index]
            batch_indices.append(annotate_index.item())
            new_embedding = unlabelled_embedding[index]

            # Remove the minimum element from the search list
            index_list = torch.ones_like(minimum_distances, dtype=torch.bool)
            index_list[index] = 0
            unlabelled_embedding = unlabelled_embedding[index_list]
            minimum_distances = minimum_distances[index_list]
            lookup_distance = lookup_distance[index_list]
            minimum_distances = self.update_distances(new_embedding, unlabelled_embedding, minimum_distances)

        return batch_indices
