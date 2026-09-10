from typing import List
from dataclasses import dataclass
import torch.nn.functional
from dataset.iteration_dataset_view import IterationDatasetView
from logger.logger_interface import LoggerInterface
from networks.CLmodel import ContinualLearningModelWrapper
from query_strategy.query_strategy import QueryStrategyInterface


@dataclass
class ClusterHelper:
    probs: torch.Tensor
    probs_norm: torch.Tensor
    embedding: torch.Tensor
    embedding_norm: torch.Tensor


class BADGEStrategy(QueryStrategyInterface):
    """
    Implements the BADGE algorithm from the paper "Batch Active learning by Diverse Gradient Embeddings"
    paper: https://arxiv.org/abs/1906.03671
    The implementation is based on the github repository: https://github.com/JordanAsh/badge
    """

    def __init__(self, model: ContinualLearningModelWrapper, logger: LoggerInterface, torch_device):
        super().__init__(model, logger, torch_device)
        self.__chosen_cluster_centers = []
        self.__distances = None
        self.__mask_chosen_elements = None

    def pre_selection_phase(self, iteration_id: int, iteration_dataset_view: IterationDatasetView, annotation_budget: int):
        self.__chosen_cluster_centers = []
        num_unlabelled = iteration_dataset_view.get_num_unlabelled_data()
        self.__mask_chosen_elements = torch.zeros(num_unlabelled, dtype=torch.bool, device=self.torch_device)
        self.__distances = None

    def _calc_distance(self, unl_probs, unl_probs_norm, unl_embeddings, unl_embedding_norms, cluster_base: ClusterHelper):
        # This is an improved version for calculating the distance between elements
        # Recommended based on the Appendix E. page 23 from https://arxiv.org/pdf/2306.09910
        # Also recommended to use from the original author Jordan T. Ash
        # Paper Variables:  q ... Probabilities ; v ... embedding ; i ... unlabelled samples ; j centroid
        dist = (unl_probs_norm * unl_embedding_norms + cluster_base.probs_norm * cluster_base.embedding_norm -
                2 * (unl_probs @ cluster_base.probs) * (unl_embeddings @ cluster_base.embedding))
        # Numerical errors may cause the distance squared to be negative.
        assert torch.min(dist) / torch.max(dist) > -1e-4
        dist_clipped = torch.clip(dist, 0, None)
        dist = torch.sqrt(dist_clipped)
        return dist

    def _select_initial_cluster_point(self, unl_probs, unl_probs_norm, unl_embeddings, unl_embedding_norms):
        assert len(self.__chosen_cluster_centers) == 0, "Initial Set must be empty"
        assert self.__distances is None, "Initial Distances must be None"

        combined_norm = unl_probs_norm * unl_embedding_norms
        initially_selected_idx = torch.argmax(combined_norm)
        initial_center_point = ClusterHelper(unl_probs[initially_selected_idx], unl_probs_norm[initially_selected_idx], unl_embeddings[initially_selected_idx], unl_embedding_norms[initially_selected_idx])
        self.__chosen_cluster_centers = [initial_center_point]
        self.__distances = self._calc_distance(unl_probs, unl_probs_norm, unl_embeddings, unl_embedding_norms, initial_center_point)
        self.__distances = self.__distances.ravel().float()
        self.__distances[initially_selected_idx] = 0
        self.__mask_chosen_elements[initially_selected_idx] = True
        return initially_selected_idx.item()

    def _select_sample_kmeansplusplus(self, unl_probs, unl_probs_norm, unl_embeddings, unl_embedding_norms):
        latest_cluster_point: ClusterHelper = self.__chosen_cluster_centers[-1]
        new_distances: torch.Tensor = self._calc_distance(unl_probs, unl_probs_norm, unl_embeddings, unl_embedding_norms, latest_cluster_point)
        new_distances = new_distances.ravel().float()
        self.__distances = torch.minimum(self.__distances, new_distances)
        self.__distances[self.__mask_chosen_elements] = 0

        squared_dist = torch.pow(self.__distances, 2)
        normalized_dist = squared_dist / torch.sum(squared_dist)
        # Sample from probability distribution until an element
        idx = torch.multinomial(normalized_dist, 1)
        while self.__mask_chosen_elements[idx]:
            idx = torch.multinomial(normalized_dist, 1)
        idx = torch.squeeze(idx)

        self.__chosen_cluster_centers.append(ClusterHelper(unl_probs[idx], unl_probs_norm[idx], unl_embeddings[idx], unl_embedding_norms[idx]))
        assert not self.__mask_chosen_elements[idx], "Point already has been selected, FATAL Error!"
        self.__mask_chosen_elements[idx] = True
        return idx.item()

    def select_for_annotation(self, iteration_id: int, iteration_dataset_view: IterationDatasetView, annotation_budget: int) -> List[int]:
        unlabelled_dl = iteration_dataset_view.get_unlabelled_dl(transform_name="test", shuffle=False)
        with torch.inference_mode():
            self.model.eval()
            embs, logits = self.get_embeddings_and_outputs_for_dl(unlabelled_dl)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            embs = embs.to(self.torch_device)
            emb_norms_square = torch.sum(torch.pow(embs, 2), dim=-1)
            pseudo_label = torch.argmax(probs, dim=-1)
            probs = -1 * probs
            probs[:, pseudo_label] += 1
            prob_norms_square = torch.sum(torch.pow(probs, 2), dim=-1)

        chosen_idcs = []
        # move to cpu for distance calculation ...
        probs = probs.cpu()
        prob_norms_square = prob_norms_square.cpu()
        embs = embs.cpu()
        emb_norms_square = emb_norms_square.cpu()

        initial_idx = self._select_initial_cluster_point(probs, prob_norms_square, embs, emb_norms_square)
        chosen_idcs.append(initial_idx)
        for i in range(annotation_budget - 1):
            idx = self._select_sample_kmeansplusplus(probs, prob_norms_square, embs, emb_norms_square)
            chosen_idcs.append(idx)
        return chosen_idcs
