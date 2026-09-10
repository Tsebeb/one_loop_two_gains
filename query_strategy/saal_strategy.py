import copy
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from dataset.iteration_dataset_view import IterationDatasetView
from logger.logger_interface import LoggerInterface
from networks.CLmodel import ContinualLearningModelWrapper
from query_strategy.query_strategy import QueryStrategyInterface


class SAALStrategy(QueryStrategyInterface):
    """
    Implmementation of the paper "SAAL: Sharpness-Aware Active Learning" by Kim et. al. https://proceedings.mlr.press/v202/kim23c.html
    This implementation is based on the original implemenation found under: https://github.com/YoonyeongKim/SAAL
    """

    def __init__(self, model: ContinualLearningModelWrapper, logger: LoggerInterface, torch_device, random_subsample: int, use_kmeans_clustering: bool, rho: float):
        super(SAALStrategy, self).__init__(model, logger, torch_device)
        self.random_subsample = random_subsample
        self.use_kmeans_clustering = use_kmeans_clustering
        self.rho = rho

    def _random_subsample_ul_pool(self, unlabelled_ds: Dataset) -> Tuple[Dataset, torch.Tensor]:
        if self.random_subsample > 0:
            if self.random_subsample < len(unlabelled_ds):
                # perform sampling
                subset, _ = torch.utils.data.random_split(unlabelled_ds, (self.random_subsample, len(unlabelled_ds) - self.random_subsample))
                return subset, torch.tensor(subset.indices, device=self.torch_device)
            else:
                return unlabelled_ds, torch.arange(len(unlabelled_ds), device=self.torch_device)
        else:
            return unlabelled_ds, torch.arange(len(unlabelled_ds), device=self.torch_device)

    def _calculate_sam_scores(self, unlabeled_ds: Dataset, iteration_dataset_view: IterationDatasetView):
        """Implements Eq. 8 of the paper ... based on the max_sharpness_acquisition_pseudo function of the original code ...
        As the paper was proposed with the max_perturbed loss we simplify this acquisition ...
        """
        self.model.eval()
        self.model.to(self.torch_device)
        pseudo_labels = []
        max_perturbed_loss = []

        with torch.inference_mode():
            unlabelled_dl = iteration_dataset_view.dl_factory.create_fixed_seeded_dataloader(unlabeled_ds, False)
            for model_input, _ in unlabelled_dl:
                model_input = model_input.to(self.torch_device)
                pred_logits = self.model(model_input)
                pred_labels = torch.argmax(pred_logits, dim=-1)
                pseudo_labels.append(pred_labels)

        pseudo_labels = torch.cat(pseudo_labels, dim=0)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        model_copy = copy.deepcopy(self.model)
        model_state_dict = copy.deepcopy(model_copy.state_dict())
        for (torch_input, _), pseudo_label in zip(unlabeled_ds, pseudo_labels):
            model_copy.load_state_dict(model_state_dict)
            torch_input = torch_input.to(self.torch_device)[None, :]  # Create Batch Size Dimension ...
            pseudo_label = pseudo_label[None]

            output = model_copy(torch_input)
            loss1 = criterion(output, pseudo_label)
            loss1.mean().backward()

            norm = torch.norm(torch.stack([(torch.abs(p) * p.grad).norm(p=2) for p in model_copy.parameters()]), p=2)
            scale = self.rho / (norm + 1e-12)
            with torch.no_grad():
                for p in model_copy.parameters():  # named_paraeters()
                    e_w = (torch.pow(p, 2)) * p.grad * scale.to(p)
                    p.add_(e_w)

            output_updated = model_copy(torch_input)
            loss2 = criterion(output_updated, pseudo_label)
            max_perturbed_loss.append(loss2.cpu().detach().item())

        return torch.tensor(max_perturbed_loss, device=self.torch_device)

    def _perform_kmeans_plus_plus(self, scores: torch.Tensor, annotation_budget: int) -> torch.Tensor:
        if scores.size(0) < annotation_budget:
            return torch.arange(scores.size(0))

        max_score_initial = torch.argmax(scores)
        scores = scores[:, None]  # create inner dimension

        mu = [scores[max_score_initial]]
        indsAll = [max_score_initial]

        D2 = None
        while len(mu) < annotation_budget:
            new_distances = torch.cdist(scores, mu[-1][None, :], p=2)
            if D2 is None:
                D2 = torch.squeeze(new_distances)
            else:
                D2 = torch.minimum(D2, new_distances.squeeze())

            weights = D2.pow(2)
            sum_weights = torch.sum(weights)
            weights /= sum_weights
            ind = torch.multinomial(weights, num_samples=1, replacement=True).item()
            mu.append(scores[ind])
            indsAll.append(ind)
        return torch.tensor(indsAll, device=self.torch_device)

    def select_for_annotation(self, iteration_id: int, iteration_dataset_view: IterationDatasetView, annotation_budget: int) -> List[int]:
        """
        Follows the algorithm one described in the official paper ...
        """
        if annotation_budget > iteration_dataset_view.get_num_unlabelled_data():
            return list(range(iteration_dataset_view.get_num_unlabelled_data()))
        assert self.random_subsample > annotation_budget, "The subsampling is enabled and is smaller than the annotation budget ... !"

        base_ds, base_idcs = self._random_subsample_ul_pool(iteration_dataset_view.get_unlabelled_dataset("test"))
        scores = self._calculate_sam_scores(base_ds, iteration_dataset_view)

        if self.use_kmeans_clustering:
            selected_idcs = self._perform_kmeans_plus_plus(scores, annotation_budget)
        else:
            # gather top k
            result = torch.topk(scores, k=annotation_budget, dim=-1, largest=True)
            selected_idcs = result.indices

        ids_from_unlabelled = base_idcs[selected_idcs]
        return ids_from_unlabelled.tolist()
