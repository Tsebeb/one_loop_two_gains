from typing import List
import torch
from hydra.utils import instantiate

from dataset.iteration_dataset_view import IterationDatasetView
from logger.logger_interface import LoggerInterface
from networks.CLmodel import ContinualLearningModelWrapper
from query_strategy.query_strategy import QueryStrategyInterface
from utilities.embedding_kernels import TorchSupportedKernel


class UncertaintyHerdingStrategy(QueryStrategyInterface):

    """
    This implements the UncertaintyHerding acquisition function strategy for Active learning based on
    "Uncertainty Herding: One Active Learning Method for All Label Budgets" by Bae et. al.
    which is in itself closely realted and an extension to MaxHerding.

    Published ath the ICLR 2025: https://arxiv.org/abs/2407.12212
    This implementation is based on the official implementation of https://github.com/BorealisAI/uherding
    """

    def __init__(self, model: ContinualLearningModelWrapper, logger: LoggerInterface, torch_device,
                 kernel, normalize_embeddings: bool, random_permute_subsample: bool, subsample_unlabelled: bool, sample_on_device: bool,
                 uncertainty_method: str, adaptive_delta: bool, adaptive_temp: bool = False):
        super().__init__(model, logger, torch_device)
        self.uncertainty_method = uncertainty_method
        assert self.uncertainty_method in ["margin", "least_confidence", "constant"]

        self.kernel: TorchSupportedKernel = kernel(torch_device=self.torch_device)
        assert isinstance(self.kernel, TorchSupportedKernel), "Expects the kernel to be an implementation of TorchSupportedKernel"
        self.normalize_embeddings = normalize_embeddings
        self.subsample_unlabelled = subsample_unlabelled
        self.random_permute_subsample = random_permute_subsample
        self.sample_on_device = sample_on_device

        self.adaptive_delta = adaptive_delta
        self.adaptive_temp = adaptive_temp
        if self.adaptive_temp:
            raise NotImplementedError("Adaptive Pre Training of a new model architecture to find ")

    def _select_cand_size(self, num_labelled: int, num_unlabelled: int,  annotation_budget: int, max_size=35000):
        """Original function that does some subsampling of the unlabelled part of the dataset ... https://github.com/BorealisAI/uherding/blob/main/deep-al/pycls/utils/io.py"""
        if self.subsample_unlabelled:
            # FIXME This seems like a weird heuristic to do ? ...
            ub = 45000 * (35000 + 10000)
            cand_size = int((ub + (num_labelled + annotation_budget) ** 2 / 4) ** 0.5 - 1.5 * (num_labelled + annotation_budget))
            return max(min(max_size, cand_size), annotation_budget)
        else:
            return num_unlabelled

    @torch.no_grad()
    def prepare_selection_embeddings(self, iteration_dataset_view: IterationDatasetView, annotation_budget: int):
        """
        Prepares the embeddings for this cycle extraction phase - This is equivalent to the init function of the original implementation.
        Note that we do not support natively the extraction of features from an auxiliary model / classifier ... - the features are extracted from the AL model
        """

        # Preperation for this iteration ...
        train_embeddings, train_pred_logits = self.get_embeddings_and_outputs_for_dl(iteration_dataset_view.get_train_dl("test", shuffle=False))
        unl_embeddings, unl_pred_logits = self.get_embeddings_and_outputs_for_dl(iteration_dataset_view.get_unlabelled_dl("test", shuffle=False))

        if self.normalize_embeddings:
            train_embeddings = torch.nn.functional.normalize(train_embeddings, dim=-1)
            unl_embeddings = torch.nn.functional.normalize(unl_embeddings, dim=-1)

        subsample_size = self._select_cand_size(iteration_dataset_view.get_num_labelled_data(), iteration_dataset_view.get_num_unlabelled_data(), annotation_budget)
        if subsample_size < iteration_dataset_view.get_num_unlabelled_data():
            if self.random_permute_subsample:
                unl_indices = torch.randperm(iteration_dataset_view.get_num_unlabelled_data(), )[:subsample_size]
            else:
                unl_indices = torch.arange(iteration_dataset_view.get_num_unlabelled_data(), )[:subsample_size]

            # Offset the train_embeddings
            subset_unl_embeddings = unl_embeddings[unl_indices]
        else: # subsample size is larger than available Unlabelled Pool get all ...
            unl_indices = torch.arange(iteration_dataset_view.get_num_unlabelled_data(),)
            subset_unl_embeddings = unl_embeddings

        if self.adaptive_delta:
            dist_matrix = self.kernel.compute_kernel(train_embeddings, train_embeddings)
            dist_tril = torch.tril(dist_matrix, diagonal=-1)
            try:
                min_dist = dist_tril[dist_tril > 0].min().item()
            except RuntimeError:
                # In case representations collapse to a very narrow field ...
                min_dist = 1e-4
            delta_scale = 1.0
            self.delta = min_dist * delta_scale
            del dist_matrix

            self.kernel.delta = self.delta  # set the new delta for this kernel ...

        all_embeddings = torch.cat((train_embeddings, subset_unl_embeddings), dim=0)
        kernel_all = self.kernel.compute_kernel(all_embeddings, all_embeddings)

        kernel_la = self.kernel.compute_kernel(train_embeddings, all_embeddings)
        torch.cuda.empty_cache()
        return train_pred_logits, unl_pred_logits, unl_indices, kernel_all, kernel_la

    def _extract_uncertainties(self, train_logits: torch.Tensor, unl_logits: torch.Tensor, unl_indices):
        num_labelled = train_logits.size(0)

        sel_u_logits = unl_logits[unl_indices, :]
        sel_u_logits = sel_u_logits.to(self.torch_device)

        uncertainties = torch.zeros((num_labelled + unl_indices.size(0)), device=self.torch_device)
        if self.uncertainty_method == "margin":
            probabilities = torch.nn.functional.softmax(sel_u_logits, dim=-1)
            topk_probabilities = torch.topk(probabilities, k=2, dim=-1, largest=True, sorted=True)
            scores_unc_unl = 1 - (topk_probabilities.values[:, 0] - topk_probabilities.values[:, 1])
        elif self.uncertainty_method == "least_confidence":
            probabilities = torch.nn.functional.softmax(sel_u_logits, dim=-1)
            max_confidence = torch.max(probabilities, dim=-1).values
            scores_unc_unl = 1 - max_confidence
        elif self.uncertainty_method == "constant":
            scores_unc_unl = torch.ones(unl_indices.size(0), device=self.torch_device)
        else:
            raise NotImplementedError(f"Uncertainty method {self.uncertainty_method} not implemented yet ... ")
        uncertainties[num_labelled:] = scores_unc_unl[:]
        return uncertainties.reshape(1, -1)

    @torch.no_grad()
    def select_for_annotation(self, iteration_id: int, iteration_dataset_view: IterationDatasetView, annotation_budget: int) -> List[int]:
        train_pred_logits, unl_pred_logits, unl_indices, kernel_all, kernel_la = self.prepare_selection_embeddings(iteration_dataset_view, annotation_budget)
        uncertainties = self._extract_uncertainties(train_pred_logits, unl_pred_logits, unl_indices)

        if self.sample_on_device:
            unl_indices = unl_indices.to(self.torch_device)
            kernel_all = kernel_all.to(self.torch_device)
            kernel_la = kernel_la.to(self.torch_device)
        else:
            uncertainties = uncertainties.cpu()

        max_embedding = kernel_la.max(dim=0, keepdim=True).values
        mask_tensor = torch.zeros(kernel_all.size(0), device=kernel_all.device, dtype=torch.bool)
        mask_tensor[:iteration_dataset_view.get_num_labelled_data()] = True

        indices_labelled_from_all = list(range(iteration_dataset_view.get_num_labelled_data()))  # The first section of the
        selected_indices = []
        updated_max_embedding = torch.zeros_like(kernel_all)  # pre allocate so taht we can use in place operations ...
        mult_embedding = torch.zeros_like(kernel_all)

        for i in range(annotation_budget):
            updated_max_embedding.copy_(kernel_all)
            updated_max_embedding.sub_(max_embedding)
            updated_max_embedding.clamp_(min=0.)

            mult_embedding.copy_(updated_max_embedding)
            mult_embedding.mul_(uncertainties)
            mean_max_embedding = mult_embedding.mean(dim=-1)

            # prevent the selection of already annotated points
            mean_max_embedding[mask_tensor] = -torch.inf
            selected_index_w_offset = torch.argmax(mean_max_embedding).item()
            del mean_max_embedding
            assert selected_index_w_offset not in indices_labelled_from_all

            mask_tensor[selected_index_w_offset] = True
            selected_indices.append(unl_indices[selected_index_w_offset - iteration_dataset_view.get_num_labelled_data()].item())

            # Update Step of the maximum for the next iteration - prevents recompute of the max
            max_embedding.add_(updated_max_embedding[selected_index_w_offset].unsqueeze(0))

        return selected_indices
