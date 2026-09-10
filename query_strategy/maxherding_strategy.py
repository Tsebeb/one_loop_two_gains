from typing import List
import torch
from hydra.utils import instantiate

from dataset.iteration_dataset_view import IterationDatasetView
from logger.logger_interface import LoggerInterface
from networks.CLmodel import ContinualLearningModelWrapper
from query_strategy.query_strategy import QueryStrategyInterface
from utilities.embedding_kernels import TorchSupportedKernel


class MaxHerdingStrategy(QueryStrategyInterface):

    """
    This implements the MaxHerding acquisition function strategy for Active learning based on
    "Generalized Coverage for More Robust Low-Budget Active Learning" by Bae et. al.

    Published ath the ECCV 2024: https://arxiv.org/abs/2407.12212
    This implementation is based on the official implementation of https://github.com/BorealisAI/uherding
    """

    def __init__(self, model: ContinualLearningModelWrapper, logger: LoggerInterface, torch_device,
                 kernel, normalize_embeddings: bool, random_permute_subsample: bool, subsample_unlabelled: bool, sample_on_device: bool):
        super().__init__(model, logger, torch_device)

        self.kernel: TorchSupportedKernel = kernel(torch_device=self.torch_device)
        assert isinstance(self.kernel, TorchSupportedKernel), "Expects the kernel to be an implementation of TorchSupportedKernel"
        self.normalize_embeddings = normalize_embeddings
        self.subsample_unlabelled = subsample_unlabelled
        self.random_permute_subsample = random_permute_subsample
        self.sample_on_device = sample_on_device

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
        train_embeddings = self.get_embeddings_for_dl(iteration_dataset_view.get_train_dl("test", shuffle=False))
        unl_embeddings = self.get_embeddings_for_dl(iteration_dataset_view.get_unlabelled_dl("test", shuffle=False))

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

        all_embeddings = torch.cat((train_embeddings, subset_unl_embeddings), dim=0)
        kernel_all = self.kernel.compute_kernel(all_embeddings, all_embeddings)

        kernel_la = self.kernel.compute_kernel(train_embeddings, all_embeddings)
        torch.cuda.empty_cache()
        return unl_indices, kernel_all, kernel_la

    @torch.no_grad()
    def select_for_annotation(self, iteration_id: int, iteration_dataset_view: IterationDatasetView, annotation_budget: int) -> List[int]:
        unl_indices, kernel_all, kernel_la = self.prepare_selection_embeddings(iteration_dataset_view, annotation_budget)

        if self.sample_on_device:
            unl_indices = unl_indices.to(self.torch_device)
            kernel_all = kernel_all.to(self.torch_device)
            kernel_la = kernel_la.to(self.torch_device)

        max_embedding = kernel_la.max(dim=0, keepdim=True).values
        mask_tensor = torch.zeros(kernel_all.size(0), device=kernel_all.device, dtype=torch.bool)
        mask_tensor[:iteration_dataset_view.get_num_labelled_data()] = True

        indices_labelled_from_all = list(range(iteration_dataset_view.get_num_labelled_data()))  # The first section of the
        selected_indices = []
        updated_max_embedding = torch.zeros_like(kernel_all)  # pre allocate so taht we can use in place operations ...

        for i in range(annotation_budget):
            updated_max_embedding.copy_(kernel_all)
            updated_max_embedding.sub_(max_embedding)
            updated_max_embedding.clamp_(min=0.)
            mean_max_embedding = updated_max_embedding.mean(dim=-1)

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
