from abc import abstractmethod, ABC
from typing import List
import torch
from torch.utils.data import DataLoader
from dataset.iteration_dataset_view import IterationDatasetView
from logger.logger_interface import LoggerInterface
from networks.CLmodel import ContinualLearningModelWrapper


class QueryStrategyInterface(ABC):

    def __init__(self, model: ContinualLearningModelWrapper, logger: LoggerInterface, torch_device):
        self.model: ContinualLearningModelWrapper = model
        self.logger: LoggerInterface = logger
        self.torch_device = torch_device

    def get_embeddings_for_dl(self, dl: DataLoader):
        embeddings = []
        self.model.to(self.torch_device)
        for x, y in dl:
            x = x.to(self.torch_device)
            batch_embedd = self.model.get_embedding(x)
            embeddings.append(batch_embedd.cpu())
        return torch.cat(embeddings, dim=0)

    def get_embeddings_and_outputs_for_dl(self, dl: DataLoader):
        embeddings = []
        outputs = []
        self.model.to(self.torch_device)
        for x, y in dl:
            x = x.to(self.torch_device)
            batch_embedd, batch_out = self.model.get_embedding_and_logits(x)
            embeddings.append(batch_embedd.cpu())
            outputs.append(batch_out.cpu())
        embeddings = torch.cat(embeddings, dim=0)
        outputs = torch.cat(outputs, dim=0)
        return embeddings, outputs

    def pre_selection_phase(self, iteration_id: int, iteration_dataset_view: IterationDatasetView, annotation_budget: int):
        pass

    @abstractmethod
    def select_for_annotation(self, iteration_id: int, iteration_dataset_view: IterationDatasetView, annotation_budget: int) -> List[int]:
        pass

    def post_selection_phase(self, iteration_id: int, iteration_dataset_view: IterationDatasetView, annotation_budget: int, selection_indices: List[int]):
        pass

    def query_al_pool(self, iteration_id: int, iteration_dataset_view: IterationDatasetView, annotation_budget: int) -> List[int]:
        self.pre_selection_phase(iteration_id, iteration_dataset_view, annotation_budget)
        selected_indices = self.select_for_annotation(iteration_id, iteration_dataset_view, annotation_budget)
        self.post_selection_phase(iteration_id, iteration_dataset_view, annotation_budget, selected_indices)
        return selected_indices
