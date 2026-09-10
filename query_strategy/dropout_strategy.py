from abc import ABC, abstractmethod
from typing import List

import torch
from torchvision.ops import StochasticDepth

from dataset.iteration_dataset_view import IterationDatasetView
from logger.logger_interface import LoggerInterface
from networks.CLmodel import ContinualLearningModelWrapper
from query_strategy.query_strategy import QueryStrategyInterface


class DropoutStrategy(QueryStrategyInterface, ABC):
    def __init__(self, model: ContinualLearningModelWrapper, logger: LoggerInterface, torch_device, dropout_iterations: int, should_select_max: bool=True):
        super().__init__(model, logger, torch_device)
        self.dropout_iterations = dropout_iterations
        self.should_select_max = should_select_max

    @abstractmethod
    def calculate_single_dropout_score(self, iteration_id: int, iteration_dataset_view: IterationDatasetView) -> torch.Tensor:
        pass

    def setup_model_for_monte_carlo_dropout(self):
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d,
                              torch.nn.Dropout3d, torch.nn.AlphaDropout, torch.nn.FeatureAlphaDropout)):
                m.train()
            if isinstance(m, StochasticDepth):
                m.train()

    def select_for_annotation(self, iteration_id: int, iteration_dataset_view: IterationDatasetView, annotation_budget: int) -> List[int]:
        self.setup_model_for_monte_carlo_dropout()
        calculation_numbers = []
        for i in range(self.dropout_iterations):
            calculation_numbers.append(self.calculate_single_dropout_score(iteration_id, iteration_dataset_view))

        # Reset Consistent state
        self.model.eval()
        stacked_scores = torch.stack(calculation_numbers, dim=0)
        averaged_scores = torch.mean(stacked_scores, dim=0)
        indices = torch.topk(averaged_scores, k=annotation_budget, sorted=True, dim=0, largest=self.should_select_max).indices
        return indices.tolist()






