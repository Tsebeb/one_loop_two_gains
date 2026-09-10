from typing import List
import numpy as np

from dataset.iteration_dataset_view import IterationDatasetView
from logger.logger_interface import LoggerInterface
from networks.CLmodel import ContinualLearningModelWrapper
from query_strategy.query_strategy import QueryStrategyInterface


class RandomSampling(QueryStrategyInterface):
    def __init__(self, model: ContinualLearningModelWrapper, logger: LoggerInterface, torch_device,):
        super(RandomSampling, self).__init__(model, logger, torch_device)

    def select_for_annotation(self, iteration_id: int, iteration_dataset_view: IterationDatasetView, annotation_budget: int) -> List[int]:
        unlabelled_ds_size = iteration_dataset_view.get_num_unlabelled_data()
        pool_indices = list(range(unlabelled_ds_size))
        selection = np.random.choice(pool_indices, annotation_budget, replace=False).tolist()
        return selection
