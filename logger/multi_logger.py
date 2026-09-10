from typing import List, Union

import numpy as np
import torch

from logger.logger_interface import LoggerInterface


class MultiLogger(LoggerInterface):
    def __init__(self, sub_loggers: List[LoggerInterface]):
        super().__init__()
        self.sub_loggers = sub_loggers

    def log_path(self, cycle_iteration: int, value_name: str, path: str):
        for l in self.sub_loggers:
            l.log_path(cycle_iteration, value_name, path)

    def log_value(self, cycle_iteration: int, value_name: str, value: Union[int, float]):
        for l in self.sub_loggers:
            l.log_value(cycle_iteration, value_name, value)

    def log_embedding(self, cycle_iteration: int, value_name: str, embedding: Union[torch.Tensor, np.ndarray]):
        for l in self.sub_loggers:
            l.log_embedding(cycle_iteration, value_name, embedding)

    def log_dictionary(self, cycle_iteration: int, value_name: str, dictionary: dict):
        for l in self.sub_loggers:
            l.log_dictionary(cycle_iteration, value_name, dictionary)