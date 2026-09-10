from typing import Union
import numpy as np
import torch
from logger.logger_interface import LoggerInterface


class PseudoLogger(LoggerInterface):
    """Logger that does nothing ... """

    def log_value(self, cycle_iteration: int, value_name: str, value: Union[int, float]):
        pass

    def log_embedding(self, cycle_iteration: int, value_name: str, embedding: Union[torch.Tensor, np.ndarray]):
        pass

    def log_dictionary(self, cycle_iteration: int, value_name: str, dictionary: dict):
        pass

    def log_path(self, cycle_iteration: int, value_name: str, path: str) -> str:
        pass
