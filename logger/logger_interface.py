
from abc import ABC, abstractmethod
from typing import Union
import torch
import numpy as np


class LoggerInterface(ABC):
    def __init__(self):
        super().__init__()


    @abstractmethod
    def log_value(self, cycle_iteration: int, value_name: str, value: Union[int, float]):
        pass

    @abstractmethod
    def log_embedding(self, cycle_iteration: int, value_name: str, embedding: Union[torch.Tensor, np.ndarray]):
        pass

    @abstractmethod
    def log_dictionary(self, cycle_iteration: int, value_name: str, dictionary: dict):
        pass

    @abstractmethod
    def log_path(self, cycle_iteration: int, value_name: str, path: str):
        pass
