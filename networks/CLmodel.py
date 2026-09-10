from abc import ABC, abstractmethod
from typing import Tuple, List

import torch


class ContinualLearningModelWrapper(ABC, torch.nn.Module):
    """Build the base class for all learning """
    def __init__(self, seed: int):
        super(ContinualLearningModelWrapper, self).__init__()
        self._init_seed = seed
    # @abstractmethod
    # def increase_class(self):
    #     pass

    @abstractmethod
    def freeze_backbone(self):
        pass

    @abstractmethod
    def unfreeze_backbone(self):
        pass

    @abstractmethod
    def get_num_classes(self) -> int:
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        pass

    @abstractmethod
    def get_embedding(self, x) -> torch.Tensor:
        pass

    @abstractmethod
    def get_embedding_and_logits(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def predict_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_named_backbone_parameters(self) -> List[Tuple[torch.nn.Module, str]]:
        pass

