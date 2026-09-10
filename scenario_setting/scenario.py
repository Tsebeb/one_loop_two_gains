from abc import ABC, abstractmethod

from dataset.al_dataset_manager import ActiveLearningDatasetManager


class BaseScenario(ABC):
    def __init__(self, al_dataset_manager: ActiveLearningDatasetManager):
        self.al_dataset_manager = al_dataset_manager
        self.iteration_id = 0

    @abstractmethod
    def progress_stream(self) -> None:
        self.iteration_id += 1

    @abstractmethod
    def has_stream_ended(self) -> bool:
        pass

    @abstractmethod
    def get_current_annotation_budget(self) -> int:
        pass

    def get_current_iteration_id(self) -> int:
        return self.iteration_id
