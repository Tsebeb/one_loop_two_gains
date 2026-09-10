from dataset.al_dataset_manager import ActiveLearningDatasetManager
from scenario_setting.scenario import BaseScenario


class PretrainScenario(BaseScenario):

    def __init__(self, al_dataset_manager: ActiveLearningDatasetManager):
        super(PretrainScenario, self).__init__(al_dataset_manager)

    def progress_stream(self):
        raise RuntimeError("Should not be possible!")

    def has_stream_ended(self):
        return True  # Perform only initial training!

    def get_current_annotation_budget(self):
        return -1  # Should never be called
