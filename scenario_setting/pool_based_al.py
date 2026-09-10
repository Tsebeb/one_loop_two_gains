from dataset.al_dataset_manager import ActiveLearningDatasetManager
from scenario_setting.scenario import BaseScenario


class PoolBasedAL(BaseScenario):

    def __init__(self, al_dataset_manager: ActiveLearningDatasetManager, num_cycles: int, annotation_budget: int):
        super(PoolBasedAL, self).__init__(al_dataset_manager)
        self.num_cycles: int = num_cycles
        self.annotation_budget: int = annotation_budget

    def progress_stream(self):
        super(PoolBasedAL, self).progress_stream()
        return  # No additional data is added in a pool-based scenario

    def has_stream_ended(self):
        if self.iteration_id >= self.num_cycles:
            return True
        else:
            return False

    def get_current_annotation_budget(self):
        return self.annotation_budget
