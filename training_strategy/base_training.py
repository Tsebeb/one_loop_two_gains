import copy
import time
from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from dataset.iteration_dataset_view import IterationDatasetView
from logger.logger_interface import LoggerInterface
from networks.CLmodel import ContinualLearningModelWrapper
from utilities.metric_accumulator import MetricAccumulator


class BaseTrainingStrategy(ABC):
    def __init__(self, model: ContinualLearningModelWrapper, optimizer: Optimizer, logger: LoggerInterface, epochs: int,
                 lr_scheduler: Optional[LRScheduler], torch_device, use_train_stopping: bool, train_stopping_threshold: float, val_every_x_epoch: int = 5):
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self._init_optimizer_state_dict = copy.deepcopy(self.optimizer.state_dict())
        self.logger = logger
        self.lr_scheduler = lr_scheduler
        if lr_scheduler is not None:
            self._lr_init_state_dict = copy.deepcopy(lr_scheduler.state_dict())

        self.torch_device = torch_device
        self._iteration_training = 0
        self.val_every_x_epoch = val_every_x_epoch
        self.use_train_stopping = use_train_stopping
        self.train_stopping_threshold: float = train_stopping_threshold
        assert 0 <= train_stopping_threshold <= 1

    def train(self, ds_manager: IterationDatasetView):
        self.pre_train_process(ds_manager)
        train_dataloader = ds_manager.get_train_dl()
        unlabelled_dataloader = ds_manager.get_unlabelled_dl()
        val_dataloader = ds_manager.get_val_dl()
        dict_statistics = MetricAccumulator()
        for i in range(1, self.epochs + 1):
            start_time = time.perf_counter()
            self.model.train()
            print(f"Train Epoch {i:3} / {self.epochs:3}", end="")
            training_statistics_epoch = self.train_epoch(i, train_dataloader, unlabelled_dataloader, ds_manager)
            needed_train_time = time.perf_counter() - start_time
            print(f" | Acc: {training_statistics_epoch['acc'] * 100.0:5.2f} | Loss: {training_statistics_epoch['loss']:8.4f} | Time {needed_train_time:11.3f} sec", end="")
            dict_statistics.update_dict({"e": i, **training_statistics_epoch})

            if i % self.val_every_x_epoch == 0 and len(val_dataloader) > 0:
                start_time = time.perf_counter()
                self.model.eval()
                validation_statistics_epoch = self.evaluate(val_dataloader)
                needed_validation_time = time.perf_counter() - start_time
                print(f" | VAL Acc: {validation_statistics_epoch['acc'] * 100.0:5.2f} | VAL Loss: {validation_statistics_epoch['loss']:8.4f} | VAL Time {needed_validation_time:11.3f} sec")
                dict_statistics.update_dict({"val_e": i, **validation_statistics_epoch})
            else:
                print(f" | skip validation ")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.use_train_stopping:
                if "acc" not in training_statistics_epoch:
                    raise RuntimeWarning("The accuracy has not been calculated but use_train_stopping is True")
                else:
                    if training_statistics_epoch["acc"] > self.train_stopping_threshold:  # Followin same principle as Jordan T. Ash with
                        print(f"Early Stopping triggered due to Train Acc > {self.train_stopping_threshold}")
                        break

        self.logger.log_dictionary(self._iteration_training, "stats_training", dict_statistics.accumulation_dict)
        self.post_train_process(ds_manager)
        self._iteration_training += 1

    def pre_train_process(self, ds_manager: IterationDatasetView):
        # self.optimizer.load_state_dict(self.__init_optimizer_state_dict) TODO think whether it is good to set this by default?
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(self._lr_init_state_dict)

    @abstractmethod
    def train_epoch(self, epoch: int, train_dl: DataLoader, unlabelled_dl: DataLoader, ds_manager: IterationDatasetView) -> dict:
        pass

    def post_train_process(self, ds_manager: IterationDatasetView):
        pass

    @abstractmethod
    @torch.inference_mode
    def evaluate(self, eval_dl: DataLoader, return_embeddings: bool = False):
        pass

    @abstractmethod
    def _calculate_loss(self, prediction: torch.Tensor, target: torch.Tensor, meta: Optional[dict] = None, is_train: bool = True) -> torch.Tensor:
        pass
