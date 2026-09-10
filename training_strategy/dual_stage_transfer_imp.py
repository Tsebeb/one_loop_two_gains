import copy
import os
import time
from typing import Optional

import torch
from torch.nn.utils.prune import is_pruned, global_unstructured, L1Unstructured
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from dataset.iteration_dataset_view import IterationDatasetView
from logger.logger_interface import LoggerInterface
from networks.CLmodel import ContinualLearningModelWrapper
from training_strategy.fine_tuning import FineTuning
from utilities.metric_accumulator import MetricAccumulator
from utilities.prune_utils import remove_all_pruning_hooks, get_current_prune_mask, apply_pruning_mask, count_parameters_masked


class DualStageTransferBase(FineTuning):
    """
    This Method performs an LTH circuit for each round of training in an step
    """

    def __init__(self, model: ContinualLearningModelWrapper, optimizer: Optimizer, logger: LoggerInterface, epochs: int,
                 lr_scheduler: LRScheduler, torch_device, final_sparsity: float, prune_per_round: float, rewind_epoch: Optional[int] = None, use_train_stopping: bool = False,
                 train_stopping_threshold: float = 0.99, ):
        super().__init__(model, optimizer, logger, epochs, lr_scheduler, torch_device, use_train_stopping, train_stopping_threshold)
        self._rewind_state_dict = copy.deepcopy(model.state_dict())
        self._rewind_state_dict_set = False
        self._rewind_is_pruned = False
        self._rewind_epoch = rewind_epoch

        self.final_sparsity = final_sparsity
        self.prune_per_round = prune_per_round

        if self._rewind_epoch is None:
            self._rewind_state_dict_set = True
        self._model_structure_helper = self.model.get_named_backbone_parameters()
        self.__transfer_mask = None

    def pre_train_process(self, ds_manager: IterationDatasetView):
        super().pre_train_process(ds_manager)
        if is_pruned(self.model):
            remove_all_pruning_hooks(self._model_structure_helper)
        self.model.load_state_dict(self._rewind_state_dict)
        self.optimizer.load_state_dict(self._init_optimizer_state_dict)

    def train_epoch(self, epoch: int, train_dl: DataLoader, unlabelled_dl: DataLoader, ds_manager: IterationDatasetView) -> dict:
        return_val = super().train_epoch(epoch, train_dl, unlabelled_dl, ds_manager)
        if self._rewind_epoch is not None and not self._rewind_state_dict_set and epoch == self._rewind_epoch:
            self._rewind_is_pruned = is_pruned(self.model)
            self._rewind_state_dict_set = True
            self._rewind_state_dict = copy.deepcopy(self.model.state_dict())
        return return_val

    def _sub_train_model(self, train_dataloader, unlabelled_dataloader: DataLoader, ds_manager: IterationDatasetView) -> MetricAccumulator:
        self.optimizer.load_state_dict(self._init_optimizer_state_dict)
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(self._lr_init_state_dict)

        if not is_pruned(self.model):
            sparsity = 0.0
        else:
            _, _, sparsity = count_parameters_masked(self.model)

        print(f"\n\nTraining sparsity: {sparsity * 100:5.2f}%\n\n")
        dict_statistics = MetricAccumulator()
        triggered_once = False
        i = 0
        for i in range(1, self.epochs + 1):
            start_time = time.perf_counter()
            self.model.train()
            print(f"Train Epoch {i:3} / {self.epochs:3}", end="")
            training_statistics_epoch = self.train_epoch(i, train_dataloader, unlabelled_dataloader, ds_manager)
            needed_train_time = time.perf_counter() - start_time
            print(f" | Acc: {training_statistics_epoch['acc'] * 100.0:5.2f} | Loss: {training_statistics_epoch['loss']:8.4f} | Time {needed_train_time:11.3f} sec")
            dict_statistics.update_dict({"e": i, **training_statistics_epoch})

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.use_train_stopping:
                if "acc" not in training_statistics_epoch:
                    raise RuntimeWarning("The accuracy has not been calculated but use_train_stopping is True")
                else:
                    if training_statistics_epoch["acc"] > self.train_stopping_threshold:  # Followin same principle as Jordan T. Ash with
                        if triggered_once:
                            return dict_statistics
                        triggered_once = True
        dict_statistics.update_dict({"total_train_epochs": i,})
        return dict_statistics

    def _do_imp_loop(self, train_dataloader, unlabelled_dataloader, ds_manager, val_dataloader) -> MetricAccumulator:
        os.makedirs(f"imp_trans_models", exist_ok=True)
        cur_train_metrics = self._sub_train_model(train_dataloader, unlabelled_dataloader, ds_manager)
        self.logger.log_dictionary(self._iteration_training, f"stats_training_round_0", cur_train_metrics.get_dict_repr())
        best_eval_statistics = self.evaluate(val_dataloader)
        self.logger.log_dictionary(self._iteration_training, f"val_round_0", best_eval_statistics)
        torch.save(self.model.state_dict(), os.path.join(f"imp_trans_models", f"sparsity_{0.0:.5f}.pth"))

        active_parameters = 1.0
        num_rounds = 0
        while active_parameters > self.final_sparsity:
            num_rounds += 1
            active_parameters *= (1-self.prune_per_round)

        for i in range(num_rounds):
            print(f"Pruning round {i + 1} of {num_rounds}")
            _, _, sparsity = count_parameters_masked(self.model)
            prune_amount_until_target_sparsity = 1 - (self.final_sparsity / sparsity)
            current_prune_amount = min(self.prune_per_round, prune_amount_until_target_sparsity)
            global_unstructured(self._model_structure_helper, L1Unstructured, amount=current_prune_amount)
            _, _, sparsity = count_parameters_masked(self.model)

            # Masked Reset for initialization ...
            current_mask = get_current_prune_mask(self.model)
            if self._rewind_is_pruned:
                self.model.load_state_dict(self._rewind_state_dict)
                remove_all_pruning_hooks(self._model_structure_helper)
                apply_pruning_mask(current_mask)
            else:
                remove_all_pruning_hooks(self._model_structure_helper)
                self.model.load_state_dict(self._rewind_state_dict)
                apply_pruning_mask(current_mask)

            # Train Model ...
            cur_train_metrics = self._sub_train_model(train_dataloader, unlabelled_dataloader, ds_manager)
            self.logger.log_dictionary(self._iteration_training, f"stats_training_round_{i + 1}", cur_train_metrics.get_dict_repr())
            torch.save(self.model.state_dict(), os.path.join(f"imp_trans_models", f"sparsity_{sparsity:.5f}.pth"))

        self.__transfer_mask = get_current_prune_mask(self.model)
        return cur_train_metrics

    def train(self, ds_manager: IterationDatasetView):
        self.pre_train_process(ds_manager)
        train_dataloader = ds_manager.get_train_dl()
        unlabelled_dataloader = ds_manager.get_unlabelled_dl()
        val_dataloader = ds_manager.get_val_dl()
        if self.__transfer_mask is None: # Do Full IMP based on the first round ...
            best_train_metric = self._do_imp_loop(train_dataloader, unlabelled_dataloader, ds_manager, val_dataloader)
        else:
            if is_pruned(self.model):
                remove_all_pruning_hooks(self._model_structure_helper)
            apply_pruning_mask(self.__transfer_mask)
            best_train_metric = self._sub_train_model(train_dataloader, unlabelled_dataloader, ds_manager)

        total_params, currently_pruned, fraction = count_parameters_masked(self.model)
        # To have same key names ...
        self.logger.log_value(self._iteration_training, "total_params_2nd_stage", total_params)
        self.logger.log_value(self._iteration_training, "pruned_params_2nd_stage", currently_pruned)
        self.logger.log_value(self._iteration_training, "fractions_2nd_stage", fraction)

        self.logger.log_dictionary(self._iteration_training, "stats_training", best_train_metric.get_dict_repr())
        self.post_train_process(ds_manager)
        self._iteration_training += 1

