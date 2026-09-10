import copy
import math
import os
import time

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import List, Optional, Tuple, Union
from torch.nn import Module
from torch import Tensor
import torch
from torch.utils.data import DataLoader

from dataset.iteration_dataset_view import IterationDatasetView
from logger.logger_interface import LoggerInterface
from networks.CLmodel import ContinualLearningModelWrapper
from training_strategy.fine_tuning import FineTuning
from torch.nn.utils.prune import is_pruned, global_unstructured, L1Unstructured, RandomUnstructured

from utilities.metric_accumulator import MetricAccumulator
from utilities.prune_utils import remove_all_pruning_hooks, apply_pruning_mask, count_parameters_masked, get_current_prune_mask, apply_pruning_structure
from utilities.reactivation_utils import build_mask_history_sum, build_mask_history_with_decay, reactivate_mask_by_weighted_aggregate, reactivate_connections_random_structured


class DualStageRetrain(FineTuning):
    def __init__(self, model: ContinualLearningModelWrapper, optimizer: Optimizer, logger: LoggerInterface, epochs: int,
                 lr_scheduler: LRScheduler, torch_device, target_sparsity: float, prune_per_round: Union[float, List[float]], prunning_method: str,
                 rewind_epoch: Optional[int] = None, use_train_stopping: bool = False, train_stopping_threshold: float = 0.99,
                 reactivation_percentage: float = 0.0, reactivation_method: str = "random_structured", patience_epochs: int = 10,
                 prune_at_start: bool = False, warmup_epochs_head_only: Optional[int] = None, warmup_head_lr: float = 1e-3):
        super().__init__(model, optimizer, logger, epochs, lr_scheduler, torch_device, use_train_stopping, train_stopping_threshold)
        self._internal_model_state_dict = copy.deepcopy(model.state_dict())

        self._iteration_ctr = -1
        self._rewind_epoch = rewind_epoch # After x epochs the rewinding point should is taken
        self._rewind_set = False
        self.patience_epochs = patience_epochs
        self.prune_at_start = prune_at_start
        self.warmup_epochs_head_only = warmup_epochs_head_only
        self.warmup_head_lr = warmup_head_lr

        if self.warmup_epochs_head_only is not None:
            self.epochs += self.warmup_epochs_head_only  # adjust epochs ...

        if self.prune_at_start:
            # Setting all the patience epochs to the the 2nd phase make it equivalent ...
            self.patience_epochs += self.epochs
            self.epochs = 0
            assert prunning_method in ["random_unstructured", "none"]

        self.reactivation_method = reactivation_method
        assert self.reactivation_method in ["random_structured", "sum_weighted_unstructured", "decay_weighted_unstructured"]
        self._reactivate_aggregation_mask = None

        assert 0.0 < prune_per_round < 1.0, "The pruning amount per round needs to be between 0.0 and 1.0."
        assert isinstance(prune_per_round, float), "The pruning amount needs to be a float"
        assert 0.0 < target_sparsity < 1.0, "The max_prune_ratio needs to be between 0.0 and 1.0."
        assert isinstance(target_sparsity, float), "The max_prune_ratio needs to be a float"

        self._target_sparsity = target_sparsity
        self._prune_per_round = prune_per_round
        assert not is_pruned(self.model), "The backbone extraction fails if that is the case ... "
        self._named_backbone_params: List[Tuple[Module, str]] = model.get_named_backbone_parameters()
        self._current_prune_mask: Optional[List[Tuple[Module, str, Tensor]]] = None
        self._reactivation_percentage = reactivation_percentage

        if prunning_method == "l1_unstructured":
            self._prune_method = L1Unstructured
        elif prunning_method == "random_unstructured":
            self._prune_method = RandomUnstructured
        elif prunning_method == "none":
            self._prune_method = None
        else:
            raise NotImplementedError(f"Implementation of pruning method {prunning_method} still missing ... ")

    def pre_train_process(self, ds_manager: IterationDatasetView):
        super().pre_train_process(ds_manager)
        self.optimizer.load_state_dict(self._init_optimizer_state_dict)
        self._iteration_ctr += 1

        if is_pruned(self.model):
            current_mask = get_current_prune_mask(self.model)

            if self._reactivation_percentage > 0.0:
                if self.reactivation_method == "random_structured":
                    adjusted_mask = reactivate_connections_random_structured(current_mask, self._reactivation_percentage)
                elif self.reactivation_method == "sum_weighted_unstructured":
                    self._reactivate_aggregation_mask = build_mask_history_sum(current_mask, self._reactivate_aggregation_mask)
                    adjusted_mask = reactivate_mask_by_weighted_aggregate(current_mask, self._reactivate_aggregation_mask, self._reactivation_percentage)
                elif self.reactivation_method == "decay_weighted_unstructured":
                    self._reactivate_aggregation_mask = build_mask_history_with_decay(current_mask, self._reactivate_aggregation_mask)
                    adjusted_mask = reactivate_mask_by_weighted_aggregate(current_mask, self._reactivate_aggregation_mask, self._reactivation_percentage)
                else:
                    raise NotImplementedError("Please implement the reactivation strategy here ... ")
            else:
                adjusted_mask = current_mask  # Nothing to change on a mask basis ...

            remove_all_pruning_hooks(self._named_backbone_params)
            self.model.load_state_dict(self._internal_model_state_dict)
            apply_pruning_mask(adjusted_mask)
        else:
            self.model.load_state_dict(self._internal_model_state_dict)

        total_params, currently_pruned, fraction = count_parameters_masked(self.model)

        os.makedirs("models", exist_ok=True)
        model_save_dir = os.path.join("models", f"start_iter_{self._iteration_ctr:02d}_model_state_dict.pth")
        torch.save(self.model.state_dict(), model_save_dir)

        self.logger.log_value(self._iteration_training, "total_params_1st_stage", total_params)
        self.logger.log_value(self._iteration_training, "pruned_params_1st_stage", currently_pruned)
        self.logger.log_value(self._iteration_training, "fractions_1st_stage", fraction)

    def train_epoch(self, epoch: int, train_dl: DataLoader, unlabelled_dl: DataLoader, ds_manager: IterationDatasetView) -> dict:
        return_value = super().train_epoch(epoch, train_dl, unlabelled_dl, ds_manager)
        if self._rewind_epoch is not None and epoch == self._rewind_epoch and not self._rewind_set:
            self._internal_model_state_dict = copy.deepcopy(self.model.state_dict())
            self._rewind_set = True
        return return_value

    def train(self, ds_manager: IterationDatasetView):
        self.pre_train_process(ds_manager)
        train_dataloader = ds_manager.get_train_dl()
        unlabelled_dataloader = ds_manager.get_unlabelled_dl()
        val_dataloader = ds_manager.get_val_dl()
        dict_statistics = MetricAccumulator()

        i = 0
        if self.warmup_epochs_head_only is not None and not self.prune_at_start:
            # Save original LRs and set constant warmup LR
            original_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.warmup_head_lr

            self.model.freeze_backbone()
            for i in range(1, self.warmup_epochs_head_only):
                start_time = time.perf_counter()
                self.model.train()
                print(f"Warmup Head Epoch {i:3} / {self.epochs:3}", end="")
                training_statistics_epoch = self.train_epoch(0, train_dataloader, unlabelled_dataloader, ds_manager)
                needed_train_time = time.perf_counter() - start_time
                print(f" | Acc: {training_statistics_epoch['acc'] * 100.0:5.2f} | Loss: {training_statistics_epoch['loss']:8.4f} | Time {needed_train_time:11.3f} sec")
                dict_statistics.update_dict({"e": i, **training_statistics_epoch})
            self.model.unfreeze_backbone()

            # Restore original LRs for main training
            for pg, lr in zip(self.optimizer.param_groups, original_lrs):
                pg['lr'] = lr

        for i in range(i+1, self.epochs + 1):
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
                print(
                    f" | VAL Acc: {validation_statistics_epoch['acc'] * 100.0:5.2f} | VAL Loss: {validation_statistics_epoch['loss']:8.4f} | VAL Time {needed_validation_time:11.3f} sec")
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

        # Perform 2nd stage pruning ...
        if self._prune_method is not None:
            print("Performing 2nd stage pruning ... ")
            _, _, fraction_pruned = count_parameters_masked(self.model)
            if not is_pruned(self.model):
                current_prune_amount = min(self._prune_per_round, 1-self._target_sparsity)
            else:
                prune_amount_until_target_sparsity = 1 - (self._target_sparsity / (1-fraction_pruned))
                current_prune_amount = min(self._prune_per_round, prune_amount_until_target_sparsity)

            global_unstructured(self._named_backbone_params, self._prune_method, amount=current_prune_amount)

        total_params, currently_pruned, fraction = count_parameters_masked(self.model)
        self.logger.log_value(self._iteration_training, "total_params_2nd_stage", total_params)
        self.logger.log_value(self._iteration_training, "pruned_params_2nd_stage", currently_pruned)
        self.logger.log_value(self._iteration_training, "fractions_2nd_stage", fraction)

        if self.warmup_epochs_head_only is not None and self.prune_at_start:
            # Save original LRs and set constant warmup LR
            original_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.warmup_head_lr

            self.model.freeze_backbone()
            for i in range(1, self.warmup_epochs_head_only):
                start_time = time.perf_counter()
                self.model.train()
                print(f"Warmup Head Epoch {i:3} / {self.epochs:3}", end="")
                training_statistics_epoch = self.train_epoch(0, train_dataloader, unlabelled_dataloader, ds_manager)
                needed_train_time = time.perf_counter() - start_time
                print(f" | Acc: {training_statistics_epoch['acc'] * 100.0:5.2f} | Loss: {training_statistics_epoch['loss']:8.4f} | Time {needed_train_time:11.3f} sec", end="")
                dict_statistics.update_dict({"e": i, **training_statistics_epoch})
            self.model.unfreeze_backbone()

            # Restore original LRs for main training
            for pg, lr in zip(self.optimizer.param_groups, original_lrs):
                pg['lr'] = lr

        print("Second Stage Patience training ... ")
        # perform second 2nd stage of training ...
        total_patience_epochs = i + self.patience_epochs + 1
        j = -1
        for j in range(i, total_patience_epochs):
            start_time = time.perf_counter()
            self.model.train()
            print(f"Patience Epoch {j+1:3} / {total_patience_epochs:3}", end="")
            training_statistics_epoch = self.train_epoch(i, train_dataloader, unlabelled_dataloader, ds_manager)
            needed_train_time = time.perf_counter() - start_time
            print(f" | Acc: {training_statistics_epoch['acc'] * 100.0:5.2f} | Loss: {training_statistics_epoch['loss']:8.4f} | Time {needed_train_time:11.3f} sec", end="")
            dict_statistics.update_dict({"e": i, **training_statistics_epoch})

            if i % self.val_every_x_epoch == 0 and len(val_dataloader) > 0:
                start_time = time.perf_counter()
                self.model.eval()
                validation_statistics_epoch = self.evaluate(val_dataloader)
                needed_validation_time = time.perf_counter() - start_time
                print(
                    f" | VAL Acc: {validation_statistics_epoch['acc'] * 100.0:5.2f} | VAL Loss: {validation_statistics_epoch['loss']:8.4f} | VAL Time {needed_validation_time:11.3f} sec")
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

        dict_statistics.update_dict({"total_train_epochs": j,})
        self.logger.log_dictionary(self._iteration_training, "stats_training", dict_statistics.accumulation_dict)
        self.post_train_process(ds_manager)
        self._iteration_training += 1
