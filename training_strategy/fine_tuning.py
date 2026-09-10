from typing import Optional

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader

from dataset.iteration_dataset_view import IterationDatasetView
from logger.logger_interface import LoggerInterface
from networks.CLmodel import ContinualLearningModelWrapper
from training_strategy.base_training import BaseTrainingStrategy
from utilities.average_metric import AverageMeter
from torchmetrics.classification import Accuracy, Recall, Precision, F1Score, ConfusionMatrix


class FineTuning(BaseTrainingStrategy):
    def __init__(self, model: ContinualLearningModelWrapper, optimizer: Optimizer, logger: LoggerInterface, epochs: int,
                 lr_scheduler: LRScheduler, torch_device, use_train_stopping: bool, train_stopping_threshold: float = 0.99):
        super().__init__(model, optimizer, logger, epochs, lr_scheduler, torch_device, use_train_stopping, train_stopping_threshold)

    def train_epoch(self, epoch: int, train_dl: DataLoader, unlabelled_dl: DataLoader, ds_manager: IterationDatasetView) -> dict:
        training_statistics = {}
        self.model.train()
        self.model.to(self.torch_device)

        loss_tracker = AverageMeter()
        acc_tracker = AverageMeter()
        for input_tensor, target in train_dl:
            input_tensor = input_tensor.to(self.torch_device)
            target = target.to(self.torch_device)
            pred = self.model(input_tensor)

            self.optimizer.zero_grad()
            loss = self._calculate_loss(pred, target, is_train=True)
            loss.backward()
            self.optimizer.step()

            predicted_class = torch.argmax(pred, dim=1)
            batch_acc = torch.mean(predicted_class == target, dtype=torch.float).item()
            loss_tracker.update(loss.item(), target.size(0))
            acc_tracker.update(batch_acc, target.size(0))

        training_statistics["loss"] = loss_tracker.avg
        training_statistics["acc"] = acc_tracker.avg
        return training_statistics

    @torch.inference_mode()
    def evaluate(self, eval_ds: DataLoader, return_embeddings: bool = False):
        self.model.eval()
        eval_statistics = {}
        self.model.to(self.torch_device)
        cached_embeddings = []

        loss_tracker = AverageMeter()
        num_classes = self.model.get_num_classes()
        metrics = {"acc": Accuracy("multiclass", num_classes=num_classes),
                   "recall": Recall("multiclass", num_classes=num_classes, average="macro"),
                   "precision": Precision("multiclass", num_classes=num_classes, average="macro"),
                   "f1": F1Score("multiclass", num_classes=num_classes, average="macro"),
                   "conf_mat": ConfusionMatrix("multiclass", num_classes=num_classes)
                   }

        for input_tensor, target in eval_ds:
            input_tensor = input_tensor.to(self.torch_device)
            target = target.to(self.torch_device)

            if return_embeddings:
                embedding, pred = self.model.get_embedding_and_logits(input_tensor)
                cached_embeddings.append(embedding.detach().cpu().numpy())
            else:
                pred = self.model(input_tensor)
            loss = self._calculate_loss(pred, target, is_train=False)

            predicted_class = torch.argmax(pred, dim=1)
            for _, m in metrics.items():
                m.update(predicted_class.cpu().detach(), target.cpu().detach())
            loss_tracker.update(loss.item(), target.size(0))

        eval_statistics["loss"] = loss_tracker.avg
        for k, m in metrics.items():
            value = m.compute().cpu()
            if len(value.size()) == 0:
                eval_statistics[k] = value.item()
            else:
                eval_statistics[k] = value.tolist()

        if return_embeddings:
            cached_embeddings = np.concatenate(cached_embeddings, axis=0)
            return eval_statistics, cached_embeddings
        return eval_statistics

    def _calculate_loss(self, prediction: torch.Tensor, target: torch.Tensor, meta: Optional[dict] = None, is_train: bool = True) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(prediction, target)
