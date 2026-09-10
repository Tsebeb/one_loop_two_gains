import os
import time
from typing import Optional, Callable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from logger.logger_interface import LoggerInterface
from networks.CLmodel import ContinualLearningModelWrapper
from pretrain_strategy.base_pretrain_strategy import BasePretrainStrategy
from torchvision.transforms.v2 import Transform

from pretrain_strategy.memory_bank import MemoryBank
from utilities.average_metric import AverageMeter
from utilities.metric_accumulator import MetricAccumulator


class SimCLRPretraining(BasePretrainStrategy):
    def __init__(self, contrastive_head: torch.nn.Module, temperature: float, projection_dimension: int, base_transform: Transform, augment_transform: Transform,
                 test_train_transform: Transform, model: ContinualLearningModelWrapper, optimizer: Callable, logger: LoggerInterface, epochs: int, lr_scheduler: Optional[Callable], torch_device):
        hydra_optimizer: Optimizer = optimizer(params=list(model.parameters()) + list(contrastive_head.parameters()))
        if lr_scheduler is not None:
            hydra_scheduler: Optional[LRScheduler] = lr_scheduler(optimizer=hydra_optimizer)
        else:
            hydra_scheduler: Optional[LRScheduler] = None
        super().__init__(model, hydra_optimizer, logger, epochs, hydra_scheduler, torch_device)
        self.contrastive_head = contrastive_head
        self.temperature = temperature
        self.projection_dimension = projection_dimension
        self.base_transform = base_transform
        assert isinstance(self.base_transform, Transform)
        self.augment_transform = augment_transform
        assert isinstance(self.augment_transform, Transform)
        self.test_train_transform = test_train_transform
        assert isinstance(self.test_train_transform, Transform)

    def _calc_simclr_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]
        output:
            - loss: loss computed according to SimCLR
        """
        b, n, dim = features.size()
        assert (n == 2)
        mask = torch.eye(b, dtype=torch.float32).to(self.torch_device, non_blocking=True)
        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature

        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()
        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).to(self.torch_device), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()
        return loss

    def _train_simclr_epoch(self, train_dl: DataLoader) -> dict:
        self.model.train()
        self.model.to(self.torch_device, non_blocking=True)
        self.contrastive_head.train()
        self.contrastive_head.to(self.torch_device, non_blocking=True)

        loss_tracker = AverageMeter()
        for image_batch, _ in train_dl:
            image_batch = image_batch.to(self.torch_device, non_blocking=True)
            image_base = self.base_transform(image_batch)
            image_augment = self.augment_transform(image_batch)

            b, c, h, w = image_batch.size()
            input_ = torch.cat([image_base.unsqueeze(1), image_augment.unsqueeze(1)], dim=1)
            input_ = input_.view(-1, c, h, w)
            embeddings = self.model.get_embedding(input_)
            output = self.contrastive_head(embeddings)
            output = torch.nn.functional.normalize(output, dim=1)
            output = output.view(b, 2, -1)
            loss = self._calc_simclr_loss(output)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_tracker.update(loss.item(), b)

        train_statistics = {"loss": loss_tracker.avg}
        return train_statistics

    @torch.inference_mode()
    def contrastive_evaluate(self, test_dl: DataLoader, memory_bank: MemoryBank) -> dict:
        acc_tracker = AverageMeter()
        self.model.eval()
        self.model.to(self.torch_device, non_blocking=True)
        self.contrastive_head.eval()
        self.contrastive_head.to(self.torch_device, non_blocking=True)

        for image, target in test_dl:
            image = image.to(self.torch_device, non_blocking=True)
            target = target.to(self.torch_device, non_blocking=True)

            embedding = self.model.get_embedding(image)
            output = self.contrastive_head(embedding)
            output = torch.nn.functional.normalize(output, dim=1)
            output = memory_bank.weighted_knn(output)

            acc1 = torch.mean(torch.eq(output, target).float())
            acc_tracker.update(acc1.item(), image.size(0))

        return {"knn_acc": acc_tracker.avg}

    @torch.inference_mode()
    def _fill_memory_bank(self, train_dl: DataLoader, memory_bank_base: MemoryBank):
        memory_bank_base.reset()
        self.model.eval()
        self.model.to(self.torch_device, non_blocking=True)
        self.contrastive_head.eval()
        self.contrastive_head.to(self.torch_device, non_blocking=True)

        for image_batch, target in train_dl:
            image_test = image_batch.to(self.torch_device, non_blocking=True)
            target = target.to(self.torch_device, non_blocking=True)
            image_test = self.test_train_transform(image_test)

            embedding = self.model.get_embedding(image_test)
            projection = self.contrastive_head(embedding)
            projection = torch.nn.functional.normalize(projection, dim=1)
            memory_bank_base.update(projection, embedding, target)

    def pretrain(self, train_dl: DataLoader, test_dl: DataLoader) -> dict:
        memory_bank = MemoryBank(n=len(train_dl.dataset),
                                 dim=self.projection_dimension,
                                 temperature=self.temperature,
                                 feature_dim=self.model.get_embedding_dim(),
                                 num_classes=self.model.get_num_classes(),
                                 torch_device=self.torch_device)

        dict_statistics = MetricAccumulator()
        for i in range(self.epochs):
            print(f"Train Epoch {i:3} / {self.epochs:3}", end="")
            start_time = time.perf_counter()
            training_statistics_epoch = self._train_simclr_epoch(train_dl)
            needed_train_time = time.perf_counter() - start_time
            print(f" | Loss: {training_statistics_epoch['loss']:8.4f} | Time {needed_train_time:11.3f} sec", end="")
            dict_statistics.update_dict({"e": i, **training_statistics_epoch})

            # Fill memory bank
            if len(test_dl) > 0:
                start_time = time.perf_counter()
                self._fill_memory_bank(train_dl, memory_bank)
                validation_statistics_epoch = self.contrastive_evaluate(test_dl, memory_bank)
                needed_validation_time = time.perf_counter() - start_time
                print(f" | VAL KNN Acc: {validation_statistics_epoch['knn_acc'] * 100.0:5.2f} | VAL Time {needed_validation_time:11.3f} sec")
                dict_statistics.update_dict({"val_e": i, **validation_statistics_epoch})
            else:
                print(f" | skip validation ")

        return dict_statistics.get_dict_repr()

    def save_checkpoint(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        opt_path = os.path.join(save_dir, "optimizer_state_dict.pth")
        model_path = os.path.join(save_dir, "model_state_dict.pth")
        project_path = os.path.join(save_dir, "project_state_dict.pth")
        lr_sched_path = os.path.join(save_dir, "scheduler_state_dict.pth")

        torch.save(self.optimizer.state_dict(), opt_path)
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.contrastive_head.state_dict(), project_path)
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), lr_sched_path)


