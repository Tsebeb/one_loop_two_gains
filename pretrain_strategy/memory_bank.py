# Authors: Wouter Van Gansbeke, Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import torch


class MemoryBank:
    def __init__(self, n, dim, num_classes, temperature, torch_device, feature_dim=512):
        self.n = n
        self.dim = dim
        self.torch_device = torch_device
        self.features = torch.zeros((self.n, self.dim), device=self.torch_device)
        self.pre_lasts = torch.zeros((self.n, feature_dim), device=self.torch_device)
        self.targets = torch.zeros((self.n), dtype=torch.long, device=self.torch_device)
        self.ptr = 0
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C, device=self.torch_device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1, -1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, self.C), yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]
        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def reset(self):
        self.ptr = 0
        self.features.zero_()
        self.pre_lasts.zero_()
        self.targets.zero_()

    def update(self, features, pre_last, targets):
        # Ensure same device ...
        features = features.to(self.torch_device, non_blocking=True)
        pre_last = pre_last.to(self.torch_device, non_blocking=True)
        targets = targets.to(self.torch_device, non_blocking=True)

        b = features.size(0)
        assert (b + self.ptr <= self.n)
        self.features[self.ptr:self.ptr + b].copy_(features.detach())
        self.pre_lasts[self.ptr:self.ptr + b].copy_(pre_last.detach())
        self.targets[self.ptr:self.ptr + b].copy_(targets.detach())
        self.ptr += b
