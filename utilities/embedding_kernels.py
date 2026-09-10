# Implementation base on on the kernels from https://github.com/BorealisAI/uherding/blob/main/deep-al/pycls/al/herding.py
import torch
from abc import ABC, abstractmethod


class TorchSupportedKernel(ABC):
    def __init__(self, torch_device, batch_size: int):
        self.device = torch_device
        self.batch_size = batch_size

    @abstractmethod
    def compute_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement this abstract method ... ")

    def _compute_norm(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1, x2 = x1.unsqueeze(0).to(self.device), x2.unsqueeze(0).to(self.device)  # 1 x n x d, 1 x n' x d
        dist_matrix = []
        batch_round = x2.shape[1] // self.batch_size + int(x2.shape[1] % self.batch_size > 0)
        for i in range(batch_round):
            # distance comparisons are done in batches to reduce memory consumption
            x2_subset = x2[:, i * self.batch_size: (i + 1) * self.batch_size]
            dist = torch.cdist(x1, x2_subset, p=2.0)
            dist_matrix.append(dist.cpu())
            del dist

        dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
        return dist_matrix

class PosNormKernel(TorchSupportedKernel):
    def __init__(self, torch_device, batch_size: int = 512):
        super().__init__(torch_device, batch_size)
    def compute_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        dist_matrix = self._compute_norm(x1, x2)
        return dist_matrix

class NegNormKernel(TorchSupportedKernel):
    def __init__(self, torch_device, batch_size: int = 512):
        super().__init__(torch_device, batch_size)
    def compute_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        dist_matrix = self._compute_norm(x1, x2)
        return -dist_matrix

class TopHatKernel(TorchSupportedKernel):
    def __init__(self, torch_device, delta: float,  batch_size: int = 512):
        super().__init__(torch_device, batch_size)
        self.delta = delta

    def compute_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        dist_matrix = self._compute_norm(x1, x2)
        k = (dist_matrix < self.delta)
        return k

class RBFKernel(TorchSupportedKernel):
    def __init__(self, torch_device, delta: float=1.0, batch_size: int = 512):
        super().__init__(torch_device, batch_size)
        self.delta = delta

    def compute_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        norm = self._compute_norm(x1, x2)
        k = torch.exp(-1.0 * (norm / self.delta) ** 2)
        return k

class StudentTKernel(TorchSupportedKernel):
    def __init__(self, torch_device, delta: float = 1.0, beta: float = 0.5, batch_size: int = 512):
        super().__init__(torch_device, batch_size)
        self.delta = delta
        self.beta = beta

    def compute_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        norms = self._compute_norm(x1, x2)
        k = (1 + ((norms / self.delta) ** 2) / self.beta) ** (-(self.beta+1)/2)
        return k

class LaplaceKernel(TorchSupportedKernel):
    def __init__(self, torch_device, delta: float = 1.0, beta: float = 1.0, batch_size: int = 512):
        super().__init__(torch_device, batch_size)
        self.delta = delta
        self.beta = beta

    def compute_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        norms = self._compute_norm(x1, x2)
        k = torch.exp(-1 / self.delta * (norms ** self.beta))
        return k

class CauchyKernel(TorchSupportedKernel):
    def __init__(self, torch_device, batch_size: int = 512):
        super().__init__(torch_device, batch_size)

    def compute_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        norms = self._compute_norm(x1, x2)
        k =  1 / (1 + norms**2)
        return k

class RationalQuadKernel(TorchSupportedKernel):
    def __init__(self, torch_device, alpha: float=1.0, batch_size: int = 512):
        super().__init__(torch_device, batch_size)
        self.alpha = alpha

    def compute_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        norms = self._compute_norm(x1, x2)
        k = (1 + norms**2 / (2 * self.alpha))**(-self.alpha)
        return k
