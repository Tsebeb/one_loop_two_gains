import torch.nn
from torch import nn


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


def get_activation_function_by_name(activation_name: str,):
    match activation_name.lower():
        case 'relu':
            return torch.nn.ReLU()
        case "swish":
            return torch.nn.SiLU()
        case "sigmoid":
            return torch.nn.Sigmoid()
        case "tanh":
            return torch.nn.Tanh()
        case 'identity':
            return torch.nn.Identity()
        case "gelu":
            return torch.nn.GELU()
        case _:
            raise RuntimeError(f"Could not resolve activation name {activation_name}. Please check spelling or add it here ...")

