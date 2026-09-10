import torch
from typing import List

from utilities.util_layer import get_activation_function_by_name


class MLP(torch.nn.Module):
    def __init__(self, input_dimension: int, layer_sizes: List[int], hidden_activation_name: str, output_activation_name: str,
                 dropout_factor: float = 0.0, with_bn_norm: bool = False, ):
        super(MLP, self).__init__()
        self.input_dimension = input_dimension
        self.layer_sizes = layer_sizes
        self.hidden_activation_name = hidden_activation_name
        self.output_activation_name = output_activation_name
        self.with_bn_norm = with_bn_norm
        assert len(layer_sizes) > 0, "The MLP sequence must at least contain a single layer"

        layers = []
        latest_in_dimension = input_dimension
        for i, l_size in enumerate(layer_sizes):
            if dropout_factor > 0.0:
                layers.append(torch.nn.Dropout(p=dropout_factor))

            layers.append(torch.nn.Linear(latest_in_dimension, l_size))
            if i == len(layer_sizes) - 1:
                layers.append(get_activation_function_by_name(output_activation_name))
            else:
                if with_bn_norm:
                    layers.append(torch.nn.BatchNorm1d(l_size))

                layers.append(get_activation_function_by_name(hidden_activation_name))
            latest_in_dimension = l_size
        self.mlp_sequence = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp_sequence(x)
