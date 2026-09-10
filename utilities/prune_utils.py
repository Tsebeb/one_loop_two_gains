import copy
import torch.nn
from timm.layers.grn import GlobalResponseNorm
from torch import Tensor
from torch.nn.modules.batchnorm import _NormBase
from torch.nn.utils.prune import remove, identity, custom_from_mask, is_pruned
from typing import List, Tuple
from torch.nn import Module, Parameter, LayerNorm, MultiheadAttention
import numpy as np


def get_prune_list(*base_modules:  Module) -> List[Tuple[Module, str]]:
    list_prune = []
    for mod in base_modules:
        for name, submod in mod.named_modules():
            if isinstance(submod, _NormBase) or isinstance(submod, LayerNorm) or isinstance(submod, GlobalResponseNorm):
                continue
            if hasattr(submod, 'weight') and isinstance(submod.weight, Parameter):
                list_prune.append((submod, "weight"))
            if hasattr(submod, 'bias') and isinstance(submod.bias, Parameter):
                list_prune.append((submod, "bias"))
            if hasattr(submod, "in_proj_weight") and isinstance(submod.in_proj_weight, Parameter):
                list_prune.append((submod, "in_proj_weight"))
            if hasattr(submod, "q_proj_weight") and isinstance(submod.q_proj_weight, Parameter):
                list_prune.append((submod, "q_proj_weight"))
            if hasattr(submod, "k_proj_weight") and isinstance(submod.k_proj_weight, Parameter):
                list_prune.append((submod, "k_proj_weight"))
            if hasattr(submod, "v_proj_weight") and isinstance(submod.v_proj_weight, Parameter):
                list_prune.append((submod, "v_proj_weight"))
    return list_prune


def get_current_prune_mask(base_model: torch.nn.Module) -> List[Tuple[Module, str, Tensor]]:
    list_cur_mask = []
    for name, mod in base_model.named_modules():
        if is_pruned(mod):
            if hasattr(mod, 'weight') and hasattr(mod, 'weight_mask'):
                list_cur_mask.append((mod, "weight", mod.weight_mask.detach().clone()))
            if hasattr(mod, 'bias') and hasattr(mod, 'bias_mask'):
                list_cur_mask.append((mod, "bias", mod.bias_mask.detach().clone()))
            if hasattr(mod, "in_proj_weight") and hasattr(mod, 'in_proj_weight_mask'):
                list_cur_mask.append((mod, "in_proj_weight", mod.in_proj_weight_mask.detach().clone()))
            if hasattr(mod, "q_proj_weight") and hasattr(mod, 'q_proj_weight_mask'):
                list_cur_mask.append((mod, "q_proj_weight", mod.q_proj_weight_mask.detach().clone()))
            if hasattr(mod, "k_proj_weight") and hasattr(mod, 'k_proj_weight_mask'):
                list_cur_mask.append((mod, "k_proj_weight", mod.k_proj_weight_mask.detach().clone()))
            if hasattr(mod, "v_proj_weight") and hasattr(mod, 'v_proj_weight_mask'):
                list_cur_mask.append((mod, "v_proj_weight", mod.v_proj_weight_mask.detach().clone()))
    return list_cur_mask


def remove_all_pruning_hooks(module_helper: List[Tuple[Module, str]]):
    for mod, name in module_helper:
        remove(mod, name)


def apply_pruning_structure(module_helper: List[Tuple[Module, str]]):
    for mod, name in module_helper:
        identity(mod, name)


def apply_pruning_mask(mask_helper: List[Tuple[Module, str, Tensor]]):
    for mod, name, mask in mask_helper:
        custom_from_mask(mod, name, mask.detach().clone())


def count_parameters_masked(module: torch.nn.Module) -> Tuple[int, int, float]:
    total_parameters_to_prune = 0
    total_parameters_pruned = 0

    for name, mod in module.named_modules():
        if is_pruned(mod):
            for parameter_name, mask_name in [("weight", "weight_mask"), ("bias", "bias_mask"), ("in_proj_weight", "in_proj_weight_mask"),
                                              ("q_proj_weight", "q_proj_weight_mask"), ("k_proj_weight", "k_proj_weight_mask"), ("v_proj_weight", "v_proj_weight_mask")]:
                if hasattr(mod, parameter_name) and hasattr(mod, mask_name):
                    mask_param = getattr(mod, mask_name)
                    total_parameters = np.prod(mask_param.size())
                    active_parameters = torch.sum(mask_param).item()
                    total_parameters_to_prune += total_parameters
                    total_parameters_pruned += (total_parameters - active_parameters)

    total_parameters_to_prune = int(total_parameters_to_prune)
    total_parameters_pruned = int(total_parameters_pruned)
    if total_parameters_to_prune > 0:
        fraction = total_parameters_pruned / total_parameters_to_prune
        return total_parameters_to_prune, total_parameters_pruned, fraction
    else:
        return 0, 0, -1.0


def split_params_helper(params_helper: List[Tuple[Module, str]]) -> Tuple[List[Tuple[Module, str]], List[Tuple[Module, str]]]:
    weight_list = []
    bias_list = []

    for mod, name, mask in params_helper:
        if "bias" in name:
            bias_list.append((mod, name))
        else:
            weight_list.append((mod, name))
    return weight_list, bias_list
