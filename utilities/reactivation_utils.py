import math

import torch
from typing import List, Tuple, Optional
from torch import Tensor
from torch.nn import Module


@torch.no_grad()
def build_mask_history_sum(current_mask: List[Tuple[Module, str, Tensor]], mask_aggregate: Optional[List[Tuple[Module, str, Tensor]]]):
    if mask_aggregate is None:
        mask_aggregate = []
        for l, name, mask in current_mask:
            mask_aggregate.append((l, name, torch.zeros_like(mask)))

    assert len(current_mask) == len(mask_aggregate), "Mask aggregation must be the same length"
    for i in range(len(current_mask)):
        l, name, weight = current_mask[i]
        a_l, a_name, a_weight = mask_aggregate[i]
        assert name == mask_aggregate[i][1], "name must be the same"
        assert l == mask_aggregate[i][0], "torch module must be the same"
        a_weight += weight
    return mask_aggregate


@torch.no_grad()
def build_mask_history_with_decay(current_mask, mask_aggregate: Optional, decay_rate: float = 0.8):
    if mask_aggregate is None:
        mask_aggregate = []
        for l, name, mask in current_mask:
            mask_aggregate.append((l, name, torch.zeros_like(mask)))

    assert len(current_mask) == len(mask_aggregate), "Mask aggregation must be the same length"
    for i in range(len(current_mask)):
        l, name, weight = current_mask[i]
        a_l, a_name, a_weight = mask_aggregate[i]
        assert name == mask_aggregate[i][1], "name must be the same"
        assert l == mask_aggregate[i][0], "torch module must be the same"
        a_weight *= decay_rate
        a_weight += weight
    return mask_aggregate


@torch.no_grad()
def reactivate_mask_by_weighted_aggregate(current_mask: List[Tuple[Module, str, torch.Tensor]], mask_aggregate: List[Tuple[Module, str, torch.Tensor]], percentage: float) -> List[Tuple[Module, str, torch.Tensor]]:
    total_parameters = 0
    active_parameters = 0
    for l, name, weight in current_mask:
        total_parameters += weight.numel()
        active_parameters += weight.sum().item()

    total_parameters_to_reactivate = math.floor(total_parameters * percentage)
    assert (total_parameters - active_parameters) > total_parameters_to_reactivate, "Total masked parameters must be greater than the total parameters to reactivate"

    # Flatten all masks and aggregate weights into contiguous tensors
    flat_masks = []
    flat_agg_weights = []
    for i in range(len(current_mask)):
        flat_masks.append(current_mask[i][2].flatten().clone())
        flat_agg_weights.append(mask_aggregate[i][2].flatten().float())

    all_masks = torch.cat(flat_masks)
    all_agg = torch.cat(flat_agg_weights)

    # Elements with mask == 1 are exempt, as they are already active
    candidate_indices = torch.where(all_masks == 0)[0]
    candidate_weights = all_agg[candidate_indices]

    # Inverse weighting: aggregate sum - higher values more often active add 1.0 for residual and chance for all ...
    weights = 1.0 + candidate_weights
    probabilities = weights / weights.sum()

    # Sample parameters to reactivate without replacement
    num_to_reactivate = min(total_parameters_to_reactivate, candidate_indices.shape[0])
    selected_local = torch.multinomial(probabilities, num_to_reactivate, replacement=False)
    selected_global = candidate_indices[selected_local]

    # Reactivate: set mask from 0 (masked) to 1 (active)
    all_masks[selected_global] = 1

    # Reconstruct per-layer masks with original shapes
    new_mask = []
    offset = 0
    for l, name, mask in current_mask:
        numel = mask.numel()
        new_mask_tensor = all_masks[offset:offset + numel].reshape(mask.shape)
        new_mask.append((l, name, new_mask_tensor))
        offset += numel

    return new_mask


@torch.no_grad()
def reactivate_connections_random_structured(current_mask, percentage: float):
    assert current_mask is not None, "Reactivation can only happen when pruned at least once ... "
    for module, name, mask in current_mask:
        mask_view = mask.view(-1)
        mask_numel = mask_view.numel()
        reactivate_params = math.floor(mask_numel * percentage)
        if reactivate_params > 0:
            zero_idcs = (mask_view == 0).nonzero(as_tuple=True)[0]
            if len(zero_idcs) > 0:
                k = min(reactivate_params, len(zero_idcs))
                perm = torch.randperm(len(zero_idcs), device=mask.device)[:k]
                chosen = zero_idcs[perm]
                mask_view[chosen] = 1
    return current_mask