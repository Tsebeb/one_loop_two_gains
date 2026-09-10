"""
sparse_flops.py
===============
Mask-aware FLOPs / MACs estimation for models pruned with
``torch.nn.utils.prune`` (unstructured, weight-level sparsity).

Supports pruning of both **weights** and **biases** — each is tracked
independently via a (module_id, param_name) lookup key so that non-uniform
per-parameter sparsity is handled correctly.

FLOPs convention
----------------
multiplier=1  → MACs  (1 multiply-add = 1 operation)   <- default
multiplier=2  → FLOPs (1 multiply-add = 2 operations)

Bias additions are always counted as 1 op each regardless of multiplier,
because they are additions only, not multiply-adds.

PyTorch's FlopCounterMode (used in the sanity-check function) counts
1 multiply-add as 2 FLOPs, so ``count_dense_flops_pytorch`` / 2 should
equal ``count_sparse_flops(..., multiplier=1)`` dense total.
"""

from __future__ import annotations
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Iterator

import torch
from torch import nn as nn


def print_flops_summary(
    sparse_ops: float,
    dense_ops: float,
    layer_stats: "OrderedDict[str, dict]",
    unit: str = "G",
) -> None:
    """
    Print a formatted per-layer FLOPs/MACs breakdown to stdout.

    The table separates weight and bias contributions so that the impact
    of bias pruning is immediately visible (typically < 0.1 % of total ops).

    Parameters
    ----------
    sparse_ops, dense_ops : float
        Totals returned by ``count_sparse_flops``.
    layer_stats : OrderedDict
        Per-layer dict returned by ``count_sparse_flops``.
    unit : str
        Display scale: ``"G"`` (giga, default), ``"M"`` (mega), ``"K"`` (kilo).
    top_n : int
        Print only the *top_n* layers ranked by dense_ops.
        Pass ``-1`` to print all layers.
    """
    scale = {"G": 1e9, "M": 1e6, "K": 1e3, "R": 1}.get(unit.upper(), 1e9)
    unit_label = f"{unit.upper()}MACs"

    savings = (1.0 - sparse_ops / dense_ops) * 100.0 if dense_ops > 0 else 0.0
    speedup = dense_ops / sparse_ops if sparse_ops > 0 else float("inf")

    # Overall weight sparsity
    total_w   = sum(s[0]["total_weights"]   for s in layer_stats.values())
    nonzero_w = sum(s[0]["nonzero_weights"] for s in layer_stats.values())
    overall_w_sparsity = 1.0 - nonzero_w / max(total_w, 1)

    # Overall bias sparsity (only layers that have a bias)
    total_b   = sum(s[0]["total_bias"] for s in layer_stats.values() if s[0]["total_bias"] is not None)
    nonzero_b = sum(s[0]["nonzero_bias"] for s in layer_stats.values() if s[0]["nonzero_bias"] is not None)
    overall_b_sparsity = 1.0 - nonzero_b / max(total_b, 1)

    print(f"\n{'='*96}")
    print(f"  Sparse FLOPs Summary  [{unit_label}]")
    print(f"{'='*96}")
    print(f"  Dense  total            : {dense_ops  / scale:>10.4f} {unit_label}")
    print(f"  Sparse total            : {sparse_ops / scale:>10.4f} {unit_label}")
    print(f"  FLOPs savings           : {savings:>10.2f} %")
    print(f"  Theoretical speedup     : {speedup:>10.4f} x")
    print(f"  Overall weight sparsity : {overall_w_sparsity * 100:.2f} %"
          f"  ({nonzero_w:,} / {total_w:,} nonzero)")
    print(f"  Overall bias sparsity   : {overall_b_sparsity * 100:.2f} %"
          f"  ({nonzero_b:,} / {total_b:,} nonzero)")
    print(f"{'='*96}")

    # Header row
    print(
        f"  {'Layer':<42} {'Type':<14}"
        f" {'W-Spar':>7} {'B-Spar':>7}"
        f" {'Dense':>9} {'Sparse':>9}"
        f"  {'W nnz/tot':>18}"
    )
    print("-" * 112)

    for name in layer_stats.keys():
        s = layer_stats[name]
        dense_ops = sum(x["dense_ops"] for x in s)
        sparse_ops = sum(x["sparse_ops"] for x in s)

        b_spar_str = (
            f"{s[0]['bias_sparsity'] * 100:6.1f}%"
            if s[0]["bias_sparsity"] is not None
            else "    n/a"
        )
        w_nnz_str = f"{s[0]['nonzero_weights']:,}/{s[0]['total_weights']:,}"
        print(
            f"  {name[:40]:<42} {s[0]['type']:<14}"
            f" {s[0]['weight_sparsity'] * 100:6.1f}%"
            f" {b_spar_str:>7}"
            f" {dense_ops  / scale:>9.4f}"
            f" {sparse_ops / scale:>9.4f}"
            f"  {w_nnz_str:>18}"
        )

    print("=" * 96)


def count_reference_dense_flops_pytorch(model: nn.Module, input_shape: Tuple[int, ...], device) -> int:
    """
    Count **dense** FLOPs using PyTorch's built-in
    ``torch.utils.flop_counter.FlopCounterMode`` (requires PyTorch >= 2.0).

    This function is intentionally sparsity-unaware -- it is provided as a
    sanity check that the dense baseline in ``count_sparse_flops`` matches
    the canonical PyTorch figure.

    Convention used by PyTorch: 1 multiply-add = **2 FLOPs**.
    To compare with ``count_sparse_flops(..., multiplier=1)`` (MACs):

        count_dense_flops_pytorch(m, s) / 2  ~=  dense_ops   # MACs

    Parameters
    ----------
    model : nn.Module
    input_shape : tuple[int, ...]
        Full input shape including batch dimension.

    Returns
    -------
    int
        Total dense FLOPs (multiply-adds counted as 2).
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as exc:
        raise ImportError(
            "FlopCounterMode requires PyTorch >= 2.0.  "
            "Install a recent release or use ptflops / fvcore as fallback."
        ) from exc
    model.eval()

    with torch.no_grad():
        dummy = torch.randn(*input_shape, device=device)
        with FlopCounterMode(display=False) as fcm:
            model(dummy)

    return fcm.get_total_flops()


PrunedEntry = Tuple[nn.Module, str, torch.Tensor]


def count_sparse_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    pruned_modules: List[PrunedEntry],
    device,
    multiplier: int = 1,
) -> Tuple[float, float, "OrderedDict[str, dict]"]:
    """
    Compute sparse and dense operation counts for a model pruned via
    ``torch.nn.utils.prune`` (mask-based, weights not physically removed).

    A single dummy forward pass is performed to capture output spatial
    dimensions from hooks.  Both weight AND bias sparsity are accounted
    for independently using the masks in *pruned_modules*.

    Parameters
    ----------
    model : nn.Module
        The pruned model.  All pruning pre-hooks must already be active
        so that ``module.weight`` and ``module.bias`` reflect the masked
        (zeroed) values.
    input_shape : tuple[int, ...]
        Full input shape **including** the batch dimension, e.g.
        ``(1, 3, 224, 224)`` for a single ImageNet sample.
    pruned_modules : list[tuple[nn.Module, str, torch.Tensor]]
        Your existing pruning tracker.  Each entry is
        ``(module, param_name, mask)`` where *param_name* is either
        ``"weight"`` or ``"bias"`` and *mask* is the corresponding binary
        ``torch.Tensor``.  Modules / parameters absent from this list are
        treated as fully dense (density = 1.0).
    multiplier : int, optional
        ``1``  -> report MACs  (1 multiply-add = 1 operation)  **[default]**
        ``2``  -> report FLOPs (1 multiply-add = 2 operations)
        Document clearly in your paper which convention is used to avoid
        the common factor-of-two ambiguity across pruning papers.

    Returns
    -------
    total_sparse_ops : float
        Sum of sparsity-adjusted operations across all Conv2d / Linear
        layers (weights scaled by weight density, biases scaled by bias
        density).
    total_dense_ops : float
        Sum of fully-dense operations -- identical to what any shape-based
        counter would report.
    layer_stats : OrderedDict[str, dict]
        Per-layer breakdown keyed by the name from ``model.named_modules()``.
        Each value is a dict with the following fields:

        Weight fields
          ``weight_sparsity``     fraction of zeroed weight elements  [0,1]
          ``weight_density``      fraction of nonzero weight elements [0,1]
          ``total_weights``       total number of weight elements
          ``nonzero_weights``     number of nonzero weight elements

        Bias fields  (None when the layer has no bias)
          ``bias_sparsity``       fraction of zeroed bias elements    [0,1]
          ``bias_density``        fraction of nonzero bias elements   [0,1]
          ``total_bias``          total number of bias elements
          ``nonzero_bias``        number of nonzero bias elements

        Operation counts
          ``dense_ops``           dense MACs/FLOPs + dense bias adds
          ``sparse_ops``          sparse MACs/FLOPs + sparse bias adds
          ``dense_weight_ops``    weight-only contribution to dense_ops
          ``sparse_weight_ops``   weight-only contribution to sparse_ops
          ``dense_bias_ops``      bias-only contribution  to dense_ops
          ``sparse_bias_ops``     bias-only contribution  to sparse_ops

        Layer metadata
          ``type``                "Conv2d" or "Linear"
          (Conv2d only) ``kernel``, ``groups``, ``in_channels``,
                        ``out_channels``, ``output_spatial``
          (Linear only) ``in_features``, ``out_features``, ``batch_tokens``

    Notes
    -----
    A global ``FlopCounterMode`` is run concurrently with the module hooks
    to capture **all** aten-level MACs -- including activation-level matrix
    multiplies like attention scores (``Q @ K^T``, ``attn @ V``) that do
    not belong to any ``nn.Module`` leaf.  The dense total is taken from
    this global counter, and the sparse total is derived by subtracting
    the dense Conv2d/Linear contribution and adding the sparsity-adjusted
    contribution:

        sparse_total = global_dense - hook_dense_sum + hook_sparse_sum

    This guarantees the dense baseline matches ``count_dense_flops_pytorch``
    exactly, while attention matmuls correctly appear at full cost in both
    dense and sparse totals (weight sparsity does not reduce them).
    """
    model.eval()

    # ------------------------------------------------------------------
    # Build lookup: (id(module), param_name) -> mask tensor
    # Using a compound key handles the case where the same module has
    # both a weight mask and a bias mask as separate tracker entries.
    # ------------------------------------------------------------------
    mask_lookup: Dict[Tuple[int, str], torch.Tensor] = {(id(mod), param_name): mask for mod, param_name, mask in pruned_modules}
    layer_stats: OrderedDict = OrderedDict()
    hooks: list = []

    # ------------------------------------------------------------------
    # Helpers: density and nonzero count for any (module, param_name)
    # ------------------------------------------------------------------
    def _density(module: nn.Module, param_name: str) -> float:
        """
        Return the fraction of nonzero elements for *param_name* on
        *module*.  Prefers the tracker mask; falls back to counting
        nonzeros in the live tensor for un-tracked parameters.
        """
        key = (id(module), param_name)
        if key in mask_lookup:
            return mask_lookup[key].float().mean().item()
        param: Optional[torch.Tensor] = getattr(module, param_name, None)
        if param is None or param.numel() == 0:
            return 1.0
        return (param != 0).float().sum().item() / param.numel()

    def _nonzero_count(module: nn.Module, param_name: str) -> int:
        """Return the integer count of nonzero elements for *param_name*."""
        key = (id(module), param_name)
        if key in mask_lookup:
            return int(mask_lookup[key].sum().item())
        param: Optional[torch.Tensor] = getattr(module, param_name, None)
        if param is None:
            return 0
        return int((param != 0).sum().item())

    # ------------------------------------------------------------------
    # Conv2d hook
    #
    # Dense weight MACs = Kh x Kw x (Cin / groups) x Cout x Hout x Wout
    # Each nonzero weight contributes exactly one MAC per output position,
    # so: sparse_weight_macs = dense_weight_macs x weight_density
    #
    # Dense bias adds  = Cout x Hout x Wout   (one add per output element)
    # Sparse bias adds = dense_bias_adds x bias_density
    # ------------------------------------------------------------------
    def _make_conv2d_hook(name: str):
        def hook(module: nn.Conv2d, inp, out):
            out_h, out_w = out.shape[2], out.shape[3]

            kh, kw = ((module.kernel_size[0], module.kernel_size[1]) if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size))
            c_in_per_group = module.in_channels // module.groups

            # ---- weight ops ------------------------------------------------
            dense_weight_macs = (kh * kw * c_in_per_group * module.out_channels * out_h * out_w)
            w_density          = _density(module, "weight") # sparsity ...
            sparse_weight_macs = dense_weight_macs * w_density
            dense_weight_ops  = dense_weight_macs  * multiplier
            sparse_weight_ops = sparse_weight_macs * multiplier

            # ---- bias ops --------------------------------------------------
            has_bias = module.bias is not None
            if has_bias:
                dense_bias_adds  = int(module.out_channels * out_h * out_w)
                b_density        = _density(module, "bias")
                sparse_bias_adds = dense_bias_adds * b_density
                b_total          = module.bias.numel()
                b_nonzero        = _nonzero_count(module, "bias")
            else:
                dense_bias_adds = sparse_bias_adds = 0.0
                b_density = b_total = b_nonzero    = 0

            result_dict = dict(
                # metadata
                type="Conv2d",
                kernel=(kh, kw),
                groups=module.groups,
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                output_spatial=(out_h, out_w),
                # weight sparsity
                weight_density=w_density,
                weight_sparsity=1.0 - w_density,
                total_weights=module.weight.numel(),
                nonzero_weights=_nonzero_count(module, "weight"),
                # bias sparsity  (None fields when no bias present)
                bias_density=(b_density        if has_bias else None),
                bias_sparsity=(1.0 - b_density  if has_bias else None),
                total_bias=(b_total            if has_bias else None),
                nonzero_bias=(b_nonzero        if has_bias else None),
                # op counts
                dense_weight_ops=float(dense_weight_ops),
                sparse_weight_ops=float(sparse_weight_ops),
                dense_bias_ops=float(dense_bias_adds),
                sparse_bias_ops=float(sparse_bias_adds),
                dense_ops=float(dense_weight_ops  + dense_bias_adds),
                sparse_ops=float(sparse_weight_ops + sparse_bias_adds),
            )

            if name in layer_stats:
                layer_stats[name].append(result_dict)
            else:
                layer_stats[name] = [result_dict]
        return hook

    # ------------------------------------------------------------------
    # Linear hook
    #
    # Dense weight MACs = in_features x out_features x batch_tokens
    # where batch_tokens accounts for both plain (B, Fin) inputs and
    # sequence (B, T, Fin) inputs (e.g. MaxViT-T attention projections).
    #
    # Dense bias adds  = out_features x batch_tokens
    # Sparse bias adds = dense_bias_adds x bias_density
    # ------------------------------------------------------------------
    def _make_linear_hook(name: str):
        def hook(module: nn.Linear, inp, out):
            # inp[0]: (B, Fin)  or  (B, T, Fin)
            # batch_tokens = total number of independent (in_features,) vectors
            batch_tokens = inp[0].numel() // inp[0].shape[-1]

            # ---- weight ops ------------------------------------------------
            dense_weight_macs  = module.in_features * module.out_features * batch_tokens
            w_density          = _density(module, "weight")
            sparse_weight_macs = dense_weight_macs * w_density

            dense_weight_ops  = dense_weight_macs  * multiplier
            sparse_weight_ops = sparse_weight_macs * multiplier

            # ---- bias ops --------------------------------------------------
            has_bias = module.bias is not None
            if has_bias:
                dense_bias_adds  = int(module.out_features * batch_tokens)
                b_density        = _density(module, "bias")
                sparse_bias_adds = dense_bias_adds * b_density
                b_total          = module.bias.numel()
                b_nonzero        = _nonzero_count(module, "bias")
            else:
                dense_bias_adds = sparse_bias_adds = 0.0
                b_density = b_total = b_nonzero    = 0


            result_dict = dict(
                # metadata
                type="Linear",
                in_features=module.in_features,
                out_features=module.out_features,
                batch_tokens=batch_tokens,
                # weight sparsity
                weight_density=w_density,
                weight_sparsity=1.0 - w_density,
                total_weights=module.weight.numel(),
                nonzero_weights=_nonzero_count(module, "weight"),
                # bias sparsity  (None fields when no bias present)
                bias_density=(b_density        if has_bias else None),
                bias_sparsity=(1.0 - b_density  if has_bias else None),
                total_bias=(b_total            if has_bias else None),
                nonzero_bias=(b_nonzero        if has_bias else None),
                # op counts
                dense_weight_ops=float(dense_weight_ops),
                sparse_weight_ops=float(sparse_weight_ops),
                dense_bias_ops=float(dense_bias_adds),
                sparse_bias_ops=float(sparse_bias_adds),
                dense_ops=float(dense_weight_ops  + dense_bias_adds),
                sparse_ops=float(sparse_weight_ops + sparse_bias_adds),
            )

            if name in layer_stats:
                layer_stats[name].append(result_dict)
            else:
                layer_stats[name] = [result_dict]
        return hook

    # ------------------------------------------------------------------
    # Register hooks on Conv2d and Linear submodules (sparse-aware)
    # ------------------------------------------------------------------
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(_make_conv2d_hook(name)))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(_make_linear_hook(name)))

    # ------------------------------------------------------------------
    # Single dummy forward pass with a GLOBAL FlopCounterMode running
    # concurrently.  The global counter captures everything (Conv2d,
    # Linear, attention bmm, element-wise ops, …).  Module hooks fire
    # inside the same pass and record per-layer sparse/dense breakdowns
    # for Conv2d / Linear only.
    # ------------------------------------------------------------------
    from torch.utils.flop_counter import FlopCounterMode

    with torch.no_grad():
        dummy = torch.randn(*input_shape, device=device)
        with FlopCounterMode(display=False) as fcm:
            model(dummy)

    # Remove hooks immediately -- never leave them on the model
    for h in hooks:
        h.remove()

    # ------------------------------------------------------------------
    # Global dense MACs from FlopCounterMode  (convention: 1 MAC = 2 FLOPs)
    # ------------------------------------------------------------------
    global_dense_macs = fcm.get_total_flops() / 2.0

    # ------------------------------------------------------------------
    # Sum hook-reported dense / sparse ops  (Conv2d + Linear only)
    # ------------------------------------------------------------------
    hook_dense_sum = 0.0
    hook_sparse_sum = 0.0
    for layer_stats_list in layer_stats.values():
        for usage_dict in layer_stats_list:
            hook_dense_sum += usage_dict["dense_ops"]
            hook_sparse_sum += usage_dict["sparse_ops"]

    # ------------------------------------------------------------------
    # "Other" ops: everything the global FlopCounterMode captured that
    # is NOT attributable to Conv2d / Linear weight+bias MACs.  This
    # includes attention matmuls (Q@K^T, attn@V), and any other aten-
    # level ops FlopCounterMode tracks.  These are activation-level
    # operations unaffected by weight sparsity, so sparse == dense.
    # ------------------------------------------------------------------
    other_ops = (global_dense_macs - hook_dense_sum) * multiplier

    if other_ops > 1e3:  # ignore negligible floating-point residuals
        layer_stats["<other (attn bmm, etc.)>"] = [dict(
            type="Other",
            dense_ops=float(other_ops),
            sparse_ops=float(other_ops),
            weight_sparsity=0.0,
            weight_density=1.0,
            total_weights=0,
            nonzero_weights=0,
            bias_sparsity=None,
            bias_density=None,
            total_bias=None,
            nonzero_bias=None,
            dense_weight_ops=float(other_ops),
            sparse_weight_ops=float(other_ops),
            dense_bias_ops=0.0,
            sparse_bias_ops=0.0,
        )]

    # ------------------------------------------------------------------
    # Totals
    #
    # Dense total is anchored to the global FlopCounterMode (exact).
    # Sparse total: only Conv2d/Linear ops are reduced by sparsity;
    # "other" ops (attn bmm, etc.) appear at full cost in both.
    #
    # sparse = global_dense - conv_linear_dense + conv_linear_sparse
    # ------------------------------------------------------------------
    total_dense_ops = global_dense_macs * multiplier
    total_sparse_ops = total_dense_ops - (hook_dense_sum * multiplier) + (hook_sparse_sum * multiplier)

    return total_sparse_ops, total_dense_ops, layer_stats
