"""Portable eager reference implementation of diagonal SAND-Muon.

The archived ODL files preserve the exact GPU implementation. This module is
an independent, device-agnostic PyTorch implementation for ordinary models.
"""
from __future__ import annotations

import math
import re
from typing import Callable

import torch
from torch import nn
from torch.optim import Optimizer

from .coefficients import S8_COEFFICIENTS, S8_MARGIN, S8_MULTIPLIER


def diagonal_inverse_sqrt(value: torch.Tensor) -> torch.Tensor:
    """Mean-normalized diagonal inverse square root, finite at zero signal."""
    mean = value.mean()
    tiny = torch.finfo(value.dtype).tiny
    normalized = value.clamp_min(mean * 1e-6) / mean.clamp_min(tiny)
    return normalized.clamp_min(1e-6).rsqrt()


def rank1_inverse_sqrt(second: torch.Tensor, mean: torch.Tensor):
    """Return D^-1/2 and the Sherman-Morrison rank-one correction."""
    scale = second.mean().clamp_min(torch.finfo(second.dtype).tiny)
    variance = (second - mean.square()).clamp_min(0.0)
    inverse = (variance / scale).clamp_min(1e-6).rsqrt()
    z = inverse * (mean / scale.sqrt())
    root = torch.sqrt(1.0 + z.square().sum())
    coefficient = -1.0 / (root * (root + 1.0))
    return inverse, z, coefficient


def rank1_whiten(matrix: torch.Tensor, left, right) -> torch.Tensor:
    """Apply the dense left/right rank-one whiteners without forming matrices."""
    li, lz, lc = left
    ri, rz, rc = right
    value = matrix * li[..., :, None] * ri[..., None, :]
    value = value + lc[..., None, None] * lz[..., :, None] * (lz[..., None, :] @ value)
    value = value + rc[..., None, None] * (value @ rz[..., :, None]) * rz[..., None, :]
    return value


def _rank1_action(value: torch.Tensor, inverse, z, coefficient, *, left: bool):
    if left:
        return value + coefficient * z[:, None] * (z[None, :] @ value)
    return value + coefficient * (value @ z[:, None]) * z[None, :]


def rank1_mapback(value: torch.Tensor, left, right) -> torch.Tensor:
    """Apply the production mapback order: rank-one factors, then diagonals."""
    li, lz, lc = left
    ri, rz, rc = right
    value = _rank1_action(value, li, lz, lc, left=True)
    value = _rank1_action(value, ri, rz, rc, left=False)
    return value * li[:, None] * ri[None, :]


def _graft_direction(value: torch.Tensor, output_rows: int) -> torch.Tensor:
    """Normalize the mapped direction to the production Frobenius graft."""
    norm = value.float().norm().clamp_min(torch.finfo(torch.float32).tiny)
    return value.float() * (math.sqrt(float(output_rows)) / norm)


def _s8_fp32(x: torch.Tensor) -> torch.Tensor:
    """Eager FP32 S8 reference; intentionally not the compensated GPU kernel."""
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    x = x.float()
    if not torch.count_nonzero(x):
        return torch.zeros_like(x.mT if transposed else x)
    norm = x.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    x = x / norm
    gram = x @ x.mT
    gram2 = gram @ gram
    scale = (S8_MARGIN * gram2.norm(dim=(-2, -1), keepdim=True).pow(0.25)).clamp_min(1e-30)
    eye = torch.eye(x.shape[-2], dtype=x.dtype, device=x.device)
    a, b, c = S8_COEFFICIENTS[0]
    p = a * eye + b * gram / scale.square() + c * gram2 / scale.pow(4)
    x = (p @ x) / scale
    for a, b, c in S8_COEFFICIENTS[1:]:
        gram = x @ x.mT
        gram2 = gram @ gram
        p = a * eye + b * gram + c * gram2
        x = p @ x
    if transposed:
        x = x.mT
    return x * S8_MULTIPLIER


class DiagonalWeightedPE(Optimizer):
    """S8 directions weighted by activation and output-cotangent diagonals.

    Statistics are accumulated from every forward/backward microbatch between
    optimizer steps. Defaults match the 103M run: beta=.97, Muon momentum=.8,
    S8 margin=1.35, auxiliary AdamW LR ratio=.1 and decay=.01.
    """

    def __init__(self, model: nn.Module, lr: float, *, rank_one_a: bool = False,
                 module_filter: Callable[[str, nn.Module], bool] | None = None):
        named_modules = dict(model.named_modules())
        named_parameters = dict(model.named_parameters())
        choose = module_filter or (lambda name, module: isinstance(module, nn.Linear))
        matrices: dict[str, nn.Parameter] = {}
        modules: dict[str, nn.Module] = {}
        for name, module in named_modules.items():
            if name and choose(name, module):
                parameter = getattr(module, "weight", None)
                if isinstance(parameter, nn.Parameter) and parameter.requires_grad and parameter.ndim == 2:
                    matrices[name + ".weight"] = parameter
                    modules[name + ".weight"] = module
        if not matrices:
            raise ValueError("No eligible trainable linear weights found")

        self.named_matrices = matrices
        self.parameter_names = {id(p): name for name, p in matrices.items()}
        self.rank_one_a = bool(rank_one_a)
        self.statistic_beta = .97
        self.s8_coefficients = S8_COEFFICIENTS
        self.s8_margin = S8_MARGIN
        self._group_constants = {}

        self.a_sources = {id(p): p for p in matrices.values()}
        # Match the production sharing rule for split Q/K/V projections.
        for name, parameter in matrices.items():
            match = re.match(r"^(transformer\.h\.\d+\.attn\.)[kv]_proj\.weight$", name)
            if match:
                q = matrices.get(match.group(1) + "q_proj.weight")
                if q is not None:
                    self.a_sources[id(parameter)] = q

        matrix_ids = {id(p) for p in matrices.values()}
        auxiliary = [p for p in model.parameters() if p.requires_grad and id(p) not in matrix_ids]
        defaults = dict(lr=float(lr), use_muon=True, momentum=.8, weight_decay=0.,
                        adamw_lr_ratio=.1, adamw_betas=(.9, .999), adamw_eps=1e-8)
        groups = [dict(params=list(matrices.values()), use_muon=True)]
        if auxiliary:
            groups.append(dict(params=auxiliary, use_muon=False, weight_decay=.01))
        super().__init__(groups, defaults)
        self.handles = []
        for name, module in modules.items():
            parameter = matrices[name]
            self.handles.append(module.register_forward_hook(
                self._capture(parameter, capture_a=self.a_sources[id(parameter)] is parameter)))

    def _capture(self, parameter, *, capture_a):
        def hook(module, inputs, output):
            if (not module.training or not torch.is_grad_enabled() or
                    not isinstance(output, torch.Tensor) or not output.requires_grad):
                return
            rows = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            state = self.state[parameter]
            with torch.no_grad():
                if capture_a:
                    if "a_sum" not in state:
                        state["a_sum"] = torch.zeros(rows.shape[-1], device=rows.device, dtype=torch.float32)
                        state["a_count"] = 0
                        if self.rank_one_a:
                            state["a_first_sum"] = torch.zeros_like(state["a_sum"])
                    state["a_sum"].add_(rows.square().sum(0))
                    if self.rank_one_a:
                        state["a_first_sum"].add_(rows.sum(0))
                    state["a_count"] += rows.shape[0]

            def cotangent_hook(gradient):
                delta = gradient.detach().reshape(-1, gradient.shape[-1]).float()
                with torch.no_grad():
                    if "b_sum" not in state:
                        state["b_sum"] = torch.zeros(delta.shape[-1], device=delta.device, dtype=torch.float32)
                        state["b_count"] = 0
                    state["b_sum"].add_(delta.square().sum(0))
                    state["b_count"] += delta.shape[0]
                return gradient
            output.register_hook(cotangent_hook)
        return hook

    def _finish_a(self, parameter):
        state = self.state[parameter]
        if state.get("a_count", 0) < 1:
            raise RuntimeError(f"Missing activation statistics: {self.parameter_names[id(parameter)]}")
        count = state["a_count"]
        second = state["a_sum"] / count
        if self.rank_one_a:
            mean = state["a_first_sum"] / count
            if "a_ema" not in state:
                state["a_ema"] = second.clone()
                state["a_mean_ema"] = mean.clone()
            else:
                state["a_ema"].mul_(self.statistic_beta).add_(second, alpha=1-self.statistic_beta)
                state["a_mean_ema"].mul_(self.statistic_beta).add_(mean, alpha=1-self.statistic_beta)
            factor = rank1_inverse_sqrt(state["a_ema"], state["a_mean_ema"])
        else:
            if "a_ema" not in state:
                state["a_ema"] = second.clone()
            else:
                state["a_ema"].mul_(self.statistic_beta).add_(second, alpha=1-self.statistic_beta)
            factor = diagonal_inverse_sqrt(state["a_ema"])
        state["a_sum"].zero_()
        if self.rank_one_a:
            state["a_first_sum"].zero_()
        state["a_count"] = 0
        return factor, count

    def _finish_b(self, parameter, expected_count):
        state = self.state[parameter]
        if state.get("b_count", 0) != expected_count:
            raise RuntimeError(f"Missing or mismatched cotangent statistics: {self.parameter_names[id(parameter)]}; expected {expected_count}, got {state.get('b_count', 0)}")
        b = state["b_sum"] / expected_count
        b = b / b.mean().clamp_min(torch.finfo(b.dtype).tiny)
        if "b_ema" not in state:
            state["b_ema"] = b.clone()
        else:
            state["b_ema"].mul_(self.statistic_beta).add_(b, alpha=1-self.statistic_beta)
        inverse = diagonal_inverse_sqrt(state["b_ema"])
        state["b_sum"].zero_()
        state["b_count"] = 0
        return inverse

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            loss = None
        else:
            with torch.enable_grad():
                loss = closure()
        matrix_items, auxiliary_items, a_cache = [], [], {}
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                state = self.state[parameter]
                if group.get("use_muon", True):
                    source = self.a_sources[id(parameter)]
                    if id(source) not in a_cache:
                        a_cache[id(source)] = self._finish_a(source)
                    a_factor, count = a_cache[id(source)]
                    inv_b = self._finish_b(parameter, count)
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(parameter, dtype=torch.float32)
                    beta = group["momentum"]
                    state["momentum_buffer"].mul_(beta).add_(gradient.float(), alpha=1-beta)
                    source_gradient = state["momentum_buffer"] * beta + gradient.float() * (1-beta)
                    rows, columns = parameter.shape
                    transposed = rows > columns
                    if transposed:
                        source_gradient = source_gradient.mT
                        if self.rank_one_a:
                            left = a_factor
                            right = (inv_b, torch.zeros_like(inv_b), torch.zeros((), device=inv_b.device))
                        else:
                            left, right = a_factor, inv_b
                    else:
                        if self.rank_one_a:
                            left = (inv_b, torch.zeros_like(inv_b), torch.zeros((), device=inv_b.device))
                            right = a_factor
                        else:
                            left, right = inv_b, a_factor
                    matrix_items.append((parameter, source_gradient, left, right, group["lr"], transposed))
                else:
                    auxiliary_items.append((parameter, gradient.float(), state, group))

        for parameter, gradient, state, group in auxiliary_items:
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
                state["step"] = 0
            state["step"] += 1
            b1, b2 = group["adamw_betas"]
            state["exp_avg"].mul_(b1).add_(gradient, alpha=1-b1)
            state["exp_avg_sq"].mul_(b2).addcmul_(gradient, gradient, value=1-b2)
            bias1, bias2 = 1-b1**state["step"], 1-b2**state["step"]
            update = (state["exp_avg"] / bias1) / ((state["exp_avg_sq"] / bias2).sqrt() + group["adamw_eps"])
            lr = group["lr"] * group["adamw_lr_ratio"]
            parameter.mul_(1 - lr * group["weight_decay"])
            parameter.add_(update.to(parameter.dtype), alpha=-lr)

        grouped = {}
        for item in matrix_items:
            parameter, source, left, right, lr, transposed = item
            key = (tuple(source.shape), parameter.device)
            grouped.setdefault(key, []).append(item)
        for entries in grouped.values():
            sources = torch.stack([item[1] for item in entries])
            if self.rank_one_a:
                li = torch.stack([item[2][0] for item in entries]); lz = torch.stack([item[2][1] for item in entries]); lc = torch.stack([item[2][2] for item in entries])
                ri = torch.stack([item[3][0] for item in entries]); rz = torch.stack([item[3][1] for item in entries]); rc = torch.stack([item[3][2] for item in entries])
                whitened = rank1_whiten(sources, (li, lz, lc), (ri, rz, rc))
            else:
                li = torch.stack([item[2] for item in entries]); ri = torch.stack([item[3] for item in entries])
                whitened = sources * li[:, :, None] * ri[:, None, :]
            q = _s8_fp32(whitened)
            directions = []
            for index, item in enumerate(entries):
                parameter, _, left, right, _, _ = item
                value = q[index]
                if self.rank_one_a:
                    value = rank1_mapback(value, (li[index], lz[index], lc[index]),
                                          (ri[index], rz[index], rc[index]))
                else:
                    value = value * li[index, :, None] * ri[index, None, :]
                rows = parameter.shape[0]
                # The historical aspect multiplier is applied before the norm
                # and therefore cancels; the graft leaves norm sqrt(rows).
                direction = _graft_direction(value, rows)
                if item[5]:
                    direction = direction.mT
                directions.append(direction)
            for direction, item in zip(directions, entries):
                item[0].add_(direction.to(item[0].dtype), alpha=-item[4])
        return loss

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class RankOneAWeightedPE(DiagonalWeightedPE):
    """Production SAND variant with exact rank-one activation mean correction."""
    def __init__(self, model: nn.Module, lr: float, *,
                 module_filter: Callable[[str, nn.Module], bool] | None = None):
        super().__init__(model, lr, rank_one_a=True, module_filter=module_filter)
