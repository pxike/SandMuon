"""SAND-Muon: diagonal sensitivity weighting around fast compensated S8."""
from __future__ import annotations

import math
import json
import re
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer

from odl.optim.registry import register_optimizer


@torch.compile(fullgraph=True, dynamic=False)
def _accumulate_sum_squares(rows: torch.Tensor, total: torch.Tensor) -> None:
    """Fuse cast, square, reduction and accumulation into persistent storage."""
    total.add_(rows.float().square().sum(0))


@torch.compile(fullgraph=True, dynamic=False)
def _accumulate_rank1_a(rows: torch.Tensor, total_square: torch.Tensor,
                        total_sum: torch.Tensor) -> None:
    """Collect both activation moments in one compiled reduction."""
    values = rows.float()
    total_square.add_(values.square().sum(0))
    total_sum.add_(values.sum(0))


def _rank1_parameters(second: torch.Tensor, mean: torch.Tensor):
    scale = second.mean().clamp_min(torch.finfo(second.dtype).tiny)
    variance = (second-mean.square()).clamp_min(0.)
    inverse = (variance/scale).clamp_min(1e-6).rsqrt()
    z = inverse*(mean/scale.sqrt())
    root = torch.sqrt(1+z.square().sum())
    coefficient = -1/(root*(root+1))
    return inverse, z, coefficient


@torch.compile(fullgraph=True, dynamic=False)
def _initialize_a_rank1(a_sum, a_first_sum, count: int):
    second = a_sum/count
    mean = a_first_sum/count
    inverse, z, coefficient = _rank1_parameters(second, mean)
    a_sum.zero_(); a_first_sum.zero_()
    return second, mean, inverse, z, coefficient


@torch.compile(fullgraph=True, dynamic=False)
def _update_a_rank1(a_sum, a_first_sum, a_ema, a_mean_ema,
                    count: int, beta: float):
    second = a_sum/count
    mean = a_first_sum/count
    a_ema.mul_(beta).add_(second, alpha=1-beta)
    a_mean_ema.mul_(beta).add_(mean, alpha=1-beta)
    inverse, z, coefficient = _rank1_parameters(a_ema, a_mean_ema)
    a_sum.zero_(); a_first_sum.zero_()
    return inverse, z, coefficient


def _normalized_inverse_sqrt(value: torch.Tensor) -> torch.Tensor:
    mean = value.mean()
    tiny = torch.finfo(value.dtype).tiny
    normalized = value.clamp_min(mean*1e-6)/mean.clamp_min(tiny)
    return normalized.rsqrt()


@torch.compile(fullgraph=True, dynamic=False)
def _initialize_a(a_sum, count: int):
    a = a_sum/count
    inverse = _normalized_inverse_sqrt(a)
    a_sum.zero_()
    return a, inverse


@torch.compile(fullgraph=True, dynamic=False)
def _update_a(a_sum, a_ema, count: int, beta: float):
    a = a_sum/count
    a_ema.mul_(beta).add_(a, alpha=1-beta)
    inverse = _normalized_inverse_sqrt(a_ema)
    a_sum.zero_()
    return inverse


@torch.compile(fullgraph=True, dynamic=False)
def _initialize_b(b_sum, count: int):
    b = b_sum/count
    b = b/b.mean().clamp_min(torch.finfo(b.dtype).tiny)
    inverse = _normalized_inverse_sqrt(b)
    b_sum.zero_()
    return b, inverse


@torch.compile(fullgraph=True, dynamic=False)
def _update_b(b_sum, b_ema, count: int, beta: float):
    b = b_sum/count
    b = b/b.mean().clamp_min(torch.finfo(b.dtype).tiny)
    b_ema.mul_(beta).add_(b, alpha=1-beta)
    inverse = _normalized_inverse_sqrt(b_ema)
    b_sum.zero_()
    return inverse


@torch.compile(fullgraph=True, dynamic=False)
def _finish_directions(q, inverse_a, inverse_b, q_scale, graft_rows):
    q = q*q_scale[:, None, None]
    raw = q*inverse_b[:, :, None]*inverse_a[:, None, :]
    norms = raw.float().norm(dim=(-2, -1), keepdim=True)
    return raw.float()*(graft_rows.sqrt()[:, None, None]
                        / norms.clamp_min(torch.finfo(torch.float32).tiny))


@torch.compile(fullgraph=True, dynamic=False)
def _rank1_whiten(sources, left_inverse, left_z, left_coefficient,
                  right_inverse, right_z, right_coefficient):
    value = sources*left_inverse[:, :, None]*right_inverse[:, None, :]
    value = value + left_coefficient[:, None, None]*left_z[:, :, None]*torch.bmm(
        left_z[:, None, :], value)
    value = value + right_coefficient[:, None, None]*torch.bmm(
        value, right_z[:, :, None])*right_z[:, None, :]
    return value


@torch.compile(fullgraph=True, dynamic=False)
def _rank1_finish(q, left_inverse, left_z, left_coefficient,
                  right_inverse, right_z, right_coefficient,
                  q_scale, graft_rows):
    value = q*q_scale[:, None, None]
    value = value + left_coefficient[:, None, None]*left_z[:, :, None]*torch.bmm(
        left_z[:, None, :], value)
    value = value + right_coefficient[:, None, None]*torch.bmm(
        value, right_z[:, :, None])*right_z[:, None, :]
    value = value*left_inverse[:, :, None]*right_inverse[:, None, :]
    norms = value.float().norm(dim=(-2, -1), keepdim=True)
    return value.float()*(graft_rows.sqrt()[:, None, None]
                          / norms.clamp_min(torch.finfo(torch.float32).tiny))


@torch.compile(fullgraph=True, dynamic=False)
def _one_side_whiten(sources, left_inverse, right_inverse, z, coefficient,
                     rank_one_left: bool):
    value = sources*left_inverse[:, :, None]*right_inverse[:, None, :]
    if rank_one_left:
        return value + coefficient[:, None, None]*z[:, :, None]*torch.bmm(
            z[:, None, :], value)
    return value + coefficient[:, None, None]*torch.bmm(
        value, z[:, :, None])*z[:, None, :]


@torch.compile(fullgraph=True, dynamic=False)
def _one_side_finish(q, left_inverse, right_inverse, z, coefficient,
                     rank_one_left: bool, q_scale, graft_rows):
    value = q*q_scale[:, None, None]
    if rank_one_left:
        value = value + coefficient[:, None, None]*z[:, :, None]*torch.bmm(
            z[:, None, :], value)
    else:
        value = value + coefficient[:, None, None]*torch.bmm(
            value, z[:, :, None])*z[:, None, :]
    value = value*left_inverse[:, :, None]*right_inverse[:, None, :]
    norms = value.float().norm(dim=(-2, -1), keepdim=True)
    return value.float()*(graft_rows.sqrt()[:, None, None]
                          / norms.clamp_min(torch.finfo(torch.float32).tiny))


class DiagonalWeightedPE(Optimizer):
    """S8 polarization in coordinates weighted by full-batch diagonal A/B EMAs.

    Matrix momentum is .8 Nesterov. A tracks raw activation second moments;
    B tracks the relative channel shape of real-loss output-cotangent second
    moments. Both use a fixed .97 EMA once per optimizer update. Auxiliary
    parameters use AdamW with LR ratio .1, betas (.9,.999), and weight decay
    .01, matching the validated 10M recipe.
    """

    def __init__(self, model: nn.Module, lr: float, *, rank_one_a: bool = False):
        named_modules = dict(model.named_modules())
        named_parameters = dict(model.named_parameters())
        matrices = {}
        modules = {}
        for name, module in named_modules.items():
            if re.match(r"^transformer\.h\.\d+\.", name) and isinstance(module, nn.Linear):
                parameter_name = name + ".weight"
                parameter = named_parameters.get(parameter_name)
                if parameter is not None and parameter.requires_grad and parameter.ndim == 2:
                    matrices[parameter_name] = parameter
                    modules[parameter_name] = module
        if not matrices:
            raise ValueError("No transformer.h.* linear weights found")
        matrix_ids = {id(parameter) for parameter in matrices.values()}
        auxiliary = [parameter for parameter in model.parameters()
                     if parameter.requires_grad and id(parameter) not in matrix_ids]
        defaults = dict(lr=float(lr), use_muon=True, momentum=.8,
                        weight_decay=0., adamw_lr_ratio=.1,
                        adamw_betas=(.9, .999), adamw_eps=1e-8)
        groups = [dict(params=list(matrices.values()), use_muon=True)]
        if auxiliary:
            groups.append(dict(params=auxiliary, use_muon=False, weight_decay=.01))
        super().__init__(groups, defaults)
        self.named_matrices = matrices
        self.parameter_names = {id(parameter): name for name, parameter in matrices.items()}
        self.statistic_beta = .97
        self.rank_one_a = bool(rank_one_a)
        manifest = Path(__file__).resolve().parents[2] / "paper/s8/data/completed/s8_manifest.json"
        payload = json.loads(manifest.read_text())
        fit = payload["s8_fit"]
        if (len(fit["coefficients"]) != 5 or float(fit["margin"]) != 1.35
                or not fit.get("compensated_first", False)):
            raise RuntimeError("Invalid deployed compensated-S8 recipe")
        self.s8_coefficients = fit["coefficients"]
        self.s8_margin = float(fit["margin"])
        self._group_constants = {}
        self.a_sources = {id(parameter): parameter for parameter in matrices.values()}
        for name, parameter in matrices.items():
            match = re.match(r"^(transformer\.h\.\d+\.attn\.)[kv]_proj\.weight$", name)
            if match:
                q_parameter = matrices.get(match.group(1) + "q_proj.weight")
                if q_parameter is None:
                    raise RuntimeError("Split attention projection is missing q_proj")
                self.a_sources[id(parameter)] = q_parameter
        self.handles = []
        for name, module in modules.items():
            self.handles.append(module.register_forward_hook(
                self._capture(
                    matrices[name],
                    capture_a=self.a_sources[id(matrices[name])] is matrices[name])))

    def _capture(self, parameter, *, capture_a):
        def hook(module, inputs, output):
            if (not module.training or not torch.is_grad_enabled()
                    or not isinstance(output, torch.Tensor) or not output.requires_grad):
                return
            rows = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            state = self.state[parameter]
            if capture_a:
                with torch.no_grad():
                    if "a_sum" not in state:
                        state["a_sum"] = torch.zeros(
                            rows.shape[1], device=rows.device, dtype=torch.float32)
                        state["a_count"] = 0
                        if self.rank_one_a:
                            state["a_first_sum"] = torch.zeros(
                                rows.shape[1], device=rows.device, dtype=torch.float32)
                    if self.rank_one_a:
                        _accumulate_rank1_a(rows, state["a_sum"], state["a_first_sum"])
                    else:
                        _accumulate_sum_squares(rows, state["a_sum"])
                    state["a_count"] += rows.shape[0]

            def delta_hook(gradient):
                delta = gradient.detach().reshape(-1, gradient.shape[-1])
                with torch.no_grad():
                    if "b_sum" not in state:
                        state["b_sum"] = torch.zeros(
                            delta.shape[1], device=delta.device, dtype=torch.float32)
                        state["b_count"] = 0
                    _accumulate_sum_squares(delta, state["b_sum"])
                    state["b_count"] += delta.shape[0]
                return gradient

            output.register_hook(delta_hook)
        return hook

    def _finish_a(self, parameter):
        state = self.state[parameter]
        if "a_sum" not in state or state.get("a_count", 0) < 1:
            raise RuntimeError("Missing activation statistics: "
                               + self.parameter_names[id(parameter)])
        count = state["a_count"]
        if "a_ema" not in state:
            state["a_ema"], inverse = _initialize_a(state["a_sum"], count)
        else:
            inverse = _update_a(
                state["a_sum"], state["a_ema"], count, self.statistic_beta)
        state["a_count"] = 0
        return inverse, count

    def _finish_b(self, parameter, expected_count):
        state = self.state[parameter]
        if "b_sum" not in state or state.get("b_count", 0) != expected_count:
            raise RuntimeError("Missing or mismatched cotangent statistics: "
                               + self.parameter_names[id(parameter)])
        count = state["b_count"]
        if "b_ema" not in state:
            state["b_ema"], inverse = _initialize_b(state["b_sum"], count)
        else:
            inverse = _update_b(
                state["b_sum"], state["b_ema"], count, self.statistic_beta)
        state["b_count"] = 0
        return inverse

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        grouped = {}
        matrix_items = []
        auxiliary_items = []
        inverse_a_cache = {}
        for group in self.param_groups:
            lr = group["lr"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                state = self.state[parameter]
                if group.get("use_muon", True):
                    a_source = self.a_sources[id(parameter)]
                    cached = inverse_a_cache.get(id(a_source))
                    if cached is None:
                        cached = self._finish_a(a_source)
                        inverse_a_cache[id(a_source)] = cached
                    inverse_a, a_count = cached
                    inverse_b = self._finish_b(parameter, a_count)
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(parameter, dtype=torch.float32)
                    matrix_items.append((parameter, gradient.float(), state["momentum_buffer"],
                                         inverse_a, inverse_b, lr))
                else:
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
                        state["step"] = torch.zeros((), device=parameter.device,
                                                    dtype=torch.float32)
                    elif not isinstance(state["step"], torch.Tensor):
                        state["step"] = torch.tensor(
                            state["step"], device=parameter.device,
                            dtype=torch.float32)
                    auxiliary_items.append((parameter, gradient.float(), state,
                                            lr, group))

        if matrix_items:
            gradients = [item[1] for item in matrix_items]
            momentums = [item[2] for item in matrix_items]
            torch._foreach_mul_(momentums, .8)
            torch._foreach_add_(momentums, gradients, alpha=.2)
            sources = torch._foreach_mul(momentums, .8)
            torch._foreach_add_(sources, gradients, alpha=.2)
            for source, (parameter, _gradient, _momentum, inverse_a,
                         inverse_b, lr) in zip(sources, matrix_items):
                rows, columns = parameter.shape
                transposed = rows > columns
                if transposed:
                    source = source.mT
                    inverse_a, inverse_b = inverse_b, inverse_a
                key = (tuple(source.shape), parameter.device)
                grouped.setdefault(key, []).append(
                    (parameter, source, inverse_a, inverse_b, lr, transposed,
                     float(rows), math.sqrt(max(1., rows/columns))))

        if auxiliary_items:
            parameters = [item[0] for item in auxiliary_items]
            gradients = [item[1] for item in auxiliary_items]
            states = [item[2] for item in auxiliary_items]
            exp_avg = [state["exp_avg"] for state in states]
            exp_avg_sq = [state["exp_avg_sq"] for state in states]
            group = auxiliary_items[0][4]
            b1, b2 = group["adamw_betas"]
            auxiliary_lr = auxiliary_items[0][3]*group["adamw_lr_ratio"]
            torch.optim._functional.adamw(
                parameters, gradients, exp_avg, exp_avg_sq, [],
                [state["step"] for state in states], foreach=False,
                capturable=False, differentiable=False, fused=True,
                grad_scale=None, found_inf=None, has_complex=False,
                amsgrad=False, beta1=b1, beta2=b2, lr=auxiliary_lr,
                weight_decay=group["weight_decay"], eps=group["adamw_eps"],
                maximize=False)

        for (shape, _device), entries in grouped.items():
            sources = torch.stack([entry[1] for entry in entries])
            inverse_a = torch.stack([entry[2] for entry in entries])
            inverse_b = torch.stack([entry[3] for entry in entries])
            whitened = sources*inverse_b[:, :, None]*inverse_a[:, None, :]
            from odl.ops.dense_s8_compensated_fast import dense_s8_compensated_fast
            q = dense_s8_compensated_fast(
                whitened, self.s8_coefficients, self.s8_margin, 1.,
                packed_prefix=True)
            constants_key = (shape, q.device, q.dtype,
                             tuple(id(entry[0]) for entry in entries))
            constants = self._group_constants.get(constants_key)
            if constants is None:
                constants = (
                    torch.tensor([entry[7] for entry in entries],
                                 device=q.device, dtype=q.dtype),
                    torch.tensor([entry[6] for entry in entries],
                                 device=q.device, dtype=q.dtype),
                )
                self._group_constants[constants_key] = constants
            q_scale, graft_rows = constants
            directions = _finish_directions(
                q, inverse_a, inverse_b, q_scale, graft_rows)
            for direction, (parameter, _source, _ia, _ib, lr, transposed,
                            _graft_rows, _q_scale) in zip(directions, entries):
                if transposed:
                    direction = direction.mT
                parameter.add_(direction.to(parameter.dtype), alpha=-lr)
        return loss

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class RankOneAWeightedPE(DiagonalWeightedPE):
    """Production SAND with exact rank-one activation geometry and diagonal B."""

    def __init__(self, model: nn.Module, lr: float, *, split_sides: bool = False):
        super().__init__(model, lr, rank_one_a=True)
        self.split_sides = bool(split_sides)

    def _finish_a_rank1(self, parameter):
        state = self.state[parameter]
        if ("a_sum" not in state or "a_first_sum" not in state
                or state.get("a_count", 0) < 1):
            raise RuntimeError("Missing activation statistics: "
                               + self.parameter_names[id(parameter)])
        count = state["a_count"]
        if "a_ema" not in state:
            (state["a_ema"], state["a_mean_ema"], inverse, z,
             coefficient) = _initialize_a_rank1(
                 state["a_sum"], state["a_first_sum"], count)
        else:
            inverse, z, coefficient = _update_a_rank1(
                state["a_sum"], state["a_first_sum"], state["a_ema"],
                state["a_mean_ema"], count, self.statistic_beta)
        state["a_count"] = 0
        return (inverse, z, coefficient), count

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        grouped = {}; matrix_items = []; auxiliary_items = []; a_cache = {}
        for group in self.param_groups:
            lr = group["lr"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                state = self.state[parameter]
                if group.get("use_muon", True):
                    a_source = self.a_sources[id(parameter)]
                    cached = a_cache.get(id(a_source))
                    if cached is None:
                        cached = self._finish_a_rank1(a_source); a_cache[id(a_source)] = cached
                    a_factor, a_count = cached
                    inverse_b = self._finish_b(parameter, a_count)
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(parameter, dtype=torch.float32)
                    matrix_items.append((parameter, gradient.float(), state["momentum_buffer"],
                                         a_factor, inverse_b, lr))
                else:
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
                        state["step"] = torch.zeros((), device=parameter.device, dtype=torch.float32)
                    elif not isinstance(state["step"], torch.Tensor):
                        state["step"] = torch.tensor(state["step"], device=parameter.device,
                                                     dtype=torch.float32)
                    auxiliary_items.append((parameter, gradient.float(), state, lr, group))

        if matrix_items:
            gradients = [x[1] for x in matrix_items]; momentums = [x[2] for x in matrix_items]
            torch._foreach_mul_(momentums, .8); torch._foreach_add_(momentums, gradients, alpha=.2)
            sources = torch._foreach_mul(momentums, .8); torch._foreach_add_(sources, gradients, alpha=.2)
            for source, (parameter, _g, _m, a_factor, inverse_b, lr) in zip(sources, matrix_items):
                rows, columns = parameter.shape; transposed = rows > columns
                a_inverse, a_z, a_coefficient = a_factor
                zero_b = torch.zeros_like(inverse_b)
                zero_coefficient = torch.zeros((), device=inverse_b.device, dtype=inverse_b.dtype)
                if transposed:
                    source = source.mT
                    left = (a_inverse, a_z, a_coefficient)
                    right = (inverse_b, zero_b, zero_coefficient)
                else:
                    left = (inverse_b, zero_b, zero_coefficient)
                    right = (a_inverse, a_z, a_coefficient)
                key = (tuple(source.shape), parameter.device)
                grouped.setdefault(key, []).append(
                    (parameter, source, left, right, lr, transposed,
                     float(rows), math.sqrt(max(1., rows/columns))))

        if auxiliary_items:
            parameters = [x[0] for x in auxiliary_items]; gradients = [x[1] for x in auxiliary_items]
            states = [x[2] for x in auxiliary_items]; group = auxiliary_items[0][4]
            b1, b2 = group["adamw_betas"]
            torch.optim._functional.adamw(
                parameters, gradients, [s["exp_avg"] for s in states],
                [s["exp_avg_sq"] for s in states], [], [s["step"] for s in states],
                foreach=False, capturable=False, differentiable=False, fused=True,
                grad_scale=None, found_inf=None, has_complex=False, amsgrad=False,
                beta1=b1, beta2=b2, lr=auxiliary_items[0][3]*group["adamw_lr_ratio"],
                weight_decay=group["weight_decay"], eps=group["adamw_eps"], maximize=False)

        from odl.ops.dense_s8_compensated_fast import dense_s8_compensated_fast
        for (shape, _device), entries in grouped.items():
            # Fixed-size S8 batches are numerically important in BF16.  Sort
            # only within the batch so one-sided maps are contiguous while
            # preserving the original batch cardinality.
            if self.split_sides:
                entries.sort(key=lambda item: item[5], reverse=True)
            sources = torch.stack([x[1] for x in entries])
            li = torch.stack([x[2][0] for x in entries]); lz = torch.stack([x[2][1] for x in entries])
            lc = torch.stack([x[2][2] for x in entries])
            ri = torch.stack([x[3][0] for x in entries]); rz = torch.stack([x[3][1] for x in entries])
            rc = torch.stack([x[3][2] for x in entries])
            if not self.split_sides:
                whitened = _rank1_whiten(sources, li, lz, lc, ri, rz, rc)
            else:
                boundary = sum(x[5] for x in entries)
                pieces = []
                if boundary:
                    pieces.append(_one_side_whiten(
                        sources[:boundary], li[:boundary], ri[:boundary],
                        lz[:boundary], lc[:boundary], True))
                if boundary < len(entries):
                    pieces.append(_one_side_whiten(
                        sources[boundary:], li[boundary:], ri[boundary:],
                        rz[boundary:], rc[boundary:], False))
                whitened = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
            q = dense_s8_compensated_fast(whitened, self.s8_coefficients,
                                          self.s8_margin, 1., packed_prefix=True)
            constants_key = (shape, q.device, q.dtype, tuple(id(x[0]) for x in entries))
            constants = self._group_constants.get(constants_key)
            if constants is None:
                constants = (torch.tensor([x[7] for x in entries], device=q.device, dtype=q.dtype),
                             torch.tensor([x[6] for x in entries], device=q.device, dtype=q.dtype))
                self._group_constants[constants_key] = constants
            if not self.split_sides:
                directions = _rank1_finish(q, li, lz, lc, ri, rz, rc, *constants)
            else:
                pieces = []
                if boundary:
                    pieces.append(_one_side_finish(
                        q[:boundary], li[:boundary], ri[:boundary],
                        lz[:boundary], lc[:boundary], True,
                        constants[0][:boundary], constants[1][:boundary]))
                if boundary < len(entries):
                    pieces.append(_one_side_finish(
                        q[boundary:], li[boundary:], ri[boundary:],
                        rz[boundary:], rc[boundary:], False,
                        constants[0][boundary:], constants[1][boundary:]))
                directions = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
            for direction, entry in zip(directions, entries):
                parameter, transposed, lr = entry[0], entry[5], entry[4]
                if transposed: direction = direction.mT
                parameter.add_(direction.to(parameter.dtype), alpha=-lr)
        return loss


@register_optimizer("sand_muon")
@register_optimizer("muon_raw_diagonal_weighted_pe")
def _create(model, *, lr, betas, weight_decay, device_type):
    del betas, weight_decay, device_type
    return DiagonalWeightedPE(model, lr=lr)


@register_optimizer("sand_muon_rank1")
def _create_rank1(model, *, lr, betas, weight_decay, device_type):
    del betas, weight_decay, device_type
    return RankOneAWeightedPE(model, lr=lr, split_sides=False)


__all__ = ["DiagonalWeightedPE", "RankOneAWeightedPE"]
