import copy

import torch
from torch import nn

from sandmuon.optimizer import (
    DiagonalWeightedPE,
    RankOneAWeightedPE,
    _s8_fp32,
    diagonal_inverse_sqrt,
    _graft_direction,
    rank1_inverse_sqrt,
    rank1_mapback,
    rank1_whiten,
)


def _dense_whitener(second, mean):
    inverse, z, coefficient = rank1_inverse_sqrt(second, mean)
    correction = torch.eye(second.numel(), dtype=second.dtype) + coefficient * torch.outer(z, z)
    return correction @ torch.diag(inverse)


def test_rank_one_whitener_matches_dense_covariance_oracle():
    torch.manual_seed(7)
    x = torch.randn(500, 6, dtype=torch.float64) + torch.randn(6, dtype=torch.float64)
    second = x.square().mean(0)
    mean = x.mean(0)
    inverse, z, coefficient = rank1_inverse_sqrt(second, mean)
    w = _dense_whitener(second, mean)
    covariance = torch.diag(second - mean.square()) + torch.outer(mean, mean)
    normalized = covariance / second.mean()
    torch.testing.assert_close(w @ normalized @ w.T, torch.eye(6, dtype=torch.float64), atol=1e-10, rtol=1e-10)

    source = torch.randn(2, 6, 4, dtype=torch.float64)
    left = (inverse, z, coefficient)
    right_inverse, right_z, right_coefficient = rank1_inverse_sqrt(second[:4], mean[:4])
    right = (right_inverse, right_z, right_coefficient)
    actual = rank1_whiten(source, left, right)
    dense_left = w
    dense_right = _dense_whitener(second[:4], mean[:4])
    expected = torch.stack([dense_left @ item @ dense_right.T for item in source])
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_mapback_applies_rank_one_factors_then_diagonal_and_graft_norm():
    torch.manual_seed(8)
    q = torch.randn(5, 3, dtype=torch.float64)
    li, lz, lc = rank1_inverse_sqrt(torch.rand(5, dtype=torch.float64) + .3, torch.randn(5, dtype=torch.float64) * .1)
    ri, rz, rc = rank1_inverse_sqrt(torch.rand(3, dtype=torch.float64) + .3, torch.randn(3, dtype=torch.float64) * .1)
    left = torch.eye(5, dtype=q.dtype) + lc * torch.outer(lz, lz)
    right = torch.eye(3, dtype=q.dtype) + rc * torch.outer(rz, rz)
    expected = torch.diag(li) @ left @ q @ right @ torch.diag(ri)
    torch.testing.assert_close(rank1_mapback(q, (li, lz, lc), (ri, rz, rc)), expected)


def test_full_effective_batch_statistics_accumulate_over_microbatches():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 3, bias=False)

        def forward(self, x):
            return self.proj(x)

    torch.manual_seed(10)
    model = Model()
    x = torch.randn(9, 4)
    target = torch.randn(9, 3)
    prediction_before = model(x).detach()
    expected_b = (2 * (prediction_before - target) / x.shape[0]).square().mean(0)
    expected_b = expected_b / expected_b.mean()
    opt = RankOneAWeightedPE(model, lr=1e-3)
    opt.zero_grad(set_to_none=True)
    for start, end in ((0, 2), (2, 6), (6, 9)):
        prediction = model(x[start:end])
        ((prediction - target[start:end]).square().sum() / x.shape[0]).backward()
    before = model.proj.weight.detach().clone()
    opt.step()
    state = opt.state[model.proj.weight]
    torch.testing.assert_close(state["a_ema"], x.square().mean(0))
    torch.testing.assert_close(state["b_ema"], expected_b)
    assert state["a_count"] == state["b_count"] == 0
    assert not torch.equal(before, model.proj.weight)
    opt.close()


def _one_update(model, opt, x, target):
    opt.zero_grad(set_to_none=True)
    pred = model(x)
    (pred - target).square().mean().backward()
    opt.step()


def test_optimizer_state_dict_resume_matches_continuous_updates():
    torch.manual_seed(11)
    model_a = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 2))
    opt_a = RankOneAWeightedPE(model_a, lr=2e-3)
    x, y = torch.randn(8, 4), torch.randn(8, 2)
    _one_update(model_a, opt_a, x, y)
    model_state = copy.deepcopy(model_a.state_dict())
    optimizer_state = copy.deepcopy(opt_a.state_dict())

    model_b = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 2))
    model_b.load_state_dict(model_state)
    opt_b = RankOneAWeightedPE(model_b, lr=2e-3)
    opt_b.load_state_dict(optimizer_state)
    _one_update(model_a, opt_a, x, y)
    _one_update(model_b, opt_b, x, y)
    for a, b in zip(model_a.parameters(), model_b.parameters()):
        torch.testing.assert_close(a, b)
    opt_a.close()
    opt_b.close()


def test_zero_signal_remains_finite():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 3, bias=False)

        def forward(self, x):
            return self.proj(x)

    model = Model()
    opt = RankOneAWeightedPE(model, lr=1e-3)
    opt.zero_grad(set_to_none=True)
    out = model(torch.zeros(6, 4))
    (out * 0).sum().backward()
    opt.step()
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()
    state = opt.state[model.proj.weight]
    assert torch.isfinite(state["a_ema"]).all()
    assert torch.isfinite(state["a_mean_ema"]).all()
    assert torch.isfinite(state["b_ema"]).all()
    opt.close()


def test_s8_reference_handles_both_orientations_and_zero_matrix():
    torch.manual_seed(13)
    for shape in ((7, 3), (3, 7), (4, 4)):
        x = torch.randn(2, *shape)
        y = _s8_fp32(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
    zero = _s8_fp32(torch.zeros(1, 4, 3))
    assert torch.equal(zero, torch.zeros_like(zero))
    assert torch.isfinite(diagonal_inverse_sqrt(torch.zeros(4))).all()


def test_qkv_share_activation_statistic_only_for_production_attention_names():
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4, bias=False)
            self.k_proj = nn.Linear(4, 4, bias=False)
            self.v_proj = nn.Linear(4, 4, bias=False)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()
            self.cross_attn = Attention()

        def forward(self, x):
            return (self.attn.q_proj(x) + self.attn.k_proj(x) + self.attn.v_proj(x)
                    + self.cross_attn.q_proj(x) + self.cross_attn.k_proj(x)
                    + self.cross_attn.v_proj(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([Block()])

        def forward(self, x):
            return self.transformer.h[0](x)

    torch.manual_seed(14)
    model = Model()
    block = model.transformer.h[0]
    opt = RankOneAWeightedPE(model, lr=1e-3)
    assert opt.a_sources[id(block.attn.k_proj.weight)] is block.attn.q_proj.weight
    assert opt.a_sources[id(block.attn.v_proj.weight)] is block.attn.q_proj.weight
    assert opt.a_sources[id(block.cross_attn.k_proj.weight)] is block.cross_attn.k_proj.weight
    x = torch.randn(5, 4)
    opt.zero_grad(set_to_none=True)
    model(x).square().mean().backward()
    opt.step()
    torch.testing.assert_close(opt.state[block.attn.q_proj.weight]["a_ema"], x.square().mean(0))
    assert "a_ema" not in opt.state[block.attn.k_proj.weight]
    opt.close()

def test_diagonal_variant_zero_signal_remains_finite():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 3, bias=False)

        def forward(self, x):
            return self.proj(x)

    model = Model()
    opt = DiagonalWeightedPE(model, lr=1e-3)
    opt.zero_grad(set_to_none=True)
    (model(torch.zeros(6, 4)) * 0).sum().backward()
    opt.step()
    assert torch.isfinite(model.proj.weight).all()
    assert torch.isfinite(opt.state[model.proj.weight]["a_ema"]).all()
    assert torch.isfinite(opt.state[model.proj.weight]["b_ema"]).all()
    opt.close()


def test_actual_update_norm_matches_frobenius_graft_for_all_aspects_and_variants():
    class Model(nn.Module):
        def __init__(self, rows, columns):
            super().__init__()
            self.proj = nn.Linear(columns, rows, bias=False)

        def forward(self, x):
            return self.proj(x)

    torch.manual_seed(15)
    lr = 1e-3
    for optimizer_type in (DiagonalWeightedPE, RankOneAWeightedPE):
        for rows, columns in ((4, 4), (7, 3), (3, 7)):
            model = Model(rows, columns)
            optimizer = optimizer_type(model, lr=lr)
            x = torch.randn(12, columns)
            target = torch.randn(12, rows)
            before = model.proj.weight.detach().clone()
            (model(x) - target).square().mean().backward()
            optimizer.step()
            update_norm = (model.proj.weight.detach() - before).norm() / lr
            torch.testing.assert_close(update_norm, torch.tensor(rows ** .5), rtol=2e-5, atol=2e-5)
            optimizer.close()


def test_step_closure_runs_with_grad_enabled_and_updates_parameters():
    torch.manual_seed(16)
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 2))
    optimizer = RankOneAWeightedPE(model, lr=1e-3)
    x, target = torch.randn(7, 4), torch.randn(7, 2)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    def closure():
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = (prediction - target).square().mean()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert torch.isfinite(loss)
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    optimizer.close()
