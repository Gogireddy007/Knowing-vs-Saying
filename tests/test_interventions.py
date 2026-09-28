"""Unit tests for the intervention hooks.

We avoid loading a real LM. Instead we construct a tiny GPT-2-shaped
mock: an object with `.transformer.h` as a torch.nn.ModuleList of N
identity blocks, and verify that the patching context manager correctly
clamps the last-token residual.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import numpy as np

from interventions.patch import (
    _block_module, _patch_hook, _add_hook, paired_sign_test_p,
)


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        # Mimic GPT-2 block: returns a tuple.
        return (x + self.lin(x),)


class _Model(nn.Module):
    def __init__(self, n_blocks, d):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([_Block(d) for _ in range(n_blocks)])
        # need a parameter so next(self.parameters()) works
        self.dummy = nn.Parameter(torch.zeros(1))


def test_block_module_resolution():
    m = _Model(4, 8)
    assert _block_module(m, 0) is m.transformer.h[0]
    assert _block_module(m, 3) is m.transformer.h[3]


def test_block_module_out_of_range_raises():
    m = _Model(4, 8)
    with pytest.raises(IndexError):
        _block_module(m, 4)


def test_patch_hook_replaces_last_token():
    m = _Model(2, 8).eval()
    x = torch.randn(1, 5, 8)
    # Run once to get baseline output.
    out_baseline = m.transformer.h[0](x)[0]
    # Now replace last token with all-ones.
    rep = torch.ones(8)
    with _patch_hook(m, 0, rep):
        out = m.transformer.h[0](x)[0]
    assert torch.allclose(out[0, -1], rep)
    # Other tokens unchanged.
    assert torch.allclose(out[0, :-1], out_baseline[0, :-1])


def test_add_hook_shifts_last_token():
    m = _Model(2, 8).eval()
    x = torch.randn(1, 5, 8)
    base = m.transformer.h[0](x)[0]
    vec = torch.ones(8)
    with _add_hook(m, 0, vec, 2.0):
        shifted = m.transformer.h[0](x)[0]
    delta = shifted[0, -1] - base[0, -1]
    assert torch.allclose(delta, 2 * vec)


def test_paired_sign_test_p_detects_consistent_drop():
    rng = np.random.RandomState(0)
    baseline = rng.uniform(0.5, 0.9, size=40)
    treated = baseline - 0.2  # consistent, large drop on every example
    out = paired_sign_test_p(baseline, treated, n_boot=2000)
    assert out["mean_delta"] < 0
    assert out["p_bootstrap"] < 0.05
    assert out["p_sign_test"] < 0.05
    assert out["n_neg"] == 40 and out["n_pos"] == 0


def test_paired_sign_test_p_null_when_no_effect():
    rng = np.random.RandomState(1)
    baseline = rng.uniform(0.3, 0.9, size=40)
    treated = baseline.copy()  # identical -> no effect at all
    out = paired_sign_test_p(baseline, treated, n_boot=2000)
    assert out["mean_delta"] == 0.0
    assert out["p_bootstrap"] > 0.5
    assert out["n_pos"] == 0 and out["n_neg"] == 0


def test_paired_sign_test_p_bounded():
    rng = np.random.RandomState(2)
    baseline = rng.uniform(0, 1, size=20)
    treated = rng.uniform(0, 1, size=20)
    out = paired_sign_test_p(baseline, treated, n_boot=500)
    assert 0.0 <= out["p_bootstrap"] <= 1.0
    assert 0.0 <= out["p_sign_test"] <= 1.0
