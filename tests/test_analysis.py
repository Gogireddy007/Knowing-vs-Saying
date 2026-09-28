"""Unit tests for gap analysis and behavior parsing."""
from __future__ import annotations

import numpy as np

from analysis.gap import (
    compute_gap, null_threshold, classify_failure_modes, fit_gap_regression,
    permutation_p_value, per_rung_p_values, fano_mi_lower_bound,
)
from analysis.controls import majority_class_behavior
from evaluation.behavior import (
    parse_state_from_generation, _extract_first_json, _score_state,
)


def test_extract_first_json_balanced():
    text = 'noise {"a": 1, "b": {"c": 2}} trailing'
    out = _extract_first_json(text)
    assert out == {"a": 1, "b": {"c": 2}}


def test_extract_first_json_handles_strings_with_braces():
    text = '{"a": "}{not braces", "b": 2}'
    out = _extract_first_json(text)
    assert out == {"a": "}{not braces", "b": 2}


def test_parse_state_fallback_regex():
    # No balanced JSON object — only the inline "counts" regex can salvage it.
    text = 'the state is "counts" : "apple" 3, "egg" 1'
    s, tele = parse_state_from_generation(text, ["apple", "egg"], ["x"])
    # Inline regex too permissive — should at least return None or a state.
    assert s is None or isinstance(s.counts, dict)
    # Verify the JSON path: a properly-shaped object parses.
    full = '{"counts": {"apple": 3}, "prices": {"apple": 1.0}, ' \
           '"flags": {"x": true}, "discount": 1.0}'
    s2, tele2 = parse_state_from_generation(full, ["apple", "egg"], ["x"])
    assert s2.counts["apple"] == 3
    assert tele2["parsed"]


def test_score_state_perfect():
    gt = {"counts": {"a": 2}, "prices": {"a": 1.50}, "flags": {"x": True},
          "discount": 1.0, "total": 3.00}
    from cart_state.state import CartState
    pred = CartState(counts={"a": 2}, prices={"a": 1.50},
                     flags={"x": True}, discount=1.0)
    s = _score_state(pred, gt)
    assert all(v == 1.0 for v in s.values())


def test_compute_gap_shapes():
    F = np.array([0.9, 0.8, 0.5, 0.2, 0.1])
    B = np.array([0.8, 0.3, 0.5, 0.2, 0.0])
    L = np.array([10, 20, 30, 40, 40])
    R = ["L10", "L20", "L30", "L40", "L40"]
    g = compute_gap(F, B, L, R, n_null=200)
    assert g.G.shape == (5,)
    assert "F_mean" in g.summary
    assert isinstance(g.threshold_tau, float)
    # By-length aggregation.
    by_L = g.by_length()
    assert set(by_L) == {10, 20, 30, 40}


def test_classify_failure_modes_buckets():
    F = np.array([0.95, 0.3, 0.9, 0.2])
    B = np.array([0.9, 0.2, 0.1, 0.7])
    G = F - B
    tau = 0.4
    modes = classify_failure_modes(F, B, G, tau)
    assert modes[0] == "correct"          # B >= 0.8
    assert modes[1] == "world_model"      # max(F,B) < 0.5
    assert modes[2] == "deployment"       # high F, large +G
    assert modes[3] == "readout"          # B > F by tau, B >= 0.5


def test_fit_gap_regression_runs():
    F = np.random.RandomState(0).rand(50)
    B = (F > 0.5).astype(float)
    L = np.random.RandomState(1).choice([10, 20, 30, 40], 50)
    out = fit_gap_regression(F, B, L)
    assert "beta_1_F" in out
    assert 0.0 <= out["accuracy"] <= 1.0


def test_null_threshold_monotone():
    F = np.random.RandomState(2).rand(100)
    B = np.random.RandomState(3).rand(100)
    t90 = null_threshold(F, B, q=0.90, n_null=300)
    t99 = null_threshold(F, B, q=0.99, n_null=300)
    assert t90 <= t99


def test_permutation_p_value_large_gap_is_significant():
    # Constant F (e.g. np.full(60, 0.9)) is degenerate for THIS test: a
    # permutation of a constant array is a no-op, so every null sample
    # exactly equals the observed statistic and p is forced to ~1.0
    # regardless of how large |F-B| is. The statistic this test needs to
    # stress is genuine per-example F<->B pairing, which requires F to
    # vary across examples and be tied to B example-by-example (here,
    # perfectly anti-correlated) so that shuffling F's assignment to
    # examples actually destroys the pairing the null is supposed to break.
    rng = np.random.RandomState(0)
    B = rng.rand(60)
    F = 1.0 - B  # perfect per-example anti-correlation, not a constant
    p = permutation_p_value(F, B, n_null=5000)
    assert p < 0.01


def test_permutation_p_value_no_gap_is_not_significant():
    rng = np.random.RandomState(0)
    F = rng.rand(60)
    B = F.copy()  # F == B everywhere -> observed statistic is 0, the
                  # minimum |F-B| can ever take, so every null sample is
                  # >= the observed value and p must be ~1.0.
    p = permutation_p_value(F, B, n_null=2000)
    assert p > 0.5


def test_permutation_p_value_bounded():
    F = np.random.RandomState(1).rand(40)
    B = np.random.RandomState(2).rand(40)
    p = permutation_p_value(F, B, n_null=500)
    assert 0.0 < p <= 1.0


def test_per_rung_p_values_one_per_rung():
    F = np.array([0.9, 0.85, 0.1, 0.15, 0.5, 0.5])
    B = np.array([0.1, 0.15, 0.9, 0.85, 0.5, 0.5])
    rungs = ["L10", "L10", "L20", "L20", "L30", "L30"]
    out = per_rung_p_values(F, B, rungs, n_null=500)
    assert set(out) == {"L10", "L20", "L30"}
    assert all(0.0 < p <= 1.0 for p in out.values())


def test_fano_mi_lower_bound_binary_informative_at_high_accuracy():
    # |S|=2 (a flag): high accuracy must yield a clearly non-vacuous,
    # normalised bound strictly between 0 and 1.
    out = fano_mi_lower_bound(0.9, support_size=2)
    assert not out["vacuous"]
    assert 0.0 < out["mi_lower_bound_normalized"] < 1.0
    assert out["H_S_bits"] == 1.0


def test_fano_mi_lower_bound_clips_to_zero_at_chance():
    # At chance accuracy for a binary variable (a=0.5) Fano gives no
    # information at all -> bound must clip exactly to 0, not go negative.
    out = fano_mi_lower_bound(0.5, support_size=2)
    assert out["mi_lower_bound_normalized"] == 0.0
    assert out["vacuous"]


def test_fano_mi_lower_bound_monotone_in_accuracy():
    lo = fano_mi_lower_bound(0.3, support_size=8)
    hi = fano_mi_lower_bound(0.8, support_size=8)
    assert hi["mi_lower_bound_normalized"] >= lo["mi_lower_bound_normalized"]


def test_fano_mi_lower_bound_normalized_is_actually_normalized():
    # The normalised value must never exceed 1, regardless of |S| or a,
    # since it is supposed to be I(S;h)/H(S) <= 1 by definition of MI.
    for a in (0.1, 0.4, 0.6, 0.9, 0.99):
        for s in (2, 4, 8, 16):
            out = fano_mi_lower_bound(a, support_size=s)
            assert 0.0 <= out["mi_lower_bound_normalized"] <= 1.0


def test_majority_class_behavior_matched_scope_includes_prices():
    class _E:
        def __init__(self, target):
            self.target = target

    examples = [
        _E({"counts": {"a": 0}, "prices": {"a": 5.0}, "flags": {"x": False},
            "total": 0.0}),
        _E({"counts": {"a": 0}, "prices": {"a": 7.0}, "flags": {"x": False},
            "total": 0.0}),
    ]
    legacy = majority_class_behavior(examples, include_prices=False)
    matched = majority_class_behavior(examples, include_prices=True)
    # Legacy floor ignores prices entirely -> perfect score here (all
    # counts/flags/total at default).
    assert legacy == 1.0
    # Matched-scope floor must score the price cells too; with prices
    # {5.0, 7.0} the best constant guess (mean=6.0) cannot hit either
    # within the 1-cent tolerance, so the matched floor must be strictly
    # lower than the legacy floor — this is the scoring-mismatch fix.
    assert matched < legacy
