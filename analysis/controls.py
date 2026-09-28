"""Base-rate controls and a stratified permutation null.

Reviewer objection (C2): G = F - B compares two different scoring
rules, and a chunk of B may come from "guess zero" on zero-default
variables. Reviewer objection (C5): permuting F across all examples
pins tau to the marginal difference between mean(F) and mean(B).

This module provides:
    majority_class_behavior  : B you get by always predicting the
                               base-rate value of each variable
    excl_zero_default_gap    : F, B, G restricted to variables whose
                               ground-truth value is non-default
    stratified_null          : permutation null computed *within*
                               length strata, so tau is not pinned to
                               the global marginal gap
"""
from __future__ import annotations

import numpy as np


def majority_class_behavior(examples, base_state=None,
                            include_prices: bool = True) -> float:
    """Mean behavioural correctness of a trivial agent that always emits
    the per-variable majority/default value (counts=0, flags=False,
    discount=1, total=0).

    ``evaluation.behavior._score_state`` — the function that scores every
    real model's B — grades four variable families: counts, prices,
    flags, total. A floor that is supposed to be compared against B must
    be graded on the *same* variable set, or the comparison is not
    apples-to-apples (prices are ~35% of all scored cells on
    Cart-State; omitting them from the floor while crediting the model
    on them inflates the apparent floor-vs-model gap).

    With ``include_prices=True`` (default) the trivial agent's price
    guess is the single constant that minimises 0/1 loss against a
    *fixed* tolerance-band scorer when prices are drawn from a wide,
    roughly uniform range with no recoverable mode: the dataset's
    global mean per-item price (computed once, here, from ``examples``
    themselves — the most favourable constant guess, not a strawman).
    Pass ``include_prices=False`` to recover the legacy (under-scoped)
    floor for comparison.
    """
    correct, total = 0, 0
    price_vals = []
    if include_prices:
        for e in examples:
            price_vals.extend(float(v) for v in e.target["prices"].values())
        price_guess = float(np.mean(price_vals)) if price_vals else 1.0
    for e in examples:
        t = e.target
        for it, v in t["counts"].items():
            correct += int(int(v) == 0); total += 1
        for fl, v in t["flags"].items():
            correct += int(bool(v) is False); total += 1
        if include_prices:
            for it, v in t["prices"].items():
                correct += int(abs(float(v) - price_guess) <= 0.01)
                total += 1
        # total default 0
        correct += int(abs(float(t["total"])) < 1e-9); total += 1
    return correct / max(1, total)


def excl_zero_default_gap(F_per_var: dict, B_per_var: dict,
                          examples) -> dict:
    """Recompute mean F, B, G using ONLY (example, variable) cells whose
    ground-truth value is non-default. Removes the 'guess zero' inflation
    of B. Inputs are dicts var -> per-example correctness array (NaN where
    N/A)."""
    F_keep, B_keep = [], []
    var_keys = set(F_per_var) & set(B_per_var)
    for k in var_keys:
        fa = np.asarray(F_per_var[k], dtype=float)
        ba = np.asarray(B_per_var[k], dtype=float)
        for i, e in enumerate(examples):
            if i >= len(fa) or i >= len(ba):
                continue
            if np.isnan(fa[i]) or np.isnan(ba[i]):
                continue
            # Determine if this var is at its default for this example.
            default = _is_default(k, e.target)
            if default:
                continue
            F_keep.append(fa[i]); B_keep.append(ba[i])
    F_keep = np.asarray(F_keep); B_keep = np.asarray(B_keep)
    if F_keep.size == 0:
        return {"n": 0}
    return {
        "n": int(F_keep.size),
        "F_mean": float(F_keep.mean()),
        "B_mean": float(B_keep.mean()),
        "G_mean": float((F_keep - B_keep).mean()),
    }


def _is_default(var_key: str, target: dict) -> bool:
    if var_key == "total":
        return abs(float(target["total"])) < 1e-9
    kind, _, name = var_key.partition(":")
    if kind == "counts":
        return int(target["counts"].get(name, 0)) == 0
    if kind == "flag":
        return bool(target["flags"].get(name, False)) is False
    if kind == "prices":
        return False   # prices have no natural 'default' that B can guess
    return False


def stratified_null(F: np.ndarray, B: np.ndarray, strata,
                    q: float = 0.95, n_null: int = 1000,
                    seed: int = 0) -> float:
    """Permutation null that shuffles F *within* each stratum (e.g.
    trajectory length), so the null preserves the per-stratum marginal
    of F and tau reflects genuine F-B pairing structure rather than the
    global mean difference."""
    rng = np.random.RandomState(seed)
    F = np.nan_to_num(np.asarray(F, float), nan=np.nanmean(F))
    B = np.nan_to_num(np.asarray(B, float), nan=np.nanmean(B))
    strata = np.asarray(strata)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    out = np.empty(n_null)
    for j in range(n_null):
        Fp = F.copy()
        for g in groups:
            Fp[g] = F[g][rng.permutation(len(g))]
        out[j] = float(np.mean(np.abs(Fp - B)))
    return float(np.percentile(out, 100 * q))
