"""Advanced probe controls used to make F honest:

  * RandomLabel  — train on shuffled labels. Establishes a chance baseline;
                   anything the probe achieves above this is genuine signal.
  * MLP          — 2-layer MLP with ReLU; a *non-linear* upper bound on what
                   the residual stream contains. If MLP ≫ Linear, the linear
                   bottleneck is the reason F looks small and the signed gap
                   may flip sign.
  * Selectivity  — Hewitt-Liang style: gap between true-label probe accuracy
                   and random-label probe accuracy on the same architecture.

These controls let us write: "an observed |G| ≥ τ persists even when the
probe is given strictly more capacity than what the model itself uses."
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Suppress degenerate-fit overflows (see probing/train_probes.py).
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning,
                        module="sklearn")

from probing.train_probes import (
    _kfold_traj, _select_examples, _y_for, ProbeResult,
)


def _train_one_layer_mlp(H_layer: np.ndarray, y: np.ndarray,
                         folds, kind: str,
                         hidden: tuple[int, ...] = (256,)) -> tuple[float, np.ndarray]:
    """Same interface as train_probes._train_one_layer but with MLP probes."""
    correct = np.full(H_layer.shape[0], np.nan, dtype=np.float64)
    scores = []
    for tr_idx, te_idx in folds:
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue
        Xtr, Xte = H_layer[tr_idx], H_layer[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        if kind == "classification":
            if len(np.unique(ytr)) < 2:
                pred = np.full_like(yte, ytr[0] if len(ytr) else 0)
            else:
                m = MLPClassifier(hidden_layer_sizes=hidden,
                                  max_iter=80, random_state=0,
                                  early_stopping=False,
                                  alpha=1e-2)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(Xtr, ytr)
                pred = m.predict(Xte)
            acc = float((pred == yte).mean())
            scores.append(acc)
            correct[te_idx] = (pred == yte).astype(float)
        else:
            m = MLPRegressor(hidden_layer_sizes=hidden,
                             max_iter=80, random_state=0,
                             early_stopping=False,
                             alpha=1e-2)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(Xtr, ytr)
            pred = m.predict(Xte)
            ss_res = float(((yte - pred) ** 2).sum())
            ss_tot = float(((yte - yte.mean()) ** 2).sum()) or 1.0
            r2 = max(-1.0, 1.0 - ss_res / ss_tot)
            scores.append(r2)
            tol = max(0.5, 0.1 * (np.abs(yte).max() + 1e-6))
            correct[te_idx] = (np.abs(pred - yte) <= tol).astype(float)
    if not scores:
        return float("nan"), correct
    return float(np.mean(scores)), correct


def _train_one_layer_random(H_layer, y, folds, kind, seed=0):
    """Permute labels within each training fold to build a chance baseline."""
    rng = np.random.RandomState(seed)
    correct = np.full(H_layer.shape[0], np.nan, dtype=np.float64)
    scores = []
    for tr_idx, te_idx in folds:
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue
        y_perm = y.copy()
        y_perm[tr_idx] = y[tr_idx][rng.permutation(len(tr_idx))]
        Xtr, Xte = H_layer[tr_idx], H_layer[te_idx]
        ytr = y_perm[tr_idx]; yte = y[te_idx]
        if kind == "classification":
            if len(np.unique(ytr)) < 2:
                pred = np.full_like(yte, ytr[0] if len(ytr) else 0)
            else:
                m = LogisticRegression(C=1.0, max_iter=1000,
                                       solver="liblinear").fit(Xtr, ytr)
                pred = m.predict(Xte)
            acc = float((pred == yte).mean())
            scores.append(acc)
            correct[te_idx] = (pred == yte).astype(float)
        else:
            r = Ridge(alpha=1.0).fit(Xtr, ytr)
            pred = r.predict(Xte)
            ss_res = float(((yte - pred) ** 2).sum())
            ss_tot = float(((yte - yte.mean()) ** 2).sum()) or 1.0
            r2 = max(-1.0, 1.0 - ss_res / ss_tot)
            scores.append(r2)
            tol = max(0.5, 0.1 * (np.abs(yte).max() + 1e-6))
            correct[te_idx] = (np.abs(pred - yte) <= tol).astype(float)
    if not scores:
        return float("nan"), correct
    return float(np.mean(scores)), correct


def train_advanced_probe(
    H: torch.Tensor, examples, var_kind: str, key: str, kind: str,
    method: str, folds, hidden=(256,), seed=0,
    only_layers: list[int] | None = None,
) -> ProbeResult:
    """Train an advanced (mlp/random) probe.

    To keep MLP cost bounded, callers should set ``only_layers`` to the
    set of layers worth checking (e.g. the linear-probe best layer ± 2).
    If None, every layer is checked.
    """
    sel = _select_examples(examples, var_kind, key)
    var_name = key if var_kind == "total" else f"{var_kind}:{key}"
    n_layers = H.shape[1]
    if len(sel) < 10:
        return ProbeResult(variable=var_name, kind=kind, n_examples=len(sel),
                           per_layer=[float("nan")] * n_layers,
                           best_layer=-1, best_score=float("nan"),
                           per_example=[float("nan")] * len(examples))
    sel = np.asarray(sel)
    sel_set = set(sel.tolist())
    pos_of = {g: l for l, g in enumerate(sel.tolist())}
    sub_folds = []
    for tr_g, te_g in folds:
        tr_l = np.array([pos_of[g] for g in tr_g.tolist() if g in sel_set])
        te_l = np.array([pos_of[g] for g in te_g.tolist() if g in sel_set])
        sub_folds.append((tr_l, te_l))
    y = _y_for(examples, var_kind, key, sel)
    per_layer = [float("nan")] * n_layers
    per_example = np.full(len(examples), np.nan)
    best_score, best_layer, best_correct = -1e18, -1, None
    layers_iter = only_layers if only_layers is not None else range(n_layers)
    for L in layers_iter:
        if L < 0 or L >= n_layers:
            continue
        Hl = H[sel, L].numpy()
        if method == "mlp":
            score, correct = _train_one_layer_mlp(Hl, y, sub_folds, kind, hidden)
        elif method == "random":
            score, correct = _train_one_layer_random(Hl, y, sub_folds, kind, seed)
        else:
            raise ValueError(method)
        per_layer[L] = score
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_layer = L
            best_correct = correct
    if best_correct is not None:
        for li, gi in enumerate(sel):
            per_example[gi] = best_correct[li]
    return ProbeResult(
        variable=var_name, kind=kind, n_examples=int(len(sel)),
        per_layer=per_layer, best_layer=int(best_layer),
        best_score=float(best_score),
        per_example=per_example.tolist(),
    )


def train_advanced_suite(H, examples, items, flags, method,
                         n_splits=5, seed=0, hidden=(256,),
                         layer_hint: dict | None = None,
                         layer_window: int = 1) -> dict:
    """Train advanced probes per variable.

    `layer_hint`: dict of var_name -> best_linear_layer. We then evaluate
    the advanced probe only in a ±`layer_window` band around the linear
    best layer (default ±1 → 3 layers per variable). Massive speed-up
    over scanning all 13–25 layers.
    """
    folds = _kfold_traj(examples, k=n_splits, seed=seed)
    out = {}

    def _layers_for(name: str) -> list[int] | None:
        if not layer_hint:
            return None
        b = layer_hint.get(name)
        if b is None or b < 0:
            return None
        return list(range(max(0, b - layer_window),
                          min(H.shape[1], b + layer_window + 1)))

    for it in items:
        name = f"counts:{it}"
        out[name] = train_advanced_probe(
            H, examples, "counts", it, "regression",
            method, folds, hidden=hidden, seed=seed,
            only_layers=_layers_for(name))
        name = f"prices:{it}"
        out[name] = train_advanced_probe(
            H, examples, "prices", it, "regression",
            method, folds, hidden=hidden, seed=seed,
            only_layers=_layers_for(name))
    for fl in flags:
        name = f"flag:{fl}"
        out[name] = train_advanced_probe(
            H, examples, "flag", fl, "classification",
            method, folds, hidden=hidden, seed=seed,
            only_layers=_layers_for(name))
    out["total"] = train_advanced_probe(
        H, examples, "total", "", "regression",
        method, folds, hidden=hidden, seed=seed,
        only_layers=_layers_for("total"))
    return out


def selectivity(linear: dict, random_baseline: dict) -> dict[str, float]:
    """Per-variable selectivity = score(linear) - score(random),
    clipped to a bounded range so degenerate Ridge fits don't blow up.

    For classification probes both scores are accuracies in [0, 1].
    For regression probes we use the per-example correctness rate
    (within-tolerance) which is bounded in [0, 1] — *not* the raw R²,
    which is unbounded below."""
    out = {}
    for k in linear:
        l = linear[k]
        r = random_baseline.get(k, None)
        if r is None:
            continue
        if l.kind == "classification":
            l_acc = float(l.best_score) if not np.isnan(l.best_score) else 0.0
            r_acc = float(r.best_score) if not np.isnan(r.best_score) else 0.0
        else:
            # Use bounded per-example correctness rate.
            l_pe = np.asarray(l.per_example, dtype=float)
            r_pe = np.asarray(r.per_example, dtype=float)
            l_acc = float(np.nanmean(l_pe)) if np.any(~np.isnan(l_pe)) else 0.0
            r_acc = float(np.nanmean(r_pe)) if np.any(~np.isnan(r_pe)) else 0.0
        out[k] = float(np.clip(l_acc - r_acc, -1.0, 1.0))
    return out
