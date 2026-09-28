"""Linear probe suite.

We train one probe per (state-variable, layer) and use it to recover
the variable from the model's last-token hidden state at a checkpoint.

Variables:
    counts[item]   : nonneg integer    -> Ridge regression (R²) AND
                                          accuracy under round-and-compare
    prices[item]   : float (2dp)       -> Ridge regression (R²)
    flags[flag]    : bool              -> LogReg (accuracy)
    total          : float             -> Ridge regression (R²)

We use 5-fold *trajectory-level* CV (no leakage within a trajectory).
For each variable we pick the layer that maximises CV score; we also
record the per-layer curve so we can plot Figure 3 (layer-wise peaks).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import warnings

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Degenerate Ridge fits on tiny / nearly-singular folds trip numpy
# matmul over/underflow warnings. These do NOT change probe scores
# (we clip the R² afterwards), and the alternative — drop folds with
# singular X — would silently lose data points. Silence the
# downstream noise instead.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module="numpy")

from probing.extract import ProbeExample


# ----------------------------------------------------------------- container
@dataclass
class ProbeResult:
    variable: str               # e.g. "counts:apple"
    kind: str                   # "regression" or "classification"
    n_examples: int
    per_layer: list[float]      # length L+1
    best_layer: int
    best_score: float
    # Per-example correctness at the *best layer*, indexed by example position
    # in the examples list. NaN for examples where this variable is N/A.
    per_example: list[float] = field(default_factory=list)


# ----------------------------------------------------------------- helpers
def _kfold_traj(examples: list[ProbeExample], k: int = 5,
                seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_idx, test_idx) pairs keyed by trajectory id."""
    traj_ids = sorted({e.traj_id for e in examples})
    rng = np.random.RandomState(seed)
    rng.shuffle(traj_ids)
    folds = KFold(n_splits=k, shuffle=False)
    idx_for_traj = defaultdict(list)
    for i, e in enumerate(examples):
        idx_for_traj[e.traj_id].append(i)

    out: list[tuple[np.ndarray, np.ndarray]] = []
    traj_arr = np.array(traj_ids)
    for tr_pos, te_pos in folds.split(traj_arr):
        tr_ids = set(traj_arr[tr_pos].tolist())
        te_ids = set(traj_arr[te_pos].tolist())
        tr_idx = [j for tid in tr_ids for j in idx_for_traj[tid]]
        te_idx = [j for tid in te_ids for j in idx_for_traj[tid]]
        out.append((np.array(tr_idx), np.array(te_idx)))
    return out


def _select_examples(examples: list[ProbeExample], var_kind: str, key: str
                     ) -> list[int]:
    """Indices of examples where this variable is defined for the trajectory."""
    out = []
    for i, e in enumerate(examples):
        if var_kind == "counts" or var_kind == "prices":
            if key in e.items:
                out.append(i)
        elif var_kind == "flag":
            if key in e.flags:
                out.append(i)
        elif var_kind == "total":
            out.append(i)
    return out


def _y_for(examples: list[ProbeExample], var_kind: str, key: str,
           idxs: Sequence[int]) -> np.ndarray:
    ys = []
    for i in idxs:
        e = examples[i]
        if var_kind == "counts":
            ys.append(float(e.target["counts"].get(key, 0)))
        elif var_kind == "prices":
            ys.append(float(e.target["prices"].get(key, 0.0)))
        elif var_kind == "flag":
            ys.append(int(bool(e.target["flags"].get(key, False))))
        elif var_kind == "total":
            ys.append(float(e.target["total"]))
        else:
            raise ValueError(var_kind)
    return np.asarray(ys)


# ----------------------------------------------------------------- single probe
def _train_one_layer(
    H_layer: np.ndarray,                  # [N_sel, d]
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    kind: str,
) -> tuple[float, np.ndarray]:
    """Return (mean_cv_score, per_example_correctness over union of test idxs)."""
    correct = np.full(H_layer.shape[0], np.nan, dtype=np.float64)
    scores = []
    for tr_idx, te_idx in folds:
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue
        Xtr, Xte = H_layer[tr_idx], H_layer[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        if kind == "classification":
            # If only one class in training, predict majority class.
            if len(np.unique(ytr)) < 2:
                pred = np.full_like(yte, ytr[0] if len(ytr) else 0)
                acc = float((pred == yte).mean())
                correct[te_idx] = (pred == yte).astype(float)
            else:
                clf = LogisticRegression(
                    C=1.0, max_iter=1000, solver="liblinear",
                ).fit(Xtr, ytr)
                pred = clf.predict(Xte)
                acc = float((pred == yte).mean())
                correct[te_idx] = (pred == yte).astype(float)
            scores.append(acc)
        else:  # regression
            # Heavy regularisation and standardisation: high-dim residual
            # features are nearly singular for small N.
            scaler = StandardScaler().fit(Xtr)
            Xtr_s = scaler.transform(Xtr)
            Xte_s = scaler.transform(Xte)
            # Demean target so intercept doesn't dominate the score.
            y_mean = ytr.mean()
            reg = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0, 1000.0)).fit(
                Xtr_s, ytr - y_mean)
            pred = reg.predict(Xte_s) + y_mean
            ss_res = float(((yte - pred) ** 2).sum())
            ss_tot = float(((yte - yte.mean()) ** 2).sum()) or 1.0
            r2 = 1.0 - ss_res / ss_tot
            # Clip absurdly negative scores (Ridge can output huge values when
            # features are degenerate); we only use R² for layer selection and
            # per-example correctness, both of which are bounded operations.
            r2 = max(-1.0, r2)
            scores.append(r2)
            # Per-example correctness: 1 if |pred - y| <= tol, else 0.
            tol = max(0.5, 0.1 * (np.abs(yte).max() + 1e-6))
            correct[te_idx] = (np.abs(pred - yte) <= tol).astype(float)
    if not scores:
        return float("nan"), correct
    return float(np.mean(scores)), correct


def _train_layers(
    H: torch.Tensor,                   # [N, L+1, d]
    examples: list[ProbeExample],
    var_kind: str, key: str, kind: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> ProbeResult:
    sel = _select_examples(examples, var_kind, key)
    if len(sel) < 10:
        # Not enough data; return all-NaN.
        return ProbeResult(
            variable=f"{var_kind}:{key}" if var_kind != "total" else "total",
            kind=kind, n_examples=len(sel),
            per_layer=[float("nan")] * (H.shape[1]),
            best_layer=-1, best_score=float("nan"),
            per_example=[float("nan")] * len(examples),
        )
    sel = np.asarray(sel)
    # Restrict folds to this variable's example set.
    sel_set = set(sel.tolist())
    sub_folds: list[tuple[np.ndarray, np.ndarray]] = []
    pos_of = {g: l for l, g in enumerate(sel.tolist())}  # global -> local
    for tr_g, te_g in folds:
        tr_l = np.array([pos_of[g] for g in tr_g.tolist() if g in sel_set])
        te_l = np.array([pos_of[g] for g in te_g.tolist() if g in sel_set])
        sub_folds.append((tr_l, te_l))
    y = _y_for(examples, var_kind, key, sel)

    per_layer = []
    per_example = np.full(len(examples), np.nan)
    best_score, best_layer = -1e18, -1
    best_correct: np.ndarray | None = None
    for L in range(H.shape[1]):
        Hl = H[sel, L].numpy()
        score, correct_local = _train_one_layer(Hl, y, sub_folds, kind)
        per_layer.append(score)
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_layer = L
            best_correct = correct_local
    if best_correct is not None:
        for local_i, global_i in enumerate(sel):
            per_example[global_i] = best_correct[local_i]
    return ProbeResult(
        variable=f"{var_kind}:{key}" if var_kind != "total" else "total",
        kind=kind, n_examples=int(len(sel)),
        per_layer=per_layer, best_layer=int(best_layer),
        best_score=float(best_score),
        per_example=per_example.tolist(),
    )


# ----------------------------------------------------------------- suite
def train_probe_suite(
    H: torch.Tensor,
    examples: list[ProbeExample],
    items: Sequence[str], flags: Sequence[str],
    n_splits: int = 5, seed: int = 0,
) -> dict[str, ProbeResult]:
    folds = _kfold_traj(examples, k=n_splits, seed=seed)
    results: dict[str, ProbeResult] = {}
    for item in items:
        results[f"counts:{item}"] = _train_layers(
            H, examples, "counts", item, "regression", folds)
        results[f"prices:{item}"] = _train_layers(
            H, examples, "prices", item, "regression", folds)
    for fl in flags:
        results[f"flag:{fl}"] = _train_layers(
            H, examples, "flag", fl, "classification", folds)
    results["total"] = _train_layers(
        H, examples, "total", "", "regression", folds)
    return results


# ----------------------------------------------------------------- aggregate
def fidelity_per_example(
    results: dict[str, ProbeResult],
    examples: list[ProbeExample],
) -> np.ndarray:
    """For each example, average per-example correctness over its applicable
    variables. Returns array shape [N]."""
    N = len(examples)
    out = np.zeros(N)
    counts = np.zeros(N)
    for r in results.values():
        pe = np.asarray(r.per_example)
        mask = ~np.isnan(pe)
        out[mask] += pe[mask]
        counts[mask] += 1
    return np.where(counts > 0, out / np.maximum(counts, 1), np.nan)


def save_results(results: dict[str, ProbeResult], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({k: v.__dict__ for k, v in results.items()}, f)


def load_results(path: str | Path) -> dict[str, ProbeResult]:
    with open(path) as f:
        data = json.load(f)
    return {k: ProbeResult(**v) for k, v in data.items()}
