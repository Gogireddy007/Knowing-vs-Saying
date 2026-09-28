"""Phase 5: Fidelity-behavior gap analysis.

Definitions
-----------
For example k with applicable variables V_k:
    F_k = (1 / |V_k|) Σ_v 1[probe at best layer recovers v_k]
    B_k = (1 / |V_k|) Σ_v 1[generation matches v_k]
    G_k = F_k - B_k

Null calibration
----------------
We construct a null distribution by independently shuffling the variable-wise
probe predictions across trajectories and recomputing G. The 95th percentile
of |G_null| defines τ.

Failure mode classification
---------------------------
    world-model failure  if F_k < 0.5  and G_k <  τ
    deployment failure   if F_k ≥ 0.5  and G_k ≥  τ
    correct              if B_k ≥ 0.8
    other                otherwise
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class GapStats:
    F: np.ndarray                     # per-example fidelity
    B: np.ndarray                     # per-example behavior
    G: np.ndarray                     # per-example gap
    length: np.ndarray                # per-example trajectory length
    rung: list[str]                   # per-example rung label
    threshold_tau: float
    null_distribution: np.ndarray     # samples of |G_null|
    failure_mode: list[str]           # per-example
    summary: dict = field(default_factory=dict)

    def by_length(self) -> dict[int, dict[str, float]]:
        out: dict[int, dict[str, float]] = {}
        for L in sorted(set(self.length.tolist())):
            m = self.length == L
            sub_fm = [fm for fm, mm in zip(self.failure_mode, m.tolist()) if mm]
            n = max(1, len(sub_fm))
            out[int(L)] = {
                "F_mean": float(np.nanmean(self.F[m])),
                "B_mean": float(np.nanmean(self.B[m])),
                "G_mean": float(np.nanmean(self.G[m])),
                "G_abs_mean": float(np.nanmean(np.abs(self.G[m]))),
                "n": int(m.sum()),
                "deployment_rate": sum(1 for x in sub_fm if x == "deployment") / n,
                "readout_rate": sum(1 for x in sub_fm if x == "readout") / n,
                "wm_rate": sum(1 for x in sub_fm if x == "world_model") / n,
                "correct_rate": sum(1 for x in sub_fm if x == "correct") / n,
            }
        return out


# ----------------------------------------------------------------- compute
def compute_gap(
    F: np.ndarray, B: np.ndarray,
    lengths: Sequence[int], rungs: Sequence[str],
    tau: float | None = None,
    n_null: int = 1000, seed: int = 0,
) -> GapStats:
    F = np.asarray(F, dtype=float)
    B = np.asarray(B, dtype=float)
    G = F - B
    null = _null_distribution(F, B, n_null=n_null, seed=seed)
    if tau is None:
        tau = float(np.percentile(null, 95))
    failure = classify_failure_modes(F, B, G, tau)
    return GapStats(
        F=F, B=B, G=G,
        length=np.asarray(lengths),
        rung=list(rungs),
        threshold_tau=float(tau),
        null_distribution=null,
        failure_mode=failure,
        summary={
            "F_mean": float(np.nanmean(F)),
            "B_mean": float(np.nanmean(B)),
            "G_mean": float(np.nanmean(G)),
            "G_abs_mean": float(np.nanmean(np.abs(G))),
            "tau": float(tau),
            "n": int(len(F)),
        },
    )


def _null_distribution(F: np.ndarray, B: np.ndarray,
                       n_null: int = 1000, seed: int = 0) -> np.ndarray:
    """Sample shuffled |G_null| values.

    If the fidelity-behavior relationship is purely chance, shuffling F across
    examples should produce gaps of the same magnitude. The 95th percentile is
    used as τ.
    """
    rng = np.random.RandomState(seed)
    out = np.empty(n_null)
    n = len(F)
    F_clean = np.nan_to_num(F, nan=np.nanmean(F))
    B_clean = np.nan_to_num(B, nan=np.nanmean(B))
    for i in range(n_null):
        perm = rng.permutation(n)
        out[i] = float(np.mean(np.abs(F_clean[perm] - B_clean)))
    return out


def null_threshold(F: np.ndarray, B: np.ndarray, q: float = 0.95,
                   n_null: int = 1000, seed: int = 0) -> float:
    null = _null_distribution(F, B, n_null=n_null, seed=seed)
    return float(np.percentile(null, 100 * q))


def permutation_p_value(F: np.ndarray, B: np.ndarray,
                        n_null: int = 10_000, seed: int = 0) -> float:
    """Permutation p-value for H0: F carries no pairing structure with B
    beyond their marginals (the same null _null_distribution / tau are
    built from).

    Test statistic: mean(|F - B|), matching the statistic the null
    distribution already uses (mean(|F_perm - B|)) — so this is the
    actual significance test behind a claim like "reject H0 at p < x",
    using the *same* null as the one that calibrates tau, rather than a
    second, inconsistent statistic.

    p = (#{null >= observed} + 1) / (n_null + 1), the standard
    Laplace-smoothed permutation p-value (Davison & Hinkley 1997) so
    p is never reported as exactly 0. A permutation test can only
    resolve p down to ~1/n_null; claiming p < 10^-3 requires
    n_null >= ~10,000 for that resolution to be meaningful at all,
    so the default here is raised from the tau-calibration default of
    1000 to 10,000.
    """
    F = np.asarray(F, dtype=float)
    B = np.asarray(B, dtype=float)
    observed = float(np.nanmean(np.abs(F - B)))
    null = _null_distribution(F, B, n_null=n_null, seed=seed)
    n_ge = int(np.sum(null >= observed))
    return float((n_ge + 1) / (n_null + 1))


def fano_mi_lower_bound(accuracy: float, support_size: int,
                        label_entropy_bits: float | None = None) -> dict:
    """Proposition 1's Fano lower bound, computed honestly.

    I(S;h) >= H(S) - Hb(1-a) - (1-a)*log2(|S|-1)        [bits]

    normalised:  I(S;h)/H(S) >= 1 - [Hb(1-a) + (1-a)*log2(|S|-1)] / H(S)

    Two bugs in the previous plug-in estimate (scripts/run_upgrades.py,
    pre-fix) are corrected here:

      1. |S| was a hardcoded placeholder ("~8") for every variable
         regardless of its actual cardinality. Counts, flags (|S|=2),
         and prices (continuous, effectively unbounded |S|) do not
         share a support size, so a single constant cannot be a valid
         |S| for all of them. Callers must now pass the *actual*
         observed support size for the variable(s) being summarised.
      2. The bound was reported in raw bits, not normalised by H(S) as
         the proposition's own statement requires ("F is a calibrated
         lower bound on I(S_t;h_t)/H(S_t)"). Comparing raw-bit values
         across variables with different H(S) is not meaningful; the
         normalised form is what Prop. 1 actually claims and what is
         comparable across variables.

    If ``label_entropy_bits`` is not supplied we assume a uniform
    distribution over ``support_size`` values (the maximum-entropy,
    most conservative assumption: H(S) = log2(|S|)).

    At the observed accuracy regime in this paper (a ~ 0.4, |S| ~ 8),
    Fano's bound is honestly near-vacuous (clips to 0) because Fano is
    a *weak* bound away from the high-accuracy / small-|S| regime where
    it's informative -- that is a property of Fano's inequality itself,
    not a coding bug. Reporting the bound as 0 in that regime is the
    correct, honest output; the fix here is computing it correctly, not
    making it bigger.
    """
    a = float(np.clip(accuracy, 1e-9, 1.0 - 1e-9))
    s = max(2, int(support_size))
    Hb = -(a * np.log2(a) + (1 - a) * np.log2(1 - a))
    fano_term_bits = Hb + (1 - a) * np.log2(max(1, s - 1))
    Hs_bits = float(label_entropy_bits) if label_entropy_bits is not None \
        else float(np.log2(s))
    raw_bits = max(0.0, Hs_bits - fano_term_bits)
    normalized = max(0.0, 1.0 - fano_term_bits / Hs_bits) if Hs_bits > 0 else 0.0
    return {
        "accuracy": a, "support_size": s, "H_S_bits": Hs_bits,
        "fano_term_bits": float(fano_term_bits),
        "mi_lower_bound_bits": float(raw_bits),
        "mi_lower_bound_normalized": float(normalized),
        "vacuous": bool(normalized <= 0.0),
    }


def per_rung_p_values(F: np.ndarray, B: np.ndarray, rungs,
                      n_null: int = 10_000, seed: int = 0) -> dict[str, float]:
    """permutation_p_value computed independently within each rung, so a
    claim like 'reject H0 in every rung' is backed by one test per rung
    rather than a single pooled test misreported as per-rung."""
    F = np.asarray(F, dtype=float)
    B = np.asarray(B, dtype=float)
    rungs = np.asarray(rungs)
    out: dict[str, float] = {}
    for r in sorted(set(rungs.tolist())):
        m = rungs == r
        out[str(r)] = permutation_p_value(F[m], B[m], n_null=n_null, seed=seed)
    return out


def classify_failure_modes(
    F: np.ndarray, B: np.ndarray, G: np.ndarray, tau: float,
) -> list[str]:
    """Bidirectional gap classification.

    correct           : B >= 0.8 (the model gets the answer)
    deployment        : F - B  >  tau  AND F >= 0.5
                        (representation OK, policy bottleneck)
    readout           : B - F  >  tau  AND B >= 0.5
                        (policy beats single-shot probe — the dominant
                         regime we observe empirically)
    world_model       : max(F, B) < 0.5
                        (both representation and policy are degraded)
    other             : everything else
    """
    out: list[str] = []
    for f, b, g in zip(F, B, G):
        if np.isnan(f) or np.isnan(b):
            out.append("undefined")
            continue
        if b >= 0.8:
            out.append("correct")
        elif g >= tau and f >= 0.5:
            out.append("deployment")
        elif -g >= tau and b >= 0.5:
            out.append("readout")
        elif max(f, b) < 0.5:
            out.append("world_model")
        else:
            out.append("other")
    return out


# ----------------------------------------------------------------- regression
def fit_gap_regression(F: np.ndarray, B: np.ndarray,
                       lengths: Sequence[int]) -> dict:
    """Logistic regression: P[B > 0.8] = σ(β0 + β1 F + β2 L + β3 F*L)."""
    F = np.asarray(F, dtype=float)
    B = np.asarray(B, dtype=float)
    L = np.asarray(lengths, dtype=float)
    y = (B > 0.8).astype(int)
    X = np.column_stack([F, L, F * L])
    if len(np.unique(y)) < 2:
        return {"beta": [float("nan")] * 4,
                "accuracy": float("nan"),
                "note": "single class"}
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    acc = float((clf.predict(X) == y).mean())
    return {
        "beta_0": float(clf.intercept_[0]),
        "beta_1_F": float(clf.coef_[0, 0]),
        "beta_2_L": float(clf.coef_[0, 1]),
        "beta_3_FL": float(clf.coef_[0, 2]),
        "accuracy": acc,
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
    }
