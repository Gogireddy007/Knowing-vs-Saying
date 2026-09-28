"""Cross-architecture, multi-seed pressure-test battery.

For each (model, seed) configuration we:
    1.  Generate a 5-trajectories-per-rung Cart-State dataset.
    2.  Extract hidden states with that model.
    3.  Train **three** probe families: linear, MLP-256, random-label.
    4.  Generate behavioral predictions.
    5.  Compute the signed gap, |G|, and bootstrap CIs.

We additionally run two pressure tests per configuration:
    PT1  noise:        corrupt 0/10/20% of instructions; check whether
                       behavior collapses faster than fidelity.
    PT2  OOD length:   train probes on L ≤ 30 examples; test on L = 40.

Outputs land in results/battery/{model}_{seed}/ and a top-level
results/battery_summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from itertools import product
from pathlib import Path

# Disable TF/Flax BEFORE importing transformers (segfault on this box).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import torch

from cart_state import generate_dataset, LADDER
from cart_state.generator import Rung, save_dataset
from probing.extract import Extractor, extract_hidden_states
from probing.train_probes import (
    train_probe_suite, fidelity_per_example, save_results,
)
from probing.probes_advanced import train_advanced_suite, selectivity
from evaluation.behavior import evaluate_behavior
from analysis.gap import compute_gap, fit_gap_regression
from cart_state.generator import ITEM_POOL, FLAG_POOL


ROOT = Path(__file__).resolve().parents[1]
BAT = ROOT / "results" / "battery"
BAT.mkdir(parents=True, exist_ok=True)


def _bootstrap_ci(x: np.ndarray, n_boot: int = 1000,
                  q_lo: float = 0.025, q_hi: float = 0.975,
                  seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.array([rng.choice(x, size=len(x), replace=True).mean()
                      for _ in range(n_boot)])
    return float(x.mean()), float(np.quantile(means, q_lo)), \
           float(np.quantile(means, q_hi))


# ----------------------------------------------------------------- one config
def run_one(model_name: str, seed: int, n_per: int = 5,
            n_intervene: int = 0) -> dict:
    cfg_id = f"{model_name.replace('/', '_')}_s{seed}"
    out_dir = BAT / cfg_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ladder = tuple(Rung(r.length, r.n_items, r.n_flags, n_per,
                        r.allowed, r.name) for r in LADDER)

    t0 = time.time()
    trajs = generate_dataset(seed, ladder)
    save_dataset(trajs, out_dir / "data.json")

    extractor = Extractor(model_name=model_name)
    H, examples = extract_hidden_states(
        trajs, out_dir / "hidden",
        model_name=model_name, device=extractor.device,
        show_progress=False)

    # ---- probes
    linear = train_probe_suite(H, examples, ITEM_POOL, FLAG_POOL,
                               n_splits=5, seed=seed)
    save_results(linear, out_dir / "probes_linear.json")
    # MLP/random probes only scan a 3-layer band around the linear best
    # layer (huge speed-up; we care about the CEILING, not the curve).
    layer_hint = {k: v.best_layer for k, v in linear.items()}
    mlp = train_advanced_suite(H, examples, ITEM_POOL, FLAG_POOL,
                               method="mlp", n_splits=5, seed=seed,
                               layer_hint=layer_hint, layer_window=1)
    save_results(mlp, out_dir / "probes_mlp.json")
    rand = train_advanced_suite(H, examples, ITEM_POOL, FLAG_POOL,
                                method="random", n_splits=5, seed=seed,
                                layer_hint=layer_hint, layer_window=0)
    save_results(rand, out_dir / "probes_random.json")
    sel = selectivity(linear, rand)

    F_lin = fidelity_per_example(linear, examples)
    F_mlp = fidelity_per_example(mlp, examples)
    F_rnd = fidelity_per_example(rand, examples)

    # ---- behavior (reuse extractor — no second model load)
    beh = evaluate_behavior(trajs, examples, extractor=extractor,
                            keep_raw=False, show_progress=False)
    B = np.asarray(beh.per_example_overall)
    Ls = np.asarray([e.length for e in examples])
    R = [e.rung for e in examples]

    # ---- gap with both probe ceilings
    gap_lin = compute_gap(F_lin, B, Ls, R)
    gap_mlp = compute_gap(F_mlp, B, Ls, R)

    # ---- bootstrap CIs
    F_lin_mean, F_lin_lo, F_lin_hi = _bootstrap_ci(F_lin, seed=seed)
    F_mlp_mean, F_mlp_lo, F_mlp_hi = _bootstrap_ci(F_mlp, seed=seed)
    B_mean, B_lo, B_hi = _bootstrap_ci(B, seed=seed)
    G_lin_mean, G_lin_lo, G_lin_hi = _bootstrap_ci(gap_lin.G, seed=seed)
    G_mlp_mean, G_mlp_lo, G_mlp_hi = _bootstrap_ci(gap_mlp.G, seed=seed)

    summary = {
        "model": model_name, "seed": seed, "n_examples": int(len(examples)),
        "elapsed_s": float(time.time() - t0),
        "F_linear": [F_lin_mean, F_lin_lo, F_lin_hi],
        "F_mlp": [F_mlp_mean, F_mlp_lo, F_mlp_hi],
        "F_random": [float(np.nanmean(F_rnd)),
                     float(np.nanpercentile(F_rnd, 2.5)),
                     float(np.nanpercentile(F_rnd, 97.5))],
        "B": [B_mean, B_lo, B_hi],
        "G_linear": [G_lin_mean, G_lin_lo, G_lin_hi],
        "G_mlp": [G_mlp_mean, G_mlp_lo, G_mlp_hi],
        "tau_linear": gap_lin.threshold_tau,
        "tau_mlp": gap_mlp.threshold_tau,
        "by_length_linear": gap_lin.by_length(),
        "by_length_mlp": gap_mlp.by_length(),
        "selectivity_mean": float(np.nanmean(list(sel.values()))),
        "selectivity_per_var": sel,
        "regression_linear": fit_gap_regression(F_lin, B, Ls),
        "regression_mlp": fit_gap_regression(F_mlp, B, Ls),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f)
    return summary


# ----------------------------------------------------------------- pressure tests
def pressure_test_noise(model_name: str, seed: int, n_per: int = 5,
                        noise_levels=(0.0, 0.1, 0.2)) -> dict:
    """Corrupt a fraction of instructions and re-measure F & B."""
    import random
    out: dict = {}
    extractor = Extractor(model_name=model_name)
    ladder = tuple(Rung(r.length, r.n_items, r.n_flags, n_per,
                        r.allowed, r.name) for r in LADDER)
    for p in noise_levels:
        trajs = generate_dataset(seed, ladder)
        rng = random.Random(seed * 991 + int(1000 * p))
        # Corrupt: replace some instruction text with garbage tokens.
        for t in trajs:
            for i in range(len(t.instructions)):
                if rng.random() < p:
                    t.instructions[i].args = tuple(
                        rng.choice("qzxv?#") if isinstance(a, str)
                        else (a + rng.randint(-3, 3) if isinstance(a, int)
                              else a)
                        for a in t.instructions[i].args)
        # Run lite pipeline
        H, ex = extract_hidden_states(
            trajs, BAT / f"noise_{p}_{seed}", model_name=model_name,
            device=extractor.device, show_progress=False)
        linear = train_probe_suite(H, ex, ITEM_POOL, FLAG_POOL)
        F = fidelity_per_example(linear, ex)
        beh = evaluate_behavior(trajs, ex, extractor=extractor,
                                keep_raw=False, show_progress=False)
        B = np.asarray(beh.per_example_overall)
        out[str(p)] = {"F_mean": float(np.nanmean(F)),
                       "B_mean": float(np.nanmean(B)),
                       "G_mean": float(np.nanmean(F - B))}
    return out


def pressure_test_ood(model_name: str, seed: int, n_per: int = 8) -> dict:
    """Probe trained on L ≤ 30 examples, tested on L = 40 examples only."""
    from probing.train_probes import _train_layers, _kfold_traj
    extractor = Extractor(model_name=model_name)
    ladder = tuple(Rung(r.length, r.n_items, r.n_flags, n_per,
                        r.allowed, r.name) for r in LADDER)
    trajs = generate_dataset(seed, ladder)
    H, ex = extract_hidden_states(
        trajs, BAT / f"ood_{seed}", model_name=model_name,
        device=extractor.device, show_progress=False)
    in_dist = np.array([i for i, e in enumerate(ex) if e.length <= 30])
    ood = np.array([i for i, e in enumerate(ex) if e.length == 40])
    if in_dist.size == 0 or ood.size == 0:
        return {"note": "missing in-dist or OOD subset"}
    # Train probe on in-dist examples for the 'total' variable, eval on OOD.
    from sklearn.linear_model import Ridge
    y_train = np.array([ex[i].target["total"] for i in in_dist])
    y_test = np.array([ex[i].target["total"] for i in ood])
    layer_scores = []
    for L in range(H.shape[1]):
        Xtr = H[in_dist, L].numpy()
        Xte = H[ood, L].numpy()
        r = Ridge(alpha=1.0).fit(Xtr, y_train)
        pred = r.predict(Xte)
        ss_res = float(((y_test - pred) ** 2).sum())
        ss_tot = float(((y_test - y_test.mean()) ** 2).sum()) or 1.0
        layer_scores.append(1.0 - ss_res / ss_tot)
    return {"per_layer_R2": layer_scores,
            "best_R2": float(max(layer_scores)),
            "n_in": int(in_dist.size), "n_ood": int(ood.size)}


# ----------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gpt2", "gpt2-medium"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-per", type=int, default=5)
    ap.add_argument("--pressure", action="store_true",
                    help="Also run noise + OOD pressure tests.")
    args = ap.parse_args()

    summaries = []
    for model in args.models:
        for seed in args.seeds:
            print(f"\n>>> {model}  seed={seed}")
            t0 = time.time()
            s = run_one(model, seed, n_per=args.n_per)
            print(f"    F_lin={s['F_linear'][0]:.3f}  "
                  f"F_mlp={s['F_mlp'][0]:.3f}  "
                  f"B={s['B'][0]:.3f}  "
                  f"G_mlp={s['G_mlp'][0]:+.3f}  "
                  f"τ_mlp={s['tau_mlp']:.3f}  "
                  f"sel={s['selectivity_mean']:.3f}  "
                  f"in {time.time() - t0:.0f}s")
            summaries.append(s)

    if args.pressure:
        for model in args.models:
            for seed in args.seeds[:1]:  # one seed is enough for PT
                print(f"\n>>> pressure: {model} seed={seed}")
                noise = pressure_test_noise(model, seed, n_per=args.n_per)
                ood = pressure_test_ood(model, seed, n_per=args.n_per)
                with open(BAT / f"pressure_{model.replace('/', '_')}_s{seed}.json",
                          "w") as f:
                    json.dump({"noise": noise, "ood": ood}, f)
                print(f"    noise:", noise)
                print(f"    ood   R² best: {ood.get('best_R2', float('nan')):.3f}")

    with open(ROOT / "results" / "battery_summary.json", "w") as f:
        json.dump(summaries, f)
    print(f"\nwrote battery summary -> {ROOT / 'results' / 'battery_summary.json'}")


if __name__ == "__main__":
    main()
