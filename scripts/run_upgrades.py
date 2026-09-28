"""Run the reviewer-requested upgrades end-to-end and emit real numbers.

U1  pooled-probe sign-flip : F(last) < B but F(mean)/F(attn) -> meets/exceeds B
U2  positive-control patch : nearest-state donor should hurt B less (or help)
                             than a random-state donor
U3  capacity sweep         : last-linear -> last-MLP -> mean-linear ->
                             attn-linear, mean signed gap per variant
U4  modern model           : repeat U1/U3 on Pythia-410m (GPT-NeoX family)
U5  controls               : majority-class B floor, zero-default-excluded
                             gap, stratified null, plug-in MI lower bound

Outputs: results/upgrades.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import torch

from cart_state import generate_dataset, LADDER
from cart_state.generator import Rung, ITEM_POOL, FLAG_POOL
from probing.extract import Extractor
from probing.pooled import extract_pooled
from probing.train_probes import train_probe_suite, fidelity_per_example
from probing.probes_advanced import train_advanced_suite
from evaluation.behavior import evaluate_behavior
from analysis.gap import compute_gap
from analysis.controls import (
    majority_class_behavior, excl_zero_default_gap, stratified_null,
)
from analysis.gap import fano_mi_lower_bound
from interventions.patch import activation_patch
from evaluation.behavior import _extract_first_json, _coerce_state, _score_state

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CACHE = RES / "cache"


def _score_gen(gen, e):
    obj = _extract_first_json(gen) or {}
    pred = _coerce_state(obj, e.items, e.flags) if obj else None
    return float(np.mean(list(_score_state(pred, e.target).values())))


def _state_dist(a: dict, b: dict) -> float:
    d = 0.0
    for it in a["counts"]:
        d += abs(int(a["counts"][it]) - int(b["counts"].get(it, 0)))
    for fl in a["flags"]:
        d += int(bool(a["flags"][fl]) != bool(b["flags"].get(fl, False)))
    d += abs(float(a["total"]) - float(b["total"])) / 10.0
    return d


def per_variable_correct(results, examples):
    """var -> per-example correctness array (NaN where N/A)."""
    return {k: np.asarray(v.per_example) for k, v in results.items()}


def run_model(model_name: str, trajs, n_intervene: int,
              do_patch: bool) -> dict:
    print(f"\n=== {model_name} ===", flush=True)
    t0 = time.time()
    banks, examples = extract_pooled(
        trajs, CACHE / f"pooled_{model_name.replace('/', '_')}",
        model_name=model_name, show_progress=False)
    print(f"  pooled extract: {time.time()-t0:.0f}s", flush=True)

    extractor = Extractor(model_name=model_name)
    beh = evaluate_behavior(trajs, examples, extractor=extractor,
                            keep_raw=True, show_progress=False)
    B = np.asarray(beh.per_example_overall)
    Ls = np.asarray([e.length for e in examples])
    Rs = [e.rung for e in examples]
    print(f"  behavior B mean = {np.nanmean(B):.3f}  ({time.time()-t0:.0f}s)",
          flush=True)

    # --- U3 capacity sweep -------------------------------------------------
    variants = {}
    lin_last = train_probe_suite(banks["last"], examples, ITEM_POOL, FLAG_POOL)
    variants["last-linear"] = fidelity_per_example(lin_last, examples)
    hint = {k: v.best_layer for k, v in lin_last.items()}
    mlp_last = train_advanced_suite(banks["last"], examples, ITEM_POOL,
                                    FLAG_POOL, method="mlp",
                                    layer_hint=hint, layer_window=1)
    variants["last-mlp"] = fidelity_per_example(mlp_last, examples)
    lin_mean = train_probe_suite(banks["mean"], examples, ITEM_POOL, FLAG_POOL)
    variants["mean-linear"] = fidelity_per_example(lin_mean, examples)
    lin_attn = train_probe_suite(banks["attn"], examples, ITEM_POOL, FLAG_POOL)
    variants["attn-linear"] = fidelity_per_example(lin_attn, examples)

    sweep = {}
    for name, F in variants.items():
        g = compute_gap(F, B, Ls, Rs)
        sweep[name] = {
            "F_mean": float(np.nanmean(F)),
            "B_mean": float(np.nanmean(B)),
            "G_mean": float(np.nanmean(F - B)),
            "tau": float(g.threshold_tau),
        }
        print(f"  {name:>12}: F={sweep[name]['F_mean']:.3f} "
              f"B={sweep[name]['B_mean']:.3f} "
              f"G={sweep[name]['G_mean']:+.3f}", flush=True)

    # --- U5 controls -------------------------------------------------------
    # Matched-scope floor: the trivial agent is graded on the SAME variable
    # set _score_state grades real models on (counts+prices+flags+total).
    # The legacy floor (prices excluded) is also reported so the magnitude
    # of the scoring-mismatch artefact is visible, not silently corrected
    # away.
    maj = majority_class_behavior(examples, include_prices=True)
    maj_legacy_no_prices = majority_class_behavior(examples, include_prices=False)
    pv_F = per_variable_correct(lin_mean, examples)
    pv_B = {k: np.asarray(v) for k, v in beh.per_variable.items()}
    excl = excl_zero_default_gap(pv_F, pv_B, examples)
    F_best = variants["mean-linear"]
    tau_strat = stratified_null(F_best, B, Ls)

    # Fano MI lower bound, computed PER VARIABLE TYPE with its own actual
    # support size (flags are binary, |S|=2; counts/total use the
    # dataset's observed distinct-value count as |S| rather than a
    # one-size-fits-all placeholder), then normalised by that variable's
    # own H(S) as Prop. 1 requires. A single pooled "|S|~8" constant
    # applied to every variable family (the previous approach) mixes
    # incompatible support sizes and silently clips to a vacuous 0.0 at
    # every accuracy level this paper observes -- see analysis/gap.py
    # ::fano_mi_lower_bound's docstring for the derivation.
    flag_accs = [float(np.nanmean(lin_mean[k].per_example))
                for k in lin_mean if k.startswith("flag:")
                and not np.all(np.isnan(lin_mean[k].per_example))]
    count_support = {it: max(2, len({int(e.target["counts"].get(it, 0))
                                     for e in examples if it in e.items}))
                     for it in ITEM_POOL}
    mi_counts_per_item = {}
    for it, s in count_support.items():
        key = f"counts:{it}"
        if key not in lin_mean:
            continue
        pe = lin_mean[key].per_example
        if np.all(np.isnan(pe)):
            continue
        acc = float(np.nanmean(pe))
        mi_counts_per_item[it] = fano_mi_lower_bound(acc, s)

    mi_flag = (fano_mi_lower_bound(float(np.mean(flag_accs)), 2)
              if flag_accs else None)
    mi_counts_mean_norm = (
        float(np.mean([m["mi_lower_bound_normalized"]
                      for m in mi_counts_per_item.values()]))
        if mi_counts_per_item else None)

    controls = {
        "majority_class_B": maj,
        "majority_class_B_legacy_no_prices": maj_legacy_no_prices,
        "B_mean": float(np.nanmean(B)),
        "B_beats_majority": bool(np.nanmean(B) > maj),
        "excl_zero_default": excl,
        "tau_stratified": tau_strat,
        "mi_lower_bound_flags": mi_flag,
        "mi_lower_bound_counts_mean_normalized": mi_counts_mean_norm,
        "mi_lower_bound_counts_per_item": mi_counts_per_item,
    }
    print(f"  majority-class B floor (matched scope) = {maj:.3f}  "
          f"(legacy, prices excluded = {maj_legacy_no_prices:.3f})  "
          f"model B = {np.nanmean(B):.3f}  beats={controls['B_beats_majority']}",
          flush=True)
    print(f"  zero-default-excluded gap: {excl}", flush=True)
    print(f"  stratified tau = {tau_strat:.3f}", flush=True)
    if mi_flag:
        print(f"  Fano MI lower bound (flags, |S|=2): "
              f"{mi_flag['mi_lower_bound_normalized']:.3f} normalised "
              f"({mi_flag['mi_lower_bound_bits']:.3f} bits)", flush=True)
    if mi_counts_mean_norm is not None:
        print(f"  Fano MI lower bound (counts, per-item |S|): "
              f"{mi_counts_mean_norm:.3f} normalised (mean over items)",
              flush=True)

    # --- U2 positive-control patch (gpt2 only, expensive) ------------------
    patch = None
    if do_patch:
        target_layer = max(1, lin_mean["total"].best_layer)
        H_last = banks["last"]
        order = np.argsort(B)
        recip = order[:n_intervene]            # lowest-B recipients
        donor_mask = B >= np.nanquantile(B, 0.6)
        donor_idx = np.where(donor_mask)[0]
        rng = np.random.RandomState(0)
        d_near, d_rand, base = [], [], []
        by_id = {t.traj_id: t for t in trajs}
        for k in recip:
            e = examples[int(k)]
            prompt = by_id[e.traj_id].render_query(e.checkpoint)
            base.append(float(B[k]))
            cands = [j for j in donor_idx.tolist()
                     if examples[j].traj_id != e.traj_id]
            if not cands:
                continue
            # nearest-state donor
            nb = min(cands, key=lambda j: _state_dist(e.target,
                                                      examples[j].target))
            gen_n = activation_patch(extractor, prompt,
                                     H_last[nb, target_layer], target_layer)
            d_near.append(_score_gen(gen_n, e))
            # random donor
            rj = int(rng.choice(cands))
            gen_r = activation_patch(extractor, prompt,
                                     H_last[rj, target_layer], target_layer)
            d_rand.append(_score_gen(gen_r, e))
        base = np.asarray(base); d_near = np.asarray(d_near)
        d_rand = np.asarray(d_rand)
        patch = {
            "layer": int(target_layer),
            "baseline_B": float(base.mean()),
            "nearest_state_donor_B": float(d_near.mean()),
            "random_state_donor_B": float(d_rand.mean()),
            "delta_nearest": float(d_near.mean() - base.mean()),
            "delta_random": float(d_rand.mean() - base.mean()),
            "nearest_beats_random": bool(d_near.mean() > d_rand.mean()),
            "n": int(len(base)),
        }
        print(f"  patch positive control: base={patch['baseline_B']:.3f} "
              f"near={patch['nearest_state_donor_B']:.3f} "
              f"rand={patch['random_state_donor_B']:.3f} "
              f"near>rand={patch['nearest_beats_random']}", flush=True)

    return {"model": model_name, "sweep": sweep, "controls": controls,
            "patch": patch, "elapsed_s": float(time.time() - t0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gpt2", "EleutherAI/pythia-410m"])
    ap.add_argument("--n-per", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-intervene", type=int, default=12)
    ap.add_argument("--patch-all", action="store_true",
                    help="Run the positive-control patch on every model in "
                    "--models, not just the first. The paper discusses the "
                    "patch result on more than one model (GPT-2 replicates "
                    "the predicted direction, Qwen does not) -- that "
                    "comparison requires every discussed model to actually "
                    "have a patch entry in upgrades.json, not just the "
                    "first one in the list. Off by default since the patch "
                    "control is the most expensive step per model.")
    args = ap.parse_args()

    rungs = tuple(Rung(r.length, r.n_items, r.n_flags, args.n_per,
                       r.allowed, r.name) for r in LADDER)
    trajs = generate_dataset(args.seed, rungs)
    print(f"{len(trajs)} trajectories", flush=True)

    out = []
    for i, m in enumerate(args.models):
        do_patch = args.patch_all or (i == 0)
        out.append(run_model(m, trajs, args.n_intervene, do_patch=do_patch))

    RES.mkdir(exist_ok=True)
    with open(RES / "upgrades.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {RES / 'upgrades.json'}", flush=True)


if __name__ == "__main__":
    main()
