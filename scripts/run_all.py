"""End-to-end pipeline runner.

Sequence:
    1.  generate Cart-State dataset                  (Phase 1)
    2.  extract GPT-2 hidden states                  (Phase 2)
    3.  train linear probes                          (Phase 3)
    4.  generate cart-state JSON & score behavior    (Phase 4)
    5.  compute fidelity-behavior gap and stats      (Phase 5)
    6.  generate MiniGrid dataset, extract & probe   (Phase 6)
    7.  run interventions                            (Phase 7)
    8.  emit all figures + tables                    (paper artifacts)

Two preset speeds:
    --quick : 5 trajectories per rung, 5 MiniGrid trajs/env, 20 interventions
    --full  : full ladder (50 per rung), 20 MiniGrid/env, 60 interventions

Outputs go to ``results/`` and ``paper/figures/`` and ``paper/tables/``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cart_state import generate_dataset, LADDER
from cart_state.generator import (
    Rung, save_dataset, load_dataset, ITEM_POOL, FLAG_POOL,
)
from probing import extract_hidden_states
from probing.extract import Extractor
from probing.train_probes import (
    train_probe_suite, fidelity_per_example, save_results,
)
from evaluation.behavior import evaluate_behavior, save_behavior
from analysis.gap import (
    compute_gap, fit_gap_regression, permutation_p_value, per_rung_p_values,
)
from minigrid_val.validate import (
    generate_minigrid_dataset, save_minigrid,
    train_minigrid_probes, evaluate_minigrid_behavior, minigrid_backend,
)
from minigrid_val.validate import MiniGridTrajectory
from probing.extract import ProbeExample as PE
from interventions.patch import (
    run_intervention_suite, save_intervention_results, paired_sign_test_p,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"


def banner(msg: str) -> None:
    print()
    print("=" * 78)
    print(msg)
    print("=" * 78)


def run(args) -> None:
    RESULTS.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    (ROOT / "data").mkdir(exist_ok=True)

    n_per = 5 if args.quick else 50
    mg_n = 3 if args.quick else 20
    intervene_n = 12 if args.quick else 60

    if args.skip_cart:
        trajs = load_dataset(ROOT / "data" / "cart_state.json")
    else:
        banner(f"Phase 1: generate Cart-State (n_per_rung = {n_per})")
        ladder = tuple(Rung(r.length, r.n_items, r.n_flags, n_per,
                            r.allowed, r.name) for r in LADDER)
        trajs = generate_dataset(args.seed, ladder)
        save_dataset(trajs, ROOT / "data" / "cart_state.json")
    print(f"  {len(trajs)} trajectories")

    extractor = Extractor(model_name=args.model)
    print(f"  model={args.model}  device={extractor.device}  "
          f"layers={extractor.n_layers + 1}  d={extractor.d_model}")

    banner("Phase 2: extract hidden states")
    t0 = time.time()
    H, examples = extract_hidden_states(
        trajs, CACHE / "hidden", model_name=args.model,
        device=extractor.device, every_step=False,
        show_progress=True,
    )
    print(f"  H.shape={tuple(H.shape)}  elapsed={time.time() - t0:.1f}s")

    banner("Phase 3: train linear probes")
    t0 = time.time()
    probe_results = train_probe_suite(H, examples, ITEM_POOL, FLAG_POOL)
    save_results(probe_results, RESULTS / "probes.json")
    F = fidelity_per_example(probe_results, examples)
    print(f"  trained {len(probe_results)} probes  elapsed={time.time() - t0:.1f}s")
    print(f"  F summary: mean={np.nanmean(F):.3f}  std={np.nanstd(F):.3f}")

    banner("Phase 4: behavioral generation + scoring")
    t0 = time.time()
    behavior = evaluate_behavior(trajs, examples, extractor=extractor,
                                 keep_raw=True)
    save_behavior(behavior, RESULTS / "behavior.json")
    B = np.asarray(behavior.per_example_overall)
    print(f"  B summary: mean={np.nanmean(B):.3f}  std={np.nanstd(B):.3f}")
    print(f"  elapsed={time.time() - t0:.1f}s")

    banner("Phase 5: gap analysis")
    lengths = np.array([e.length for e in examples])
    rungs = [e.rung for e in examples]
    gap = compute_gap(F, B, lengths, rungs)
    reg = fit_gap_regression(F, B, lengths)
    # Real significance tests (same null _null_distribution/tau already
    # calibrates against) — these back any "reject H0" claim in the paper.
    # n_null=10_000 so p < 1e-3 is resolvable (a permutation test cannot
    # certify a p-value finer than ~1/n_null).
    p_overall = permutation_p_value(F, B, n_null=10_000)
    p_by_rung = per_rung_p_values(F, B, rungs, n_null=10_000)
    print(f"  τ={gap.threshold_tau:.3f}  p(overall, H0: no F-B pairing)={p_overall:.4f}")
    by_L = gap.by_length()
    for L, row in by_L.items():
        p_val = p_by_rung.get(f"L{L}", float("nan"))
        print(f"  L={L:>2}  F={row['F_mean']:.2f}  B={row['B_mean']:.2f}  "
              f"G={row['G_mean']:+.2f}  WM%={row['wm_rate']:.2f}  "
              f"DEP%={row['deployment_rate']:.2f}  n={row['n']}  p={p_val:.4f}")

    # Persist gap stats for paper.
    with open(RESULTS / "gap.json", "w") as f:
        json.dump({
            "summary": gap.summary,
            "by_length": gap.by_length(),
            "tau": gap.threshold_tau,
            "regression": reg,
            "p_overall": p_overall,
            "p_by_rung": p_by_rung,
            "F": F.tolist(), "B": B.tolist(), "G": gap.G.tolist(),
            "length": lengths.tolist(), "rung": rungs,
            "failure_mode": gap.failure_mode,
            "null_distribution": gap.null_distribution.tolist(),
        }, f)

    banner("Phase 6: MiniGrid validation")
    mg_backend = minigrid_backend()
    if mg_backend == "synthetic_fallback":
        print("  WARNING: real `minigrid`/`gymnasium` packages not "
              "importable — falling back to the synthetic toy state "
              "machine. Any MiniGrid numbers from this run are NOT real "
              "MiniGrid environment data; install minigrid>=2.3 and "
              "gymnasium>=0.29 (see requirements.txt) before trusting "
              "results/minigrid.json for publication.")
    mg_trajs = generate_minigrid_dataset(n_per_env=mg_n, base_seed=args.seed)
    save_minigrid(mg_trajs, ROOT / "data" / "minigrid.json")
    print(f"  {len(mg_trajs)} MiniGrid trajectories  (backend={mg_backend})")
    # Reuse extraction pipeline on MiniGrid prompts.
    mg_examples = []
    mg_pairs = []
    for ti, t in enumerate(mg_trajs):
        for s in t.checkpoints:
            mg_examples.append(PE(
                traj_id=t.traj_id, rung=t.task, checkpoint=s,
                length=len(t.actions), target=dict(t.states[s]),
                items=[], flags=[],
            ))
            mg_pairs.append((ti, s))
    Hmg = torch.empty(len(mg_examples),
                      extractor.n_layers + 1, extractor.d_model)
    from tqdm import tqdm as _tqdm
    for k, (ti, s) in enumerate(_tqdm(mg_pairs, desc="mg-extract")):
        Hmg[k] = extractor.encode(mg_trajs[ti].render_query(s))
    torch.save(Hmg, CACHE / "minigrid_hidden.pt")
    print("  MiniGrid hidden states cached")
    mg_probes = train_minigrid_probes(Hmg, mg_examples)
    print("  MiniGrid probes: " + ", ".join(
        f"{k}={v.best_score:.2f}" for k, v in mg_probes.items()))
    mg_behavior = evaluate_minigrid_behavior(mg_trajs, mg_examples, extractor)
    with open(RESULTS / "minigrid.json", "w") as f:
        json.dump({
            "backend": mg_backend,
            "probes": {k: v.__dict__ for k, v in mg_probes.items()},
            "behavior": mg_behavior,
            "examples": [e.__dict__ for e in mg_examples],
            "trajectories": [t.to_dict() for t in mg_trajs],
        }, f)

    banner("Phase 7: interventions")
    # Pick the best layer for the "total" probe as a representative target.
    target_layer = probe_results["total"].best_layer
    if target_layer < 0:
        target_layer = H.shape[1] // 2
    print(f"  target layer = {target_layer}")
    rows = run_intervention_suite(
        extractor, trajs, examples, H, B,
        target_layer=target_layer,
        max_examples=intervene_n,
    )
    save_intervention_results(rows, RESULTS / "interventions.json")
    base_arr = np.array([r.baseline_score for r in rows])
    pa_arr = np.array([r.patched_score for r in rows])
    st_arr = np.array([r.steered_score for r in rows])
    co_arr = np.array([r.constrained_score for r in rows])
    bl = float(base_arr.mean())
    pa = float(pa_arr.mean())
    st = float(st_arr.mean())
    co = float(co_arr.mean())
    sig_patch = paired_sign_test_p(base_arr, pa_arr)
    sig_steer = paired_sign_test_p(base_arr, st_arr)
    sig_constrain = paired_sign_test_p(base_arr, co_arr)
    print(f"  mean B over {len(rows)} examples (paired bootstrap / sign-test p-values):")
    print(f"    baseline    : {bl:.3f}")
    print(f"    patched     : {pa:.3f}  (Δ {pa-bl:+.3f}, "
          f"p_boot={sig_patch['p_bootstrap']:.4f}, p_sign={sig_patch['p_sign_test']:.4f})")
    print(f"    steered     : {st:.3f}  (Δ {st-bl:+.3f}, "
          f"p_boot={sig_steer['p_bootstrap']:.4f}, p_sign={sig_steer['p_sign_test']:.4f})")
    print(f"    constrained : {co:.3f}  (Δ {co-bl:+.3f}, "
          f"p_boot={sig_constrain['p_bootstrap']:.4f}, p_sign={sig_constrain['p_sign_test']:.4f})")
    with open(RESULTS / "intervention_significance.json", "w") as f:
        json.dump({"patch": sig_patch, "steer": sig_steer,
                   "constrain": sig_constrain}, f, indent=2)

    banner("Pipeline complete")
    print(f"  results/  ->  {RESULTS}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Tiny sub-set for smoke testing.")
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-cart", action="store_true",
                    help="Re-use existing data/cart_state.json")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
