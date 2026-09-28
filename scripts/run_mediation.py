"""Causal mediation: per-layer activation patching curve.

For each layer ℓ, replace the last-token residual stream with the donor
activation from a known-good example and re-generate. The resulting
ΔB(ℓ) = B_patched(ℓ) - B_baseline curve localises the layer band that
is on the causal path from representation to output.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
from tqdm import tqdm

from cart_state.generator import (
    Rung, ITEM_POOL, FLAG_POOL,
)
from cart_state import generate_dataset, LADDER
from probing.extract import Extractor, extract_hidden_states
from probing.train_probes import train_probe_suite, fidelity_per_example
from evaluation.behavior import (
    evaluate_behavior, _extract_first_json, _coerce_state, _score_state,
)
from interventions.patch import activation_patch


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def _score_gen(gen: str, e) -> float:
    obj = _extract_first_json(gen) or {}
    pred = _coerce_state(obj, e.items, e.flags) if obj else None
    s = _score_state(pred, e.target)
    return float(np.mean(list(s.values())))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-per", type=int, default=3)
    ap.add_argument("--n-examples", type=int, default=12,
                    help="examples to mediate")
    args = ap.parse_args()

    ladder = tuple(Rung(r.length, r.n_items, r.n_flags, args.n_per,
                        r.allowed, r.name) for r in LADDER)
    trajs = generate_dataset(args.seed, ladder)
    by_id = {t.traj_id: t for t in trajs}
    extractor = Extractor(model_name=args.model)

    H, ex = extract_hidden_states(trajs, RES / "cache" / "med_hidden",
                                  model_name=args.model,
                                  device=extractor.device,
                                  show_progress=False)
    print(f"H={tuple(H.shape)}")
    beh = evaluate_behavior(trajs, ex, extractor=extractor,
                            keep_raw=False, show_progress=False)
    B = np.asarray(beh.per_example_overall)

    # Donor pool: top-quartile behavior.
    thresh = np.nanquantile(B, 0.75)
    donor_idxs = np.where(B >= thresh)[0]
    if donor_idxs.size == 0:
        donor_idxs = np.arange(len(B))

    # Recipients: pick the bottom-half by B (those most in need).
    bottom = np.argsort(B)[: args.n_examples]
    # H has shape [N, n_blocks+1, d] where index 0 is the embedding output
    # and indices 1..n_blocks are post-block outputs. Patching is only
    # meaningful at the n_blocks transformer blocks (h[0]..h[n_blocks-1]).
    n_blocks = extractor.n_layers          # property: number of layers
    curve = np.zeros((len(bottom), n_blocks))
    base_scores = []
    rng = np.random.RandomState(0)
    for ri, k in enumerate(tqdm(bottom, desc="mediation")):
        e = ex[int(k)]
        t = by_id[e.traj_id]
        prompt = t.render_query(e.checkpoint)
        base_scores.append(B[k])
        for L in range(n_blocks):
            donor_pool = [j for j in donor_idxs.tolist()
                          if ex[j].traj_id != e.traj_id]
            if not donor_pool:
                curve[ri, L] = base_scores[-1]
                continue
            donor_k = rng.choice(donor_pool)
            # H index L+1 is the post-block-L activation we want to inject
            donor_h = H[donor_k, L + 1]
            gen = activation_patch(extractor, prompt, donor_h, L,
                                   max_new_tokens=180)
            curve[ri, L] = _score_gen(gen, e)
    delta = curve - np.asarray(base_scores)[:, None]
    out = {
        "model": args.model,
        "seed": args.seed,
        "n_layers": n_blocks,
        "baseline_mean": float(np.mean(base_scores)),
        "delta_per_layer_mean": delta.mean(0).tolist(),
        "delta_per_layer_sem": (delta.std(0) /
                                max(1, np.sqrt(len(bottom)))).tolist(),
        "patched_per_layer_mean": curve.mean(0).tolist(),
    }
    with open(RES / "mediation.json", "w") as f:
        json.dump(out, f)
    print(f"wrote {RES / 'mediation.json'}")
    print(f"baseline B mean = {out['baseline_mean']:.3f}")
    print("ΔB(layer) =",
          " ".join(f"{x:+.2f}" for x in out["delta_per_layer_mean"]))


if __name__ == "__main__":
    main()
