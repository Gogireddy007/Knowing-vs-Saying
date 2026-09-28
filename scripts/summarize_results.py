"""Print a concise text summary of all experimental results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def _load(name: str) -> dict | None:
    p = RES / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main() -> None:
    print("=" * 78)
    print("Dissociation Test — experimental summary")
    print("=" * 78)

    g = _load("gap.json")
    if g:
        s = g["summary"]
        print(f"\n[Cart-State] n={s['n']} examples")
        print(f"  τ (95th pct of null |G|)  = {s['tau']:.3f}")
        print(f"  F_mean = {s['F_mean']:.3f}  "
              f"B_mean = {s['B_mean']:.3f}  "
              f"G_mean = {s['G_mean']:+.3f}  "
              f"|G|_mean = {s['G_abs_mean']:.3f}")
        print("\n  by length:")
        print("    L   |  F     B     G     |G|   WM%   DEP%  RDT%  COR%   n")
        print("    " + "-" * 65)
        for L, row in g["by_length"].items():
            print(f"    {int(L):<3} | {row['F_mean']:.3f} "
                  f"{row['B_mean']:.3f} {row['G_mean']:+.3f} "
                  f"{row['G_abs_mean']:.3f}  {row['wm_rate']:.2f}  "
                  f"{row['deployment_rate']:.2f}  "
                  f"{row.get('readout_rate', 0.0):.2f}  "
                  f"{row.get('correct_rate', 0.0):.2f}   {row['n']}")
        reg = g["regression"]
        if "beta_1_F" in reg:
            print(f"\n  P[B>0.8] regression  (acc={reg['accuracy']:.3f}):")
            print(f"    β₀ = {reg['beta_0']:+.3f}   "
                  f"β₁(F) = {reg['beta_1_F']:+.3f}")
            print(f"    β₂(L) = {reg['beta_2_L']:+.3f}   "
                  f"β₃(F·L) = {reg['beta_3_FL']:+.3f}")

    p = _load("probes.json")
    if p:
        print(f"\n[Probes] {len(p)} variables")
        scored = sorted(
            ((k, v["best_score"], v["best_layer"]) for k, v in p.items()
             if not np.isnan(v["best_score"])),
            key=lambda x: -x[1])
        for k, s, L in scored[:8]:
            print(f"  top:  {k:<22}  score={s:.3f}  best layer={L}")
        for k, s, L in scored[-3:]:
            print(f"  low:  {k:<22}  score={s:.3f}  best layer={L}")

    mg = _load("minigrid.json")
    if mg:
        print(f"\n[MiniGrid]")
        by_task = {}
        for i, ex in enumerate(mg["examples"]):
            by_task.setdefault(ex["rung"], []).append(i)
        beh = mg["behavior"]["per_example_overall"]
        for t, idxs in by_task.items():
            B = float(np.nanmean([beh[i] for i in idxs]))
            print(f"  {t:<14}  n={len(idxs):<3}  B={B:.3f}")

    iv = _load("interventions.json")
    if iv:
        print(f"\n[Interventions] n={len(iv)}")
        base = np.asarray([r["baseline_score"] for r in iv])
        pa = np.asarray([r["patched_score"] for r in iv])
        st = np.asarray([r["steered_score"] for r in iv])
        co = np.asarray([r["constrained_score"] for r in iv])
        print(f"  baseline     B = {base.mean():.3f}")
        print(f"  patched      B = {pa.mean():.3f}  (Δ {pa.mean()-base.mean():+.3f})")
        print(f"  steered      B = {st.mean():.3f}  (Δ {st.mean()-base.mean():+.3f})")
        print(f"  constrained  B = {co.mean():.3f}  (Δ {co.mean()-base.mean():+.3f})")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
