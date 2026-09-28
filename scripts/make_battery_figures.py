"""Render figures specific to the battery / pressure-test outputs:

  fig10_signed_gap_sweep.pdf      Signed G (linear & MLP probes) vs. model scale
  fig11_probe_ceiling.pdf         Per-rung F_lin / F_mlp / B
  fig12_selectivity.pdf           Per-variable selectivity (true − random)
  fig13_noise_robustness.pdf      F & B vs. instruction-noise level
  fig14_ood_length.pdf            Per-layer R² on OOD length
  fig15_mediation_curve.pdf       ΔB(layer) from per-layer activation patching
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BAT = RES / "battery"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

sns.set_context("paper", font_scale=1.05)
sns.set_style("whitegrid")
PAL = sns.color_palette("colorblind")


def load_battery() -> list[dict]:
    p = RES / "battery_summary.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def fig10_signed_gap_sweep():
    rows = load_battery()
    if not rows:
        return
    # Group by model.
    models = sorted({r["model"] for r in rows})
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    xs = np.arange(len(models))
    g_lin = [np.mean([r["G_linear"][0] for r in rows if r["model"] == m])
             for m in models]
    g_lin_lo = [np.mean([r["G_linear"][1] for r in rows if r["model"] == m])
                for m in models]
    g_lin_hi = [np.mean([r["G_linear"][2] for r in rows if r["model"] == m])
                for m in models]
    g_mlp = [np.mean([r["G_mlp"][0] for r in rows if r["model"] == m])
             for m in models]
    g_mlp_lo = [np.mean([r["G_mlp"][1] for r in rows if r["model"] == m])
                for m in models]
    g_mlp_hi = [np.mean([r["G_mlp"][2] for r in rows if r["model"] == m])
                for m in models]
    ax.errorbar(xs - 0.07, g_lin,
                yerr=[np.array(g_lin) - np.array(g_lin_lo),
                      np.array(g_lin_hi) - np.array(g_lin)],
                fmt="o-", color=PAL[0], label="linear probe")
    ax.errorbar(xs + 0.07, g_mlp,
                yerr=[np.array(g_mlp) - np.array(g_mlp_lo),
                      np.array(g_mlp_hi) - np.array(g_mlp)],
                fmt="s-", color=PAL[3], label="MLP probe")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("Mean signed gap  $G = F - B$")
    ax.set_title("Cross-architecture signed gap (mean ± 95% CI)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig10_signed_gap_sweep.pdf"); plt.close(fig)


def fig11_probe_ceiling():
    rows = load_battery()
    if not rows:
        return
    models = sorted({r["model"] for r in rows})
    Ls = [10, 20, 30, 40]
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.0),
                             squeeze=False)
    for ax_idx, m in enumerate(models):
        ax = axes[0, ax_idx]
        # Average per-length B from linear/MLP gap dicts.
        F_lin, F_mlp, B = [], [], []
        for L in Ls:
            v_lin, v_mlp, v_b = [], [], []
            for r in rows:
                if r["model"] != m:
                    continue
                d_lin = r["by_length_linear"][str(L)]
                d_mlp = r["by_length_mlp"][str(L)]
                v_lin.append(d_lin["F_mean"])
                v_mlp.append(d_mlp["F_mean"])
                v_b.append(d_lin["B_mean"])
            F_lin.append(np.mean(v_lin)); F_mlp.append(np.mean(v_mlp))
            B.append(np.mean(v_b))
        ax.plot(Ls, F_lin, "o-", color=PAL[0], label="$F$ linear", lw=2)
        ax.plot(Ls, F_mlp, "^-", color=PAL[5], label="$F$ MLP", lw=2)
        ax.plot(Ls, B, "s--", color=PAL[3], label="$B$", lw=2)
        ax.set_title(m); ax.set_xlabel("L")
        if ax_idx == 0:
            ax.set_ylabel("Score")
        ax.set_ylim(0, 1.02)
        if ax_idx == len(models) - 1:
            ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig11_probe_ceiling.pdf"); plt.close(fig)


def fig12_selectivity():
    rows = load_battery()
    if not rows:
        return
    # Aggregate selectivity per variable across all configs.
    sel_lists: dict[str, list[float]] = {}
    for r in rows:
        for k, v in r["selectivity_per_var"].items():
            if not np.isnan(v):
                sel_lists.setdefault(k, []).append(float(v))
    if not sel_lists:
        return
    items = sorted(sel_lists.items(), key=lambda x: -np.mean(x[1]))
    items = items[:20]
    names = [k for k, _ in items]
    means = [float(np.mean(v)) for _, v in items]
    sems = [float(np.std(v) / max(1, np.sqrt(len(v)))) for _, v in items]
    fig, ax = plt.subplots(figsize=(5.4, max(3.0, 0.22 * len(names))))
    ax.barh(range(len(names)), means, xerr=sems,
            color=PAL[2], edgecolor="black", lw=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Selectivity = $F_{\\text{true}} - F_{\\text{random}}$")
    ax.set_title("Probe selectivity (positive = probe reads real signal)")
    fig.tight_layout()
    fig.savefig(FIG / "fig12_selectivity.pdf"); plt.close(fig)


def fig13_noise_robustness():
    files = list(BAT.glob("pressure_*.json"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        noise = d.get("noise", {})
        if not noise:
            continue
        ps = sorted(float(k) for k in noise)
        Fv = [noise[str(p)]["F_mean"] for p in ps]
        Bv = [noise[str(p)]["B_mean"] for p in ps]
        label = f.stem.replace("pressure_", "")
        ax.plot(ps, Fv, "o-", lw=2, label=f"{label}  F")
        ax.plot(ps, Bv, "s--", lw=2, label=f"{label}  B")
    ax.set_xlabel("Instruction-noise fraction")
    ax.set_ylabel("Score")
    ax.set_title("Noise robustness: how do F and B degrade?")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig13_noise_robustness.pdf"); plt.close(fig)


def fig14_ood_length():
    files = list(BAT.glob("pressure_*.json"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        ood = d.get("ood", {})
        per = ood.get("per_layer_R2")
        if not per:
            continue
        # Clip pathological negative R² (degenerate Ridge fits).
        per = [max(-1.0, min(1.0, float(v))) for v in per]
        label = f.stem.replace("pressure_", "")
        ax.plot(range(len(per)), per, "o-", lw=2, label=label)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("OOD $R^2$ on total (clipped to $[-1,1]$)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Generalisation to unseen trajectory lengths")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig14_ood_length.pdf"); plt.close(fig)


def fig15_mediation_curve():
    p = RES / "mediation.json"
    if not p.exists():
        return
    with open(p) as fh:
        d = json.load(fh)
    delta = d["delta_per_layer_mean"]
    sem = d["delta_per_layer_sem"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    xs = range(len(delta))
    ax.errorbar(xs, delta, yerr=sem, fmt="o-",
                color=PAL[1], lw=2, capsize=3)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Patched layer ℓ")
    ax.set_ylabel("$\\Delta B$ after patch")
    ax.set_title(f"Per-layer mediation  ({d['model']})  baseline $B$ = {d['baseline_mean']:.2f}")
    fig.tight_layout()
    fig.savefig(FIG / "fig15_mediation_curve.pdf"); plt.close(fig)


def main() -> None:
    fig10_signed_gap_sweep()
    fig11_probe_ceiling()
    fig12_selectivity()
    fig13_noise_robustness()
    fig14_ood_length()
    fig15_mediation_curve()
    print(f"wrote battery/pressure figures -> {FIG}")


if __name__ == "__main__":
    main()
