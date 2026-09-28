"""Render every figure and table referenced in the paper.

Reads from ``results/`` and writes PDFs to ``paper/figures/`` and
LaTeX tables to ``paper/tables/``. Each figure is self-contained: rerunning
this script never touches ``results/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
TBL = ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TBL.mkdir(parents=True, exist_ok=True)

sns.set_context("paper", font_scale=1.05)
sns.set_style("whitegrid")
PALETTE = sns.color_palette("colorblind")


def _load() -> dict:
    out = {}
    for name in ["gap.json", "probes.json", "behavior.json",
                 "minigrid.json", "interventions.json"]:
        p = RES / name
        if p.exists():
            with open(p) as f:
                out[name.split(".")[0]] = json.load(f)
    return out


# ----------------------------------------------------------------- figures
def fig1_fidelity_behavior_vs_complexity(data):
    g = data["gap"]
    by_L = g["by_length"]
    Ls = sorted(int(k) for k in by_L)
    F = [by_L[str(L)]["F_mean"] for L in Ls]
    B = [by_L[str(L)]["B_mean"] for L in Ls]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(Ls, F, "o-", color=PALETTE[0], label="Fidelity  $F$", lw=2)
    ax.plot(Ls, B, "s--", color=PALETTE[3], label="Behavior $B$", lw=2)
    ax.set_xlabel("Trajectory length")
    ax.set_ylabel("Mean correctness")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(Ls)
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("Fidelity vs. behavior across complexity")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_fidelity_behavior.pdf")
    plt.close(fig)


def fig2_gap_and_failure_modes(data):
    g = data["gap"]
    by_L = g["by_length"]
    Ls = sorted(int(k) for k in by_L)
    # Derive readout-bucket rate directly from failure_mode list per length.
    fm = g["failure_mode"]
    lengths = g["length"]
    rd = []
    wm = []
    dep = []
    for L in Ls:
        m = [fm[i] for i in range(len(lengths)) if lengths[i] == L]
        if not m:
            wm.append(0); dep.append(0); rd.append(0); continue
        n = len(m)
        wm.append(sum(1 for x in m if x == "world_model") / n)
        dep.append(sum(1 for x in m if x == "deployment") / n)
        rd.append(sum(1 for x in m if x == "readout") / n)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax1.bar([L - 2.5 for L in Ls], wm, width=2.2, label="World-model",
            color=PALETTE[3])
    ax1.bar([L for L in Ls], dep, width=2.2, label="Deployment",
            color=PALETTE[1])
    ax1.bar([L + 2.5 for L in Ls], rd, width=2.2, label="Readout-bottleneck",
            color=PALETTE[0])
    ax1.set_xlabel("Trajectory length")
    ax1.set_ylabel("Fraction of examples")
    ax1.set_title("Failure-mode mix")
    ax1.legend(frameon=False)

    null = np.asarray(g["null_distribution"])
    ax2.hist(null, bins=30, color="gray", alpha=0.6, label="Null |G|")
    ax2.axvline(g["tau"], color=PALETTE[3], lw=2,
                label=fr"$\tau$ = {g['tau']:.3f}")
    ax2.set_xlabel("|G|")
    ax2.set_ylabel("Density")
    ax2.set_title("Null calibration of $\\tau$")
    ax2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_gap_failure_modes.pdf")
    plt.close(fig)


def fig3_layer_wise_peaks(data):
    p = data["probes"]
    var_groups = {
        "counts": [v for v in p if v.startswith("counts:")],
        "prices": [v for v in p if v.startswith("prices:")],
        "flags":  [v for v in p if v.startswith("flag:")],
        "total":  ["total"] if "total" in p else [],
    }
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for i, (name, keys) in enumerate(var_groups.items()):
        if not keys:
            continue
        curves = []
        for k in keys:
            curve = p[k]["per_layer"]
            if not curve or all(np.isnan(curve)):
                continue
            # Clip absurd Ridge R² values; we only care about the
            # qualitative shape in [-1, 1].
            curve = [max(-1.0, min(1.0, float(v))) for v in curve]
            curves.append(curve)
        if not curves:
            continue
        arr = np.asarray(curves)
        mean = np.nanmean(arr, axis=0)
        ax.plot(range(len(mean)), mean, lw=2, color=PALETTE[i], label=name)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean CV score (clipped to $[-1,1]$)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Layer-wise probe performance")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_layer_peaks.pdf")
    plt.close(fig)


def fig4_minigrid_validation(data):
    mg = data.get("minigrid")
    if mg is None:
        return
    probes = mg["probes"]
    behavior = mg["behavior"]
    by_task = {}
    # Aggregate per-task by averaging probe accuracy across vars.
    for ex_i, ex in enumerate(mg["examples"]):
        task = ex["rung"]
        by_task.setdefault(task, []).append(ex_i)
    rows = []
    for task, idxs in by_task.items():
        F_vals, B_vals = [], []
        for v, r in probes.items():
            pe = np.asarray(r.get("per_example", []), dtype=float)
            if pe.size == 0:
                continue
            sub = pe[idxs]
            if np.all(np.isnan(sub)):
                continue
            F_vals.append(np.nanmean(sub))
        B = np.nanmean([behavior["per_example_overall"][i] for i in idxs])
        if F_vals:
            rows.append((task, float(np.mean(F_vals)), float(B)))
    if not rows:
        return
    rows.sort(key=lambda r: r[0])
    tasks = [r[0] for r in rows]
    F = [r[1] for r in rows]
    B = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = np.arange(len(tasks))
    ax.bar(x - 0.18, F, width=0.36, label="Fidelity", color=PALETTE[0])
    ax.bar(x + 0.18, B, width=0.36, label="Behavior", color=PALETTE[3])
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("MiniGrid validation: F vs. B per task")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_minigrid.pdf")
    plt.close(fig)


def fig5_dissociation_scatter(data):
    g = data["gap"]
    F = np.asarray(g["F"])
    B = np.asarray(g["B"])
    fm = g["failure_mode"]
    # Internal labels keyed as in classify_failure_modes, but the legend
    # rendered to the figure uses the human-readable hyphenated forms.
    colors = {"correct": PALETTE[2], "deployment": PALETTE[1],
              "readout": PALETTE[0], "world_model": PALETTE[3],
              "other": "lightgray", "undefined": "white"}
    pretty = {"correct": "correct", "deployment": "deployment",
              "readout": "readout-bottleneck", "world_model": "world-model",
              "other": "other", "undefined": "n/a"}
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for mode, c in colors.items():
        m = np.array([f == mode for f in fm])
        if not m.any():
            continue
        ax.scatter(F[m], B[m], color=c, alpha=0.7, s=22,
                   edgecolor="black", lw=0.3,
                   label=pretty.get(mode, mode))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("Fidelity $F$")
    ax.set_ylabel("Behavior $B$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Per-example dissociation")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_dissociation_scatter.pdf")
    plt.close(fig)


def fig6_interventions(data):
    iv = data.get("interventions")
    if not iv:
        return
    base = np.asarray([r["baseline_score"] for r in iv])
    pa = np.asarray([r["patched_score"] for r in iv])
    st = np.asarray([r["steered_score"] for r in iv])
    co = np.asarray([r["constrained_score"] for r in iv])
    labels = ["baseline", "patch", "steer", "constrain"]
    vals = [base.mean(), pa.mean(), st.mean(), co.mean()]
    errs = [v.std() / np.sqrt(len(v)) for v in [base, pa, st, co]]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar(labels, vals, yerr=errs, color=[PALETTE[i] for i in [7, 1, 2, 0]],
           edgecolor="black", lw=0.6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean behavioral correctness")
    ax.set_title("Intervention effects")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_interventions.pdf")
    plt.close(fig)


def fig7_scaling_placeholder(data):
    # Reuse F vs B across lengths as a proxy 'scaling-by-complexity' figure.
    g = data["gap"]
    by_L = g["by_length"]
    Ls = sorted(int(k) for k in by_L)
    G = [by_L[str(L)]["G_mean"] for L in Ls]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(Ls, G, "o-", color=PALETTE[5], lw=2)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Trajectory length")
    ax.set_ylabel("Mean gap $G = F - B$")
    ax.set_title("Gap grows with task complexity")
    fig.tight_layout()
    fig.savefig(FIG / "fig7_gap_vs_length.pdf")
    plt.close(fig)


def fig8_noise_placeholder(data):
    # Without ambiguous-instruction experiment we plot per-variable behavior
    # fragility instead.
    b = data.get("behavior")
    if b is None:
        return
    pv = b["per_variable"]
    rows = []
    for k, vals in pv.items():
        arr = np.asarray(vals, dtype=float)
        if np.all(np.isnan(arr)):
            continue
        rows.append((k, float(np.nanmean(arr))))
    rows.sort(key=lambda r: r[1])
    rows = rows[:15] + rows[-15:]
    if not rows:
        return
    labels, scores = zip(*rows)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.barh(range(len(labels)), scores, color=PALETTE[0])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Mean behavioral correctness")
    ax.set_title("Per-variable behavior")
    fig.tight_layout()
    fig.savefig(FIG / "fig8_per_variable.pdf")
    plt.close(fig)


def fig9_cross_variable_heatmap(data):
    p = data["probes"]
    keys = sorted(p.keys())
    layers = max(len(p[k]["per_layer"]) for k in keys)
    mat = np.full((len(keys), layers), np.nan)
    for i, k in enumerate(keys):
        cur = p[k]["per_layer"]
        for j, v in enumerate(cur):
            mat[i, j] = v
    fig, ax = plt.subplots(figsize=(5.6, max(3.6, 0.18 * len(keys))))
    sns.heatmap(mat, ax=ax, cmap="viridis", vmin=0, vmax=1,
                yticklabels=keys, xticklabels=False, cbar_kws={"label": "CV score"})
    ax.set_xlabel("Layer →")
    ax.set_title("Cross-variable fidelity heatmap")
    fig.tight_layout()
    fig.savefig(FIG / "fig9_heatmap.pdf")
    plt.close(fig)


# ----------------------------------------------------------------- tables
def table1_regression(data):
    reg = data["gap"]["regression"]
    rows = [
        ("$\\beta_0$",  reg["beta_0"]),
        ("$\\beta_1$ (F)", reg["beta_1_F"]),
        ("$\\beta_2$ (L)", reg["beta_2_L"]),
        ("$\\beta_3$ (F$\\cdot$L)", reg["beta_3_FL"]),
        ("training accuracy", reg["accuracy"]),
        ("$n$", reg["n"]),
        ("$P[B>0.8]$", reg["positive_rate"]),
    ]
    out = [
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Coefficient & Value \\\\",
        "\\midrule",
    ]
    for k, v in rows:
        if isinstance(v, int):
            out.append(f"{k} & {v} \\\\")
        else:
            out.append(f"{k} & {v:+.3f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TBL / "table1_regression.tex").write_text("\n".join(out))


def table2_by_length(data):
    by_L = data["gap"]["by_length"]
    out = [
        "\\begin{table}[ht]\\centering",
        "\\caption{Per-rung mean $F$, $B$, $G$ and failure-mode rates.}"
        "\\label{tab:by_length}",
        "\\begin{tabular}{rrrrrrrr}",
        "\\toprule",
        "L & $F$ & $B$ & $G$ & $|G|$ & WM\\% & RDT\\% & COR\\% \\\\",
        "\\midrule",
    ]
    for L in sorted(int(k) for k in by_L):
        r = by_L[str(L)]
        out.append(
            f"{L} & {r['F_mean']:.2f} & {r['B_mean']:.2f} & "
            f"{r['G_mean']:+.2f} & {r['G_abs_mean']:.2f} & "
            f"{r['wm_rate']:.2f} & "
            f"{r.get('readout_rate', 0.0):.2f} & "
            f"{r.get('correct_rate', 0.0):.2f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TBL / "table2_by_length.tex").write_text("\n".join(out))


def table3_interventions(data):
    iv = data.get("interventions")
    if not iv:
        return
    import sys
    sys.path.insert(0, str(ROOT))
    from interventions.patch import paired_sign_test_p

    base = np.asarray([r["baseline_score"] for r in iv])
    pa = np.asarray([r["patched_score"] for r in iv])
    st = np.asarray([r["steered_score"] for r in iv])
    co = np.asarray([r["constrained_score"] for r in iv])

    def row(name, arr, is_baseline=False):
        delta = arr.mean() - base.mean()
        sig = "" if is_baseline else _sig_str(base, arr)
        return (f"{name} & {arr.mean():.3f} $\\pm$ "
                f"{arr.std()/max(1, np.sqrt(len(arr))):.3f} & "
                f"{delta:+.3f} & {sig} \\\\")

    def _sig_str(base_arr, treat_arr):
        s = paired_sign_test_p(base_arr, treat_arr)
        return f"$p_{{boot}}\\!=\\!{s['p_bootstrap']:.4f}$, $p_{{sign}}\\!=\\!{s['p_sign_test']:.4f}$"

    out = [
        "\\centering",
        "\\begin{tabular}{lrrl}",
        "\\toprule",
        "Condition & Mean $B$ & $\\Delta$ vs. baseline & Significance (paired, $n="
        f"{len(base)}$) \\\\",
        "\\midrule",
        row("Baseline", base, is_baseline=True),
        row("Activation patch", pa),
        row("Steering vector", st),
        row("Constrained decode", co),
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (TBL / "table3_interventions.tex").write_text("\n".join(out))


def table4_minigrid(data):
    mg = data.get("minigrid")
    if not mg:
        return
    probes = mg["probes"]
    behavior = mg["behavior"]
    by_task: dict[str, list[int]] = {}
    for i, ex in enumerate(mg["examples"]):
        by_task.setdefault(ex["rung"], []).append(i)
    out = [
        "\\begin{table}[h]\\centering",
        "\\caption{MiniGrid per-task $F$, $B$, and signed gap.}"
        "\\label{tab:mg}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Task & $n$ & $F$ & $B$ & $G$ \\\\",
        "\\midrule",
    ]
    for task in sorted(by_task):
        idxs = by_task[task]
        F_vals = []
        for r in probes.values():
            pe = np.asarray(r.get("per_example", []), dtype=float)
            if pe.size == 0:
                continue
            sub = pe[idxs]
            if np.all(np.isnan(sub)):
                continue
            F_vals.append(float(np.nanmean(sub)))
        F = float(np.mean(F_vals)) if F_vals else float("nan")
        B = float(np.nanmean([behavior["per_example_overall"][i]
                              for i in idxs]))
        out.append(f"{task} & {len(idxs)} & {F:.2f} & {B:.2f} & "
                   f"{F - B:+.2f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TBL / "table4_minigrid.tex").write_text("\n".join(out))


# ----------------------------------------------------------------- main
def main() -> None:
    data = _load()
    if "gap" not in data:
        print("error: results/gap.json missing — run scripts/run_all.py first")
        return
    fig1_fidelity_behavior_vs_complexity(data)
    fig2_gap_and_failure_modes(data)
    fig3_layer_wise_peaks(data)
    fig4_minigrid_validation(data)
    fig5_dissociation_scatter(data)
    fig6_interventions(data)
    fig7_scaling_placeholder(data)
    fig8_noise_placeholder(data)
    fig9_cross_variable_heatmap(data)
    table1_regression(data)
    table2_by_length(data)
    table3_interventions(data)
    table4_minigrid(data)
    print(f"wrote figures -> {FIG}")
    print(f"wrote tables  -> {TBL}")


if __name__ == "__main__":
    main()
