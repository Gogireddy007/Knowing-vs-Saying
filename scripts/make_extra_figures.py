"""Render the post-references / appendix figures.

  fig_arch_overview.pdf            System architecture (data → probes → gap)
  fig_decision_tree.pdf            Decision-tree view of the failure classifier
  fig_compute_breakdown.pdf        Per-phase compute cost (proxied by elapsed s)
  fig_signed_gap_density.pdf       Density of G(t) by length, all configs
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BAT = RES / "battery"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

sns.set_context("paper", font_scale=1.0)
sns.set_style("white")
PAL = sns.color_palette("colorblind")


def _box(ax, x, y, w, h, label, fc, fontsize=9):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.05",
        fc=fc, ec="black", lw=0.9))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize)


def _arrow(ax, x0, y0, x1, y1, label: str | None = None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=1.1,
                                mutation_scale=12, color="black"))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.18, label,
                ha="center", fontsize=8, color="black",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.85, pad=1))


# ----------------------------------------------------------------- arch
def fig_arch_overview() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")

    _box(ax, 0.15, 2.2, 1.7, 1.2, "Cart-State\ngenerator", "white")
    _box(ax, 2.1, 3.5, 1.9, 1.2, "MiniGrid\ngenerator", "white")
    _box(ax, 2.1, 0.9, 1.9, 1.2, "Trajectory\nDB (JSON)", "#f6f6f6")

    _box(ax, 4.5, 2.2, 1.9, 1.2, "Frozen LM\nGPT-2 s / m", "#dde7f5")

    _box(ax, 6.9, 4.0, 1.9, 0.9, "linear probes\n(RidgeCV / LogReg)",
         "#eaf6e2")
    _box(ax, 6.9, 2.7, 1.9, 0.9, "MLP probes\n(256, ReLU)", "#eaf6e2")
    _box(ax, 6.9, 1.4, 1.9, 0.9, "random-label\n(chance baseline)", "#eaf6e2")
    _box(ax, 6.9, 0.1, 1.9, 0.9, "JSON\ngeneration", "#fde8d8")

    _box(ax, 9.1, 2.2, 1.7, 1.2,
         "gap analysis\n$G, \\tau$,\nfailure mode", "#f4e2f4")

    _arrow(ax, 1.85, 2.8, 2.1, 1.5)
    _arrow(ax, 4.0, 4.1, 4.5, 3.2)
    _arrow(ax, 4.0, 1.5, 4.5, 2.6)
    _arrow(ax, 6.4, 3.0, 6.9, 4.3)
    _arrow(ax, 6.4, 2.8, 6.9, 3.0)
    _arrow(ax, 6.4, 2.6, 6.9, 1.7)
    _arrow(ax, 6.4, 2.4, 6.9, 0.4)
    _arrow(ax, 8.8, 4.3, 9.1, 3.0)
    _arrow(ax, 8.8, 3.0, 9.1, 2.9)
    _arrow(ax, 8.8, 1.7, 9.1, 2.7)
    _arrow(ax, 8.8, 0.5, 9.1, 2.4)

    ax.text(5.5, -0.1,
            "every reported number flows left-to-right through this pipeline",
            ha="center", fontsize=8, style="italic")

    fig.tight_layout()
    fig.savefig(FIG / "fig_arch_overview.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- decision tree
def fig_decision_tree() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    _box(ax, 3.6, 5.2, 2.8, 0.6, "$(F, B, G, \\tau)$", "white")
    _box(ax, 3.6, 4.1, 2.8, 0.6, "$B \\geq 0.8$?", "#f4f4f4")
    _box(ax, 0.4, 3.0, 2.0, 0.6, "$G \\geq \\tau$?", "#f4f4f4")
    _box(ax, 7.6, 3.0, 2.0, 0.6, "$-G \\geq \\tau$?", "#f4f4f4")

    _box(ax, 0.1, 1.6, 2.4, 0.6, "deployment", "#fde7c8")
    _box(ax, 0.1, 0.5, 2.4, 0.6, "world-model", "#f7d6cf")
    _box(ax, 7.5, 1.6, 2.4, 0.6, "readout-\nbottleneck", "#d4e8fb")
    _box(ax, 7.5, 0.5, 2.4, 0.6, "other", "#eeeeee")
    _box(ax, 3.6, 3.0, 2.8, 0.6, "correct", "#d8efd8")

    _arrow(ax, 5.0, 5.2, 5.0, 4.75)
    _arrow(ax, 4.8, 4.1, 4.5, 3.6)
    _arrow(ax, 5.2, 4.1, 5.5, 3.6)
    _arrow(ax, 1.4, 4.1, 1.4, 3.6)
    _arrow(ax, 8.6, 4.1, 8.6, 3.6)
    _arrow(ax, 1.0, 3.0, 1.0, 2.2)
    _arrow(ax, 1.8, 3.0, 1.8, 1.1)
    _arrow(ax, 8.2, 3.0, 8.2, 2.2)
    _arrow(ax, 9.0, 3.0, 9.0, 1.1)

    ax.text(4.55, 3.85, "yes", fontsize=8, color="black")
    ax.text(5.45, 3.85, "no", fontsize=8, color="black")
    ax.text(1.05, 2.55, "yes", fontsize=8)
    ax.text(1.85, 2.55, "no", fontsize=8)
    ax.text(8.25, 2.55, "yes", fontsize=8)
    ax.text(9.05, 2.55, "no", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG / "fig_decision_tree.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- compute
def fig_compute_breakdown() -> None:
    p = RES / "battery_summary.json"
    if not p.exists():
        return
    with open(p) as f:
        rows = json.load(f)
    by_model: dict[str, list[float]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r["elapsed_s"])
    if not by_model:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    models = sorted(by_model)
    means = [np.mean(by_model[m]) for m in models]
    sems = [np.std(by_model[m]) / max(1, np.sqrt(len(by_model[m])))
            for m in models]
    ax.bar(models, means, yerr=sems, color=[PAL[0], PAL[3]],
           edgecolor="black", lw=0.6)
    for i, v in enumerate(means):
        ax.text(i, v + 8, f"{v:.0f}s", ha="center", fontsize=9)
    ax.set_ylabel("Elapsed wall-clock per config (s)")
    ax.set_title("End-to-end cost on Apple-Silicon MPS")
    fig.tight_layout()
    fig.savefig(FIG / "fig_compute_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- density
def fig_signed_gap_density() -> None:
    g = RES / "gap.json"
    if not g.exists():
        return
    with open(g) as f:
        d = json.load(f)
    Gs = np.asarray(d["G"]); Ls = np.asarray(d["length"])
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for i, L in enumerate(sorted(set(Ls.tolist()))):
        sub = Gs[Ls == L]
        if sub.size == 0:
            continue
        sns.kdeplot(sub, ax=ax, color=PAL[i], lw=2,
                    label=f"L = {int(L)}", fill=True, alpha=0.10)
    ax.axvline(0, color="gray", lw=0.8)
    ax.axvline(d["tau"], color="black", lw=0.6, ls=":")
    ax.axvline(-d["tau"], color="black", lw=0.6, ls=":")
    ax.set_xlabel("signed gap  $G(t) = F - B$")
    ax.set_ylabel("density")
    ax.set_title("Per-example signed-gap density by trajectory length")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_signed_gap_density.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_arch_overview()
    fig_decision_tree()
    fig_compute_breakdown()
    fig_signed_gap_density()
    print(f"wrote extra figures -> {FIG}")


if __name__ == "__main__":
    main()
