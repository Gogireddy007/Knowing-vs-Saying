"""Render the conceptual schematic of the dissociation test (Figure 1).

Two stacked panels:
 (a) The pipeline: trajectory → frozen LM → (probe, generation) → gap.
 (b) The decision rule: four-quadrant F-vs-B plot with the failure modes
     coloured and labelled inside the quadrants.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

sns.set_context("paper", font_scale=1.05)
sns.set_style("white")
PAL = sns.color_palette("colorblind")

CORRECT = PAL[2]
DEPLOY = PAL[1]
READOUT = PAL[0]
WORLD = PAL[3]


# ----------------------------------------------------------------- helpers
def _rounded(ax, x, y, w, h, label, fc, ec="black", fontsize=10,
             txt_color="black"):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.05",
        fc=fc, ec=ec, lw=0.9, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=txt_color, zorder=3)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="black",
                                lw=1.2, mutation_scale=14), zorder=2)


# ----------------------------------------------------------------- figure
def main() -> None:
    fig = plt.figure(figsize=(8.8, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.22)
    ax_pipe = fig.add_subplot(gs[0, 0])
    ax_quad = fig.add_subplot(gs[0, 1])

    # ----- (a) Pipeline --------------------------------------------------
    ax_pipe.set_xlim(0, 10); ax_pipe.set_ylim(0, 6); ax_pipe.axis("off")

    _rounded(ax_pipe, 0.1, 2.6, 2.0, 1.4,
             "trajectory\n$s_0, a_1, \\ldots, a_t$", "white")
    _rounded(ax_pipe, 2.6, 2.6, 1.9, 1.4, "frozen\nLM $\\pi_\\theta$",
             "#dde7f5")
    _rounded(ax_pipe, 5.1, 4.0, 2.4, 1.4,
             "probe ceiling\n$F^{\\mathrm{lin}}, F^{\\mathrm{MLP}}$",
             "#eaf6e2")
    _rounded(ax_pipe, 5.1, 1.0, 2.4, 1.4, "generation\n$B$", "#fde8d8")
    _rounded(ax_pipe, 7.9, 2.6, 2.0, 1.4,
             "signed gap\n$G = F - B$\nthreshold $\\tau$", "#f4e2f4")

    _arrow(ax_pipe, 2.1, 3.3, 2.6, 3.3)
    _arrow(ax_pipe, 4.5, 3.7, 5.1, 4.4)
    _arrow(ax_pipe, 4.5, 2.9, 5.1, 2.2)
    _arrow(ax_pipe, 7.5, 4.4, 7.9, 3.7)
    _arrow(ax_pipe, 7.5, 2.2, 7.9, 2.9)

    # legend strip below
    ax_pipe.text(5.0, 0.3,
                 r"per timestep $t$:  ($F(t)$ from probes,  $B(t)$ from"
                 r" generation,  $G(t)\!=\!F\!-\!B$)",
                 ha="center", va="center", fontsize=9, style="italic")

    # ----- (b) Decision rule quadrants ----------------------------------
    ax_quad.set_xlim(0, 1); ax_quad.set_ylim(0, 1)
    ax_quad.set_aspect("equal")
    ax_quad.set_xlabel("Fidelity  $F$")
    ax_quad.set_ylabel("Behavior  $B$")
    ax_quad.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax_quad.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax_quad.grid(alpha=0.3)

    # Diagonal F = B
    ax_quad.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    # B = 0.8 cut for "correct"
    ax_quad.axhline(0.8, color="black", lw=0.6, ls=":")

    # Shaded regions
    ax_quad.add_patch(mpatches.Polygon(
        [(0, 0.8), (1, 0.8), (1, 1), (0, 1)], closed=True,
        fc=CORRECT, alpha=0.18, zorder=0))
    ax_quad.add_patch(mpatches.Polygon(
        [(0, 0.5), (0.5, 0.5), (0.5, 0.8), (0, 0.8)], closed=True,
        fc=READOUT, alpha=0.18, zorder=0))
    ax_quad.add_patch(mpatches.Polygon(
        [(0.5, 0), (1, 0), (1, 0.5), (0.5, 0.5)], closed=True,
        fc=DEPLOY, alpha=0.18, zorder=0))
    ax_quad.add_patch(mpatches.Polygon(
        [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)], closed=True,
        fc=WORLD, alpha=0.18, zorder=0))

    # Region labels
    ax_quad.text(0.5, 0.92, "correct  ($B\\geq 0.8$)",
                 ha="center", va="center", fontsize=9,
                 color=CORRECT, weight="bold")
    ax_quad.text(0.22, 0.66, "readout-\nbottleneck\n($B\\!-\\!F\\!\\geq\\!\\tau$)",
                 ha="center", va="center", fontsize=9,
                 color=READOUT, weight="bold")
    ax_quad.text(0.78, 0.32, "deployment\n($F\\!-\\!B\\!\\geq\\!\\tau$)",
                 ha="center", va="center", fontsize=9,
                 color=DEPLOY, weight="bold")
    ax_quad.text(0.25, 0.22, "world-model\n($\\max(F,B)\\!<\\!0.5$)",
                 ha="center", va="center", fontsize=9,
                 color=WORLD, weight="bold")
    ax_quad.text(0.92, 0.94, "$F\\!=\\!B$", ha="right", va="top",
                 fontsize=8, color="gray", rotation=45)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    # Figure-level titles at identical absolute y so they're guaranteed
    # to share the same row regardless of per-panel axis heights.
    fig.text(0.04, 0.95, "(a) Pipeline",
             ha="left", va="center", fontsize=12, weight="bold")
    fig.text(0.59, 0.95, "(b) Failure-mode decision rule",
             ha="left", va="center", fontsize=12, weight="bold")
    fig.savefig(FIG / "fig0_concept.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "fig0_concept.pdf")


if __name__ == "__main__":
    main()
