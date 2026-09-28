"""Figures for the reviewer-upgrade results (results/upgrades.json).

  fig_signflip.pdf     F and B per probe variant; G crosses zero
  fig_poscontrol.pdf   nearest-state vs random-state donor patch ΔB
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
FIG.mkdir(parents=True, exist_ok=True)
sns.set_context("paper", font_scale=1.0)
sns.set_style("whitegrid")
PAL = sns.color_palette("colorblind")

VARIANT_ORDER = ["last-linear", "last-mlp", "mean-linear", "attn-linear"]
VARIANT_LABEL = {"last-linear": "last\nlinear", "last-mlp": "last\nMLP",
                 "mean-linear": "mean-pool\nlinear",
                 "attn-linear": "attn-pool\nlinear"}


def _load():
    p = RES / "upgrades.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def fig_signflip():
    data = _load()
    if not data:
        return
    fig, axes = plt.subplots(1, len(data), figsize=(4.2 * len(data), 3.2),
                             squeeze=False)
    for ax_i, rec in enumerate(data):
        ax = axes[0, ax_i]
        sw = rec["sweep"]
        xs = [v for v in VARIANT_ORDER if v in sw]
        F = [sw[v]["F_mean"] for v in xs]
        B = [sw[v]["B_mean"] for v in xs]
        G = [sw[v]["G_mean"] for v in xs]
        x = np.arange(len(xs))
        ax.plot(x, F, "o-", color=PAL[0], lw=2, label="$F$ (probe)")
        ax.plot(x, B, "s--", color=PAL[3], lw=2, label="$B$ (behaviour)")
        ax.bar(x, G, width=0.5, color=PAL[2], alpha=0.35,
               label="$G=F-B$")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_LABEL[v] for v in xs], fontsize=8)
        ax.set_title(rec["model"].split("/")[-1])
        ax.set_ylim(min(-0.35, min(G) - 0.05), 1.0)
        if ax_i == 0:
            ax.set_ylabel("score")
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Probe-capacity sweep: the gap does NOT close as the probe "
                 "reads more context (distributed read-out)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "fig_signflip.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_poscontrol():
    """Plot the positive-control patch for EVERY model that has one, not
    just the first. The honest "suggestive, not conclusive" framing in
    the paper text (GPT-2 replicates the predicted direction, Qwen does
    not) needs the Qwen non-replication visible in the figure too --
    previously only the first model in the run list (GPT-2) was ever
    plotted here, so the null result existed in upgrades.json and in
    prose but never in a figure a reader could see at a glance."""
    data = _load()
    if not data:
        return
    recs = [r for r in data if r.get("patch")]
    if not recs:
        return
    fig, axes = plt.subplots(1, len(recs), figsize=(4.0 * len(recs), 3.0),
                             squeeze=False)
    labels = ["baseline", "nearest-state\ndonor", "random-state\ndonor"]
    colors = ["lightgray", PAL[2], PAL[3]]
    for ax_i, rec in enumerate(recs):
        ax = axes[0, ax_i]
        p = rec["patch"]
        vals = [p["baseline_B"], p["nearest_state_donor_B"],
                p["random_state_donor_B"]]
        ax.bar(labels, vals, color=colors, edgecolor="black", lw=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
        verdict = ("replicates" if p["nearest_beats_random"]
                  else "does not replicate")
        ax.set_title(f"{rec['model'].split('/')[-1]}\n"
                     f"layer {p['layer']}, $n={p['n']}$: {verdict}",
                     fontsize=9)
        if ax_i == 0:
            ax.set_ylabel("mean behaviour $B$ after patch")
        ax.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 1.0)
    fig.suptitle("Positive control: does a state-matched donor hurt less "
                 "than a random-state donor?", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIG / "fig_poscontrol.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    fig_signflip()
    fig_poscontrol()
    print(f"wrote upgrade figures -> {FIG}")


if __name__ == "__main__":
    main()
