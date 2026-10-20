"""Render the public benchmark charts from the shipped JSON results.

Optional tooling: requires matplotlib (`pip install matplotlib`). It reads only the
public files in this directory and writes PNGs to ../docs/charts/. No model calls.
"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "docs" / "charts"
OURS = "#1f4e79"
REF = "#9aa5b1"
ACCENT = "#c55a11"
GRID = "#d9dee5"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": "#5b6570",
    "axes.labelcolor": "#222222",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def style(ax, xlabel=None, ylabel=None, title=None):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", pad=12)


def head_to_head():
    rows = [
        ("Legacy JevBench, 213 items\n3 repeats, mean", 5.63, (2.19, 9.55), "paired, p = 0.002 to 0.008"),
        ("AutoTrust JEV-27B, 231 items\nfamily macro, mean", 4.80, None, "descriptive, not paired"),
        ("ChaosNLI, 1,599 items\nplurality accuracy", 2.94, (0.44, 5.25), "paired, p = 0.0028"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ys = range(len(rows))
    for y, (label, value, ci, note) in zip(ys, rows):
        ax.barh(y, value, height=0.52, color=OURS, alpha=0.92 if ci else 0.55,
                edgecolor=OURS, hatch=None if ci else "//")
        if ci:
            ax.errorbar(value, y, xerr=[[value - ci[0]], [ci[1] - value]], fmt="none",
                        ecolor="#222222", capsize=4, linewidth=1.4)
            ax.text(ci[1] + 0.35, y, f"+{value:.1f} pp  (95% CI +{ci[0]:.1f} to +{ci[1]:.1f})",
                    va="center", fontsize=9)
        else:
            ax.text(value + 0.35, y, f"+{value:.1f} pp  (descriptive)", va="center", fontsize=9)
        ax.text(0.12, y - 0.34, note, va="center", fontsize=7.5, color="#555555")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 14)
    style(ax, xlabel="Accuracy advantage over the published reference (percentage points)",
          title="Accuracy advantage over published references")
    ax.text(0, -0.28, "Bars show our score minus the reference on the same items. Hatched bar has no paired test.",
            transform=ax.transAxes, fontsize=7.5, color="#555555")
    fig.tight_layout()
    fig.savefig(OUT / "head-to-head.png", dpi=220)
    plt.close(fig)


def legacy_repeat():
    runs = [198, 198, 199]
    ref = 186
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = [1, 2, 3]
    ax.plot(x, runs, "o-", color=OURS, linewidth=2, markersize=9, label="This readout")
    ax.axhline(ref, color=REF, linewidth=2, linestyle="--", label="Jev 1.13.0 published (186)")
    for xi, v in zip(x, runs):
        ax.annotate(f"{v}", (xi, v), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=10, color=OURS, fontweight="bold")
    ax.annotate(f"{ref}", (0.62, ref), textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=9, color="#5b6570")
    ax.set_xticks(x)
    ax.set_xticklabels(["Repeat 1", "Repeat 2", "Repeat 3"])
    ax.set_ylim(178, 206)
    ax.set_yticks([180, 185, 190, 195, 200, 205])
    style(ax, ylabel="Correct of 213 items", title="Legacy strict subset: 12 to 13 more correct items per repeat")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.text(0, -0.22, "Vertical axis starts at 178 of 213 so the gap is visible. Same public items in every repeat.",
            transform=ax.transAxes, fontsize=7.5, color="#555555")
    fig.tight_layout()
    fig.savefig(OUT / "legacy-repeat.png", dpi=220)
    plt.close(fig)


def chaos_paired():
    labels = ["Both correct", "Ours only", "Reference only", "Both wrong"]
    values = [880, 143, 96, 480]
    colors = ["#a9c1d6", OURS, REF, "#e8ebef"]
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    for b, v in zip(bars, values):
        ax.annotate(f"{v}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1000)
    style(ax, ylabel="ChaosNLI rows of 1,599",
          title="Paired outcomes on ChaosNLI: +47 net rows")
    ax.text(0, -0.22, "Net +47 rows, exact McNemar p = 0.0028. Different tie rule from the API is reported separately.",
            transform=ax.transAxes, fontsize=7.5, color="#555555")
    fig.tight_layout()
    fig.savefig(OUT / "chaos-paired.png", dpi=220)
    plt.close(fig)


def autotrust():
    repeats = [93.95, 92.96, 93.59]
    pub = 88.70
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = [1, 2, 3]
    ax.plot(x, repeats, "o-", color=OURS, linewidth=2, markersize=9, label="This readout, family macro")
    ax.axhline(pub, color=REF, linewidth=2, linestyle="--", label="AutoTrust published (88.70%)")
    for xi, v in zip(x, repeats):
        ax.annotate(f"{v:.2f}%", (xi, v), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9.5, color=OURS, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Repeat 1", "Repeat 2", "Repeat 3"])
    ax.set_ylim(84, 97)
    style(ax, ylabel="Family-macro accuracy, 231 items (%)",
          title="AutoTrust comparison: 93.50% mean against 88.70% published")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.text(0, -0.22, "Descriptive: our 18 score items are choice surrogates and the public tasks were used during development.",
            transform=ax.transAxes, fontsize=7.5, color="#555555")
    fig.tight_layout()
    fig.savefig(OUT / "autotrust.png", dpi=220)
    plt.close(fig)


def scaling():
    ks = [16, 32, 64, 128, 256]
    acc = [73.67, 71.33, 64.67, 61.33, 57.00]
    lo = [65.0, 62.67, 55.33, 51.67, 47.67]
    hi = [82.0, 79.67, 73.67, 70.67, 66.33]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.fill_between(ks, lo, hi, color=OURS, alpha=0.15, linewidth=0)
    ax.plot(ks, acc, "o-", color=OURS, linewidth=2, markersize=9, label="Primary accuracy")
    for k, v in zip(ks, acc):
        ax.annotate(f"{v:.1f}%", (k, v), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9.5, color=OURS, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(40, 90)
    style(ax, xlabel="Options per request", ylabel="Accuracy, 300 primary calls per size (%)",
          title="Option-count sensitivity on our own panel")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.text(0, -0.25, "Shaded band: item-clustered 95% CI. Same 100 texts at every size, so these are paired observations.",
            transform=ax.transAxes, fontsize=7.5, color="#555555")
    fig.tight_layout()
    fig.savefig(OUT / "scaling.png", dpi=220)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    head_to_head()
    legacy_repeat()
    chaos_paired()
    autotrust()
    scaling()
    for p in sorted(OUT.glob("*.png")):
        print(f"{p.name} {p.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
