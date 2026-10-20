"""Render the two public comparison charts from the shipped JSON results.

Optional tooling: requires matplotlib (`pip install matplotlib`). Reads only the
public files in this directory and writes PNGs to ../docs/charts/. No model calls.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "docs" / "charts"

OURS = "#1f4e79"
REF = "#9aa5b1"
TEXT = "#222222"
MUTED = "#5b6570"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def bars(title, subtitle, rows, footnote, name):
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    y = range(len(rows))
    ax.barh(y, values, height=0.5, color=colors)
    for yi, v in zip(y, values):
        ax.text(v + 1.0, yi, f"{v:.1f}%", va="center", fontsize=13, fontweight="bold", color=TEXT)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=11, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Accuracy (%)", color=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e3e7ec", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=TEXT, pad=16)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=MUTED)
    fig.text(0.01, 0.02, footnote, fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / name, dpi=220)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bars(
        "Legacy JevBench, 213 public items",
        "",
        [("QwenFN Hybrid Decision Engine", 93.1, OURS), ("Jev 1.13.0, published", 87.3, REF)],
        "Paired exact McNemar p = 0.002 to 0.008. Public items, reused during development.",
        "legacy-213.png",
    )
    bars(
        "Legacy JevBench, 231 items, family macro",
        "AutoTrust's own reported metric and item set.",
        [("QwenFN Hybrid Decision Engine", 93.5, OURS), ("AutoTrust JEV-27B, published", 88.7, REF)],
        "Descriptive: our 18 score items are choice surrogates. No paired AutoTrust outputs are public.",
        "autotrust-231.png",
    )
    for p in sorted(OUT.glob("*.png")):
        print(f"{p.name} {p.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
