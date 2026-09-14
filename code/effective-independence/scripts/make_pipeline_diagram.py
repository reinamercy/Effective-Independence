"""Generates figures/pipeline_diagram.png -- a static end-to-end architecture
diagram of the experimental harness (dataset loading -> config construction
-> cached/rate-limited generation -> grading -> N_eff analysis -> paper).

Unlike src/analysis/*.py's figures, this diagram does not depend on
results/error_matrix.csv -- it documents the harness structure itself, so
it only needs to be regenerated if the harness's module boundaries change,
not on every generation/analysis run.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGURES_DIR = REPO_ROOT / "figures"


def box(ax, xy, w, h, text, facecolor="#EAF1FB", edgecolor="#4C72B0", fontsize=8.5, textweight="normal"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.2, edgecolor=edgecolor, facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
             fontweight=textweight, wrap=True)
    return patch


def arrow(ax, start, end, color="black"):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                         linewidth=1.2, color=color, shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(15, 3.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    stages = [
        ("3 benchmark datasets\nGPQA-Diamond, MATH L4-5,\nSimpleQA (30 questions total)",
         "#EAF1FB", "#4C72B0"),
        ("build_configs.py\n12 configs x 4 axes:\nprompt / temperature /\nsize / family",
         "#FBEFEA", "#DD8452"),
        ("api_client.py\ncached, rate-limited\n(provider, model_id)-keyed;\nGoogle AI Studio + OpenRouter\n(free tier)",
         "#EAF6EC", "#55A868"),
        ("Per-dataset graders\nexact-match (GPQA) /\nmath_verify (MATH) /\ncontainment + LLM-judge\n(SimpleQA)",
         "#EAF1FB", "#4C72B0"),
        ("results/\nerror_matrix.csv\n(one row per\n(question, config) cell)",
         "#F5F5F5", "#666666"),
        ("src/analysis/\n$N_\\mathrm{eff}$ + bootstrap CI,\ncorrelation matrix,\nconfident-correlated-\nerror rate",
         "#EAF1FB", "#4C72B0"),
        ("paper/main.tex\ntables + figures\n(this document)",
         "#F3EAF6", "#8064A2"),
    ]

    n = len(stages)
    w, h = 1.85, 2.4
    gap = (15 - n * w) / (n + 1)
    y0 = (3.6 - h) / 2 - 0.15
    xs = []
    for i, (text, fc, ec) in enumerate(stages):
        x = gap + i * (w + gap)
        xs.append(x)
        box(ax, (x, y0), w, h, text, facecolor=fc, edgecolor=ec, fontsize=7.8)

    for i in range(n - 1):
        arrow(ax, (xs[i] + w, y0 + h / 2), (xs[i + 1], y0 + h / 2))

    ax.set_title(
        "Effective Independence experimental harness: dataset loading through cached,\n"
        "rate-limited generation, per-dataset grading, and $N_\\mathrm{eff}$ analysis",
        fontsize=10, pad=14,
    )

    out_path = FIGURES_DIR / "pipeline_diagram.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved {out_path}")

    from src.analysis.extra_plots import sync_paper_figures
    synced = sync_paper_figures()
    print(f"Synced {len(synced)} figure(s) to paper/figures/ (used by paper/main.tex)")


if __name__ == "__main__":
    main()
