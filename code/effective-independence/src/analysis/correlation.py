"""Full 12x12 pairwise error-correlation matrix, blocked/ordered by
diversity axis, and its heatmap figure (figures/correlation_heatmap.png).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis.neff import pairwise_phi

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "figures"

AXIS_ORDER = ["prompt_diverse", "temperature_diverse", "size_diverse", "family_diverse"]


def _axis_of(config_id: str) -> str:
    for axis in AXIS_ORDER:
        if config_id.startswith(axis):
            return axis
    return "unknown"


def build_correlation_matrix(vectors: dict[str, list]) -> tuple[list[str], np.ndarray]:
    """Orders configs by axis block, then alphabetically within the block.
    Diagonal is 1.0 by definition; off-diagonal is NaN where undefined
    (one of the pair was constant over the overlapping graded questions)."""
    config_ids = sorted(
        vectors.keys(),
        key=lambda c: (AXIS_ORDER.index(_axis_of(c)) if _axis_of(c) in AXIS_ORDER else 99, c),
    )
    n = len(config_ids)
    mat = np.full((n, n), np.nan)
    for i in range(n):
        mat[i, i] = 1.0
        for j in range(i + 1, n):
            rho = pairwise_phi(vectors[config_ids[i]], vectors[config_ids[j]])
            if rho is not None:
                mat[i, j] = rho
                mat[j, i] = rho
    return config_ids, mat


def plot_heatmap(config_ids: list[str], matrix: np.ndarray, out_path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = out_path or (FIGURES_DIR / "correlation_heatmap.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(config_ids)))
    ax.set_yticks(range(len(config_ids)))
    ax.set_xticklabels(config_ids, rotation=90, fontsize=7)
    ax.set_yticklabels(config_ids, fontsize=7)
    fig.colorbar(im, ax=ax, label="pairwise error correlation (phi)")
    ax.set_title("Error correlation across configs (blocked by diversity axis)\ngray cells = undefined (a config was constant over overlapping questions)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_neff_vs_n(question_ids: list[str], vectors: dict[str, list], out_path: Path | None = None) -> Path:
    """Cumulative N_eff as configs are added one at a time, round-robin
    across axes (prompt, temperature, size, family, prompt, ...) -- shows
    where the diversity gain from adding another same-base-model voter
    saturates vs. a genuinely different-family voter."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.analysis.neff import mean_pairwise_correlation, n_eff

    axes_slots = {
        "prompt_diverse": [f"prompt_diverse_{i}" for i in range(3)],
        "temperature_diverse": [f"temperature_diverse_{i}" for i in range(3)],
        "size_diverse": ["size_diverse_small", "size_diverse_medium", "size_diverse_flagship"],
        "family_diverse": [f"family_diverse_{i}" for i in range(3)],
    }
    order = []
    for slot in range(3):
        for axis in AXIS_ORDER:
            cfg = axes_slots[axis][slot]
            if cfg in vectors:
                order.append(cfg)

    xs, ys = [], []
    for k in range(2, len(order) + 1):
        subset = {c: vectors[c] for c in order[:k]}
        mean_rho, _, _ = mean_pairwise_correlation(subset)
        xs.append(k)
        ys.append(n_eff(k, mean_rho))

    out_path = out_path or (FIGURES_DIR / "neff_vs_n.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(xs, ys, marker="o")
    ax.plot(xs, xs, linestyle="--", color="gray", label="N_eff = N (fully independent)")
    ax.set_xlabel("Number of ensemble members (N)")
    ax.set_ylabel("Effective number of independent voters (N_eff)")
    ax.set_title("N_eff saturation as same-base-model voters are added")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
