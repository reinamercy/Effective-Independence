"""Additional figures for the paper beyond the two core analysis plots in
correlation.py: a per-axis N_eff bar chart with bootstrap CI error bars, a
confident-correlated-error-rate bar chart, and a per-config accuracy bar
chart grouped by axis. Kept separate from correlation.py since these are
paper-presentation figures rather than core analysis outputs consumed by
anything else.
"""
from __future__ import annotations

from pathlib import Path

from src.analysis.correlation import AXIS_ORDER, _axis_of

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "figures"
PAPER_FIGURES_DIR = REPO_ROOT / "paper" / "figures"


def sync_paper_figures() -> list[Path]:
    """Copies every PNG in figures/ (gitignored, regenerated each analysis
    run) into paper/figures/ (tracked in git, referenced by main.tex's
    \\includegraphics paths). main.tex intentionally does NOT reference
    ../figures/ directly -- a plain relative path outside the paper/
    directory silently fails to embed on several online LaTeX renderers
    that only see files inside the uploaded/compiled directory. Call this
    after any script that (re)writes figures/*.png."""
    import shutil

    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in FIGURES_DIR.glob("*.png"):
        dst = PAPER_FIGURES_DIR / src.name
        shutil.copyfile(src, dst)
        copied.append(dst)
    return copied

AXIS_COLORS = {
    "prompt_diverse": "#4C72B0",
    "temperature_diverse": "#DD8452",
    "size_diverse": "#55A868",
    "family_diverse": "#C44E52",
}
AXIS_LABELS = {
    "prompt_diverse": "Prompt",
    "temperature_diverse": "Temperature",
    "size_diverse": "Size",
    "family_diverse": "Family",
}


def plot_neff_bar(summary_rows: list[dict], out_path: Path | None = None) -> Path:
    """summary_rows: list of dicts with axis, n_eff, n_eff_ci95 (tuple),
    mean_rho -- same shape as bootstrap_neff() output plus an 'axis' key."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path = out_path or (FIGURES_DIR / "neff_bar.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    axes_present = [r for r in summary_rows if r["axis"] != "all_configs"]
    all_row = next((r for r in summary_rows if r["axis"] == "all_configs"), None)
    rows = axes_present + ([all_row] if all_row else [])

    labels = [AXIS_LABELS.get(r["axis"], r["axis"]) for r in rows]
    values = [r["n_eff"] for r in rows]
    colors = [AXIS_COLORS.get(r["axis"], "#888888") for r in rows]
    ci_lo = [r["n_eff_ci95"][0] if r["n_eff_ci95"][0] is not None else r["n_eff"] for r in rows]
    ci_hi = [r["n_eff_ci95"][1] if r["n_eff_ci95"][1] is not None else r["n_eff"] for r in rows]
    err_lo = [max(0.0, v - lo) for v, lo in zip(values, ci_lo)]
    err_hi = [max(0.0, hi - v) for v, hi in zip(values, ci_hi)]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, yerr=[err_lo, err_hi], capsize=5, color=colors, edgecolor="black", linewidth=0.5)
    n_nominal = [3 if r["axis"] != "all_configs" else 12 for r in rows]
    for xi, n in zip(x, n_nominal):
        ax.plot([xi - 0.4, xi + 0.4], [n, n], linestyle="--", color="gray", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("$N_\\mathrm{eff}$ (bootstrap 95% CI)")
    ax.set_title("Effective independent voters per axis\n(dashed line = nominal ensemble size)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_cce_bar(cce_by_group: dict[str, dict], out_path: Path | None = None) -> Path:
    """cce_by_group: {group_label: cce_result_dict} where cce_result_dict has
    'rate', 'n_confident_wrong', 'n_eligible_questions' (from
    error_matrix.confident_correlated_error_rate)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path = out_path or (FIGURES_DIR / "cce_rate_bar.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(cce_by_group.keys())
    rates = [cce_by_group[label]["rate"] * 100 for label in labels]
    colors = [AXIS_COLORS.get(label, "#666666") if label in AXIS_COLORS else "#666666" for label in labels]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, rates, color=colors, edgecolor="black", linewidth=0.5)
    for xi, label in zip(x, labels):
        d = cce_by_group[label]
        ax.text(xi, d["rate"] * 100 + 0.5, f"{d['n_confident_wrong']}/{d['n_eligible_questions']}",
                ha="center", va="bottom", fontsize=8)
    display_labels = [AXIS_LABELS.get(label, label.replace("_", " ").title()) for label in labels]
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.set_ylabel("Confident correlated-error rate (%)")
    ax.set_title("Fraction of questions where $\\geq$80% of the ensemble\nis wrong together, per axis")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


ALLOCATION_LABELS = {
    "same_family_prompt": "Same model,\n3 prompts",
    "same_family_temperature": "Same model,\n3 temperatures",
    "same_family_size": "Same family,\n3 sizes",
    "cross_family": "3 different\nfamilies",
    "mixed_one_per_axis": "Mixed,\none per axis",
}
ALLOCATION_COLORS = {
    "same_family_prompt": AXIS_COLORS["prompt_diverse"],
    "same_family_temperature": AXIS_COLORS["temperature_diverse"],
    "same_family_size": AXIS_COLORS["size_diverse"],
    "cross_family": AXIS_COLORS["family_diverse"],
    "mixed_one_per_axis": "#8172B3",
}


def plot_budget_allocation(results: dict[str, dict | None], out_path: Path | None = None) -> Path:
    """Phase 4 figure: N_eff for each budget-matched (k=3) allocation strategy.

    results: output of budget_allocation.compare_allocations(). Every bar is
    the same k, so the height difference is purely how the fixed budget was
    spent, which is the practitioner-facing question this phase asks.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path = out_path or (FIGURES_DIR / "budget_allocation.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = [n for n in ALLOCATIONS_ORDER if results.get(n) is not None]
    values = [results[n]["n_eff"] for n in names]
    ci_lo = [results[n]["n_eff_ci95"][0] if results[n]["n_eff_ci95"][0] is not None else results[n]["n_eff"] for n in names]
    ci_hi = [results[n]["n_eff_ci95"][1] if results[n]["n_eff_ci95"][1] is not None else results[n]["n_eff"] for n in names]
    err_lo = [max(0.0, v - lo) for v, lo in zip(values, ci_lo)]
    err_hi = [max(0.0, hi - v) for v, hi in zip(values, ci_hi)]
    colors = [ALLOCATION_COLORS.get(n, "#888888") for n in names]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(names))
    ax.bar(x, values, yerr=[err_lo, err_hi], capsize=5, color=colors,
           edgecolor="black", linewidth=0.5)
    ax.axhline(3.0, linestyle="--", color="gray", linewidth=1)
    ax.text(len(names) - 0.5, 3.02, "fully independent (k=3)", ha="right",
            va="bottom", fontsize=7, color="gray")
    ax.axhline(1.0, linestyle=":", color="black", linewidth=1)
    ax.text(len(names) - 0.5, 0.94, "one effective voter (no ensemble benefit)",
            ha="right", va="top", fontsize=7)
    # Label above each error bar's upper whisker, not above the bar, so the
    # value never collides with the CI it belongs to.
    for xi, v, hi in zip(x, values, ci_hi):
        ax.text(xi, hi + 0.06, f"{v:.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([ALLOCATION_LABELS.get(n, n) for n in names], fontsize=8)
    ax.set_ylabel("$N_\\mathrm{eff}$ (bootstrap 95% CI)")
    ax.set_ylim(0, 3.3)
    ax.set_title("Where to spend a fixed 3-call budget\n(every bar is the same number of API calls)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


ALLOCATIONS_ORDER = [
    "same_family_prompt",
    "same_family_temperature",
    "same_family_size",
    "cross_family",
    "mixed_one_per_axis",
]


def plot_nesting_by_pair_type(by_type: dict, out_path: Path | None = None) -> Path:
    """Headline structural figure: how often one member's errors are a strict
    subset of the other's, as the two members become genuinely more different.

    by_type: output of nesting.nesting_by_pair_type().
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from src.analysis.nesting import PAIR_TYPES

    out_path = out_path or (FIGURES_DIR / "nesting_by_pair_type.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kinds = [k for k in PAIR_TYPES if k in by_type and by_type[k]["n_pairs"] > 0]
    rates = [by_type[k]["rate"] * 100 for k in kinds]
    counts = [(by_type[k]["n_nested"], by_type[k]["n_pairs"]) for k in kinds]
    labels = [k.replace(", ", ",\n") for k in kinds]
    colors = ["#C44E52", "#DD8452", "#55A868"][: len(kinds)]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(kinds))
    ax.bar(x, rates, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    for xi, r, (nn, nt) in zip(x, rates, counts):
        ax.text(xi, r + 1.5, f"{r:.0f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax.text(xi, 3, f"{nn}/{nt}", ha="center", va="bottom", fontsize=8, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Pairs with fully nested errors (%)")
    ax.set_ylim(0, 112)
    ax.set_title("How often one member's errors are a subset of the other's\n"
                 "(nested = the weaker member recovers nothing)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_phi_vs_q(summary_rows: list[dict], out_path: Path | None = None) -> Path:
    """Headline robustness figure: N_eff computed with phi against N_eff
    computed with Yule's Q, per axis.

    phi is capped by the two members' error rates, so it credits an ensemble
    for accuracy heterogeneity that has nothing to do with failing on
    different questions. Q is not. Plotting the pair shows both that the axis
    ordering is preserved and that the phi-based numbers are uniformly the
    more flattering of the two.

    summary_rows: rows carrying 'axis', 'n_eff', and 'n_eff_q'.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path = out_path or (FIGURES_DIR / "phi_vs_q.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [r for r in summary_rows if r.get("n_eff_q") is not None]
    if not rows:
        raise ValueError("no rows carry n_eff_q; run the Yule's Q pass first")

    labels = [AXIS_LABELS.get(r["axis"], "All 12") for r in rows]
    phi_vals = [r["n_eff"] for r in rows]
    q_vals = [r["n_eff_q"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, phi_vals, w, label=r"$N_\mathrm{eff}$ from $\phi$ (marginal-capped)",
           color="#B0B7C3", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, q_vals, w, label=r"$N_\mathrm{eff}$ from Yule's $Q$ (marginal-robust)",
           color="#C44E52", edgecolor="black", linewidth=0.5)
    for xi, v in zip(x - w / 2, phi_vals):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + w / 2, q_vals):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.axhline(1.0, linestyle=":", color="black", linewidth=1)
    ax.text(-0.45, 0.955, "one effective voter", ha="left", va="top", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"$N_\mathrm{eff}$")
    ax.set_ylim(0, max(max(phi_vals), max(q_vals)) * 1.35)
    ax.set_title("Correcting for accuracy differences makes every ensemble\nlook less independent, not more")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pair_overlap_diagnostic(vectors: dict[str, list], out_path: Path | None = None) -> Path:
    """Honesty/diagnostic figure: each configuration pair's error correlation
    plotted against the number of questions BOTH members actually have graded.

    Under partial coverage the pairwise-complete correlation for a thin pair
    rests on as few as 8 questions, where phi is both unstable and prone to
    saturating at exactly +-1. Plotting rho against overlap makes that
    dependence visible instead of burying it, so a reader can see which
    points in the headline comparison are well-sampled and which are not.
    """
    import itertools

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.analysis.neff import pairwise_phi

    out_path = out_path or (FIGURES_DIR / "pair_overlap_diagnostic.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    seen_axes = set()
    # Several pairs land on identical (overlap, rho) coordinates -- notably
    # more than one thin pair saturating at exactly rho=1.0 -- so points are
    # jittered slightly along x to keep every pair individually visible.
    plotted: dict[tuple[int, float], int] = {}
    for a, b in itertools.combinations(sorted(vectors.keys()), 2):
        axis_a, axis_b = _axis_of(a), _axis_of(b)
        if axis_a != axis_b:
            continue  # within-axis pairs only: these are what Table I reports
        rho = pairwise_phi(vectors[a], vectors[b])
        if rho is None:
            continue
        overlap = sum(1 for x, y in zip(vectors[a], vectors[b]) if x is not None and y is not None)
        key = (overlap, round(rho, 3))
        n_prior = plotted.get(key, 0)
        plotted[key] = n_prior + 1
        ax.scatter(overlap + 0.28 * n_prior, rho, s=70, alpha=0.85,
                   color=AXIS_COLORS.get(axis_a, "#888888"),
                   edgecolor="black", linewidth=0.5, zorder=3,
                   label=AXIS_LABELS.get(axis_a) if axis_a not in seen_axes else None)
        seen_axes.add(axis_a)

    ax.axhline(1.0, linestyle=":", color="gray", linewidth=1)
    ax.set_xlabel("Questions graded for BOTH members of the pair")
    ax.set_ylabel("Pairwise error correlation $\\rho$")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("Correlation vs. how much data each pair actually has\n(thin pairs saturate at $\\rho=1$)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_accuracy_per_config(rows: list[dict], out_path: Path | None = None) -> Path:
    """rows: raw error_matrix.csv rows (list of dicts with config_id, correct)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path = out_path or (FIGURES_DIR / "accuracy_per_config.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_config: dict[str, list[bool]] = {}
    for r in rows:
        if r["correct"] not in ("True", "False"):
            continue
        by_config.setdefault(r["config_id"], []).append(r["correct"] == "True")

    config_ids = sorted(
        by_config.keys(),
        key=lambda c: (AXIS_ORDER.index(_axis_of(c)) if _axis_of(c) in AXIS_ORDER else 99, c),
    )
    accuracies = [100.0 * sum(by_config[c]) / len(by_config[c]) for c in config_ids]
    ns = [len(by_config[c]) for c in config_ids]
    colors = [AXIS_COLORS.get(_axis_of(c), "#888888") for c in config_ids]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(config_ids))
    ax.bar(x, accuracies, color=colors, edgecolor="black", linewidth=0.5)
    for xi, n in zip(x, ns):
        ax.text(xi, 2, f"n={n}", ha="center", va="bottom", fontsize=7, rotation=90, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(config_ids, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Per-configuration accuracy (colored by diversity axis)")
    from matplotlib.patches import Patch
    handles = [Patch(color=AXIS_COLORS[a], label=AXIS_LABELS[a]) for a in AXIS_ORDER]
    ax.legend(handles=handles, loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
