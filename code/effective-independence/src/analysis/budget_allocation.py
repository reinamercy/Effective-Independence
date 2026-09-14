"""Phase 4: budget-matched allocation comparison.

Honesty note (logged in full in logs/decisions.md): the original spec frames
this as a comparison "at matched total API cost". Every model in this
project's final config is free-tier ($0), so dollar cost cannot be the
matching resource -- there is nothing to match. The real scarce, binding
resource throughout this project has been *API calls against a per-model
daily quota* (see config/models.yaml), and the harness's own design (one
sample per (question, config) pair) means "number of ensemble members, k"
already IS that resource. This module therefore matches on k instead of $,
which is both the honest choice given the free-tier pivot and a direct
test of the paper's actual thesis: at fixed k, is an ensemble built from
one base model (varying prompt, temperature, or size) as reliable as one
spanning distinct model families?

Uses ONLY cached results already in results/error_matrix.csv -- generates
no new data, per the "never silently expand scope" engineering rule.
"""
from __future__ import annotations

from src.analysis.neff import bootstrap_neff

# Fixed, documented allocation choices (k=3, the max available per axis in
# this project's 4-axis-x-3-config design). "mixed_one_per_axis" is one
# arbitrary but fixed representative pick spanning 3 of the 4 axes.
ALLOCATIONS = {
    "same_family_prompt": ["prompt_diverse_0", "prompt_diverse_1", "prompt_diverse_2"],
    "same_family_temperature": ["temperature_diverse_0", "temperature_diverse_1", "temperature_diverse_2"],
    "same_family_size": ["size_diverse_small", "size_diverse_medium", "size_diverse_flagship"],
    "cross_family": ["family_diverse_0", "family_diverse_1", "family_diverse_2"],
    "mixed_one_per_axis": ["prompt_diverse_0", "temperature_diverse_1", "size_diverse_flagship"],
}


def compare_allocations(
    question_ids: list[str], vectors: dict[str, list], n_resamples: int = 1000
) -> dict[str, dict | None]:
    """Compare allocations under phi, under Yule's Q, and by error nesting.

    All three are reported together deliberately. Ranking allocations by the
    phi-based N_eff alone would contradict this project's own finding that phi
    is inflated by differences in member accuracy (see neff.max_attainable_phi
    and analysis/nesting.py): the phi column makes the spread between
    allocations look far larger than the marginal-robust column supports.
    """
    import itertools

    from src.analysis.neff import mean_pairwise_correlation, n_eff, pairwise_yules_q
    from src.analysis.nesting import pair_nesting

    results = {}
    for name, config_ids in ALLOCATIONS.items():
        subset = {c: vectors[c] for c in config_ids if c in vectors}
        if len(subset) < 2:
            results[name] = None
            continue
        r = bootstrap_neff(question_ids, subset, n_resamples=n_resamples)

        q_mean, _, _ = mean_pairwise_correlation(subset, measure=pairwise_yules_q)
        r["mean_q"] = q_mean
        r["n_eff_q"] = n_eff(len(subset), q_mean)

        n_nested = n_pairs = 0
        for a, b in itertools.combinations(config_ids, 2):
            if a not in vectors or b not in vectors:
                continue
            info = pair_nesting(vectors[a], vectors[b])
            if info is None:
                continue
            n_pairs += 1
            n_nested += int(info["nested"])
        r["n_nested"] = n_nested
        r["n_pairs"] = n_pairs
        results[name] = r
    return results


def format_comparison_table(results: dict[str, dict | None]) -> str:
    lines = [
        f"{'allocation':26s} {'k':>2s} {'rho':>6s} {'Neff(phi)':>9s} {'95% CI':>14s} "
        f"{'Q':>6s} {'Neff(Q)':>8s} {'nested':>7s}"
    ]
    for name, r in results.items():
        if r is None:
            lines.append(f"{name:26s}  -      -         -              -      -        -       -")
            continue
        ci = r["n_eff_ci95"]
        ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci[0] is not None else "n/a"
        nested = f"{r.get('n_nested', 0)}/{r.get('n_pairs', 0)}"
        lines.append(
            f"{name:26s} {r['n_configs']:>2d} {r['mean_rho']:>6.3f} {r['n_eff']:>9.3f} "
            f"{ci_str:>14s} {r.get('mean_q', float('nan')):>6.3f} "
            f"{r.get('n_eff_q', float('nan')):>8.3f} {nested:>7s}"
        )
    lines.append(
        "\nRead the Neff(Q) and nested columns, not Neff(phi) alone: phi is capped by "
        "differences in member accuracy and overstates the spread between allocations."
    )
    return "\n".join(lines)
