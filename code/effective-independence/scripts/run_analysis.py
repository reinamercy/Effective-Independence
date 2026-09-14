"""Phase 3: analysis + required figures.

Reads results/error_matrix.csv (written by run_full_generation.py, which is
safe to call at any coverage level -- it always regrades whatever's cached
and overwrites the CSV). This script therefore works whether generation is
30% or 100% done; it just reports coverage honestly so a partial-data run
is never mistaken for a final one.

Produces:
  figures/neff_vs_n.png
  figures/correlation_heatmap.png
  results/summary_stats.csv
and prints the headline per-axis N_eff numbers (with bootstrap 95% CIs)
directly to the terminal -- this is the actual go/no-go signal per
phase_gates.phase_3 ("decision gate for project owner -- do not
auto-proceed to phase 4").
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.analysis import correlation, extra_plots, neff, nesting, voting
from src.analysis.error_matrix import (
    RESULTS_PATH,
    confident_correlated_error_rate,
    coverage_summary,
    error_vectors,
    load_rows,
    pivot_correctness,
)
from src.analysis.neff import bootstrap_neff

AXIS_CONFIGS = {
    "prompt_diverse": [f"prompt_diverse_{i}" for i in range(3)],
    "temperature_diverse": [f"temperature_diverse_{i}" for i in range(3)],
    "size_diverse": ["size_diverse_small", "size_diverse_medium", "size_diverse_flagship"],
    "family_diverse": [f"family_diverse_{i}" for i in range(3)],
}


def main():
    if not RESULTS_PATH.exists():
        print(
            f"[ERROR] {RESULTS_PATH} does not exist yet. Run "
            "`python scripts/run_full_generation.py` at least once first."
        )
        return

    rows = load_rows()
    summary = coverage_summary(rows)
    print("=== Coverage ===")
    print(
        f"{summary['n_cells_graded']}/{summary['n_cells_total']} cells graded "
        f"({summary['n_questions']} questions x {summary['n_configs']} configs), "
        f"overall accuracy {summary['accuracy_pct']:.1f}%."
    )
    if summary["n_questions"] < 30 or summary["n_cells_missing"] > 0:
        print(
            "[NOTE] This is a PARTIAL run (full design target: 30 questions x 12 configs = "
            "360 cells). All N_eff / correlation numbers below are preliminary and their "
            "confidence intervals will be wide until generation completes -- do not treat "
            "them as final. See config/project_spec.yaml's small-N caveat and "
            "logs/decisions.md for the statistical-power discussion."
        )

    question_ids, config_ids, matrix = pivot_correctness(rows)
    vectors = error_vectors(question_ids, config_ids, matrix)

    print("\n=== N_eff per axis (bootstrap 95% CI, 1000 resamples) ===")
    summary_rows = []
    for axis, configs_in_axis in AXIS_CONFIGS.items():
        subset = {c: vectors[c] for c in configs_in_axis if c in vectors}
        if len(subset) < 2:
            print(f"{axis:22s}  insufficient data")
            continue
        r = bootstrap_neff(question_ids, subset)
        ci = r["n_eff_ci95"]
        ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci[0] is not None else "n/a"
        print(
            f"{axis:22s}  N_eff={r['n_eff']:.3f}  95% CI={ci_str}  "
            f"mean_rho={r['mean_rho']:.3f}  (pairs used={r['n_pairs_used']}, "
            f"undefined={r['n_pairs_undefined']})"
        )
        summary_rows.append({"axis": axis, **r})

    print("\n=== N_eff across all 12 configs (whole ensemble) ===")
    r_all = bootstrap_neff(question_ids, vectors)
    ci = r_all["n_eff_ci95"]
    ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci[0] is not None else "n/a"
    print(
        f"all_configs            N_eff={r_all['n_eff']:.3f}  95% CI={ci_str}  "
        f"mean_rho={r_all['mean_rho']:.3f}  (pairs used={r_all['n_pairs_used']}, "
        f"undefined={r_all['n_pairs_undefined']})"
    )
    summary_rows.append({"axis": "all_configs", **r_all})

    # Error-nesting structure. This is the mechanism behind the numbers above:
    # a "nested" pair is one where the weaker member fails on a superset of
    # what the stronger one fails on, so it contributes no question the
    # ensemble could recover by voting.
    print("\n=== Error nesting (is one member's error set a subset of the other's?) ===")
    model_map = nesting.config_model_map(rows)
    by_type = nesting.nesting_by_pair_type(config_ids, vectors, model_map)
    print(nesting.format_nesting_table(by_type))
    n_nested, n_pairs, rate = nesting.overall_nesting(by_type)
    print(
        f"A nested pair has zero complementary errors: the weaker member gets nothing "
        f"right that the stronger member gets wrong, so majority voting over the two "
        f"cannot recover any question."
    )
    lo, hi = nesting.bootstrap_nesting_rate(config_ids, vectors)
    print(f"Overall nesting rate {rate*100:.0f}%, bootstrap 95% CI [{lo*100:.0f}%, {hi*100:.0f}%].")

    # Chance baseline. Sparse error tables can produce empty off-diagonal
    # cells by luck, so the raw rate is uninterpretable without this control.
    null = nesting.permutation_null(config_ids, vectors, model_map)
    print(
        f"Permutation null ({null['n_permutations']} shuffles, marginals and coverage held fixed):\n"
        f"  all pairs    observed {null['observed_all']*100:.0f}%  vs chance "
        f"{null['null_all_mean']*100:.0f}% "
        f"[{null['null_all_ci'][0]*100:.0f}%, {null['null_all_ci'][1]*100:.0f}%]  "
        f"p={null['p_all']:.4f}\n"
        f"  same model   observed {null['observed_same_model']*100:.0f}%  vs chance "
        f"{null['null_same_mean']*100:.0f}% "
        f"[{null['null_same_ci'][0]*100:.0f}%, {null['null_same_ci'][1]*100:.0f}%]  "
        f"p={null['p_same_model']:.4f}"
    )

    # Does voting actually help? The outcome the whole paper is about, measured
    # directly rather than inferred from N_eff.
    print("\n=== Majority vote vs. individual members (no new API calls) ===")
    vote_reports = {}
    for axis, configs_in_axis in AXIS_CONFIGS.items():
        present = [c for c in configs_in_axis if c in config_ids]
        vote_reports[axis] = voting.majority_vote_report(question_ids, present, matrix)
    vote_reports["all_configs"] = voting.majority_vote_report(question_ids, config_ids, matrix)
    print(voting.format_voting_table(vote_reports))

    # Nesting rate uncertainty and the trend behind the word "monotonic".
    print("\n=== Nesting rate intervals and trend ===")
    for kind in nesting.PAIR_TYPES:
        b = by_type.get(kind)
        if not b or b["n_pairs"] == 0:
            continue
        lo_w, hi_w = nesting.wilson_interval(b["n_nested"], b["n_pairs"])
        print(
            f"{kind:28s} {b['n_nested']:2d}/{b['n_pairs']:<2d} = {b['rate']*100:5.1f}%  "
            f"Wilson 95% CI [{lo_w*100:.0f}%, {hi_w*100:.0f}%]"
        )
    tt = nesting.trend_test(by_type)
    print(
        f"Cochran-Armitage trend test: z={tt['z']:.2f}, two-sided p={tt['p_two_sided']:.5f} "
        f"(z<0 means nesting falls as members get more different)"
    )

    # Marginal-robust recomputation. phi is capped by the two configs' error
    # rates, so a phi-based N_eff is inflated whenever ensemble members differ
    # in accuracy. Yule's Q is not subject to that cap, which makes this the
    # load-bearing robustness check on every number printed above: if the axis
    # ordering holds under Q, it is not an artifact of accuracy heterogeneity.
    print("\n=== Marginal-robust recomputation (Yule's Q instead of phi) ===")
    print(f"{'axis':22s} {'Q':>6s} {'N_eff(Q)':>9s} {'95% CI':>16s}   {'N_eff(phi)':>10s}  {'inflation':>9s}")
    for row in summary_rows:
        axis = row["axis"]
        subset = vectors if axis == "all_configs" else {
            c: vectors[c] for c in AXIS_CONFIGS[axis] if c in vectors
        }
        if len(subset) < 2:
            continue
        rq = bootstrap_neff(question_ids, subset, measure=neff.pairwise_yules_q)
        ciq = rq["n_eff_ci95"]
        ciq_str = f"[{ciq[0]:.2f}, {ciq[1]:.2f}]" if ciq[0] is not None else "n/a"
        inflation = row["n_eff"] - rq["n_eff"]
        print(
            f"{axis:22s} {rq['mean_rho']:6.3f} {rq['n_eff']:9.3f} {ciq_str:>16s}   "
            f"{row['n_eff']:10.3f}  {inflation:+9.3f}"
        )
        row["mean_q"] = rq["mean_rho"]
        row["n_eff_q"] = rq["n_eff"]
        row["n_eff_q_ci_lo"] = ciq[0]
        row["n_eff_q_ci_hi"] = ciq[1]
    print(
        "'inflation' is N_eff(phi) - N_eff(Q): how much the phi-based estimate "
        "overstates effective independence because phi cannot exceed the ceiling "
        "set by the members' differing error rates."
    )

    print("\n=== Confident correlated errors (>=80% of ensemble wrong together) ===")
    cce_all = confident_correlated_error_rate(question_ids, config_ids, matrix, threshold=0.8)
    print(
        f"all_configs (k={len(config_ids)}): {cce_all['n_confident_wrong']}/{cce_all['n_eligible_questions']} "
        f"eligible questions ({cce_all['rate']*100:.1f}%), {cce_all['n_excluded_questions']} questions excluded "
        f"(fewer than 3 graded configs)."
    )
    cce_by_axis = {"all_configs": cce_all}
    for axis, configs_in_axis in AXIS_CONFIGS.items():
        present = [c for c in configs_in_axis if c in config_ids]
        if len(present) < 3:
            continue
        cce_axis = confident_correlated_error_rate(question_ids, present, matrix, threshold=0.8, min_graded=3)
        cce_by_axis[axis] = cce_axis
        print(
            f"{axis:22s}: {cce_axis['n_confident_wrong']}/{cce_axis['n_eligible_questions']} eligible "
            f"({cce_axis['rate']*100:.1f}%)"
        )

    # Marginal-bound diagnostic. Phi cannot exceed a ceiling set by the two
    # configs' error rates, so a low measured correlation is only evidence of
    # real error decorrelation if it sits well BELOW that ceiling. Printed per
    # within-axis pair because it materially changes how Table I reads: a pair
    # at ratio ~1.00 is already as correlated as its marginals allow.
    print("\n=== Observed phi vs. marginal-bound ceiling (within-axis pairs) ===")
    print(f"{'pair':46s} {'n':>3s} {'rho':>6s} {'max':>6s} {'ratio':>6s}")
    at_ceiling = 0
    total_pairs = 0
    for axis, configs_in_axis in AXIS_CONFIGS.items():
        present = [c for c in configs_in_axis if c in config_ids]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = present[i], present[j]
                rho = neff.pairwise_phi(vectors[a], vectors[b])
                mx = neff.max_attainable_phi(vectors[a], vectors[b])
                if rho is None or mx is None or mx == 0:
                    continue
                n_ov = sum(1 for x, y in zip(vectors[a], vectors[b]) if x is not None and y is not None)
                ratio = rho / mx
                total_pairs += 1
                if ratio >= 0.999:
                    at_ceiling += 1
                label = f"{a.replace('_diverse','')} x {b.replace('_diverse','')}"
                print(f"{label:46s} {n_ov:3d} {rho:6.3f} {mx:6.3f} {ratio:6.3f}")
    if total_pairs:
        print(
            f"{at_ceiling}/{total_pairs} within-axis pairs sit at their marginal-bound ceiling "
            f"(ratio >= 0.999): their phi is as high as differing accuracy permits, so a low "
            f"phi there is NOT evidence of genuine error decorrelation."
        )

    # Figures
    corr_ids, corr_mat = correlation.build_correlation_matrix(vectors)
    heatmap_path = correlation.plot_heatmap(corr_ids, corr_mat)
    print(f"\nSaved {heatmap_path}")

    try:
        neff_curve_path = correlation.plot_neff_vs_n(question_ids, vectors)
        print(f"Saved {neff_curve_path}")
    except Exception as e:  # noqa: BLE001 -- a figure failing to render shouldn't hide the printed numbers above
        print(f"[WARN] Could not render neff_vs_n.png: {e}")

    try:
        neff_bar_path = extra_plots.plot_neff_bar(summary_rows)
        print(f"Saved {neff_bar_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Could not render neff_bar.png: {e}")

    try:
        cce_bar_path = extra_plots.plot_cce_bar(cce_by_axis)
        print(f"Saved {cce_bar_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Could not render cce_rate_bar.png: {e}")

    try:
        acc_bar_path = extra_plots.plot_accuracy_per_config(rows)
        print(f"Saved {acc_bar_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Could not render accuracy_per_config.png: {e}")

    try:
        nesting_path = extra_plots.plot_nesting_by_pair_type(by_type)
        print(f"Saved {nesting_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Could not render nesting_by_pair_type.png: {e}")

    try:
        phi_q_path = extra_plots.plot_phi_vs_q(summary_rows)
        print(f"Saved {phi_q_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Could not render phi_vs_q.png: {e}")

    try:
        overlap_path = extra_plots.plot_pair_overlap_diagnostic(vectors)
        print(f"Saved {overlap_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Could not render pair_overlap_diagnostic.png: {e}")

    synced = extra_plots.sync_paper_figures()
    print(f"Synced {len(synced)} figure(s) to paper/figures/ (used by paper/main.tex)")

    # Summary CSV
    summary_path = REPO_ROOT / "results" / "summary_stats.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["axis", "n_configs", "n_questions", "n_pairs_used", "n_pairs_undefined",
                           "mean_rho", "mean_rho_ci_lo", "mean_rho_ci_hi", "n_eff", "n_eff_ci_lo", "n_eff_ci_hi",
                           "mean_q", "n_eff_q", "n_eff_q_ci_lo", "n_eff_q_ci_hi"]
        )
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(
                {
                    "axis": r["axis"], "n_configs": r["n_configs"], "n_questions": r["n_questions"],
                    "n_pairs_used": r["n_pairs_used"], "n_pairs_undefined": r["n_pairs_undefined"],
                    "mean_rho": r["mean_rho"],
                    "mean_rho_ci_lo": r["mean_rho_ci95"][0], "mean_rho_ci_hi": r["mean_rho_ci95"][1],
                    "n_eff": r["n_eff"],
                    "n_eff_ci_lo": r["n_eff_ci95"][0], "n_eff_ci_hi": r["n_eff_ci95"][1],
                    "mean_q": r.get("mean_q"), "n_eff_q": r.get("n_eff_q"),
                    "n_eff_q_ci_lo": r.get("n_eff_q_ci_lo"), "n_eff_q_ci_hi": r.get("n_eff_q_ci_hi"),
                }
            )
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
