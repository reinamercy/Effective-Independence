"""Phase 4: budget-matched allocation experiment.

Per phase_gates.phase_4, this only runs after explicit go-ahead following
Phase 3. Uses only cached results already in results/error_matrix.csv --
generates no new data (see src/analysis/budget_allocation.py's module
docstring for why "budget" is matched on k, not $, in this project).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.budget_allocation import compare_allocations, format_comparison_table
from src.analysis.error_matrix import RESULTS_PATH, error_vectors, load_rows, pivot_correctness
from src.analysis.extra_plots import plot_budget_allocation, sync_paper_figures


def main():
    if not RESULTS_PATH.exists():
        print(f"[ERROR] {RESULTS_PATH} does not exist yet. Run run_full_generation.py first.")
        return

    rows = load_rows()
    question_ids, config_ids, matrix = pivot_correctness(rows)
    vectors = error_vectors(question_ids, config_ids, matrix)

    results = compare_allocations(question_ids, vectors)
    print("=== Budget-matched allocation comparison (k=3 members, matched by API-call count) ===\n")
    print(format_comparison_table(results))

    out_path = REPO_ROOT / "results" / "budget_allocation.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["allocation", "n_configs", "n_eff", "n_eff_ci_lo", "n_eff_ci_hi", "mean_rho", "mean_q", "n_eff_q", "n_nested", "n_pairs"])
        for name, r in results.items():
            if r is None:
                writer.writerow([name, "", "", "", "", "", "", "", "", ""])
            else:
                writer.writerow([name, r["n_configs"], r["n_eff"], r["n_eff_ci95"][0], r["n_eff_ci95"][1], r["mean_rho"], r.get("mean_q"), r.get("n_eff_q"), r.get("n_nested"), r.get("n_pairs")])
    print(f"\nSaved {out_path}")

    fig_path = plot_budget_allocation(results)
    print(f"Saved {fig_path}")
    synced = sync_paper_figures()
    print(f"Synced {len(synced)} figure(s) to paper/figures/ (used by paper/main.tex)")


if __name__ == "__main__":
    main()
