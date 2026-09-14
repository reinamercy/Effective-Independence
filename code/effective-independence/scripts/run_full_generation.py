"""Phase 2: full generation over the entire (shrunk-per-quota) 30-question
dataset x all 12 configs = 360 cells. Writes results/error_matrix.csv.

Fully cache-based and idempotent, exactly like run_pilot.py -- re-running
this script only issues calls for (question, config) cells not already
cached. Because dataset_loader.load_all() is the SAME deterministic 30-
question set (seed=0) that sample_pilot_set() draws its 9-question subset
from, every pilot cell already generated is reused here automatically; this
script only needs to fill the remaining ~252 cells.

Per project_spec.yaml phase_gates.phase_2: pause for confirmation if
projected cost exceeds full_generation_pause_threshold_usd ($150). Every
model in the current config is free-tier ($0), so this will never trigger
in practice, but the check is kept live rather than removed, in case a
paid model is ever reintroduced to config/models.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from src.analysis.error_matrix import coverage_summary, write_error_matrix_csv
from src.build_configs import build_configs
from src.config_runner import generate
from src.dataset_loader import load_all
from src.graders import gpqa_grader, math_grader, simpleqa_grader
from src.utils.cost_tracker import CostTracker, FULL_GENERATION_PAUSE_THRESHOLD_USD


def grade_row(row: dict, ground_truth: str, judge_config: dict | None, cost_tracker: CostTracker) -> dict:
    if row["error"] is not None:
        return {"correct": None, "extracted_answer": None}  # null, not wrong

    dataset = row["dataset"]
    if dataset == "gpqa_diamond":
        result = gpqa_grader.grade(row["raw_response"], ground_truth)
    elif dataset == "math":
        result = math_grader.grade(row["raw_response"], ground_truth)
    elif dataset == "simpleqa":
        result = simpleqa_grader.grade(
            row["raw_response"], ground_truth, question_id=row["question_id"],
            judge_config=judge_config, cost_tracker=cost_tracker,
        )
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return result


def main():
    print("Loading full question set (30 questions across 3 datasets)...")
    questions = load_all()
    print(f"Loaded {len(questions)} questions.")

    configs = build_configs()
    n_calls = len(questions) * len(configs)
    print(f"Full matrix: {n_calls} (question, config) cells. Already-cached cells (e.g. from the "
          f"Phase 1 pilot, which draws from this same 30-question set) are skipped automatically.")

    cost_tracker = CostTracker()

    # Pre-flight cost check per phase_gates.phase_2 -- always $0 on the
    # current free-tier config, but kept live rather than hardcoded away.
    projected_remaining = n_calls  # worst case: none cached yet
    avg_cost_per_call = (cost_tracker.total_usd / max(1, n_calls))  # rough, historical
    if avg_cost_per_call * projected_remaining > FULL_GENERATION_PAUSE_THRESHOLD_USD:
        print(
            f"\n[PAUSE] Projected cost for remaining cells exceeds "
            f"${FULL_GENERATION_PAUSE_THRESHOLD_USD} pause threshold. Per phase_gates.phase_2, "
            "halting for explicit owner confirmation before continuing."
        )
        return

    print("Running generation...")
    results = generate(configs, questions, cost_tracker=cost_tracker, max_workers=8)

    gt_by_qid = {q["question_id"]: q["ground_truth"] for q in questions}
    mid_tier = configs[0]
    judge_config = {
        "model_id": mid_tier["model_id"],
        "provider": mid_tier["provider"],
        "price_per_m_tokens": mid_tier["price_per_m_tokens"],
    }

    graded = []
    for row in results:
        gt = gt_by_qid.get(row["question_id"])
        grading = grade_row(row, gt, judge_config, cost_tracker)
        graded.append({**row, **grading, "ground_truth": gt})

    write_error_matrix_csv(graded)
    print(f"\nWrote results/error_matrix.csv ({len(graded)} rows).")

    summary = coverage_summary(
        [
            {**r, "correct": "" if r["correct"] is None else str(r["correct"])}
            for r in graded
        ]
    )
    n_missing = summary["n_cells_missing"]
    print(
        f"Coverage: {summary['n_cells_graded']}/{summary['n_cells_total']} cells graded "
        f"({summary['n_questions']} questions x {summary['n_configs']} configs), "
        f"accuracy {summary['accuracy_pct']:.1f}%."
    )

    cost_tracker.print_dashboard()

    if n_missing > 0:
        print(
            f"\nINCOMPLETE: {n_missing} cells still missing (daily quota / transient errors -- "
            "see logs/decisions.md for the current quota situation). Cached progress is saved -- "
            "re-run `python scripts/run_full_generation.py` again to retry."
        )
    else:
        print(
            "\nCOMPLETE: no null cells remain. Per phase_gates.phase_3, run "
            "`python scripts/run_analysis.py` next."
        )


if __name__ == "__main__":
    main()
