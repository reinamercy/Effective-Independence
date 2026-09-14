"""Phase 1 pilot: ~50 questions x all 12 configs, then print a random sample
of graded responses per model for hand-audit before full generation begins.

Per project_spec.yaml phase_gates.phase_1: do not proceed to Phase 2 until
the project owner confirms grading accuracy on this sample.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

# Windows consoles default stdout to cp1252, which raises UnicodeEncodeError on
# ordinary model output (e.g. U+2011 non-breaking hyphen) -- reconfigure to
# UTF-8 so a single odd character can't kill the run mid-report.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from src.build_configs import build_configs
from src.config_runner import generate
from src.dataset_loader import sample_pilot_set
from src.graders import gpqa_grader, math_grader, simpleqa_grader
from src.utils.cost_tracker import CostTracker, SINGLE_RUN_FLAG_THRESHOLD_USD

AUDIT_SAMPLE_SIZE = 50


def grade_row(row: dict, ground_truth: str, judge_config: dict | None, cost_tracker: CostTracker) -> dict:
    if row["error"] is not None:
        return {"correct": None, "extracted_answer": None}  # null, not wrong — per engineering rules

    dataset = row["dataset"]
    if dataset == "gpqa_diamond":
        result = gpqa_grader.grade(row["raw_response"], ground_truth)
    elif dataset == "math":
        result = math_grader.grade(row["raw_response"], ground_truth)
    elif dataset == "simpleqa":
        result = simpleqa_grader.grade(
            row["raw_response"],
            ground_truth,
            question_id=row["question_id"],
            judge_config=judge_config,
            cost_tracker=cost_tracker,
        )
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return result


def main():
    print("Loading pilot question set (~9 questions across 3 datasets)...")
    questions = sample_pilot_set(n_per_dataset=3, seed=0)
    print(f"Loaded {len(questions)} pilot questions.")

    configs = build_configs()
    n_calls = len(questions) * len(configs)
    print(f"Will issue up to {n_calls} (question, config) calls (fewer if any are already cached).")

    cost_tracker = CostTracker()

    print("Running generation...")
    results = generate(configs, questions, cost_tracker=cost_tracker, max_workers=8)

    if cost_tracker.total_usd > SINGLE_RUN_FLAG_THRESHOLD_USD:
        print(
            f"\n[FLAG] This pilot run cost ${cost_tracker.total_usd:.2f}, "
            f"above the ${SINGLE_RUN_FLAG_THRESHOLD_USD} single-run flag threshold.\n"
        )

    gt_by_qid = {q["question_id"]: q["ground_truth"] for q in questions}
    mid_tier = build_configs()[0]  # reuse prompt_diverse_0's model/provider/pricing as the SimpleQA judge
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

    n_errors = sum(1 for r in graded if r["error"] is not None)
    n_graded = len(graded) - n_errors
    n_correct = sum(1 for r in graded if r.get("correct") is True)
    n_quota_exhausted = sum(
        1 for r in graded if r["error"] is not None and "RESOURCE_EXHAUSTED" in (r["error"] or "") and "PerDay" in (r["error"] or "")
    )
    print(f"\nGeneration errors (null cells): {n_errors} / {len(graded)}  (of which {n_quota_exhausted} are daily-quota-exhausted)")
    print(f"Graded: {n_graded}, correct: {n_correct} ({(n_correct / n_graded * 100) if n_graded else 0:.1f}%)")

    print("\n=== Hand-audit sample ===")
    rng = random.Random(1)
    audit_pool = [r for r in graded if r["error"] is None]
    sample = rng.sample(audit_pool, min(AUDIT_SAMPLE_SIZE, len(audit_pool)))
    for r in sample:
        print("-" * 80)
        print(f"question_id={r['question_id']} dataset={r['dataset']} config_id={r['config_id']}")
        print(f"model_version={r['model_version']}")
        print(f"ground_truth={r['ground_truth']!r}")
        print(f"extracted_answer={r.get('extracted_answer')!r} correct={r.get('correct')}")
        print(f"raw_response (truncated): {(r['raw_response'] or '')[:300]!r}")

    cost_tracker.print_dashboard()

    if n_errors > 0:
        n_other = n_errors - n_quota_exhausted
        print(
            f"\nINCOMPLETE: {n_errors} cells are still missing ({n_quota_exhausted} daily-quota-"
            f"exhausted, {n_other} other transient errors -- e.g. 503 model-overloaded, or "
            "per-minute rate limits that outlasted retries). Cached progress is saved -- re-run "
            "`python scripts/run_pilot.py` again to retry (daily quota resets roughly at day "
            "rollover; transient/overload errors may clear on the very next attempt, no waiting "
            "needed). The audit sample above reflects only what completed so far."
        )
    else:
        print(
            "\nCOMPLETE: no null cells remain. Per phase_gates.phase_1 in config/project_spec.yaml, "
            "do not run run_full_generation.py until the project owner confirms grading accuracy "
            "on the sample above."
        )


if __name__ == "__main__":
    main()
