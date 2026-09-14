"""Reads/writes results/error_matrix.csv and pivots it into per-config error
vectors for downstream N_eff / correlation analysis.

error_matrix.csv schema (one row per (question, config) cell attempted):
question_id, dataset, config_id, axis, model_id, model_version, correct,
extracted_answer, ground_truth, error, cost_usd, timestamp.
`correct` is "" (empty) for any cell with a non-null `error` -- null, not
wrong, per the project's engineering rules.
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = REPO_ROOT / "results" / "error_matrix.csv"

FIELDNAMES = [
    "question_id", "dataset", "config_id", "axis", "model_id", "model_version",
    "correct", "extracted_answer", "ground_truth", "error", "cost_usd", "timestamp",
]


def write_error_matrix_csv(graded_rows: list[dict], path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in graded_rows:
            writer.writerow(
                {
                    "question_id": r["question_id"],
                    "dataset": r["dataset"],
                    "config_id": r["config_id"],
                    "axis": r.get("axis"),
                    "model_id": r.get("model_id"),
                    "model_version": r.get("model_version"),
                    "correct": "" if r.get("correct") is None else str(r["correct"]),
                    "extracted_answer": r.get("extracted_answer"),
                    "ground_truth": r.get("ground_truth"),
                    "error": r.get("error") or "",
                    "cost_usd": r.get("cost_usd", 0.0),
                    "timestamp": r.get("timestamp"),
                }
            )


def load_rows(path: Path = RESULTS_PATH) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_correct(raw: str):
    if raw == "True":
        return True
    if raw == "False":
        return False
    return None


def pivot_correctness(rows: list[dict]) -> tuple[list[str], list[str], dict]:
    """Returns (question_ids, config_ids, matrix) with matrix[question_id][config_id]
    in {True, False, None} (None = ungraded/error)."""
    question_ids = sorted({r["question_id"] for r in rows})
    config_ids = sorted({r["config_id"] for r in rows})
    lookup = {(r["question_id"], r["config_id"]): _parse_correct(r["correct"]) for r in rows}
    matrix = {q: {c: lookup.get((q, c)) for c in config_ids} for q in question_ids}
    return question_ids, config_ids, matrix


def error_vectors(question_ids: list[str], config_ids: list[str], matrix: dict) -> dict[str, list]:
    """config_id -> [1.0 if wrong, 0.0 if correct, None if ungraded] in question_ids order."""
    vectors = {}
    for c in config_ids:
        vec = []
        for q in question_ids:
            v = matrix[q][c]
            vec.append(None if v is None else (0.0 if v else 1.0))
        vectors[c] = vec
    return vectors


def confident_correlated_error_rate(
    question_ids: list[str], config_ids: list[str], matrix: dict, threshold: float = 0.8, min_graded: int = 3
) -> dict:
    """Fraction of questions where >= `threshold` of the graded configs in
    the ensemble are simultaneously wrong -- a confident, correlated error
    that a majority vote over this exact ensemble cannot detect or correct.
    Questions with fewer than `min_graded` graded configs are excluded
    (not enough configs to assess an 80% agreement meaningfully) and
    counted separately so the denominator is never silently inflated.
    """
    n_confident_wrong = 0
    n_eligible = 0
    n_excluded = 0
    per_question = {}
    for q in question_ids:
        graded = [matrix[q][c] for c in config_ids if matrix[q][c] is not None]
        if len(graded) < min_graded:
            n_excluded += 1
            continue
        n_eligible += 1
        wrong_frac = sum(1 for v in graded if v is False) / len(graded)
        is_confident_wrong = wrong_frac >= threshold
        per_question[q] = wrong_frac
        if is_confident_wrong:
            n_confident_wrong += 1
    return {
        "threshold": threshold,
        "n_eligible_questions": n_eligible,
        "n_excluded_questions": n_excluded,
        "n_confident_wrong": n_confident_wrong,
        "rate": (n_confident_wrong / n_eligible) if n_eligible else 0.0,
        "per_question_wrong_frac": per_question,
    }


def coverage_summary(rows: list[dict]) -> dict:
    """How complete is the matrix -- for honestly reporting CI-width caveats."""
    n_total = len(rows)
    n_graded = sum(1 for r in rows if r["correct"] in ("True", "False"))
    n_correct = sum(1 for r in rows if r["correct"] == "True")
    n_questions = len({r["question_id"] for r in rows})
    n_configs = len({r["config_id"] for r in rows})
    return {
        "n_cells_total": n_total,
        "n_cells_graded": n_graded,
        "n_cells_missing": n_total - n_graded,
        "n_correct": n_correct,
        "accuracy_pct": (n_correct / n_graded * 100) if n_graded else 0.0,
        "n_questions": n_questions,
        "n_configs": n_configs,
    }
