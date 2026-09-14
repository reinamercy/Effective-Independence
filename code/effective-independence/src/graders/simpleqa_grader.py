"""SimpleQA grader: normalized string containment, with an LLM-judge fallback
for ambiguous cases (~10-15% expected per project_spec.yaml).

Every fallback case is appended to logs/simpleqa_judge_audit.jsonl so it can
be hand-audited later — the LLM judge's verdict is never treated as silently
authoritative.
"""
from __future__ import annotations

import json
import re
import string
import threading
from pathlib import Path
from typing import Optional

from src.api_client import get_response
from src.utils.cost_tracker import CostTracker

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_LOG_PATH = REPO_ROOT / "logs" / "simpleqa_judge_audit.jsonl"

_audit_lock = threading.Lock()

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

JUDGE_SYSTEM_PROMPT = (
    "You are grading a short-answer factual question. Respond with exactly one "
    "word: CORRECT if the candidate answer matches the reference answer in "
    "meaning, or INCORRECT if it does not."
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().translate(_PUNCT_TABLE)).strip()


def _string_containment_match(raw_response: str, ground_truth: str) -> bool:
    return _normalize(ground_truth) in _normalize(raw_response)


def _log_audit_case(question_id, raw_response, ground_truth, judge_verdict) -> None:
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "question_id": question_id,
        "raw_response": raw_response,
        "ground_truth": ground_truth,
        "judge_verdict": judge_verdict,
    }
    with _audit_lock:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def grade(
    raw_response: str,
    ground_truth: str,
    question_id: Optional[str] = None,
    judge_config: Optional[dict] = None,
    cost_tracker: Optional[CostTracker] = None,
) -> dict:
    """judge_config, if provided, must have: model_id, provider, price_per_m_tokens.
    If omitted, ambiguous cases are left ungraded (correct=False) with the
    audit record still written — callers running the pilot must pass a
    judge_config to exercise the fallback path."""
    if not raw_response:
        return {"correct": False, "extracted_answer": None, "raw_response": raw_response}

    if _string_containment_match(raw_response, ground_truth):
        return {"correct": True, "extracted_answer": raw_response.strip(), "raw_response": raw_response}

    judge_verdict = None
    correct = False
    if judge_config is not None:
        judge_prompt = f"Reference answer: {ground_truth}\nCandidate answer: {raw_response}"
        result = get_response(
            model_id=judge_config["model_id"],
            provider=judge_config["provider"],
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=judge_prompt,
            temperature=0.0,
            max_tokens=8,
            price_per_m_tokens=judge_config.get("price_per_m_tokens"),
            dataset="simpleqa_judge",
            cost_tracker=cost_tracker,
        )
        judge_verdict = (result.get("text") or "").strip().upper()
        correct = judge_verdict.startswith("CORRECT")

    _log_audit_case(question_id, raw_response, ground_truth, judge_verdict)

    return {"correct": correct, "extracted_answer": raw_response.strip(), "raw_response": raw_response}
