"""GPQA-Diamond grader: exact letter match (A/B/C/D).

Assumes the prompt enforces a constrained final-answer format, e.g.
"Final Answer: <letter>" on its own line at the end of the response.
Never discards the raw response, even when extraction fails.
"""
from __future__ import annotations

import re

FINAL_ANSWER_RE = re.compile(r"final answer\s*:?\s*\(?([A-D])\)?", re.IGNORECASE)
BARE_LETTER_RE = re.compile(r"\b([A-D])\b")


def extract_answer(raw_response: str) -> str | None:
    if not raw_response:
        return None
    match = FINAL_ANSWER_RE.search(raw_response)
    if match:
        return match.group(1).upper()
    # Fallback: last standalone A-D letter in the response.
    matches = BARE_LETTER_RE.findall(raw_response)
    return matches[-1].upper() if matches else None


def grade(raw_response: str, ground_truth: str) -> dict:
    extracted = extract_answer(raw_response)
    correct = extracted is not None and extracted == ground_truth.strip().upper()
    return {
        "correct": correct,
        "extracted_answer": extracted,
        "raw_response": raw_response,
    }
