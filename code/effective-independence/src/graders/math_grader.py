"""MATH (levels 4-5) grader: symbolic/numeric equivalence via math_verify.

Extracts the answer from a \\boxed{...} final-answer convention (standard
for the MATH dataset) and checks equivalence with math_verify rather than a
hand-rolled parser, per project engineering rules.
"""
from __future__ import annotations

import re

BOXED_START_RE = re.compile(r"\\boxed\{")


def extract_boxed(raw_response: str) -> str | None:
    """Find the LAST \\boxed{...} and return its contents, matching braces by
    depth rather than a non-greedy regex -- MATH answers routinely nest braces
    (e.g. \\boxed{\\frac{6}{7}}), and a naive `\\boxed\\{(.+?)\\}` stops at the
    first inner `}` and truncates the answer (confirmed on live pilot output:
    \\boxed{\\frac{6}{7}} was extracted as just '\\frac{6')."""
    if not raw_response:
        return None
    starts = list(BOXED_START_RE.finditer(raw_response))
    if not starts:
        return None
    start = starts[-1].end()  # last \boxed{ in the response
    depth = 1
    i = start
    while i < len(raw_response) and depth > 0:
        if raw_response[i] == "{":
            depth += 1
        elif raw_response[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None  # unbalanced braces -- truncated response, no usable answer
    return raw_response[start : i - 1].strip()


def grade(raw_response: str, ground_truth: str) -> dict:
    extracted = extract_boxed(raw_response)
    correct = False
    if extracted is not None:
        from math_verify import parse, verify  # lazy import — only needed when grading

        # parsing_timeout / timeout_seconds MUST be 0 (or None) on Windows: math_verify's
        # default timeout wrapper uses multiprocessing with a locally-defined closure, which
        # can't be pickled by Windows' spawn-based process start method (confirmed via
        # raise_on_error=True: AttributeError "Can't pickle local object
        # timeout.<locals>.decorator.<locals>.wrapper.<locals>.run_func"). With the default
        # raise_on_error=False, this failure was being silently swallowed and parse()/verify()
        # always returned an empty/False result -- every MATH grade in this project was wrong
        # until this fix (confirmed live: exact-match \boxed{} answers were graded incorrect).
        # Disabling the timeout falls back to math_verify's synchronous (non-multiprocess)
        # path, per its own source (utils.py: `if timeout_seconds is None or timeout_seconds
        # <= 0: return no_timeout_decorator`). This does mean a pathological input could hang
        # grading with no timeout protection -- acceptable at this project's scale (dozens,
        # not millions, of MATH questions).
        try:
            correct = bool(
                verify(
                    parse(ground_truth, parsing_timeout=0),
                    parse(extracted, parsing_timeout=0),
                    timeout_seconds=0,
                )
            )
        except Exception:
            correct = False
    return {
        "correct": correct,
        "extracted_answer": extracted,
        "raw_response": raw_response,
    }
