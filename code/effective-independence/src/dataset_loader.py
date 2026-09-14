"""Loads and normalizes the 3 benchmark datasets into a common question schema:
{question_id, dataset, prompt, ground_truth}.

Raw downloads are cached to data/raw/ and normalized questions to
data/processed/<dataset>.jsonl so repeated runs never re-download or
re-process. Loader functions are idempotent: if the processed file already
exists, it's read directly and no network call is made.
"""
from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

from src.utils import ssl_bootstrap  # noqa: F401 — must run before any requests/httpx call

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

GPQA_ANSWER_LETTERS = ["A", "B", "C", "D"]

GPQA_PROMPT_TEMPLATE = """Answer the following multiple-choice question. Think step by step, \
then on the final line write exactly: "Final Answer: <letter>" where <letter> is one of A, B, C, or D.

Question: {question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}
"""

MATH_PROMPT_TEMPLATE = """Solve the following math problem. Think step by step, then give your \
final answer inside \\boxed{{}} on the last line.

Problem: {problem}
"""

SIMPLEQA_PROMPT_TEMPLATE = """Answer the following question as concisely as possible.

Question: {question}
"""


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_gpqa_diamond(force_refresh: bool = False, target_n: int = 8) -> list[dict]:
    processed_path = DATA_PROCESSED / "gpqa_diamond.jsonl"
    if processed_path.exists() and not force_refresh:
        return _read_jsonl(processed_path)

    # NOTE: datasets.load_dataset() raises a spurious DatasetNotFoundError /
    # "gated dataset" error on this repo even with a token that has confirmed
    # gated-repo access (verified via hf_hub_download succeeding with the same
    # token) -- a library quirk on this specific gated repo, not an access
    # problem. Downloading the raw CSV directly via huggingface_hub sidesteps it.
    from huggingface_hub import hf_hub_download  # lazy import

    hf_token = os.environ.get("HF_TOKEN")
    csv_path = hf_hub_download(
        repo_id="Idavidrein/gpqa", repo_type="dataset", filename="gpqa_diamond.csv", token=hf_token
    )
    (DATA_RAW / "gpqa_diamond.csv").parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy(csv_path, DATA_RAW / "gpqa_diamond.csv")

    with open(csv_path, "r", encoding="utf-8") as f:
        items = list(csv.DictReader(f))

    rng = random.Random(0)
    rng.shuffle(items)
    items = items[:target_n]

    rows = []
    for i, item in enumerate(items):
        choices = [
            item["Correct Answer"],
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [choices[j] for j in order]
        correct_letter = GPQA_ANSWER_LETTERS[order.index(0)]
        prompt = GPQA_PROMPT_TEMPLATE.format(
            question=item["Question"],
            choice_a=shuffled[0],
            choice_b=shuffled[1],
            choice_c=shuffled[2],
            choice_d=shuffled[3],
        )
        rows.append(
            {
                "question_id": f"gpqa_diamond_{i}",
                "dataset": "gpqa_diamond",
                "prompt": prompt,
                "ground_truth": correct_letter,
            }
        )
    _write_jsonl(processed_path, rows)
    return rows


def load_math(force_refresh: bool = False, levels: tuple[int, ...] = (4, 5), target_n: int = 11) -> list[dict]:
    processed_path = DATA_PROCESSED / "math.jsonl"
    if processed_path.exists() and not force_refresh:
        return _read_jsonl(processed_path)

    from datasets import load_dataset  # lazy import

    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
    ds.save_to_disk(str(DATA_RAW / "math_hf"))

    level_names = {f"Level {lvl}" for lvl in levels}
    filtered = [item for item in ds if item.get("level") in level_names]

    rng = random.Random(0)
    rng.shuffle(filtered)
    filtered = filtered[:target_n]

    rows = []
    for i, item in enumerate(filtered):
        rows.append(
            {
                "question_id": f"math_{i}",
                "dataset": "math",
                "prompt": MATH_PROMPT_TEMPLATE.format(problem=item["problem"]),
                "ground_truth": item["solution"],
            }
        )
    _write_jsonl(processed_path, rows)
    return rows


def load_simpleqa(force_refresh: bool = False, target_n: int = 11) -> list[dict]:
    processed_path = DATA_PROCESSED / "simpleqa.jsonl"
    if processed_path.exists() and not force_refresh:
        return _read_jsonl(processed_path)

    import requests  # lazy import

    raw_path = DATA_RAW / "simpleqa.csv"
    if not raw_path.exists() or force_refresh:
        url = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(resp.content)

    with open(raw_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    rng = random.Random(0)
    rng.shuffle(reader)
    reader = reader[:target_n]

    rows = []
    for i, item in enumerate(reader):
        rows.append(
            {
                "question_id": f"simpleqa_{i}",
                "dataset": "simpleqa",
                "prompt": SIMPLEQA_PROMPT_TEMPLATE.format(question=item["problem"]),
                "ground_truth": item["answer"],
            }
        )
    _write_jsonl(processed_path, rows)
    return rows


def load_all(force_refresh: bool = False) -> list[dict]:
    return load_gpqa_diamond(force_refresh) + load_math(force_refresh) + load_simpleqa(force_refresh)


def sample_pilot_set(n_per_dataset: int = 3, seed: int = 0) -> list[dict]:
    """~9 questions total, drawn evenly across the 3 datasets for the Phase 1 pilot.

    Deliberately tiny (not the ~50 in the original spec, and smaller than the
    already-shrunk ~21): the *real observed* free-tier quota for Gemini
    models is 20 requests/day/model (RPM=5) -- far below the 250-1500/day
    public docs suggested. 8 of the 12 configs share one model's daily
    budget, so even a 21-question pilot would take ~9 days; a 9-question
    pilot takes ~4 days at 20/day (8 configs x 9 questions = 72 calls). See
    logs/decisions.md.
    """
    rng = random.Random(seed)
    questions = load_all()
    by_dataset: dict[str, list[dict]] = {}
    for q in questions:
        by_dataset.setdefault(q["dataset"], []).append(q)
    pilot = []
    for dataset, items in by_dataset.items():
        pilot.extend(rng.sample(items, min(n_per_dataset, len(items))))
    return pilot
