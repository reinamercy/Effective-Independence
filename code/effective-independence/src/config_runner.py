"""Runs a list of configs x a list of questions through api_client.get_response.

Resumable by construction: api_client checks the disk cache before every
call, so re-invoking generate() with the same configs/questions only issues
network calls for (question, config) cells not already cached — nothing is
redone or duplicated.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

from src.api_client import get_response
from src.utils.cost_tracker import CostTracker


def _run_one(config: dict, question: dict, cost_tracker: Optional[CostTracker]) -> dict:
    system_prompt = config.get("system_prompt", "")
    user_prompt = question["prompt"]

    result = get_response(
        model_id=config["model_id"],
        provider=config["provider"],
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=config["temperature"],
        seed=config.get("seed"),
        max_tokens=config.get("max_tokens", 4096),
        price_per_m_tokens=config.get("price_per_m_tokens"),
        dataset=question["dataset"],
        cost_tracker=cost_tracker,
    )

    return {
        "question_id": question["question_id"],
        "dataset": question["dataset"],
        "config_id": config["config_id"],
        "axis": config.get("axis"),
        "model_id": config["model_id"],
        "model_version": result["model_version"],
        "raw_response": result["text"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "cost_usd": result["cost_usd"],
        "cached": result["cached"],
        "error": result["error"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def generate(
    configs: list[dict],
    questions: list[dict],
    cost_tracker: Optional[CostTracker] = None,
    max_workers: int = 8,
) -> list[dict]:
    """Run every (question, config) pair. Returns one result dict per pair,
    in no particular order. A failed call yields a row with error != None and
    raw_response = None rather than raising — never crashes the batch on a
    single failure (per project engineering rules)."""
    tasks = [(config, question) for config in configs for question in questions]
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one, config, question, cost_tracker): (config["config_id"], question["question_id"])
            for config, question in tasks
        }
        for future in as_completed(futures):
            config_id, question_id = futures[future]
            try:
                results.append(future.result())
            except Exception as e:  # noqa: BLE001 — a bug here is infra, not data quality; log and skip
                results.append(
                    {
                        "question_id": question_id,
                        "dataset": None,
                        "config_id": config_id,
                        "axis": None,
                        "model_id": None,
                        "model_version": None,
                        "raw_response": None,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "cost_usd": 0.0,
                        "cached": False,
                        "error": f"unhandled exception in config_runner: {e}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
    return results
