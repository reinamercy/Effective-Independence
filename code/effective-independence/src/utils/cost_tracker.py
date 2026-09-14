"""Cumulative cost tracking and budget enforcement.

Persists a running ledger to cache/cost_ledger.json so cost survives across
resumed runs (the ledger is rebuilt from logs/api_calls.jsonl on demand via
`rebuild_from_call_log`, but normal operation just accumulates in place).
"""
from __future__ import annotations

import json
import threading
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "cache" / "cost_ledger.json"

SINGLE_RUN_FLAG_THRESHOLD_USD = 20
FULL_GENERATION_PAUSE_THRESHOLD_USD = 150
COST_CEILING_USD = 300


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int, price_per_m_tokens: dict) -> float:
    """price_per_m_tokens: {"input": float, "output": float} — USD per 1M tokens."""
    return (
        prompt_tokens * price_per_m_tokens["input"]
        + completion_tokens * price_per_m_tokens["output"]
    ) / 1_000_000


class CostTracker:
    """Tracks cumulative spend broken down by model and by dataset.

    Not thread-safe across processes — intended for use within a single
    config_runner.py invocation, persisting to disk after each update so a
    resumed run picks up the correct running total.
    """

    def __init__(self, ledger_path: Path = LEDGER_PATH):
        self.ledger_path = ledger_path
        self.by_model: dict[str, float] = defaultdict(float)
        self.by_dataset: dict[str, float] = defaultdict(float)
        self.total_usd: float = 0.0
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self.ledger_path.exists():
            data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            self.by_model = defaultdict(float, data.get("by_model", {}))
            self.by_dataset = defaultdict(float, data.get("by_dataset", {}))
            self.total_usd = data.get("total_usd", 0.0)

    def _save(self) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text(
            json.dumps(
                {
                    "by_model": dict(self.by_model),
                    "by_dataset": dict(self.by_dataset),
                    "total_usd": self.total_usd,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def add(self, model_id: str, dataset: str, cost_usd: float) -> None:
        with self._lock:
            self.by_model[model_id] += cost_usd
            self.by_dataset[dataset] += cost_usd
            self.total_usd += cost_usd
            self._save()

    def check_ceiling(self) -> None:
        if self.total_usd > COST_CEILING_USD:
            raise RuntimeError(
                f"Cumulative project spend ${self.total_usd:.2f} has exceeded the "
                f"${COST_CEILING_USD} cost ceiling. Halting — requires explicit sign-off "
                f"to continue (see config/project_spec.yaml cost_ceiling_usd)."
            )

    def print_dashboard(self) -> None:
        print("\n=== Cumulative Cost Dashboard ===")
        print(f"Total spend: ${self.total_usd:.4f} / ${COST_CEILING_USD:.2f} ceiling")
        print("By model:")
        for model_id, cost in sorted(self.by_model.items(), key=lambda kv: -kv[1]):
            print(f"  {model_id:30s} ${cost:.4f}")
        print("By dataset:")
        for dataset, cost in sorted(self.by_dataset.items(), key=lambda kv: -kv[1]):
            print(f"  {dataset:30s} ${cost:.4f}")
        print("==================================\n")
