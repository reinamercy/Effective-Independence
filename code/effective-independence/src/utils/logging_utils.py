"""Structured call logging shared by api_client.py and config_runner.py.

Every API call (cache hit or miss) gets one JSON line appended to
logs/api_calls.jsonl. Failures are logged the same way with an "error" field
set, never raised past this layer.
"""
from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CALL_LOG_PATH = REPO_ROOT / "logs" / "api_calls.jsonl"

_lock = threading.Lock()


def log_call(record: dict) -> None:
    """Append one structured call record to logs/api_calls.jsonl.

    Expected keys: timestamp, provider, requested_model_id,
    returned_model_version, cache_hit, prompt_tokens, completion_tokens,
    cost_usd, error (None on success).
    """
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    CALL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        with open(CALL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def log_error(message: str, **context) -> None:
    """Fail-loud helper for data-quality issues (per project engineering rules).

    Prints to stderr immediately so it isn't lost in batch output, in addition
    to being captured via log_call for anything call-shaped.
    """
    print(f"[ERROR] {message} | context={context}", file=sys.stderr)
