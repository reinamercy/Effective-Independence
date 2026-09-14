"""Cached, rate-limited, retry-wrapped API client.

Public entry point: get_response(...). Every (model_id, system_prompt,
user_prompt, temperature, seed) tuple is hashed to a cache key; the cache is
checked before any network call and written after every successful call, so
re-running a script never re-calls the API for a tuple already on disk.

Provider SDKs (anthropic / openai / google-genai) are imported lazily inside
each _call_<provider> function so that importing this module never requires
all three to be installed, and so no API calls can happen as a side effect
of import.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import yaml

from src.utils import ssl_bootstrap  # noqa: F401 — must run before any requests/httpx call
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from src.utils.cost_tracker import CostTracker, estimate_cost_usd
from src.utils.logging_utils import log_call, log_error

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "cache" / "responses"
MODELS_CONFIG_PATH = REPO_ROOT / "config" / "models.yaml"

DEFAULT_MAX_TOKENS = 1024
MAX_RETRY_ATTEMPTS = 5
HTTP_TIMEOUT_SECONDS = 90  # per-request HTTP timeout, all providers. Without this, a stalled
# network call (confirmed live: one pilot run hung for 28+ minutes at only 25s of CPU time --
# i.e. idle, not computing -- almost certainly a hung request with no client-side timeout) can
# block the whole batch indefinitely, since tenacity's retry logic never gets a chance to fire
# if the underlying call never returns in the first place.


class APICallError(RuntimeError):
    """Raised internally after retries are exhausted; caught inside get_response
    so a single failed call never propagates out of a batch."""


class RateLimitedError(Exception):
    """Raised by _call_* wrappers on 429 / transient errors, to trigger retry."""


class DailyQuotaExhaustedError(Exception):
    """Raised on a confirmed per-day (not per-minute) quota error. Deliberately
    NOT retried -- retrying within the same day cannot succeed, so retrying
    would just burn ~2-4 minutes of exponential backoff per call for nothing.
    Confirmed live 2026-08-12: Gemini's error message distinguishes
    'GenerateRequestsPerMinutePerProjectPerModel-FreeTier' (retry-worthy) from
    'GenerateRequestsPerDayPerProjectPerModel-FreeTier' (not retry-worthy)."""


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_models_config_cache: Optional[dict] = None


def _load_models_config() -> dict:
    global _models_config_cache
    if _models_config_cache is None:
        with open(MODELS_CONFIG_PATH, "r", encoding="utf-8") as f:
            _models_config_cache = yaml.safe_load(f)
    return _models_config_cache


def _provider_config(provider: str) -> dict:
    cfg = _load_models_config()
    providers = cfg.get("providers", {})
    if provider not in providers:
        raise ValueError(f"Unknown provider '{provider}' — not in config/models.yaml providers")
    return providers[provider]


# ---------------------------------------------------------------------------
# Rate limiting — simple per-provider sliding-window limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.rpm = requests_per_minute
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            while self._timestamps and now - self._timestamps[0] > 60:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.rpm:
                sleep_for = 60 - (now - self._timestamps[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] > 60:
                    self._timestamps.popleft()
            self._timestamps.append(time.monotonic())


_rate_limiters: dict[tuple[str, str], RateLimiter] = {}
_rate_limiters_lock = threading.Lock()


def _get_rate_limiter(provider: str, model_id: str) -> RateLimiter:
    """Keyed by (provider, model_id), not just provider: Google's real quota is
    per-model (confirmed live -- gemini-3.5-flash hit RESOURCE_EXHAUSTED with
    quotaId GenerateRequestsPerMinutePerProjectPerModel-FreeTier while
    gemini-3.5-flash-lite kept working), so a provider-wide limiter would
    needlessly throttle models that still have quota available."""
    key = (provider, model_id)
    with _rate_limiters_lock:
        if key not in _rate_limiters:
            rpm = _provider_config(provider).get("rate_limit_rpm", 30)
            _rate_limiters[key] = RateLimiter(rpm)
        return _rate_limiters[key]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache_key(model_id: str, system_prompt: str, user_prompt: str, temperature: float, seed, max_tokens: int) -> str:
    payload = json.dumps(
        {
            "model_id": model_id,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _read_cache(key: str) -> Optional[dict]:
    path = _cache_path(key)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _write_cache(key: str, record: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(key).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Provider call wrappers — each returns (text, model_version, prompt_tokens, completion_tokens)
# ---------------------------------------------------------------------------


def _call_anthropic(model_id, system_prompt, user_prompt, temperature, max_tokens, seed):
    import anthropic

    client = anthropic.Anthropic(
        api_key=os.environ[_provider_config("anthropic")["api_key_env"]], timeout=HTTP_TIMEOUT_SECONDS
    )
    try:
        resp = client.messages.create(
            model=model_id,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except anthropic.RateLimitError as e:
        raise RateLimitedError(str(e)) from e
    text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
    return text, resp.model, resp.usage.input_tokens, resp.usage.output_tokens


def _call_openai(model_id, system_prompt, user_prompt, temperature, max_tokens, seed):
    import openai

    client = openai.OpenAI(
        api_key=os.environ[_provider_config("openai")["api_key_env"]], timeout=HTTP_TIMEOUT_SECONDS
    )
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
    except openai.RateLimitError as e:
        raise RateLimitedError(str(e)) from e
    text = resp.choices[0].message.content or ""
    return text, resp.model, resp.usage.prompt_tokens, resp.usage.completion_tokens


def _call_openrouter(model_id, system_prompt, user_prompt, temperature, max_tokens, seed):
    import openai

    client = openai.OpenAI(
        api_key=os.environ[_provider_config("openrouter")["api_key_env"]],
        base_url="https://openrouter.ai/api/v1",
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except openai.RateLimitError as e:
        raise RateLimitedError(str(e)) from e
    text = resp.choices[0].message.content or ""
    prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
    completion_tokens = resp.usage.completion_tokens if resp.usage else 0
    return text, resp.model, prompt_tokens, completion_tokens


def _call_google(model_id, system_prompt, user_prompt, temperature, max_tokens, seed):
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=os.environ[_provider_config("google")["api_key_env"]],
        http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_SECONDS * 1000),  # genai wants milliseconds
    )
    try:
        resp = client.models.generate_content(
            model=model_id,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
                seed=seed,
            ),
        )
    except Exception as e:  # google-genai raises a generic ClientError on 429
        if "RESOURCE_EXHAUSTED" in str(e) and "PerDay" in str(e):
            raise DailyQuotaExhaustedError(str(e)) from e
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            raise RateLimitedError(str(e)) from e
        raise
    text = resp.text or ""
    usage = resp.usage_metadata
    model_version = getattr(resp, "model_version", model_id)
    return text, model_version, usage.prompt_token_count, usage.candidates_token_count


_PROVIDER_DISPATCH = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "google": _call_google,
    "openrouter": _call_openrouter,
}


@retry(
    retry=retry_if_exception_type(RateLimitedError),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
    reraise=True,
)
def _call_with_retry(provider, model_id, system_prompt, user_prompt, temperature, max_tokens, seed):
    _get_rate_limiter(provider, model_id).acquire()
    return _PROVIDER_DISPATCH[provider](model_id, system_prompt, user_prompt, temperature, max_tokens, seed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_response(
    model_id: str,
    provider: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    seed: Optional[int] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    price_per_m_tokens: Optional[dict] = None,
    dataset: Optional[str] = None,
    cost_tracker: Optional[CostTracker] = None,
) -> dict:
    """Return a dict with keys: text, model_version, prompt_tokens,
    completion_tokens, cost_usd, cached, error.

    On failure (after retries exhausted), returns error != None and text = None
    rather than raising — callers (config_runner) must check `error` and mark
    that (question, config) cell as null, never as an incorrect answer.
    """
    key = _cache_key(model_id, system_prompt, user_prompt, temperature, seed, max_tokens)
    cached = _read_cache(key)
    if cached is not None:
        result = {**cached, "cached": True}
        log_call(
            {
                "provider": provider,
                "requested_model_id": model_id,
                "returned_model_version": cached.get("model_version"),
                "cache_hit": True,
                "prompt_tokens": cached.get("prompt_tokens"),
                "completion_tokens": cached.get("completion_tokens"),
                "cost_usd": 0.0,
                "dataset": dataset,
                "error": None,
            }
        )
        return result

    try:
        text, model_version, prompt_tokens, completion_tokens = _call_with_retry(
            provider, model_id, system_prompt, user_prompt, temperature, max_tokens, seed
        )
    except Exception as e:  # noqa: BLE001 — deliberate: never propagate a single call failure
        log_error(
            f"API call failed after retries: {e}",
            provider=provider,
            model_id=model_id,
            dataset=dataset,
        )
        log_call(
            {
                "provider": provider,
                "requested_model_id": model_id,
                "returned_model_version": None,
                "cache_hit": False,
                "prompt_tokens": None,
                "completion_tokens": None,
                "cost_usd": 0.0,
                "dataset": dataset,
                "error": str(e),
            }
        )
        return {
            "text": None,
            "model_version": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "cost_usd": 0.0,
            "cached": False,
            "error": str(e),
        }

    cost_usd = 0.0
    if price_per_m_tokens is not None:
        # Some providers return None for usage fields on edge-case responses (e.g. a
        # safety-filtered or empty completion) even though the call itself didn't raise --
        # confirmed live: completion_tokens=None crashed estimate_cost_usd with an unhandled
        # TypeError, killing the whole batch script mid-run. Treat missing counts as 0 rather
        # than letting one odd response crash a multi-hour run (per engineering rules: fail
        # quiet on infrastructure, not on a single response's missing metadata).
        cost_usd = estimate_cost_usd(prompt_tokens or 0, completion_tokens or 0, price_per_m_tokens)
        if cost_tracker is not None:
            cost_tracker.add(model_id, dataset or "unknown", cost_usd)
            cost_tracker.check_ceiling()

    record = {
        "text": text,
        "model_version": model_version,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": cost_usd,
        "error": None,
    }
    _write_cache(key, record)
    log_call(
        {
            "provider": provider,
            "requested_model_id": model_id,
            "returned_model_version": model_version,
            "cache_hit": False,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "dataset": dataset,
            "error": None,
        }
    )
    return {**record, "cached": False}
