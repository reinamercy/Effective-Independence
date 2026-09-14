"""Builds the 12 config definitions (4 axes x 3 configs) from
config/project_spec.yaml and config/models.yaml.

Each config dict has: config_id, axis, model_id, provider, system_prompt,
temperature, price_per_m_tokens, seed. Consumed by config_runner.generate().
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SPEC_PATH = REPO_ROOT / "config" / "project_spec.yaml"
MODELS_PATH = REPO_ROOT / "config" / "models.yaml"

PERSONA_PROMPTS = {
    "careful analyst": (
        "You are a careful analyst. Work through the problem methodically, "
        "double-checking each step before committing to a final answer."
    ),
    "confident expert": (
        "You are a confident domain expert. Give a direct, decisive answer "
        "based on your expertise."
    ),
    "skeptical reviewer": (
        "You are a skeptical reviewer. Question your own reasoning and look "
        "for mistakes before finalizing your answer."
    ),
}

NEUTRAL_PROMPT = "Answer the question accurately."


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_configs() -> list[dict]:
    spec = _load_yaml(PROJECT_SPEC_PATH)
    models = _load_yaml(MODELS_PATH)

    configs = []

    # Axis 1: prompt_diverse — fixed mid-tier model, 3 personas, temperature 0.7
    mid_tier = models["mid_tier_model"]
    for i, persona in enumerate(spec["configurations"]["axes"][0]["personas"]):
        configs.append(
            {
                "config_id": f"prompt_diverse_{i}",
                "axis": "prompt_diverse",
                "provider": mid_tier["provider"],
                "model_id": mid_tier["model_id"],
                "price_per_m_tokens": mid_tier["price_per_m_tokens"],
                "system_prompt": PERSONA_PROMPTS[persona],
                "temperature": 0.7,
                "seed": None,
            }
        )

    # Axis 2: temperature_diverse — fixed mid-tier model, neutral prompt, temps {0.3, 0.7, 1.0}
    for i, temp in enumerate([0.3, 0.7, 1.0]):
        configs.append(
            {
                "config_id": f"temperature_diverse_{i}",
                "axis": "temperature_diverse",
                "provider": mid_tier["provider"],
                "model_id": mid_tier["model_id"],
                "price_per_m_tokens": mid_tier["price_per_m_tokens"],
                "system_prompt": NEUTRAL_PROMPT,
                "temperature": temp,
                "seed": None,
            }
        )

    # Axis 3: size_diverse — one family, 3 sizes, neutral prompt, temperature 0.7
    size_family = models["size_family"]
    for tier in ["small", "medium", "flagship"]:
        tier_cfg = size_family[tier]
        configs.append(
            {
                "config_id": f"size_diverse_{tier}",
                "axis": "size_diverse",
                "provider": size_family["provider"],
                "model_id": tier_cfg["model_id"],
                "price_per_m_tokens": tier_cfg["price_per_m_tokens"],
                "system_prompt": NEUTRAL_PROMPT,
                "temperature": 0.7,
                "seed": None,
            }
        )

    # Axis 4: family_diverse — 3 labs, neutral prompt, temperature 0.7
    for i, model_cfg in enumerate(models["family_diverse_models"]):
        configs.append(
            {
                "config_id": f"family_diverse_{i}",
                "axis": "family_diverse",
                "provider": model_cfg["provider"],
                "model_id": model_cfg["model_id"],
                "price_per_m_tokens": model_cfg["price_per_m_tokens"],
                "system_prompt": NEUTRAL_PROMPT,
                "temperature": 0.7,
                "seed": None,
            }
        )

    assert len(configs) == 12, f"expected 12 configs, built {len(configs)}"
    return configs
