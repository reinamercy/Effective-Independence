# Effective Independence

Quantifying the Reliability Ceiling of Same-Base-Model Ensembles in LLM Agent Pipelines.

Debate, self-consistency, and LLM-as-judge pipelines assume ensemble diversity they don't
actually have when built from one base model. This project measures the effective number of
independent voters ($N_\text{eff}$) across four diversity axes (prompt, temperature, size,
model family) and shows where each axis saturates.

API-access only: no local model weights, no fine-tuning, no activation access, no GPU
assumptions.

## Status

Phase 0 (scaffold) complete. See `config/project_spec.yaml` for the frozen project spec and
`logs/decisions.md` for the running decision log.

Model list (`config/models.yaml`) and paper format are unconfirmed — Phase 1 will not begin
until both are approved by the project owner.

## Setup

```
cp .env.example .env   # fill in real API keys, never commit .env
pip install -r requirements.txt
```

## Structure

See `config/project_spec.yaml` for the full frozen spec (datasets, configs, grading rules,
phase gates, engineering rules).
