"""Model evaluation suite: does a candidate model meet the app's requirements?

``uv run llm-eval`` scores a model per ROLE (llm / recall / vlm) against the
exact contracts the application's call sites impose — intent routing accuracy,
chat rule compliance, recall factuality over a known journal, VLM decoration
coverage and leakage, latency budgets — and persists a JSON verdict per run.

Unlike ``llm-harness --bench`` (print-only, latency + eyeball) this suite is
the pass/fail gate for swapping any of the three ``OLLAMA_*_MODEL`` env keys.
"""
