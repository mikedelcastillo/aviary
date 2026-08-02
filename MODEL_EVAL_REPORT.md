# Aviary Model Evaluation Report

Goal: find the **lowest-parameter-count** ollama models that fully meet the app's
LLM/VLM requirements, per role. Maintained live during the 2026-08-02 model search.

## Model roles (from `.env` + call sites)

| Role | Env key | Current model | Serves |
|------|---------|---------------|--------|
| LLM | `OLLAMA_LLM_MODEL` | gemma3:4b | intent classification, chat replies, streaming Telegram replies |
| VLM | `OLLAMA_VLM_MODEL` | qwen2.5vl:7b | observation decoration, scene descriptions, camera naming, memory frames |
| Recall | `OLLAMA_RECALL_MODEL` | gemma3:12b | complex memory recall (day/week summaries, per-bird tallies) |

## Serving topology

Olla proxy `:40114` (strict model→endpoint routing) over:
- `pascal-gpu0` = 192.168.1.62:11434, GTX 1050 Ti 4GB (shares box with YOLO)
- `pascal-gpu1` = 192.168.1.62:11435, GTX 1060 6GB (VLM's usual home; `--no-mmproj-offload` = vision encoder on CPU)
- `rig-1-2` = 192.168.1.2:11434, RTX 5060 8GB (big rig, on intermittently)
- `node-1-168` = offline

Production usage since 2026-07-04 (`data/server/llm_usage.json`):
qwen2.5vl:7b 22,463 calls (~9.4s busy/call avg), gemma3:4b 5,973 calls (~2.8s), gemma3:12b 5 calls.

## Eval suite

`uv run llm-eval --role llm|recall|vlm [--model NAME] [--base-url URL]` — exits 0
only when every task meets its requirement. Results persist to
`data/server/model_evals/<role>/<model>.json` (last 10 runs kept, with git rev).

| Role | Task | What it scores | Requirement |
|------|------|----------------|-------------|
| llm | intent | action accuracy + argument fidelity + follow-up inheritance over 60 labeled messages (incl. adversarial boundary cases mined from the router prompt's own rules) | ≥90%, p90 ≤12s |
| llm | chat | persona contract (1 sentence <35w, no species words, no bird-vocative, no markdown) + grounding against fabricated state blocks (visible birds, paused, night, degraded cameras) + care-question vet deflection | ≥80%, p90 ≤25s |
| llm | sleep | morning one-liner from sleep facts: non-fallback, no invented numbers, disturbance/fright mentioned only when real | ≥67%, p90 ≤30s |
| recall | recall_qa | factual Q&A over notes + COUNTS blocks with computable answers: yes/no polarity vs explicit VERDICT, flock ranking names the right bird, zero-count scans, health→vet, anti-confabulated digits, no meta-words | ≥75%, p90 ≤45s |
| recall | summary | bullet contract (2-4 "• " lines <16w), subject-first, no species words | ≥75%, p90 ≤45s |
| vlm | analyze | observation decoration replayed from real journal frames (`annotate_for_vlm` → `analyze_frame`, byte-identical to backfill): content rate, label coverage, overlay/species leakage, evasion (hidden/unknown) rate, activity agreement vs stored 7b silver + hand-curated golden set | ≥70%, p90 ≤90s |
| vlm | scene | `/find`-style captions on collect sidecar frames with detection grounding: names the bird, concrete activity word, ≤32w, never "no bird" when detected | ≥75%, p90 ≤60s |
| vlm | camera_names | view naming across distinct cameras: non-empty after cleaning, ≤2 words, stability across frames, uniqueness across cameras | ≥70%, p90 ≤60s |

## Candidates tried

_(one entry per model: why tried, scores, verdict, kept/deleted)_

## Current recommendation

_(pending)_
