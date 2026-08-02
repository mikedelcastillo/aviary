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

## Search plan

Per-role, smallest-first; a candidate wins its role by PASSING every requirement
at a lower parameter count than the incumbent (quality ties break toward fewer B,
then lower latency). Incumbents are baselined first so requirements reflect what
the app demonstrably needs today.

Candidate queue (B = params; ✔ = already on disk):

- **LLM** (needs: intent JSON schema, chat persona, instruct behavior):
  gemma3:1b (1B), llama3.2:1b (1B), llama3.2:3b (3B), granite3.3:2b (2B),
  phi4-mini (3.8B), gemma3:4b ✔ incumbent (4B)
- **VLM** (needs: vision + format-schema JSON + grounded captions):
  moondream (1.8B), granite3.2-vision (2B), qwen2.5vl:3b ✔ (3B),
  gemma3:4b ✔ (4B — multimodal; could serve BOTH llm+vlm roles on one card),
  llava-phi3 (3.8B), minicpm-v (8B), qwen2.5vl:7b ✔ incumbent (7B)
- **Recall** (needs: numeric fidelity over COUNTS blocks):
  llama3.2:3b (3B), gemma3:4b ✔ (4B), qwen2.5:7b (7B), llama3.1:8b (8B),
  gemma3:12b ✔ incumbent (12B)

## Incumbent baselines (2026-08-02, aviary stopped, direct Pascal endpoints)

### gemma3:4b — LLM role — effectively PASS
intent **98.3%** (58/59, p50 1.8s / p90 2.1s) — only miss: "where is everyone?" → find instead of activity.
chat **90%** (9/10, p50 3.1s, 0 fallbacks) — reproducible flaw: **invents bird sightings while cameras are PAUSED**.
sleep 2/3 at first run was a suite scoring bug (12-hour clock normalization, fixed in 94e8c69); with the fix the reply was compliant.

### gemma3:12b — Recall role — PASS
recall_qa **100%** (6/6, p50 21.1s / p90 26.5s), summary **100%** (2/2, p50 14.6s).
Slow but well inside the 45s budget even running mostly on CPU (8.1GB model, 4GB card).

### qwen2.5vl:7b — VLM role
_(running)_

## Candidates tried

### gemma3:1b (1B) — LLM role — ❌ REJECTED
Why tried: smallest gemma3; shares family with incumbent so prompt style transfers.
Scores: intent **66.1%** (39/59) ✗ · chat 80.0% ✓ (borderline) · sleep 100% ✓ · p50 ~1.7-2.9s.
Rejection: intent routing is unsafe — "stop the cams" → stop_find and "turn the cameras off" → stop_find mean a **privacy request would leave cameras recording**; "play" → pause inverts resume. Disqualifying regardless of speed.
Disk: kept temporarily for cross-checks, will delete at cleanup.

### llama3.2:1b (1B) — LLM role — ❌ REJECTED
Why tried: 1B tier, different family in case gemma-style failures were family-specific.
Scores: intent **69.5%** ✗ · chat **30%** ✗ · p50 1.7-3.6s.
Rejection: worst-in-class safety — "hello there"/"good morning" → **home** (physically slews the PTZ cameras on a greeting); chat invents the wrong visible birds ("Jynx and Draft" when the state block says Percy+Pizza) and leaks the raw state block verbatim. Deleted.

### llama3.2:3b (3B) — LLM role — ❌ REJECTED
Why tried: 3B tier; llama family's stronger instruct tune.
Scores: intent **96.6%** ✓ (p50 **0.9s** — fastest seen) · chat **60%** ✗ · sleep **33%** ✗.
Rejection: chat invents sightings ("Percy, Matcha, Jynx, Bambi and Draft" visible when the state says Percy+Pizza; ignores paused state), no vet deflection; sleep summary miscalculates durations ("9 hours 55" for a 655-minute night). Fast but ungrounded. Deleted.

## Current recommendation

_(pending)_
