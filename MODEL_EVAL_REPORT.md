# Aviary Model Evaluation Report

Goal: find the **lowest-parameter-count** ollama models that fully meet the app's
LLM/VLM requirements, per role. Maintained live during the 2026-08-02 model search.

## Final verdict (2026-08-02)

**All three incumbents survive a 10-candidate challenge. No .env change recommended.**

| Role | Winner | Challengers beaten | Deciding evidence |
|------|--------|--------------------|-------------------|
| LLM | **gemma3:4b** (keep) | gemma3:1b, llama3.2:1b, llama3.2:3b, qwen2.5:3b, granite3.3:2b, phi4-mini | every sub-4B passes intent (94-97%) but fails chat grounding — invented sightings, ignored paused state, diet advice, run-ons |
| Recall | **gemma3:12b** (keep) | gemma3:4b (16.7%), qwen2.5:7b (33.3%) | accuracy cliff below 12B: confabulated clock times, dropped Yes/No polarity, misgendering |
| VLM | **qwen2.5vl:7b** (keep) | qwen2.5vl:3b, moondream, granite3.2-vision, llava-phi3, minicpm-v (8B) | incumbent's analyze 50% doubles the field's best (25%); smaller models collapse on coverage or describe the annotation overlay |

Verification is reproducible: `uv run llm-eval --all` (suite added in this work;
per-model JSON history in `data/server/model_evals/`). All losing models deleted
from disk; the aviary service was stopped during benchmarks and restarted and
health-verified after. Known incumbent flaws the suite now tracks (improvement
backlog, prompt-level not model-level): chat invents sightings while paused;
VLM golden agreement 0.479 with species-word leakage; camera-name uniqueness 0.4.

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

### qwen2.5vl:7b — VLM role — best available, but fails the aspirational bar
analyze **50%** (12/24, p50 32.2s / p90 42.4s): coverage 0.931, golden agreement **0.479**,
species words in 1/3 of scenes ("parrot", "the lovebirds"), occasional identity swaps
(labels a Draft frame's scene "Percy perches calmly…"). scene **83.3%** ✓ (both misses were
cat/dog frames from a sampler bug, since fixed — bird-only score 10/10). camera_names 80% ✓
(uniqueness only 0.4 — many cameras get the same name, matching the odd names in the journal).

## Phase 2: quantization / bigger-models-on-small-cards (2026-08-02)

Directive: try aggressive quants (q3/q2/IQ2/IQ3, QAT), HF GGUF imports, KV-cache
quantization, dual-GPU split, partial CPU offload — a well-quantized larger model
may beat a small full-precision one.

### colibri — ❌ RESEARCHED, NOT APPLICABLE
"Colibri" = [JustVugg/colibri](https://github.com/JustVugg/colibri) (Jul 2026): a
pure-C inference engine that runs GLM-5.2 (744B MoE) on ~25GB-RAM machines by
streaming routed experts from SSD. Real and impressive, but measured at
**0.05–0.1 tokens/sec** — a 30-token intent JSON would take 5–10 minutes against
the app's 45s interactive gate, and GLM-5.2 has no vision. Wrong tool for a
latency-bound Telegram bot on this box. Not pulled.

### gemma3:4b (4B, multimodal) — VLM role — ⭐ BEATS INCUMBENT ON DECORATION
Why tried: phase-2 insight — the LLM incumbent is multimodal, nobody had tested it as the VLM.
Scores: analyze **62.5%** (vs incumbent 50%) — coverage 0.986, golden agreement **0.597**
(vs 0.479), **zero** species/overlay leaks, evasion 1.0, p50 **10.1s** (vs 32.2s, 3x faster) ·
scene **66.7%** ✗ (grounded but prefixes "Here's a description of the scene:" — a fixable
prompt/post-processing issue, not a capability gap) · camera_names **40%** ✗ (stability 1.0
but uniqueness 0.2 — names most views identically).
Takeaway: best decoration model available at any size tested, and decoration is 95%+ of
production VLM volume. Loses the two low-volume tasks. See phase-2 conclusion for the
split-role recommendation.

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

### qwen2.5:3b (3B) — LLM role — ❌ REJECTED (closest sub-4B so far)
Why tried: strong instruct family, 3B tier.
Scores: intent **94.9%** ✓ (p50 0.9s) · chat **60%** ✗ · sleep **100%** ✓.
Rejection: chat grounding/policy — reports "all 4 cameras healthy" when the state block says 3/4, glosses over paused mode, and **gives diet advice** ("avocados are toxic; avoid feeding") where the persona requires deflecting to an avian vet. Intent+sleep are genuinely good; chat discipline isn't. Deleted at cleanup unless needed for cross-checks.

### granite3.3:2b (2B) — LLM role — ❌ REJECTED
Why tried: 2B tier, IBM instruct tune reputed good at structured tasks.
Scores: intent **96.6%** ✓ (p50 **0.7s** — fastest of all) · chat **60%** ✗ · sleep **100%** ✓.
Rejection: chat persona — invents a full scene while nothing supports it, writes species words ("budgerariums"), no vet deflection, greets birds by name. Same failure signature as every other sub-4B: structured tasks fine, free-chat discipline absent. Deleted.

### phi4-mini (3.8B) — LLM role — ❌ REJECTED
Why tried: largest sub-incumbent option; Microsoft instruct tune.
Scores: intent **94.9%** ✓ (p50 1.2s) · chat **30%** ✗ · sleep **100%** ✓.
Rejection: chat rambles 50-82 words against a 35-word contract and invents which birds are visible. Deleted.

### gemma3:4b (4B) — Recall role — ❌ REJECTED
Why tried: already the configured fallback; passing would retire the 12b (8GB + ~4x faster answers).
Scores: recall_qa **16.7%** ✗ (p50 4.9s) · summary 50% ✗.
Rejection: confabulates clock times not present in the data ("9:30 am"), quotes meta-words ("feeding observations recorded"), bullets omit the subject's name. Confirms the .env comment: a 4B confabulates over per-bird tallies. The 12b's 100% under identical scoring is real headroom, not slack requirements.

### qwen2.5:7b (7B) — Recall role — ❌ REJECTED
Why tried: halfway point between failed 4B and passing 12B.
Scores: recall_qa **33.3%** ✗ (p50 7.1s) · summary **100%** ✓.
Rejection: correct facts but poor instruction-following — omits the required Yes/No opener, misgenders Draft ("she"), skips the vet suggestion, quotes meta-words. Deleted.

### qwen2.5vl:3b (3B) — VLM role — ❌ REJECTED
Why tried: already on disk; 3x faster than the 7b (p50 8.9s vs 32.2s); same family.
Scores: analyze **12.5%** ✗ (coverage 0.458 — skips half the labeled birds; golden agreement 0.167) · scene 83.3% ✓ · camera_names 100% but stability 0.0 (new name every frame).
Rejection: decoration coverage collapse — the exact quality gap Mike's earlier "keep 7b" decision predicted. Kept on disk for now (useful A/B reference), candidate for deletion at cleanup.

### moondream (1.8B) — VLM role — ❌ REJECTED
Why tried: smallest vision model on ollama; "fastest" per old harness notes.
Scores: analyze **0%** (0/24; names_birds 0.125, overlay leakage — literally describes "colored boxes with names printed on them") · scene **8.3%** (empty captions, echoes the prompt text back).
Rejection: not viable for any VLM task in this app. Deleted immediately.

### granite3.2-vision (2B) — VLM role — ❌ REJECTED
Why tried: 2B document/vision tune; decent JSON discipline expected.
Scores: analyze **16.7%** ✗ (coverage 0.91 — best of the small models — but golden agreement 0.285, species words, invented identities: "vibrant yellow parakeet named Percy" on a bambi/matcha/percy frame) · scene **41.7%** ✗.
Rejection: hallucinated identity + species leakage. Deleted.

### llava-phi3 (3.8B) — VLM role — ❌ REJECTED
Why tried: llava tune on phi3; last sub-incumbent candidate.
Scores: analyze **20.8%** ✗ (overlay leakage 0.417 — "birds sitting in their boxes"; names_birds 0.167) · scene **66.7%** ✗ (fast though: p50 2.5s).
Rejection: describes the annotation overlay, rarely uses the given bird names. Deleted.

### minicpm-v (8B) — VLM role — ❌ REJECTED
Why tried: one tier above the incumbent; if nothing ≤7B meets the bar, an 8B that
does would be the smallest model meeting all requirements.
Scores: analyze **25%** ✗ (golden agreement **0.514** — the best of any model — but
names_birds 0.25, invents a bird called "Lynx", describes "colorful boxes overlayed") ·
scene **16.7%** ✗ · fast (p50 ~7s). Deleted.

## VLM role conclusion (2026-08-02)

**qwen2.5vl:7b stays.** Nothing at any size met the 70% analyze bar, and the
incumbent leads everything tested at 50% (next best: minicpm-v 8B at 25%,
llava-phi3 20.8%, granite3.2-vision 16.7%, qwen2.5vl:3b 12.5%, moondream 0%).
The operative rule — beat the incumbent's subscores at lower parameters — was
never approached: sub-7B models collapse on either coverage (3b: 0.458) or
identity/overlay discipline (all others). Recommendation: keep
`OLLAMA_VLM_MODEL=qwen2.5vl:7b`. The suite also documents the incumbent's real
flaws (golden agreement 0.479, species leakage, camera-name uniqueness 0.4) —
those are prompt/post-processing opportunities, not model-swap opportunities.

## Recall role conclusion (2026-08-02)

**gemma3:12b stays.** 4B scored 16.7%, 7B scored 33.3%, the 12B scores 100% with
p90 26.5s — the accuracy cliff between 7B and 12B is exactly why the role exists.
The role is also the lowest-volume call site (5 production calls since 2026-07-04),
so shrinking it buys nearly nothing. Recommendation: keep `OLLAMA_RECALL_MODEL=gemma3:12b`.

## LLM role conclusion (2026-08-02)

**gemma3:4b (incumbent) is the smallest model on ollama that meets the LLM-role
requirements.** Five candidates below it (gemma3:1b, llama3.2:1b/3b, qwen2.5:3b,
granite3.3:2b, phi4-mini — five distinct families, 1B→3.8B) all pass intent
routing (94-97%) and mostly sleep, and all FAIL the chat persona/grounding
contract the Telegram bot depends on (invented sightings, ignored paused state,
species words, diet advice, run-on replies). The failure is capability-shaped,
not family-shaped. Recommendation: keep `OLLAMA_LLM_MODEL=gemma3:4b`.

## Current recommendation

_(pending)_
