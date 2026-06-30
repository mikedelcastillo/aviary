"""End-to-end test + model benchmark for the aviary LLM/VLM (``uv run llm-harness``).

Three modes:

  * single-shot   ``uv run llm-harness "where is percy"`` — run one message
    through the REAL pipeline (intent classification -> target expansion ->
    simulated dispatch / live chat reply) and print every stage.
  * REPL          ``uv run llm-harness`` — same, interactively, line by line.
  * benchmark     ``uv run llm-harness --bench`` — score every installed text
    model on an intent test-set (accuracy + latency) and time every vision
    model on a sample frame, then recommend the best for each role.

It talks to the same Ollama the server uses (``OLLAMA_BASE_URL``) and never
needs the camera/ML stack, so it's a fast way to tune prompts and pick models.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time

import requests

from lib.ai.chat import chat_reply
from lib.ai.client import OllamaClient
from lib.ai.intent import classify_intent
from lib.ai.vlm import describe_scene, name_camera_view
from lib.config import _ollama_config
from lib.labels import pretty
from lib.roster import expand_targets, load_species_members


# The live model's findable labels (kept in sync with live-019). Override on the
# CLI with --birds if you retrain with a different roster.
DEFAULT_BIRDS = [
    "bambi", "budgie", "cockatiel", "draft", "jynx",
    "lovebird", "matcha", "percy", "pizza", "unknown_bird",
]

# Labeled intent cases: (message, expected_action). Covers commands, groups,
# "any", and the find-vs-history distinction.
INTENT_TESTS: list[tuple[str, str]] = [
    ("stop the cams", "pause"),
    ("pause for 1 minute", "pause"),
    ("go private for an hour", "pause"),
    ("turn the cameras off", "pause"),
    ("resume", "resume"),
    ("play", "resume"),
    ("turn the cameras back on", "resume"),
    ("where is percy?", "find"),
    ("find matcha", "find"),
    ("find the cockatiels", "find"),
    ("where are the lovebirds", "find"),
    ("find any bird", "find"),
    ("is bambi out", "find"),
    ("stop looking", "stop_find"),
    ("cancel the search", "stop_find"),
    ("reload cams", "discover"),
    ("rescan the cameras", "discover"),
    ("how are the cameras?", "status"),
    ("what's the status", "status"),
    ("take a snapshot", "snapshot"),
    ("show me the cameras", "snapshot"),
    ("what did percy do today?", "activity"),
    ("what are the birds doing right now?", "activity"),
    ("did pizza eat today?", "activity"),
    ("did matcha and jynx spend time together?", "activity"),
    ("when did the birds go in their cage?", "activity"),
    ("are the birds asleep?", "activity"),
    ("hello there", "chat"),
    ("good morning", "chat"),
]

# A representative few for --quick.
QUICK_TESTS = INTENT_TESTS[:1] + INTENT_TESTS[4:5] + INTENT_TESTS[7:9] + INTENT_TESTS[13:14] + INTENT_TESTS[19:21]

# Vision models worth trying that the user may not have pulled yet.
SUGGESTED_VLMS = "llama3.2-vision:11b (richer scenes), moondream (fastest), minicpm-v"
SUGGESTED_LLMS = "qwen3:1.7b (faster routing), llama3.1:8b"


def _models(client: OllamaClient) -> list[dict]:
    response = requests.get(f"{client.base_url}/api/tags", timeout=10)
    response.raise_for_status()
    return response.json().get("models", [])


def _capabilities(model: dict) -> list[str]:
    return model.get("capabilities") or model.get("details", {}).get("capabilities") or []


def _sample_image() -> bytes | None:
    paths = sorted(
        glob.glob("data/server/collect/snapshots/*.jpg")
        + glob.glob("data/server/collect/**/*.jpg", recursive=True),
        reverse=True,
    )
    if not paths:
        return None
    with open(paths[0], "rb") as handle:
        return handle.read()


# -- single message pipeline ------------------------------------------------


def run_pipeline(client, cfg, birds, species_members, text: str) -> None:
    print(f"\n\033[1mYOU>\033[0m {text}")
    started = time.time()
    try:
        intent = classify_intent(client, cfg.llm_model, text, birds)
    except Exception as exc:
        print(f"  intent: ERROR talking to Ollama: {exc}")
        return
    dt = time.time() - started
    print(f"  intent  ({dt:.1f}s): action={intent.action!r} argument={intent.argument!r}")

    if intent.action == "find":
        targets = expand_targets(intent.argument, birds, species_members)
        print(f"  expand: {[pretty(t) for t in targets] or 'NOTHING (would ask user)'}")
        print(f"  dispatch: would /find -> search for {len(targets)} label(s)")
    elif intent.action == "pause":
        print(f"  dispatch: would PAUSE (duration arg {intent.argument!r})")
    elif intent.action in ("resume", "discover", "status", "snapshot"):
        print(f"  dispatch: would run /{intent.action}")
    else:  # chat
        t0 = time.time()
        try:
            reply = chat_reply(client, cfg.llm_model, text)
        except Exception as exc:
            print(f"  chat: ERROR: {exc}")
            return
        print(f"  chat    ({time.time()-t0:.1f}s): {reply!r}")


def run_repl(client, cfg, birds, species_members) -> None:
    print("Type a message (or 'quit'). Empty line exits.")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in {"quit", "exit"}:
            break
        run_pipeline(client, cfg, birds, species_members, text)


# -- benchmarks -------------------------------------------------------------


def benchmark_llms(client, models, birds, tests) -> None:
    text_models = [
        m["name"]
        for m in models
        if "completion" in _capabilities(m) and "embedding" not in _capabilities(m)
    ]
    print(f"\n\033[1m=== LLM intent benchmark ({len(tests)} prompts) ===\033[0m")
    print(f"models: {', '.join(text_models)}\n")
    results = []
    for name in text_models:
        correct = 0
        total_latency = 0.0
        failures = 0
        for message, expected in tests:
            t0 = time.time()
            try:
                intent = classify_intent(client, name, message, birds)
                if intent.action == expected:
                    correct += 1
            except Exception:
                failures += 1
            total_latency += time.time() - t0
        accuracy = correct / len(tests) * 100
        avg = total_latency / len(tests)
        results.append((name, accuracy, avg, failures))
        print(f"  {name:18s}  acc={accuracy:5.1f}%  avg={avg:5.2f}s/req  fails={failures}")
    if results:
        best = sorted(results, key=lambda r: (-r[1], r[2]))[0]
        print(f"\n  \033[1m→ best LLM: {best[0]}\033[0m (acc {best[1]:.0f}%, {best[2]:.2f}s/req)")
        print(f"  not installed but worth trying: {SUGGESTED_LLMS}")


def benchmark_vlms(client, models, image: bytes | None) -> None:
    vision_models = [m["name"] for m in models if "vision" in _capabilities(m)]
    print(f"\n\033[1m=== VLM vision benchmark ===\033[0m")
    if image is None:
        print("  no sample image found under data/server/collect — skip (run /snapshot first)")
        return
    print(f"models: {', '.join(vision_models)} (sample {len(image)//1024} KB)\n")
    results = []
    for name in vision_models:
        t0 = time.time()
        try:
            scene = describe_scene(client, name, image, timeout_seconds=120)
            label = name_camera_view(client, name, image, timeout_seconds=120)
            dt = time.time() - t0
            results.append((name, dt, len(scene)))
            print(f"  {name:18s}  {dt:5.1f}s")
            print(f"      scene: {scene[:200]!r}")
            print(f"      name : {label!r}")
        except Exception as exc:
            print(f"  {name:18s}  ERROR: {exc}")
    if results:
        fastest = min(results, key=lambda r: r[1])
        print(f"\n  \033[1m→ fastest VLM: {fastest[0]}\033[0m ({fastest[1]:.1f}s)")
        print("  quality is subjective — eyeball the scenes above; qwen2.5vl tends to be most")
        print(f"  detailed, moondream fastest. Not installed: {SUGGESTED_VLMS}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aviary LLM/VLM test + benchmark harness")
    parser.add_argument("message", nargs="*", help="a single message to run through the pipeline")
    parser.add_argument("--bench", action="store_true", help="benchmark installed models")
    parser.add_argument("--quick", action="store_true", help="use a small prompt set for --bench")
    parser.add_argument("--image", help="image path for the VLM benchmark")
    parser.add_argument("--birds", help="comma-separated findable labels override")
    args = parser.parse_args()

    cfg = _ollama_config()
    client = OllamaClient(cfg.base_url, timeout_seconds=cfg.timeout_seconds)
    birds = [b.strip() for b in args.birds.split(",")] if args.birds else DEFAULT_BIRDS
    species_members = load_species_members()

    print(f"Ollama: {cfg.base_url}  llm={cfg.llm_model}  vlm={cfg.vlm_model}")
    if not client.is_available():
        print("\033[31mOllama is NOT reachable — is it running? (OLLAMA_BASE_URL)\033[0m")
        sys.exit(1)

    if args.bench:
        models = _models(client)
        tests = QUICK_TESTS if args.quick else INTENT_TESTS
        benchmark_llms(client, models, birds, tests)
        image = open(args.image, "rb").read() if args.image else _sample_image()
        benchmark_vlms(client, models, image)
        return

    if args.message:
        run_pipeline(client, cfg, birds, species_members, " ".join(args.message))
    else:
        run_repl(client, cfg, birds, species_members)


if __name__ == "__main__":
    main()
