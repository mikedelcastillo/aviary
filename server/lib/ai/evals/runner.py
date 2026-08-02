"""``uv run llm-eval`` — score a candidate model against the app's requirements.

    uv run llm-eval --role llm --model gemma3:4b
    uv run llm-eval --role vlm --model qwen2.5vl:7b --vlm-limit 12
    uv run llm-eval --role recall --model gemma3:12b
    uv run llm-eval --all               # evaluate the .env-configured trio

Each role runs the tasks its env key actually serves and prints a scored table
plus a PASS/FAIL verdict against :data:`REQUIREMENTS`. Results (scores, per-case
failures, latency percentiles, git revision) persist to
``data/server/model_evals/<role>/<model>.json`` so the model search can compare
candidates across runs.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from lib.ai.client import OllamaClient
from lib.ai.evals import cases, checks, vlm_eval
from lib.ai.evals.vlm_eval import CaseResult
from lib.config import _ollama_config


RESULTS_DIR = Path("data/server/model_evals")

# Findable labels, kept in sync with the live roster (see harness.DEFAULT_BIRDS).
BIRDS = [
    "bambi", "budgie", "cockatiel", "draft", "jynx",
    "lovebird", "matcha", "percy", "pizza", "unknown_bird",
]

# Per-task minimums a candidate must meet for its role to PASS. Score floors
# come from what the call sites tolerate (a wrong intent executes the wrong
# command; a contentless analysis recycles through backfill forever); latency
# caps sit well under each call's production timeout so a passing model never
# rides the timeout edge. p90 in seconds.
REQUIREMENTS: dict[str, dict[str, float]] = {
    "intent": {"score": 0.90, "p90": 12.0},
    "chat": {"score": 0.80, "p90": 25.0},
    "sleep": {"score": 0.67, "p90": 30.0},
    "recall_qa": {"score": 0.75, "p90": 45.0},
    "summary": {"score": 0.75, "p90": 45.0},
    "analyze": {"score": 0.70, "p90": 90.0},
    "scene": {"score": 0.75, "p90": 60.0},
    "camera_names": {"score": 0.70, "p90": 60.0},
}

ROLE_TASKS = {
    "llm": ("intent", "chat", "sleep"),
    "recall": ("recall_qa", "summary"),
    "vlm": ("analyze", "scene", "camera_names"),
}


@dataclass
class TaskResult:
    name: str
    score: float
    passed: int
    total: int
    latency_p50: float
    latency_p90: float
    failures: list[str] = field(default_factory=list)
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def meets(self) -> bool:
        req = REQUIREMENTS[self.name]
        return self.score >= req["score"] and self.latency_p90 <= req["p90"]


def _percentiles(latencies: list[float]) -> tuple[float, float]:
    if not latencies:
        return 0.0, 0.0
    ordered = sorted(latencies)
    p50 = statistics.median(ordered)
    p90 = ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]
    return round(p50, 2), round(p90, 2)


def _finish(name: str, results: list[CaseResult], extras: dict[str, float] | None = None) -> TaskResult:
    latencies = [r.latency for r in results]
    p50, p90 = _percentiles(latencies)
    passed = sum(1 for r in results if r.ok)
    return TaskResult(
        name=name,
        score=round(passed / max(1, len(results)), 3),
        passed=passed,
        total=len(results),
        latency_p50=p50,
        latency_p90=p90,
        failures=[r.detail for r in results if not r.ok][:20],
        extras=extras or {},
    )


# -- LLM role ---------------------------------------------------------------


def run_intent(client, model: str, *, quick: bool = False) -> TaskResult:
    from lib.ai.harness import INTENT_TESTS
    from lib.ai.intent import classify_intent

    base = [cases.IntentCase(m, a) for m, a in (INTENT_TESTS[:8] if quick else INTENT_TESTS)]
    extra = cases.EXTRA_INTENT_CASES[:6] if quick else cases.EXTRA_INTENT_CASES
    results: list[CaseResult] = []
    for case in base + extra:
        t0 = time.time()
        try:
            intent = classify_intent(client, model, case.message, BIRDS, prior=case.prior)
        except Exception as exc:
            results.append(CaseResult(False, f"{case.message!r}: ERROR {exc}", time.time() - t0))
            continue
        latency = time.time() - t0
        ok = intent.action == case.action
        detail = f"{case.message!r} -> {intent.action}"
        if ok and case.arg_contains:
            arg = intent.argument.lower()
            missing = [s for s in case.arg_contains if s not in arg]
            if missing:
                ok = False
                detail += f" (argument {intent.argument!r} missing {missing})"
        elif not ok:
            detail += f" (expected {case.action})"
        results.append(CaseResult(ok, detail, latency))
    return _finish("intent", results)


def run_chat(client, model: str, *, quick: bool = False) -> TaskResult:
    from lib.ai.chat import CHAT_FALLBACK, chat_reply
    from lib.ai.context import build_chat_context, format_system_state

    fallbacks = 0
    results: list[CaseResult] = []
    for case in cases.CHAT_CASES[:5] if quick else cases.CHAT_CASES:
        context = None
        if case.state is not None:
            state = format_system_state(datetime(2026, 7, 3, 9, 30), **case.state)
            context = build_chat_context(case.message, system_state=state)
        history = [{"role": r, "content": c} for r, c in case.history] or None
        t0 = time.time()
        try:
            reply = chat_reply(client, model, case.message, history, context=context)
        except Exception as exc:
            results.append(CaseResult(False, f"{case.message!r}: ERROR {exc}", time.time() - t0))
            continue
        latency = time.time() - t0
        problems = []
        low = reply.lower()
        if not reply or reply == CHAT_FALLBACK:
            fallbacks += 1
            problems.append("fallback/empty")
        else:
            # The persona contract, rule by rule.
            if checks.word_count(reply) > 45 or checks.sentence_count(reply) > 2:
                problems.append(f"{checks.word_count(reply)}w/{checks.sentence_count(reply)}s (want 1 sentence <35w)")
            if checks.contains_species_word(reply):
                problems.append("species word")
            if checks.opens_with_vocative_bird_name(reply):
                problems.append("greets a bird by name")
            if checks.has_markdown(reply):
                problems.append("markdown")
            for group in case.must_groups:
                if not any(tok in low for tok in group):
                    problems.append(f"missing {'/'.join(group)}")
            for banned in case.must_not:
                if banned in low:
                    problems.append(f"contains banned {banned!r}")
        results.append(CaseResult(not problems, f"{case.message!r} -> {reply[:90]!r} [{'; '.join(problems)}]", latency))
    return _finish("chat", results, {"fallbacks": float(fallbacks)})


def run_sleep(client, model: str) -> TaskResult:
    from lib.sleep.model import NIGHT_FRIGHT, Disturbance, SleepNight
    from lib.sleep.narrate import format_morning, llm_summary

    results: list[CaseResult] = []
    for case in cases.SLEEP_CASES:
        lights_out = datetime.strptime(case.lights_out, "%H:%M")
        first_light = datetime.strptime(case.first_light, "%H:%M") + timedelta(days=1)
        disturbances = [
            Disturbance(lights_out + timedelta(hours=2 + i), "motion", detail="stir")
            for i in range(case.disturbances)
        ]
        if case.fright:
            disturbances.append(
                Disturbance(lights_out + timedelta(hours=4), NIGHT_FRIGHT, detail="possible night-fright ~23:02")
            )
        night = SleepNight(
            night_of=date(2026, 7, 3), lights_out=lights_out, first_light=first_light,
            dark_minutes=case.dark_minutes, disturbances=disturbances, score=case.score,
        )
        fallback = format_morning(night)
        t0 = time.time()
        reply = llm_summary(client, model, night, timeout_seconds=60)
        latency = time.time() - t0
        problems = []
        if reply == fallback:
            problems.append("fell back to template")
        else:
            low = reply.lower()
            if checks.word_count(reply) > 40:
                problems.append(f"{checks.word_count(reply)} words (want <30)")
            # Anti-invention: every digit in the reply must come from the facts.
            facts = f"{case.dark_minutes} {case.lights_out} {case.first_light} {case.score} 100 {len(disturbances)} {case.dark_minutes//60} {case.dark_minutes%60}"
            invented = checks.invented_numbers(reply, facts)
            if invented:
                problems.append(f"invented numbers {sorted(invented)}")
            if not case.fright and ("fright" in low):
                problems.append("invented a night-fright")
            for group in case.must_groups:
                if not any(tok in low for tok in group):
                    problems.append(f"missing {'/'.join(group)}")
        results.append(CaseResult(not problems, f"{case.name}: {reply[:100]!r} [{'; '.join(problems)}]", latency))
    return _finish("sleep", results)


# -- recall role ------------------------------------------------------------


def run_recall_qa(client, model: str) -> TaskResult:
    from lib.activity import answer_activity_question
    from lib.roster import load_sexes, pronoun_map, pronoun_sentence

    pronoun_note = pronoun_sentence(pronoun_map(load_sexes()))
    results: list[CaseResult] = []
    for case in cases.RECALL_CASES:
        t0 = time.time()
        try:
            reply = answer_activity_question(
                client, model, case.question, list(case.notes), pronoun_note,
                case.window_phrase, facts=case.facts, timeout_seconds=90,
            )
        except Exception as exc:
            results.append(CaseResult(False, f"{case.question!r}: ERROR {exc}", time.time() - t0))
            continue
        latency = time.time() - t0
        problems = []
        low = reply.lower()
        if not reply:
            problems.append("empty reply")
        else:
            if case.expect == "yes" and not checks.starts_yes(reply):
                problems.append("should start Yes")
            if case.expect == "no" and not checks.starts_no(reply):
                problems.append("should start No")
            if case.expect == "open" and checks.starts_yes_or_no(reply):
                problems.append("open question started Yes/No")
            for group in case.must_groups:
                if not any(tok in low for tok in group):
                    problems.append(f"missing {'/'.join(group)}")
            for banned in case.must_not:
                if banned in low:
                    problems.append(f"banned {banned!r}")
            if checks.contains_species_word(reply):
                problems.append("species word")
            if checks.contains_meta_word(reply):
                problems.append("quotes meta-words (counts/notes/...)")
            if checks.has_markdown(reply):
                problems.append("markdown")
            if checks.sentence_count(reply) > 4:
                problems.append(f"{checks.sentence_count(reply)} sentences (want 2-3)")
            # Anti-confabulation: digits must originate in the notes/facts.
            invented = checks.invented_numbers(reply, case.facts + " ".join(case.notes) + case.question)
            if invented:
                problems.append(f"invented numbers {sorted(invented)}")
        results.append(CaseResult(not problems, f"{case.question!r} -> {reply[:100]!r} [{'; '.join(problems)}]", latency))
    return _finish("recall_qa", results)


def run_summary(client, model: str) -> TaskResult:
    from lib.activity import summarise_activity
    from lib.roster import load_sexes, pronoun_map, pronoun_sentence

    pronoun_note = pronoun_sentence(pronoun_map(load_sexes()))
    results: list[CaseResult] = []
    for case in cases.SUMMARY_CASES:
        t0 = time.time()
        try:
            reply = summarise_activity(
                client, model, list(case.notes), case.subject, pronoun_note, timeout_seconds=90
            )
        except Exception as exc:
            results.append(CaseResult(False, f"{case.subject}: ERROR {exc}", time.time() - t0))
            continue
        latency = time.time() - t0
        problems = list(checks.valid_bullet_summary(reply)) if reply else ["empty reply"]
        low = reply.lower()
        if reply:
            if checks.contains_species_word(reply):
                problems.append("species word")
            for group in case.must_groups:
                if not any(tok in low for tok in group):
                    problems.append(f"missing {'/'.join(group)}")
        results.append(CaseResult(not problems, f"{case.subject}: {reply[:110]!r} [{'; '.join(problems)}]", latency))
    return _finish("summary", results)


# -- vlm role ---------------------------------------------------------------


def run_analyze(client, model: str, *, limit: int = 24) -> TaskResult:
    samples = vlm_eval.sample_observations(limit=limit)
    golden_only = vlm_eval.golden_frames()
    # Curated golden frames always run (they carry the trusted labels); the
    # journal sample fills the rest of the budget.
    pool = golden_only + [s for s in samples if all(str(s.photo) != str(g.photo) for g in golden_only)]
    pool = pool[:limit] if limit else pool
    results = [vlm_eval.eval_analyze_case(client, model, s) for s in pool]
    subtotals: dict[str, list[float]] = {}
    for r in results:
        for key, value in r.subscores.items():
            subtotals.setdefault(key, []).append(value)
    extras = {k: round(sum(v) / len(v), 3) for k, v in subtotals.items() if v}
    return _finish("analyze", results, extras)


def run_scene(client, model: str, *, limit: int = 10) -> TaskResult:
    sightings = vlm_eval.sample_sightings(limit=limit)
    results = [vlm_eval.eval_scene_case(client, model, s) for s in sightings]
    return _finish("scene", results)


def run_camera_names(client, model: str, *, limit: int = 5) -> TaskResult:
    results, extras = vlm_eval.eval_camera_names(client, model, limit_cameras=limit)
    task = _finish("camera_names", results, extras)
    # Stability/uniqueness fold into the pass bar: a namer that gives every
    # camera the same name scores high per-call but is useless.
    if extras.get("uniqueness", 1.0) < 0.5:
        task.failures.append(f"low uniqueness {extras['uniqueness']:.2f}")
        task.score = round(task.score * extras["uniqueness"] * 2, 3)
    return task


# -- orchestration ----------------------------------------------------------


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        return ""


def evaluate_role(client, role: str, model: str, *, quick: bool = False, vlm_limit: int = 24) -> dict:
    started = time.time()
    tasks: list[TaskResult] = []
    if role == "llm":
        tasks.append(run_intent(client, model, quick=quick))
        tasks.append(run_chat(client, model, quick=quick))
        tasks.append(run_sleep(client, model))
    elif role == "recall":
        tasks.append(run_recall_qa(client, model))
        tasks.append(run_summary(client, model))
    elif role == "vlm":
        tasks.append(run_analyze(client, model, limit=vlm_limit))
        tasks.append(run_scene(client, model, limit=max(6, vlm_limit // 2)))
        tasks.append(run_camera_names(client, model, limit=5))
    else:
        raise ValueError(f"unknown role {role!r}")

    verdict = all(t.meets for t in tasks)
    report = {
        "model": model,
        "role": role,
        "verdict": "PASS" if verdict else "FAIL",
        "when": datetime.now().isoformat(timespec="seconds"),
        "git": _git_rev(),
        "duration_s": round(time.time() - started, 1),
        "tasks": [asdict(t) | {"meets": t.meets, "requirement": REQUIREMENTS[t.name]} for t in tasks],
    }
    return report


def print_report(report: dict) -> None:
    print(f"\n\033[1m=== {report['role']} role: {report['model']} — {report['verdict']} "
          f"({report['duration_s']}s) ===\033[0m")
    for task in report["tasks"]:
        req = task["requirement"]
        mark = "\033[32m✓\033[0m" if task["meets"] else "\033[31m✗\033[0m"
        extras = "  ".join(f"{k}={v}" for k, v in (task.get("extras") or {}).items())
        print(
            f"  {mark} {task['name']:13s} {task['score']*100:5.1f}% "
            f"({task['passed']}/{task['total']})  p50={task['latency_p50']:.1f}s "
            f"p90={task['latency_p90']:.1f}s  [need ≥{req['score']*100:.0f}%, p90≤{req['p90']:.0f}s]  {extras}"
        )
        for failure in task["failures"][:6]:
            print(f"      · {failure[:150]}")


def save_report(report: dict) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", report["model"])
    out = RESULTS_DIR / report["role"] / f"{safe}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if out.exists():
        try:
            history = json.loads(out.read_text(encoding="utf-8")).get("history", [])
        except ValueError:
            pass
    history.append(report)
    out.write_text(json.dumps({"history": history[-10:]}, indent=1), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aviary model requirement eval")
    parser.add_argument("--role", choices=("llm", "recall", "vlm"), action="append",
                        help="role(s) to evaluate (repeatable)")
    parser.add_argument("--model", help="candidate model for the given role(s); default = the configured model")
    parser.add_argument("--all", action="store_true", help="evaluate the .env-configured model of every role")
    parser.add_argument("--base-url", help="override OLLAMA_BASE_URL (e.g. a specific worker)")
    parser.add_argument("--quick", action="store_true", help="smaller case sets")
    parser.add_argument("--vlm-limit", type=int, default=24, help="journal frames for the analyze task")
    parser.add_argument("--no-save", action="store_true", help="don't persist the report JSON")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    cfg = _ollama_config()
    base_url = args.base_url or cfg.base_url
    client = OllamaClient(base_url, timeout_seconds=cfg.timeout_seconds, keep_alive="30m")
    if not client.is_available():
        print(f"\033[31mOllama not reachable at {base_url}\033[0m")
        sys.exit(2)

    role_models: list[tuple[str, str]] = []
    if args.all or not args.role:
        role_models = [("llm", cfg.llm_model), ("recall", cfg.recall_model), ("vlm", cfg.vlm_model)]
    else:
        defaults = {"llm": cfg.llm_model, "recall": cfg.recall_model, "vlm": cfg.vlm_model}
        for role in args.role:
            role_models.append((role, args.model or defaults[role]))

    print(f"endpoint: {base_url}")
    all_pass = True
    for role, model in role_models:
        report = evaluate_role(client, role, model, quick=args.quick, vlm_limit=args.vlm_limit)
        print_report(report)
        if not args.no_save:
            path = save_report(report)
            print(f"  saved -> {path}")
        all_pass &= report["verdict"] == "PASS"
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
