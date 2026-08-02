"""Eval fixtures: labeled cases per LLM task, derived from the live prompts.

Every case encodes an expectation a call site actually relies on. Intent cases
mine the router prompt's own tie-break rules (find-vs-activity, weather-vs-chat,
sleep-vs-chat, machine-vs-status, photo-vs-snapshot); chat cases pair a message
with a fabricated ``format_system_state`` block and check the reply against the
persona contract; recall cases carry notes + a COUNTS block whose right answer
is computable, so confabulation is measurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lib.ai.evals import checks


# -- intent -----------------------------------------------------------------


@dataclass(frozen=True)
class IntentCase:
    message: str
    action: str
    # When set, the returned argument (lowercased) must contain EVERY substring.
    arg_contains: tuple[str, ...] = ()
    # Previous (message, action) — exercises the follow-up inheritance path.
    prior: tuple[str, str] | None = None


# The llm-harness INTENT_TESTS set is imported by the runner and merged with
# these; here live only the additions — argument checks, follow-ups, and
# adversarial boundary cases from the system prompt's rules.
EXTRA_INTENT_CASES: list[IntentCase] = [
    # Argument fidelity (the router prompt: argument = exactly what was named).
    IntentCase("find percy and matcha", "find", arg_contains=("percy", "matcha")),
    IntentCase("where's jynx?", "find", arg_contains=("jynx",)),
    IntentCase("pause for 20 minutes", "pause", arg_contains=("20",)),
    IntentCase("what did draft do this morning?", "activity", arg_contains=("draft",)),
    # Follow-up inheritance: a subject swap keeps the prior action.
    IntentCase("what about jynx?", "find", arg_contains=("jynx",), prior=("where is percy?", "find")),
    IntentCase("and bambi?", "activity", arg_contains=("bambi",), prior=("what did pizza do today?", "activity")),
    # A follow-up that is NOT a subject swap must be classified fresh.
    IntentCase("thanks!", "chat", prior=("where is percy?", "find")),
    # Boundary: greeting is chat, never a command (the "home" trap).
    IntentCase("good night", "chat"),
    IntentCase("hey!", "chat"),
    # Boundary: machine (server hardware) vs status (cameras).
    IntentCase("how hot is the gpu?", "machine"),
    IntentCase("is the cluster healthy?", "machine"),
    IntentCase("are the cameras ok?", "status"),
    # Boundary: weather (forecast) vs chat (care/live state).
    IntentCase("will it rain later?", "weather"),
    IntentCase("is it too cold for them?", "chat"),
    IntentCase("is it dark yet?", "chat"),
    # Boundary: sleep (last night) vs chat (live / how-to).
    IntentCase("how did they sleep last night?", "sleep"),
    IntentCase("how long should they sleep?", "chat"),
    IntentCase("did anyone have a night fright?", "sleep"),
    # Boundary: photos are activity, snapshot is live-capture only.
    IntentCase("show me photos of pizza taking a bath", "activity", arg_contains=("pizza",)),
    IntentCase("what do the cameras see now?", "snapshot"),
    # Boundary: find (locate one) vs activity (summary of all).
    IntentCase("where is everyone?", "activity"),
    IntentCase("is draft out?", "find", arg_contains=("draft",)),
    # Commands with less common phrasing.
    IntentCase("switch the cameras to stream2", "quality", arg_contains=("stream2",)),
    IntentCase("turn autofind off", "autofind", arg_contains=("disable",)),
    IntentCase("point the cameras home", "home"),
    IntentCase("restart aviary", "restart"),
    IntentCase("never mind, stop searching", "stop_find"),
    IntentCase("go private for 2 hours", "pause"),
    IntentCase("did percy and bambi hang out this week?", "activity", arg_contains=("percy", "bambi")),
    IntentCase("who was most active today?", "activity"),
]


# -- chat -------------------------------------------------------------------


# State-block ingredients per scenario (passed to format_system_state).
DAY_STATE = dict(cameras_total=4, cameras_healthy=4,
                 visible_text="Percy (Big Cage); Pizza (Play Gym)", daylight="day", autofind_on=True)
DEGRADED_STATE = dict(cameras_total=4, cameras_healthy=3, visible_text="", daylight="day", autofind_on=True)
NIGHT_STATE = dict(cameras_total=4, cameras_healthy=4, visible_text="", daylight="night", autofind_on=False)
PAUSED_STATE = dict(paused=True, pause_status="Privacy mode is ON — paused for another 42 minutes.")


@dataclass(frozen=True)
class ChatCase:
    message: str
    state: dict | None
    # Reply must contain at least one word from EVERY group (lowercased).
    must_groups: tuple[tuple[str, ...], ...] = ()
    # Reply must contain none of these words.
    must_not: tuple[str, ...] = ()
    history: tuple[tuple[str, str], ...] = ()  # (role, content) prior turns


CHAT_CASES: list[ChatCase] = [
    # Grounding: who is visible comes straight from the state block.
    ChatCase("which birds can you see right now?", DAY_STATE,
             must_groups=(("percy",), ("pizza",))),
    # A bird absent from the state: the prompt says offer "find <bird>".
    ChatCase("where is matcha right now?", DAY_STATE,
             must_groups=(("find", "look", "search"),)),
    # Night state: the reply should reflect roosting/dark, not invent activity.
    ChatCase("are the birds asleep?", NIGHT_STATE,
             must_groups=(("sleep", "asleep", "roost", "rest", "night", "dark"),)),
    # Camera health from the state block ("3 of 4").
    ChatCase("are all the cameras working?", DEGRADED_STATE,
             must_groups=(("3", "three", "one"),)),
    # Paused state must be surfaced, not glossed over.
    ChatCase("what's happening in the aviary?", PAUSED_STATE,
             must_groups=(("privacy", "paused", "pause", "not watching", "off"),)),
    # Care questions: deflect to a vet, no diet advice.
    ChatCase("can the birds eat avocado?", DAY_STATE,
             must_groups=(("vet", "veterinarian", "avian"),)),
    ChatCase("why is percy fluffed up?", DAY_STATE,
             must_groups=(("vet", "veterinarian", "watch", "monitor", "keep an eye"),)),
    # Plain small talk: just needs to obey the format contract.
    ChatCase("good morning! how are things?", DAY_STATE),
    ChatCase("thanks for watching them", DAY_STATE),
    # History continuity: "her" refers to Percy (she) from the prior turn.
    ChatCase("is she usually out at this time?", DAY_STATE,
             must_groups=(("percy", "she", "her"),),
             history=(("user", "tell me about percy"),
                      ("assistant", "Percy is out on the Big Cage perch, looking lively."))),
]


# -- recall (activity Q&A over notes + counts) ------------------------------


@dataclass(frozen=True)
class RecallCase:
    question: str
    notes: tuple[str, ...]
    facts: str
    window_phrase: str = "today"
    must_groups: tuple[tuple[str, ...], ...] = ()
    must_not: tuple[str, ...] = ()
    # "yes"/"no": must start with that word; "open": must NOT start Yes/No;
    # "any": polarity unchecked (negated yes/no questions read either way).
    expect: str = "open"


_NOTES_DAY: tuple[str, ...] = (
    "(9 hours ago) [pizza]: Pizza cracked seeds at the food bowl, eating hungrily.",
    "(7 hours ago) [jynx, matcha]: Jynx and Matcha preened side by side on the window perch.",
    "(6 hours ago) [bambi]: Bambi splashed in the water dish, having a proper bath.",
    "(4 hours ago) [draft]: Draft climbed back into the cage and settled on a branch.",
    "(3 hours ago) [percy]: Percy napped quietly, tucked on the perch.",
    "(1 hour ago) [jynx]: Jynx played with the little bell toy.",
)

RECALL_CASES: list[RecallCase] = [
    # Together verdict YES: the COUNTS block is explicit; the answer must not
    # contradict it (the exact failure the VERDICT line was added to stop).
    RecallCase(
        "did jynx and matcha spend time together today?",
        _NOTES_DAY,
        "Over today:\n"
        "VERDICT: YES, Jynx and Matcha DID spend time together (14 shared moments, "
        "e.g. 09:12, 09:47, 10:03) — even though they were often apart too. Any "
        "'were they together' answer is YES.\n"
        "Jynx + Matcha same-frame/view observations: 14 (09:12, 09:47, 10:03).\n"
        "Jynx + Matcha apart/only-one observations: 22 (Jynx only 12, Matcha only 10).",
        must_groups=(("jynx",), ("matcha",)),
        expect="yes",
    ),
    # Together verdict NO.
    RecallCase(
        "did percy and draft spend time together today?",
        _NOTES_DAY,
        "Over today:\n"
        "VERDICT: NO, Percy and Draft were never seen together.\n"
        "Percy + Draft same-frame/view observations: 0.\n"
        "Percy + Draft apart/only-one observations: 31 (Percy only 18, Draft only 13).",
        expect="no",
    ),
    # Flock ranking: the counts name the top bird; the answer must repeat it.
    RecallCase(
        "who was most active today?",
        _NOTES_DAY,
        "Over today:\n"
        "Jynx: 41 observations from 07:02 to 16:44; with other birds 12, alone/solo 29; "
        "activity tags: playing x18, climbing x9, preening x8; health: no explicit health-concern words recorded.\n"
        "Pizza: 22 observations from 07:30 to 15:10; activity tags: feeding x9, resting x8.\n"
        "Percy: 12 observations from 09:00 to 14:00; activity tags: resting x10, preening x2.\n"
        "Bambi: 8 observations from 10:15 to 12:40; activity tags: bathing x3, resting x5.",
        must_groups=(("jynx",),),
        expect="open",
    ),
    # Zero-count scan: which bird has NO feeding recorded (bambi + percy don't;
    # accept either being named, require bambi as the clearest one).
    RecallCase(
        "did any bird not eat today?",
        _NOTES_DAY,
        "Over today:\n"
        "Pizza: 22 observations; activity tags: feeding x9, resting x8.\n"
        "Jynx: 41 observations; activity tags: playing x18, feeding x4.\n"
        "Matcha: 15 observations; activity tags: preening x7, feeding x3.\n"
        "Draft: 13 observations; activity tags: climbing x6, feeding x2.\n"
        "Percy: 12 observations; activity tags: resting x10, preening x2 (no feeding recorded).\n"
        "Bambi: 8 observations; activity tags: bathing x3, resting x5 (no feeding recorded).",
        must_groups=(("bambi", "percy"),),
        expect="any",
    ),
    # Health flag: must name the bird and suggest a vet, gently.
    RecallCase(
        "is any bird not doing well?",
        _NOTES_DAY,
        "Over today:\n"
        "Draft: 13 observations; activity tags: resting x9; health: fluffed (fluffed up on the cage floor, 11:20).\n"
        "Jynx: 41 observations; health: no explicit health-concern words recorded.\n"
        "Pizza: 22 observations; health: no explicit health-concern words recorded.",
        must_groups=(("draft",), ("vet", "veterinarian", "avian", "eye on", "watch")),
        expect="any",
    ),
    # Open question about one bird: right bird, right activity, no Yes/No start.
    RecallCase(
        "what did percy do today?",
        _NOTES_DAY,
        "Over today:\n"
        "Percy: 12 observations from 09:00 to 14:00; with other birds 2, alone/solo 10; "
        "activity tags: resting x8, sleeping x2, preening x2; health: no explicit health-concern words recorded.",
        must_groups=(("percy",), ("rest", "resting", "nap", "napped", "sleep", "slept", "quiet", "preen")),
        expect="open",
    ),
]


# -- activity summary (bulleted) --------------------------------------------


@dataclass(frozen=True)
class SummaryCase:
    subject: str
    notes: tuple[str, ...]
    must_groups: tuple[tuple[str, ...], ...] = ()


SUMMARY_CASES: list[SummaryCase] = [
    SummaryCase(
        "Pizza",
        (
            "(3 hours ago) [pizza]: Pizza cracked seeds at the food bowl, eating hungrily.",
            "(2 hours ago) [pizza, draft]: Pizza and Draft perched together near the mirror.",
            "(1 hour ago) [pizza]: Pizza chewed on the rope toy by the cage door.",
        ),
        must_groups=(("pizza",), ("seed", "seeds", "ate", "eating", "food")),
    ),
    SummaryCase(
        "Percy",
        (
            "(5 hours ago) [percy]: Percy preened her wing feathers on the window perch.",
            "(2 hours ago) [percy]: Percy napped quietly, tucked on the perch.",
        ),
        must_groups=(("percy",), ("preen", "preened", "preening", "nap", "napped", "rest", "groom")),
    ),
]


# -- sleep morning one-liner ------------------------------------------------


@dataclass(frozen=True)
class SleepCase:
    name: str
    # Keyword args for lib.sleep.tracker.SleepNight construction happen in the
    # runner (import locality); here just the deterministic facts string parts.
    dark_minutes: int
    lights_out: str  # "HH:MM"
    first_light: str
    score: int
    disturbances: int
    fright: bool
    must_groups: tuple[tuple[str, ...], ...] = ()


# Keyword bars are deliberately loose: the contract is "one warm sentence from
# these facts, invent nothing", not a fixed phrasing. A good night needs any
# positive-quality word; a rough night needs SOME trace of the disturbance
# facts (a count, a duration, or a disturbance word).
SLEEP_CASES: list[SleepCase] = [
    SleepCase("good night", 701, "18:41", "06:22", 92, 0, False,
              must_groups=(("92", "well", "solid", "good", "great", "restful",
                            "peaceful", "quiet", "comfort", "undisturbed", "sound"),)),
    SleepCase("short night", 512, "20:10", "04:42", 61, 3, False,
              must_groups=(("61", "short", "disturb", "interrupt", "restless",
                            "movement", "stir", "3", "8"),)),
    SleepCase("night fright", 655, "19:02", "05:57", 55, 2, True,
              must_groups=(("fright", "startle", "disturb", "55", "rough", "stir",
                            "movement", "restless"),)),
]
