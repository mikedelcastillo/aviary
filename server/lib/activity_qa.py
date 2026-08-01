"""Answer activity questions from the persistent day memory (the journal).

``/activity``, "what did percy do today?", "what did percy and pizza do lately?"
all read the Markdown memory the caretaker keeps (:mod:`lib.journal`) — which
already holds VLM-written notes and the photos it looked at — and fold the
relevant entries into a <=3-sentence report, with a few of the best photos. If a
specific bird hasn't been logged recently and the question is about *now*, it
kicks off a live ``/find`` instead of shrugging.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable

from lib.activity import answer_activity_question, summarise_activity
from lib.ai.client import OllamaUnavailableError
from lib.clock import now_ph
from lib.find import pretty_phrase
from lib.suntimes import sun_times
from lib.journal import MemoryEntry, MemoryObservation, humanize_ago, load_recent
from lib.labels import pretty
from lib.roster import ALL_BIRD_WORDS, DEFAULT_SPECIES_MEMBERS, expand_targets


LOGGER = logging.getLogger("lib.activity_qa")

MAX_QA_PHOTOS = 4
# A plain recall answer ("what did percy do", "how is percy") gets ONE relevant
# photo alongside the text — an album is reserved for an explicit photo request.
RECALL_PHOTOS = 1
# Cap how many raw note lines reach the LLM. A busy day logs 100-200+ notes
# (~20k chars) — far too much for a small instruct model, which then drowns and
# rambles off-subject. The whole-day aggregate is already captured compactly in
# the structured facts, so the notes only need a representative temporal spread.
MAX_QA_NOTES = 30
SUMMARY_TIMEOUT_SECONDS = 60.0
CAPTION_LIMIT = 1024  # Telegram album/photo caption cap.

# Whole words that make an empty result trigger a live search instead of
# "haven't seen". Matched against word tokens (not substrings) so "now" doesn't
# fire inside "know"/"snow".
_LIVE_WORDS = frozenset({"now", "currently", "doing", "where"})


def _wants_live(text: str) -> bool:
    return bool(_LIVE_WORDS & set(re.findall(r"[a-z]+", text.lower())))

# A message starting with one of these (or containing "?") is treated as a
# specific question to answer, not a request for a generic activity summary.
_QUESTION_WORDS = (
    "did", "do", "does", "is", "are", "was", "were", "when", "what", "where",
    "who", "how", "has", "have", "had", "can", "could", "will", "should", "any",
)
_PHOTO_WORDS = frozenset({"photo", "photos", "picture", "pictures", "pic", "pics", "image", "images"})
_PHOTO_INTENT_WORDS = frozenset({"show", "send", "see", "get", "give"})


def _is_question(text: str) -> bool:
    stripped = text.strip().lstrip("/").lower()
    if "?" in stripped:
        return True
    first = stripped.split()[0] if stripped.split() else ""
    return first in _QUESTION_WORDS


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _is_photo_request(text: str) -> bool:
    words = _tokens(text)
    return bool(words & _PHOTO_WORDS) and bool(words & _PHOTO_INTENT_WORDS)


# Query words -> the v3 activity they should match, so "show me photos of birds
# eating" / "percy taking a bath" can filter the saved shots by what a bird was
# actually doing (the structured detection activity), not just who was present.
_ACTIVITY_QUERY_WORDS: dict[str, tuple[str, ...]] = {
    "feeding": ("eat", "eats", "eating", "feeding", "feed", "fed", "ate"),
    "drinking": ("drink", "drinking"),
    "preening": ("preen", "preening", "groom", "grooming"),
    "bathing": ("bath", "bathe", "bathing", "splash", "splashing", "bathtime"),
    "playing": ("play", "playing"),
    "sleeping": ("sleep", "sleeping", "asleep", "roosting", "roost"),
    "climbing": ("climb", "climbing", "hanging"),
    "flying": ("fly", "flying"),
    "vocalizing": ("sing", "singing", "chirp", "chirping", "vocalizing", "calling"),
    "exploring": ("explore", "exploring", "forage", "foraging"),
}


def _requested_activities(text: str) -> set[str]:
    """The v3 activities a photo query asks for ("eating" -> {feeding}), or empty."""
    words = _tokens(text)
    return {act for act, kws in _ACTIVITY_QUERY_WORDS.items() if words & set(kws)}


def _obs_matches_activity(obs, activities: set[str], targets: set[str] | None) -> bool:
    """True when a (target) bird in ``obs`` was doing one of the wanted activities."""
    for det in getattr(obs, "detections", None) or []:
        if targets and det.label not in targets:
            continue
        if det.activity in activities or (activities & set(det.tags or [])):
            return True
    return False


def _looks_like_pure_photo_request(text: str) -> bool:
    """True for requests that want images, not an activity explanation."""
    if not _is_photo_request(text):
        return False
    words = _tokens(text)
    analysis_words = {
        "what", "did", "doing", "up", "summary", "summarize", "tell",
        "happened", "today", "morning", "afternoon", "evening", "week",
        "yesterday", "together", "with",
    }
    return not bool(words & analysis_words)


def _explicit_individual_targets(text: str, argument: str, known_labels: list[str]) -> list[str]:
    """Named individual birds only, without species fallback labels.

    ``expand_targets("percy")`` intentionally adds ``lovebird`` so a live search
    can still match an IR species outline. For memory Q&A that broadens too much:
    it pulls every generic lovebird note into "what did Percy do?" answers. We
    first try exact individual names, then fall back to the broader expansion
    only when no exact memories exist.
    """
    words = _tokens(f"{argument} {text}")
    species = set(DEFAULT_SPECIES_MEMBERS)
    blocked = species | ALL_BIRD_WORDS | {"unknown_bird"}
    result: list[str] = []
    for label in known_labels:
        low = label.lower()
        if low in blocked:
            continue
        if low in words and low not in result:
            result.append(low)
    return result


def _resolved_bird_text(text: str, argument: str, known_labels: list[str]) -> tuple[str, list[str] | None]:
    """Resolve activity targets, falling back from router argument to user text.

    The LLM router can correctly pick ``activity`` but omit ``argument``. In that
    case, recover named birds/groups from the original message so "what did Percy
    do today?" never widens to all birds.
    """
    bird_text, _ = parse_activity_arg(argument)
    targets = expand_targets(bird_text, known_labels) if bird_text.strip() else []
    if not targets:
        text_targets = expand_targets(text, known_labels)
        if text_targets:
            words = _tokens(text)
            # "birds/everyone/any" intentionally means all birds, so keep the
            # display generic. Otherwise use the explicit bird/group tokens from
            # the message; don't pass the whole question through to captions.
            if not (words & ALL_BIRD_WORDS):
                known = {label.lower() for label in known_labels}
                groups = set(DEFAULT_SPECIES_MEMBERS)
                named = [
                    word for word in re.findall(r"[a-z]+", text.lower())
                    if word in known and word != "unknown_bird"
                ]
                if not named:
                    named = [word for word in words if word in groups]
                bird_text = bird_text or " ".join(named)
            targets = text_targets
    return bird_text, (targets if targets else None)


def _entry_observations(entry: MemoryEntry) -> list[MemoryObservation]:
    if entry.observations:
        return entry.observations
    # Legacy entry with no structured observations: synthesize one observation
    # PER saved photo so the photo-selection loop can surface all of them, not
    # just photos[0]. Keep a single no-photo observation when the entry has none.
    if not entry.photos:
        return [MemoryObservation(birds=entry.birds, note=entry.note, photo="")]
    return [
        MemoryObservation(birds=entry.birds, note=entry.note, photo=photo)
        for photo in entry.photos
    ]


def _cap_notes(notes: list[str], limit: int) -> list[str]:
    """Even temporal sample of the notes so a long day fits a small model's context.

    Keeps a representative spread across the window (and always the most recent
    note) rather than every line — the full-day aggregate lives in the structured
    facts, so the raw notes only need to illustrate, not enumerate.
    """
    if len(notes) <= limit:
        return notes
    step = len(notes) / limit
    sampled = [notes[int(i * step)] for i in range(limit)]
    sampled[-1] = notes[-1]  # always include the latest note
    return sampled


def _recall_name(label: str) -> str:
    """Render a bird label for a recall answer, resolving species-only detections.

    The detector sometimes IDs a bird only to species, not an individual. Per the
    user's choice we resolve when unique and group otherwise, never leaking a bare
    species word: budgie -> "Bambi" (its only member); cockatiel/lovebird ->
    "one of the cockatiels" / "one of the lovebirds"; an unidentified detection ->
    "an unidentified bird". A named individual passes through capitalised."""
    low = label.lower()
    members = DEFAULT_SPECIES_MEMBERS.get(low)
    if members:
        return pretty(members[0]) if len(members) == 1 else f"one of the {low}s"
    if low in ("unknown_bird", "bird"):
        return "an unidentified bird"
    return pretty(label)


def _bird_list(labels: list[str]) -> str:
    return ", ".join(_recall_name(b) for b in labels) if labels else "quiet"


def _activity_for_bird(obs, target: str) -> str:
    """The v3 structured activity recorded for ``target`` in ``obs``, or ""."""
    for det in getattr(obs, "detections", None) or []:
        if det.label == target and det.activity:
            return det.activity
    return ""


def _bird_activity_list(obs) -> str:
    """Birds in an observation with their structured activity when available
    ("Percy: preening, Matcha: resting"), else just the names. Giving the LLM the
    per-bird activity directly — instead of only a prose blob — is what makes v3
    recall accurate and on-subject."""
    dets = getattr(obs, "detections", None) or []
    if any(d.activity for d in dets):
        return ", ".join(
            f"{_recall_name(d.label)}: {d.activity}" if d.activity else _recall_name(d.label)
            for d in dets
        )
    return _bird_list(obs.birds)


def _note_lines(entries: list[MemoryEntry], now: datetime, targets: set[str] | None) -> list[str]:
    lines: list[str] = []
    for entry in entries:
        observations = _entry_observations(entry)
        relevant = [
            obs for obs in observations
            if not targets or any(b in targets for b in obs.birds)
        ]
        if not relevant and (not targets or any(b in targets for b in entry.birds)):
            relevant = [
                MemoryObservation(
                    birds=entry.birds,
                    note=entry.note,
                    photo=entry.photos[0] if entry.photos else "",
                )
            ]
        for obs in relevant:
            camera = f" @ {obs.camera}" if obs.camera else ""
            lines.append(
                f"({humanize_ago(entry.time, now)}) "
                f"[{_bird_activity_list(obs)}]{camera}: {obs.note}"
            )
    return lines


_ACTIVITY_TAG_WORDS: dict[str, tuple[str, ...]] = {
    "feeding/drinking": ("eat", "eating", "ate", "seed", "food", "bowl", "drink", "water"),
    "preening": ("preen", "preening", "groom", "grooming"),
    "resting/calm": ("rest", "resting", "nap", "napping", "sleep", "sleeping", "calm", "stationary", "settled"),
    "play/enrichment": ("play", "playing", "toy", "bell", "rope", "chew", "chewing"),
    "movement/exploration": ("move", "moving", "climb", "climbing", "hang", "hanging", "explore", "exploring", "fly", "flying", "walk", "walking"),
    "bathing": ("bath", "bathing", "splash", "splashing"),
    "social": ("together", "nearby", "beside", "side by side", "interacting", "with "),
    "alone": ("alone", "by herself", "by himself", "no other", "not interacting"),
}

_HEALTH_CONCERN_WORDS = (
    "injur", "limp", "droop", "letharg", "puffed", "fluffed", "labored",
    "breathing", "tail bob", "sick", "weak", "bleeding", "wound",
)


def _tags_for(text: str) -> set[str]:
    low = text.lower()
    tags: set[str] = set()
    for tag, words in _ACTIVITY_TAG_WORDS.items():
        if any(word in low for word in words):
            tags.add(tag)
    return tags


def _has_health_concern(text: str) -> bool:
    low = text.lower()
    return any(word in low for word in _HEALTH_CONCERN_WORDS)


def _clock_times(records: list[tuple[MemoryEntry, MemoryObservation]], limit: int = 5) -> str:
    seen: list[str] = []
    for entry, _ in records:
        stamp = entry.time.strftime("%H:%M")
        if stamp not in seen:
            seen.append(stamp)
        if len(seen) >= limit:
            break
    return ", ".join(seen) if seen else "none"


# Words that make a question a WHOLE-FLOCK comparison/exception ("who was most
# active", "any bird didn't eat", "any bird not doing well") rather than a
# single-bird lookup. These get a per-bird ranking table folded into the facts.
_FLOCK_WORDS = (
    "any bird", "any of them", "anyone", "which bird", "which one", "which of",
    "who ", "who's", "whos ", "each bird", "every bird", "everyone", "all birds",
    "all the birds", "most", "least", "busiest", "quietest", "laziest", "calmest",
    "compare", "rank", "more than", "less than",
)


def _is_flock_query(text: str) -> bool:
    low = f" {text.lower()} "
    return any(w in low for w in _FLOCK_WORDS)


def _flock_comparison(records, known_labels: list[str], question: str = "") -> str:
    """A per-bird comparison table (activity, social, health) for flock questions.

    Lets the LLM answer "who was most active", "any bird didn't eat", "any bird not
    doing well" from exact tallies rather than guessing. Absence of an activity means
    it was NOT tagged — not proof it never happened (the VLM misses some) — and the
    header says so, so the model hedges honestly.
    """
    individuals = {
        b for b in known_labels if b not in DEFAULT_SPECIES_MEMBERS and b != "unknown_bird"
    }
    seen: Counter = Counter()
    social: Counter = Counter()
    alone: Counter = Counter()
    activity: dict[str, Counter] = {}
    health: dict[str, Counter] = {}
    pair: Counter = Counter()
    for _entry, obs in records:
        dets = [d for d in (getattr(obs, "detections", None) or []) if d.label in individuals]
        present = {d.label for d in dets} or {b for b in obs.birds if b in individuals}
        for d in dets:
            if d.activity:
                activity.setdefault(d.label, Counter())[d.activity] += 1
            if d.health:
                health.setdefault(d.label, Counter())[d.health] += 1
        for b in present:
            seen[b] += 1
            (social if len(present) >= 2 else alone)[b] += 1
        ordered = sorted(present)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                pair[(ordered[i], ordered[j])] += 1
    if not seen:
        return ""
    lines = [
        "Per-bird tallies (a missing activity means it was NOT recorded, not proof it "
        "never happened):"
    ]
    for bird, count in seen.most_common():
        acts = ", ".join(f"{a} x{n}" for a, n in activity.get(bird, Counter()).most_common(6))
        h = ", ".join(f"{k} x{n}" for k, n in health.get(bird, Counter()).items())
        lines.append(
            f"  {pretty(bird)}: seen {count} (with others {social[bird]}, alone {alone[bird]}); "
            f"did: {acts or 'nothing tagged'}; health: {h or 'no concern seen'}"
        )
    missing = [pretty(b) for b in sorted(individuals) if b not in seen]
    if missing:
        lines.append(f"  Never seen this period: {', '.join(missing)}")
    # The top-pairs line only helps a togetherness question; otherwise it lures the
    # model into answering co-occurrence for an unrelated flock question.
    wants_pairs = any(w in question.lower() for w in ("together", "with each", "hung out", "beside", "interact", "pair", "closest"))
    if pair and wants_pairs:
        lines.append(
            "  Most time together: "
            + ", ".join(f"{pretty(x)}+{pretty(y)} {n}" for (x, y), n in pair.most_common(3))
        )
    return "\n".join(lines)


def _structured_facts(
    question: str,
    entries: list[MemoryEntry],
    now: datetime,
    targets: list[str],
    window_phrase: str,
    *,
    known_labels: list[str] | None = None,
) -> str:
    if not entries:
        return ""
    target_order = []
    for target in targets:
        if target and target not in target_order:
            target_order.append(target)

    records: list[tuple[MemoryEntry, MemoryObservation]] = [
        (entry, obs)
        for entry in entries
        for obs in _entry_observations(entry)
    ]
    if not records:
        return ""

    lines = [f"Over {window_phrase or 'the requested time'}:"]

    # Whole-flock comparison for "who was most active" / "any bird didn't eat" /
    # "any bird not doing well" — a per-bird table the LLM ranks/scans over.
    if known_labels and _is_flock_query(question):
        comparison = _flock_comparison(records, known_labels, question)
        if comparison:
            lines.append(comparison)

    individual_targets = [
        target for target in target_order
        if target not in DEFAULT_SPECIES_MEMBERS and target != "unknown_bird"
    ]
    pair = individual_targets[:2]
    if len(pair) == 2:
        a, b = pair
        together = [
            (entry, obs) for entry, obs in records
            if a in obs.birds and b in obs.birds
        ]
        apart = [
            (entry, obs) for entry, obs in records
            if (a in obs.birds) ^ (b in obs.birds)
        ]
        a_only = sum(1 for _, obs in apart if a in obs.birds)
        b_only = sum(1 for _, obs in apart if b in obs.birds)
        separate_same_report: list[tuple[MemoryEntry, MemoryObservation]] = []
        for entry in entries:
            obs = _entry_observations(entry)
            saw_a = any(a in o.birds for o in obs)
            saw_b = any(b in o.birds for o in obs)
            same_view = any(a in o.birds and b in o.birds for o in obs)
            if saw_a and saw_b and not same_view:
                separate_same_report.append((entry, obs[0]))
        # An explicit verdict the LLM must not contradict — small models otherwise
        # answer "no" to "did they spend time together" whenever the pair was ALSO
        # apart a lot, even with hundreds of shared moments.
        if together:
            lines.append(
                f"VERDICT: YES, {pretty(a)} and {pretty(b)} DID spend time together "
                f"({len(together)} shared moments, e.g. {_clock_times(together)}) — even "
                f"though they were often apart too. Any 'were they together' answer is YES."
            )
        else:
            lines.append(
                f"VERDICT: NO, {pretty(a)} and {pretty(b)} were never seen together."
            )
        lines.append(
            f"{pretty(a)} + {pretty(b)} same-frame/view observations: "
            f"{len(together)} ({_clock_times(together)})."
        )
        lines.append(
            f"{pretty(a)} + {pretty(b)} apart/only-one observations: {len(apart)} "
            f"({pretty(a)} only {a_only}, {pretty(b)} only {b_only}; "
            f"separate views in same report {len(separate_same_report)} at "
            f"{_clock_times(separate_same_report)})."
        )

    profile_targets = individual_targets or target_order
    for target in profile_targets[:4]:
        seen = [
            (entry, obs) for entry, obs in records
            if target in obs.birds
        ]
        if not seen:
            continue
        first = min(entry.time for entry, _ in seen).strftime("%H:%M")
        last = max(entry.time for entry, _ in seen).strftime("%H:%M")
        with_others = sum(1 for _, obs in seen if len(obs.birds) >= 2)
        alone = sum(
            1 for _, obs in seen
            if len(obs.birds) == 1 or "alone" in obs.note.lower()
        )
        tag_counts: Counter[str] = Counter()
        health_examples: list[str] = []
        for _, obs in seen:
            # Prefer the v3 structured activity recorded for THIS bird; fall back to
            # parsing the prose note only when the observation predates the schema.
            structured = _activity_for_bird(obs, target)
            if structured:
                tag_counts[structured] += 1
            else:
                tag_counts.update(_tags_for(obs.note))
            det_health = next(
                (d.health for d in (getattr(obs, "detections", None) or [])
                 if d.label == target and d.health),
                "",
            )
            if (det_health or _has_health_concern(obs.note)) and len(health_examples) < 3:
                health_examples.append(det_health or obs.note)
        tag_text = (
            ", ".join(f"{tag} x{count}" for tag, count in tag_counts.most_common(5))
            if tag_counts else "no specific activity tags"
        )
        health = (
            "; ".join(health_examples)
            if health_examples else "no explicit health-concern words recorded"
        )
        lines.append(
            f"{pretty(target)}: {len(seen)} observations from {first} to {last}; "
            f"with other birds {with_others}, alone/solo {alone}; "
            f"activity tags: {tag_text}; health: {health}."
        )

    return "\n".join(lines)


def _photo_caption(bird_text: str, window_phrase: str, count: int) -> str:
    who = pretty_phrase(bird_text) if bird_text.strip() else "the birds"
    noun = "photo" if count == 1 else "photos"
    phrase = window_phrase.removeprefix("in ")
    return f"{count} {noun} of {who} from {phrase}."


def parse_activity_arg(argument: str) -> tuple[str, bool]:
    """Split a /activity argument into (bird text, today?). "percy today" -> ("percy", True)."""
    tokens = argument.strip().split()
    today = any(t.lower() in ("today", "day") for t in tokens)
    birds = " ".join(t for t in tokens if t.lower() not in ("today", "day"))
    return birds.strip(), today


class ActivityResponder:
    def __init__(
        self,
        memories_dir,
        client,
        llm_model: str,
        known_labels: Callable[[], list[str]],
        *,
        notify: Callable[[int, str], None],
        send_album: Callable[[int, list[tuple[bytes, str | None]]], object] | None = None,
        find: Callable[[int, str], None] | None = None,
        pronoun_note: str = "",
        now: Callable[[], datetime] = now_ph,
        fallback_model: str = "",
        latitude: float = 0.0,
        longitude: float = 0.0,
        utc_offset_hours: float = 0.0,
        health=None,
        stream_factory: Callable[[int], object] | None = None,
    ) -> None:
        self._memories_dir = Path(memories_dir)
        self._client = client
        self._llm_model = llm_model
        # A cheaper/smaller model to fall back to if the big recall model is down or
        # unavailable on the cluster (the main node isn't always up) — the answer
        # degrades in quality but the user still gets a real reply.
        self._fallback_model = fallback_model
        self._known_labels = known_labels
        self._notify = notify
        self._send_album = send_album
        self._find = find
        self._pronoun_note = pronoun_note
        self._now = now
        # Optional lib.ai.health.OllamaHealth: skip a model the cluster doesn't
        # serve right now (instead of a doomed 60s request), and queue the
        # question for replay when the whole cluster is down.
        self._health = health
        # Set by main after the NL router exists (they reference each other):
        # called with (chat_id, text) when NO model could answer because the
        # cluster is unreachable — the question is saved + replayed later.
        self._on_unavailable: Callable[[int, str], None] | None = None
        # Whether the CURRENT thread is answering a replayed (queued) message —
        # suppresses the live-camera find fallback for stale questions.
        self._replaying: Callable[[], bool] | None = None
        # Location for anchoring "when the sun rose / at sunset" activity windows.
        self._latitude = latitude
        self._longitude = longitude
        self._utc_offset_hours = utc_offset_hours
        # Optional lib.telegram.stream.StreamedReply factory: when set, recall
        # answers stream into a growing message (recall runs on the BIG model,
        # whose full answer can take tens of seconds — the first words matter).
        self._stream_factory = stream_factory

    def set_on_unavailable(self, callback: Callable[[int, str], None]) -> None:
        """Late-bind the queue-for-replay hook (the router is built after us)."""
        self._on_unavailable = callback

    def set_replaying(self, callback: Callable[[], bool]) -> None:
        """Late-bind the router's is-this-a-replay check (built after us)."""
        self._replaying = callback

    def _sun_anchor(self, day: date, *, rise: bool) -> time:
        """Local sunrise (or sunset) time for ``day`` at the configured location,
        falling back to a 6am/6pm clock anchor when the location is unset."""
        sunrise, sunset = sun_times(
            day, self._latitude, self._longitude, self._utc_offset_hours
        )
        anchor = sunrise if rise else sunset
        if anchor is not None:
            return anchor
        return time(6, 0) if rise else time(18, 0)

    def _window(self, text: str, argument: str, now: datetime, question: bool) -> tuple[datetime, datetime, str]:
        """Resolve the lookback window ``(since, until, phrase)``, honouring
        time-of-day granularity.

        Explicit phrasing wins (morning/afternoon/evening/today/last hour).
        Otherwise a question defaults to the whole day ("did pizza eat?" means
        today), while a bare /activity defaults to the last hour. Windows are
        clamped so they never extend past ``now``.
        """
        t = f"{text} {argument}".lower()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

        def window(since: datetime, until: datetime, phrase: str) -> tuple[datetime, datetime, str]:
            # A period that hasn't started yet today (since >= until) would be an
            # inverted, always-empty range — fall back to the whole day instead.
            if since >= until:
                return midnight, now, "today"
            return since, until, phrase

        # Anchored to the sun: "when the sun rose", "at sunrise", "as the sun set".
        # Computed from the location's actual sunrise/sunset, not a clock guess.
        sun_rise = any(p in t for p in ("sunrise", "sun rose", "sun rise", "sun came up", "sun come up", "sun coming up"))
        sun_set = any(p in t for p in ("sunset", "sun set", "sun went down", "sun going down", "sun down"))
        if sun_rise or sun_set:
            day = (midnight - timedelta(days=1)).date() if "yesterday" in t else midnight.date()
            anchor = datetime.combine(day, self._sun_anchor(day, rise=sun_rise or not sun_set))
            if sun_rise:
                return window(anchor - timedelta(minutes=45), min(anchor + timedelta(minutes=75), now), "around sunrise")
            return window(anchor - timedelta(minutes=75), min(anchor + timedelta(minutes=45), now), "around sunset")

        # "last night" is the PREVIOUS evening through this morning — handled
        # before the bare "night" branch so it isn't mistaken for this evening.
        if "last night" in t or "overnight" in t:
            return window(
                midnight - timedelta(days=1) + timedelta(hours=18),
                min(midnight + timedelta(hours=6), now),
                "last night",
            )
        if "morning" in t:
            return window(midnight.replace(hour=5), min(midnight.replace(hour=12), now), "this morning")
        if "afternoon" in t:
            return window(midnight.replace(hour=12), min(midnight.replace(hour=17), now), "this afternoon")
        if "evening" in t or "tonight" in t or "night" in t:
            return window(midnight.replace(hour=17), now, "this evening")
        if "week" in t:
            return midnight - timedelta(days=6), now, "this week"
        if "yesterday" in t:
            return midnight - timedelta(days=1), midnight, "yesterday"
        if "today" in t or "all day" in t or "whole day" in t or "so far" in t:
            return midnight, now, "today"
        if "hour" in t or "recently" in t or "just now" in t or "lately" in t:
            return now - timedelta(hours=1), now, "in the last hour"
        # Questions and photo/together requests mean the whole day; only a bare
        # /activity summary defaults to the last hour.
        broad = question or any(
            w in t for w in ("photo", "picture", "show", "together", "with ", "spend")
        )
        if broad:
            return midnight, now, "today"
        return now - timedelta(hours=1), now, "in the last hour"

    def respond(self, chat_id: int, text: str, argument: str) -> None:
        known_labels = self._known_labels()
        bird_text, targets = _resolved_bird_text(text, argument, known_labels)
        exact_targets = _explicit_individual_targets(text, argument, known_labels)
        now = self._now()
        question = _is_question(text)
        pure_photo_request = _looks_like_pure_photo_request(text)
        since, until, window_phrase = self._window(text, argument, now, question)

        # A whole-flock question ("who was most active", "any bird didn't eat") must
        # look at EVERY bird, so ignore any single-bird target the router guessed.
        flock_query = _is_flock_query(text)
        filter_targets = set() if flock_query else set(exact_targets or targets or [])
        note_targets: set[str] | None = filter_targets if filter_targets else None
        entries = load_recent(
            self._memories_dir, since, until, note_targets
        )
        used_broad_fallback = False
        if not entries and exact_targets and targets:
            broad_targets = set(targets)
            if broad_targets != set(exact_targets):
                entries = load_recent(self._memories_dir, since, until, broad_targets)
                if entries:
                    note_targets = broad_targets
                    used_broad_fallback = True
        if not entries:
            who = pretty_phrase(bird_text) if bird_text.strip() else "the birds"
            # No live camera search on a REPLAYED question: "what is percy doing
            # right now?" asked during an outage must not fire a PTZ patrol hours
            # later (the same reason `find` isn't a replay-safe action).
            live_ok = self._replaying is None or not self._replaying()
            if targets and self._find is not None and _wants_live(text) and live_ok:
                self._notify(chat_id, f"I haven't logged {who} {window_phrase} — let me check the cameras…")
                self._find(chat_id, bird_text)
                return
            self._notify(chat_id, f"I haven't logged any activity for {who} {window_phrase}.")
            return

        # Each note carries when it happened, which birds were seen, and (for new
        # memories) the specific camera observation rather than only the report's
        # aggregate bird list. This keeps "what did Percy do?" centered on Percy
        # even when the report also saw other birds.
        notes = _cap_notes(_note_lines(entries, now, note_targets), MAX_QA_NOTES)
        analysis_targets = exact_targets or (targets or [])
        facts = _structured_facts(
            text, entries, now, analysis_targets, window_phrase, known_labels=known_labels
        )
        if used_broad_fallback:
            fallback_line = (
                "Exact individual sightings were not logged; broader species/group "
                "fallback notes were included and should be described as less certain."
            )
            facts = f"{facts}\n{fallback_line}" if facts else fallback_line
        # Stream the answer into a growing message when the notifier can edit
        # (a photo-only request has no LLM text to stream). Every path below
        # must end in deliver()/finalize so the registry slot is released.
        stream = (
            self._stream_factory(chat_id)
            if self._stream_factory is not None and not pure_photo_request
            else None
        )

        def deliver(message: str) -> None:
            if stream is not None:
                stream.finalize(message)
            elif message:
                self._notify(chat_id, message)

        def _ask(model: str) -> str:
            if pure_photo_request:
                return ""
            on_partial = stream.update if stream is not None else None
            cancelled = stream.cancelled if stream is not None else None
            if question:
                return answer_activity_question(
                    self._client, model, text, notes, self._pronoun_note, window_phrase,
                    facts=facts, timeout_seconds=SUMMARY_TIMEOUT_SECONDS,
                    on_partial=on_partial, cancelled=cancelled,
                )
            subject = pretty_phrase(bird_text) if bird_text.strip() else ""
            return summarise_activity(
                self._client, model, notes, subject, self._pronoun_note,
                timeout_seconds=SUMMARY_TIMEOUT_SECONDS,
                on_partial=on_partial, cancelled=cancelled,
            )

        # Try the big recall model, then fall back to the small one if it's down /
        # unavailable on the cluster — a slightly weaker answer beats no answer.
        summary = ""
        unavailable = False
        asked_any = False
        for model in (self._llm_model, self._fallback_model):
            if not model or (model == self._llm_model and pure_photo_request):
                continue
            if self._health is not None and not self._health.has_model(model):
                # The cluster doesn't serve this model right now (e.g. only the
                # small worker is up) — skip straight to the fallback instead of
                # burning a doomed 60s request.
                LOGGER.info("Skipping %s for recall (not served by the cluster)", model)
                continue
            try:
                asked_any = True
                summary = _ask(model)
                if summary or pure_photo_request:
                    break
            except OllamaUnavailableError:
                unavailable = True
                LOGGER.warning("Activity answer skipped on %s: Ollama down", model)
            except Exception:
                LOGGER.warning("Activity answer failed on %s; trying fallback", model, exc_info=True)
        if (
            (unavailable or not asked_any)  # every candidate down or unserved
            and not summary
            and not pure_photo_request
            and self._on_unavailable is not None
        ):
            # No model could run at all — save the question for replay instead of
            # degrading to a raw journal line the user didn't ask for.
            if stream is not None:
                stream.finalize("")
            self._on_unavailable(chat_id, text)
            return
        # A newer message superseded this answer mid-stream: freeze whatever
        # partial is on screen and stop — the newer turn is being answered.
        if stream is not None and stream.cancelled():
            stream.finalize("")
            return
        summary = summary or ("" if pure_photo_request else notes[-1])

        # Pick the photos that go with the answer. For a "together"/"with other
        # birds" request, prefer moments where two or more birds were seen at once
        # (the actual together-shots); otherwise the most recent relevant photos.
        wants_apart = any(
            phrase in text.lower()
            for phrase in ("apart", "separate", "separately", "away from each other")
        )
        wants_together = not wants_apart and any(
            phrase in text.lower()
            for phrase in ("together", "with each other", "with other", "spend", "spending",
                           "with ", "beside", "next to", "side by side")
        )
        # Only filter photos by activity for an actual PHOTO request. A plain
        # question like "any bird that didn't play today?" mentions an activity but
        # must be ANSWERED (from the tallies), not turned into a photo search.
        want_acts = _requested_activities(text) if _is_photo_request(text) else set()
        # For "bambi and percy beside each other", require BOTH named birds in the
        # same shot — not merely two birds, one of which is a target.
        pair_targets = set(exact_targets) if wants_together and len(exact_targets) >= 2 else set()
        # An explicit photo request gets a small album; a plain recall answer gets
        # just one relevant photo tucked alongside the text.
        photo_limit = MAX_QA_PHOTOS if pure_photo_request else RECALL_PHOTOS
        chosen: list[str] = []
        seen: set[str] = set()
        for entry in reversed(entries):
            for obs in _entry_observations(entry):
                if note_targets and obs.birds and not any(b in note_targets for b in obs.birds):
                    continue
                if pair_targets and not pair_targets.issubset(set(obs.birds)):
                    continue
                if wants_together and not pair_targets and len(obs.birds) < 2:
                    continue
                if want_acts and not _obs_matches_activity(obs, want_acts, note_targets):
                    continue
                photos = [obs.photo] if obs.photo else entry.photos
                for photo in photos:
                    if photo and photo not in seen and Path(photo).exists():
                        seen.add(photo)
                        chosen.append(photo)
                    if len(chosen) >= photo_limit:
                        break
                if len(chosen) >= photo_limit:
                    break
            if len(chosen) >= photo_limit:
                break

        # Send photos as an album, but keep the activity answer as text. Telegram
        # captions are capped at 1024 chars; sending the full answer separately
        # prevents "what did Percy do today?" from being cut off. A plain recall
        # photo needs no "N photos of X" caption — the text answer is the message;
        # that caption is only for an explicit photo request.
        if self._send_album is not None and chosen:
            items: list[tuple[bytes, str | None]] = []
            caption = _photo_caption(bird_text, window_phrase, len(chosen)) if pure_photo_request else None
            if caption and len(caption) > CAPTION_LIMIT:
                caption = caption[: CAPTION_LIMIT - 1] + "…"
            for index, photo in enumerate(reversed(chosen)):
                try:
                    items.append((Path(photo).read_bytes(), caption if index == 0 else None))
                except Exception:
                    LOGGER.exception("Reading activity photo failed")
            if items:
                deliver(summary)
                self._send_album(chat_id, items)
                return
        # No photos (or no album sender). For an activity/together photo search that
        # matched nothing, say so specifically rather than "found activity but no
        # photos" — the point of the query was the filter.
        who = pretty_phrase(bird_text) if bird_text.strip() else "the birds"
        if not chosen and want_acts:
            deliver(f"I don't have any saved photos of {who} {' or '.join(sorted(want_acts))} {window_phrase}.")
            return
        if not chosen and pair_targets:
            deliver(f"I don't have any saved photos of {who} together {window_phrase}.")
            return
        deliver(summary or f"I found activity for {who} {window_phrase}, but no saved photos to show.")
