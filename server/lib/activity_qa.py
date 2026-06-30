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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from lib.activity import answer_activity_question, summarise_activity
from lib.clock import now_ph
from lib.find import pretty_phrase
from lib.journal import MemoryEntry, MemoryObservation, humanize_ago, load_recent
from lib.labels import pretty
from lib.roster import ALL_BIRD_WORDS, DEFAULT_SPECIES_MEMBERS, expand_targets


LOGGER = logging.getLogger("lib.activity_qa")

MAX_QA_PHOTOS = 4
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


def _bird_list(labels: list[str]) -> str:
    return ", ".join(pretty(b) for b in labels) if labels else "quiet"


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
                f"[{_bird_list(obs.birds)}]{camera}: {obs.note}"
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


def _structured_facts(
    question: str,
    entries: list[MemoryEntry],
    now: datetime,
    targets: list[str],
    window_phrase: str,
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

    lines = [f"Window: {window_phrase or 'the requested time'}."]

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
            tag_counts.update(_tags_for(obs.note))
            if _has_health_concern(obs.note) and len(health_examples) < 3:
                health_examples.append(obs.note)
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
        care_answer: Callable[[str], str | None] | None = None,
    ) -> None:
        self._memories_dir = Path(memories_dir)
        self._client = client
        self._llm_model = llm_model
        self._known_labels = known_labels
        self._notify = notify
        self._send_album = send_album
        self._find = find
        self._pronoun_note = pronoun_note
        self._now = now
        # Given the message text, returns a grounded care answer if it's actually
        # a care question (else None) — so a care Q that got routed here instead of
        # to chat still gets a real answer rather than "I haven't logged that".
        self._care_answer = care_answer

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

        filter_targets = set(exact_targets or targets or [])
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
            # A care question routed to the activity path ("is it too cold for percy
            # now?") is answered from care knowledge FIRST — before the live-find
            # branch, so a care question carrying a live word ("now") isn't sent off
            # to the cameras instead of being answered.
            if self._care_answer is not None:
                try:
                    answer = self._care_answer(text)
                except Exception:
                    LOGGER.exception("Care fallback failed")
                    answer = None
                if answer:
                    self._notify(chat_id, answer)
                    return
            if targets and self._find is not None and _wants_live(text):
                self._notify(chat_id, f"I haven't logged {who} {window_phrase} — let me check the cameras…")
                self._find(chat_id, bird_text)
                return
            self._notify(chat_id, f"I haven't logged any activity for {who} {window_phrase}.")
            return

        # Each note carries when it happened, which birds were seen, and (for new
        # memories) the specific camera observation rather than only the report's
        # aggregate bird list. This keeps "what did Percy do?" centered on Percy
        # even when the report also saw other birds.
        notes = _note_lines(entries, now, note_targets)
        analysis_targets = exact_targets or (targets or [])
        facts = _structured_facts(text, entries, now, analysis_targets, window_phrase)
        if used_broad_fallback:
            fallback_line = (
                "Exact individual sightings were not logged; broader species/group "
                "fallback notes were included and should be described as less certain."
            )
            facts = f"{facts}\n{fallback_line}" if facts else fallback_line
        try:
            if pure_photo_request:
                summary = ""
            elif question:
                summary = answer_activity_question(
                    self._client, self._llm_model, text, notes,
                    self._pronoun_note, window_phrase, facts=facts,
                    timeout_seconds=SUMMARY_TIMEOUT_SECONDS,
                )
            else:
                subject = pretty_phrase(bird_text) if bird_text.strip() else ""
                summary = summarise_activity(
                    self._client, self._llm_model, notes, subject,
                    self._pronoun_note, timeout_seconds=SUMMARY_TIMEOUT_SECONDS,
                )
        except Exception:
            LOGGER.exception("Activity response failed")
            summary = ""
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
            for phrase in ("together", "with each other", "with other", "spend", "spending", "with ")
        )
        chosen: list[str] = []
        seen: set[str] = set()
        for entry in reversed(entries):
            for obs in _entry_observations(entry):
                if note_targets and obs.birds and not any(b in note_targets for b in obs.birds):
                    continue
                if wants_together and len(obs.birds) < 2:
                    continue
                photos = [obs.photo] if obs.photo else entry.photos
                for photo in photos:
                    if photo and photo not in seen and Path(photo).exists():
                        seen.add(photo)
                        chosen.append(photo)
                    if len(chosen) >= MAX_QA_PHOTOS:
                        break
                if len(chosen) >= MAX_QA_PHOTOS:
                    break
            if len(chosen) >= MAX_QA_PHOTOS:
                break

        # Send photos as an album, but keep the activity answer as text. Telegram
        # captions are capped at 1024 chars; sending the full answer separately
        # prevents "what did Percy do today?" from being cut off.
        if self._send_album is not None and chosen:
            items: list[tuple[bytes, str | None]] = []
            caption = _photo_caption(bird_text, window_phrase, len(chosen))
            if len(caption) > CAPTION_LIMIT:
                caption = caption[: CAPTION_LIMIT - 1] + "…"
            for index, photo in enumerate(reversed(chosen)):
                try:
                    items.append((Path(photo).read_bytes(), caption if index == 0 else None))
                except Exception:
                    LOGGER.exception("Reading activity photo failed")
            if items:
                if summary:
                    self._notify(chat_id, summary)
                self._send_album(chat_id, items)
                return
        # No photos (or no album sender) — just the text.
        self._notify(chat_id, summary or f"I found activity for {pretty_phrase(bird_text) if bird_text.strip() else 'the birds'} {window_phrase}, but no saved photos to show.")
