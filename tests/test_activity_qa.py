from __future__ import annotations

from datetime import datetime

from lib.activity_qa import (
    ActivityResponder,
    _obs_matches_activity,
    _requested_activities,
    parse_activity_arg,
)
from lib.journal import MemoryDetection, MemoryEntry, MemoryObservation, append_entry

KNOWN = ["bambi", "draft", "matcha", "percy", "pizza"]


def test_requested_activities_maps_query_words() -> None:
    assert _requested_activities("show me photos of birds eating") == {"feeding"}
    assert _requested_activities("a picture of percy taking a bath") == {"bathing"}
    assert _requested_activities("photos of birds playing") == {"playing"}
    assert _requested_activities("show me photos of percy") == set()  # no activity word


def test_obs_matches_activity_uses_structured_detections() -> None:
    obs = MemoryObservation(
        birds=["percy", "matcha"],
        detections=[
            MemoryDetection(label="percy", activity="feeding"),
            MemoryDetection(label="matcha", activity="resting"),
        ],
    )
    assert _obs_matches_activity(obs, {"feeding"}, None)  # some bird is feeding
    assert not _obs_matches_activity(obs, {"bathing"}, None)  # nobody bathing
    assert _obs_matches_activity(obs, {"feeding"}, {"percy"})  # percy specifically
    assert not _obs_matches_activity(obs, {"feeding"}, {"matcha"})  # matcha isn't feeding


class FakeClient:
    def __init__(self) -> None:
        self.system = ""
        self.user = ""

    def chat(self, model, messages, **kwargs):
        self.system = messages[0]["content"]
        self.user = messages[1]["content"]
        return "Percy preened on the perch while Matcha napped nearby."


class BoomClient:
    def chat(self, model, messages, **kwargs):
        raise AssertionError("pure photo requests should not call the LLM")


def _responder(memories_dir, notify, find=None, send_album=None, now=None, client=None):
    # Fix "now" so the window is deterministic.
    when = now or datetime(2026, 6, 25, 15, 0)
    return ActivityResponder(
        memories_dir,
        client or FakeClient(),
        "gemma3:4b",
        lambda: KNOWN,
        notify=notify,
        send_album=send_album,
        find=find,
        pronoun_note="Percy and Bambi are female (use she/her for them).",
        now=lambda: when,
    )


def test_parse_activity_arg() -> None:
    assert parse_activity_arg("percy today") == ("percy", True)
    assert parse_activity_arg("today") == ("", True)
    assert parse_activity_arg("percy") == ("percy", False)
    assert parse_activity_arg("") == ("", False)


def test_respond_summarises_recent_memory(tmp_path) -> None:
    photo = tmp_path / "p.jpg"
    photo.write_bytes(b"\xff\xd8jpeg")
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 14, 40), ["percy", "matcha"], "Percy preens.", [str(photo)]))
    sent: list = []
    albums: list = []
    _responder(tmp_path, lambda c, t: sent.append(t), send_album=lambda c, items: albums.append(items)).respond(
        7, "/activity percy", "percy"
    )
    # Full summary is sent as text, and the album caption stays short so it
    # cannot truncate the activity answer at Telegram's caption limit.
    assert sent and "preened" in sent[0]
    assert len(albums) == 1
    items = albums[0]
    assert len(items) == 1  # a plain recall answer gets ONE relevant photo…
    assert items[0][1] is None  # …with no caption — the text answer is the message


def test_respond_recovers_bird_from_text_when_router_argument_empty(tmp_path) -> None:
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 8, 0), ["bambi"], "Bambi ate early."))
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 14, 0), ["percy"], "Percy napped on the perch."))
    client = FakeClient()
    _responder(tmp_path, lambda c, t: None, client=client).respond(
        7, "what did percy do today?", ""
    )
    assert "Percy napped" in client.user
    assert "Bambi ate" not in client.user


def test_individual_question_uses_relevant_structured_observation(tmp_path) -> None:
    append_entry(
        tmp_path,
        MemoryEntry(
            datetime(2026, 6, 25, 14, 0),
            ["percy", "matcha"],
            "two-camera report",
            observations=[
                MemoryObservation("Big Cage", ["percy"], "Percy preened alone."),
                MemoryObservation("Desk", ["matcha"], "Matcha chewed a toy."),
            ],
        ),
    )
    client = FakeClient()
    _responder(tmp_path, lambda c, t: None, client=client).respond(
        7, "what did percy do today?", ""
    )
    assert "Percy preened alone" in client.user
    assert "Matcha chewed" not in client.user


def test_pair_apart_question_includes_structured_counts(tmp_path) -> None:
    append_entry(
        tmp_path,
        MemoryEntry(
            datetime(2026, 6, 25, 9, 0),
            ["draft"],
            "draft only",
            observations=[MemoryObservation("Play Gym", ["draft"], "Draft perched alone.")],
        ),
    )
    append_entry(
        tmp_path,
        MemoryEntry(
            datetime(2026, 6, 25, 10, 0),
            ["pizza"],
            "pizza only",
            observations=[MemoryObservation("Cage", ["pizza"], "Pizza ate seeds alone.")],
        ),
    )
    append_entry(
        tmp_path,
        MemoryEntry(
            datetime(2026, 6, 25, 11, 0),
            ["draft", "pizza"],
            "together",
            observations=[MemoryObservation("Cage", ["draft", "pizza"], "Draft and Pizza stood side by side.")],
        ),
    )
    append_entry(
        tmp_path,
        MemoryEntry(
            datetime(2026, 6, 25, 12, 0),
            ["draft", "pizza"],
            "separate views",
            observations=[
                MemoryObservation("Play Gym", ["draft"], "Draft climbed on the play gym."),
                MemoryObservation("Cage", ["pizza"], "Pizza rested in the cage."),
            ],
        ),
    )
    client = FakeClient()
    _responder(tmp_path, lambda c, t: None, client=client).respond(
        7, "Did draft and pizza spend time apart today?", ""
    )
    assert "Draft + Pizza same-frame/view observations: 1" in client.user
    assert "Draft + Pizza apart/only-one observations: 4" in client.user
    assert "separate views in same report 1 at 12:00" in client.user


def test_pure_photo_request_sends_photos_without_llm_reasoning(tmp_path) -> None:
    photo = tmp_path / "percy.jpg"
    photo.write_bytes(b"percy-bytes")
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 14, 0), ["percy"], "Percy napped on the perch.", [str(photo)]))
    sent: list = []
    albums: list = []
    _responder(
        tmp_path,
        lambda c, t: sent.append(t),
        send_album=lambda c, items: albums.append(items),
        client=BoomClient(),
    ).respond(7, "show me picture of percy", "")
    assert sent == []
    assert len(albums) == 1
    assert albums[0][0] == (b"percy-bytes", "1 photo of Percy from today.")


def test_respond_today_window(tmp_path) -> None:
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 8, 0), ["bambi"], "Bambi ate early."))
    sent: list = []
    # "today" -> since midnight, so the 08:00 entry is included even though it's
    # well outside the last hour.
    _responder(tmp_path, lambda c, t: sent.append(t)).respond(7, "/activity bambi today", "bambi today")
    assert sent  # a summary was produced from the morning entry


def test_respond_triggers_find_when_unlogged_and_live(tmp_path) -> None:
    found: list = []
    sent: list = []
    _responder(tmp_path, lambda c, t: sent.append(t), find=lambda cid, arg: found.append((cid, arg))).respond(
        7, "what is draft doing right now?", "draft"
    )
    assert found == [(7, "draft")]
    assert any("check the cameras" in t for t in sent)


def test_respond_says_unlogged_when_not_live(tmp_path) -> None:
    sent: list = []
    _responder(tmp_path, lambda c, t: sent.append(t), find=lambda cid, arg: None).respond(
        7, "/activity matcha", "matcha"
    )
    assert sent and "haven't logged" in sent[0].lower()


def test_question_uses_qa_path_with_birds_and_question(tmp_path) -> None:
    # Whole-day co-occurrence question -> QA prompt, notes carry each entry's birds.
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 9, 0), ["jynx", "matcha"], "Together on the perch."))
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 13, 0), ["pizza"], "Pizza alone."))
    client = FakeClient()
    _responder(tmp_path, lambda c, t: None, client=client).respond(
        7, "did jynx and matcha spend time together today?", "jynx and matcha"
    )
    # The QA system prompt (not the bullet-summary one) was used.
    assert "answering one question" in client.system.lower()
    # The user payload includes the question and the per-entry bird list.
    assert "spend time together" in client.user
    assert "Jynx, Matcha" in client.user


def test_question_defaults_to_whole_day_window(tmp_path) -> None:
    # "did pizza eat" (no "today") still looks at the whole day, not just last hour.
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 8, 30), ["pizza"], "Pizza at the food bowl."))
    client = FakeClient()
    sent: list = []
    _responder(tmp_path, lambda c, t: sent.append(t), client=client).respond(
        7, "did pizza eat?", "pizza"
    )
    assert "food bowl" in client.user  # the 08:30 entry was in-window
    assert "from today" in client.user


def test_morning_window_excludes_afternoon(tmp_path) -> None:
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 8, 0), ["bambi"], "Morning bath."))
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 14, 0), ["bambi"], "Afternoon nap."))
    client = FakeClient()
    _responder(tmp_path, lambda c, t: None, client=client).respond(
        7, "what did bambi do this morning?", "bambi"
    )
    assert "Morning bath" in client.user
    assert "Afternoon nap" not in client.user
    assert "this morning" in client.user


def test_together_request_prefers_multi_bird_photos(tmp_path) -> None:
    duo = tmp_path / "duo.jpg"; duo.write_bytes(b"duo-bytes")
    solo = tmp_path / "solo.jpg"; solo.write_bytes(b"solo-bytes")
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 9, 0), ["bambi", "jynx"], "Together.", [str(duo)]))
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 13, 0), ["bambi"], "Bambi alone.", [str(solo)]))
    albums: list = []
    _responder(tmp_path, lambda c, t: None, send_album=lambda c, items: albums.append(items)).respond(
        7, "show me photos of bambi with other birds", "bambi"
    )
    assert albums
    sent = [img for img, _ in albums[0]]
    assert duo.read_bytes() in sent and solo.read_bytes() not in sent


def test_week_window_includes_earlier_days(tmp_path) -> None:
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 22, 9, 0), ["jynx", "bambi"], "Earlier this week together."))
    client = FakeClient()
    _responder(tmp_path, lambda c, t: None, client=client).respond(
        7, "has bambi spent time with jynx this week?", "bambi and jynx"
    )
    assert "Earlier this week" in client.user


def test_evening_window_before_5pm_falls_back_to_today(tmp_path) -> None:
    # Asked at 10am, "tonight" would be an inverted 17:00->10:00 range; must not
    # invert — fall back to the whole day.
    r = _responder(tmp_path, lambda c, t: None, now=datetime(2026, 6, 25, 10, 0))
    since, until, phrase = r._window("are the birds asleep tonight", "", datetime(2026, 6, 25, 10, 0), True)
    assert since <= until and phrase == "today"


def test_last_night_uses_previous_evening(tmp_path) -> None:
    r = _responder(tmp_path, lambda c, t: None)
    now = datetime(2026, 6, 25, 9, 0)
    since, until, phrase = r._window("what happened last night", "", now, True)
    assert phrase == "last night"
    assert since.day == 24 and since < until  # spans into the previous day


def _located_responder(tmp_path, now):
    # Manila — a real location so sun_times returns actual sunrise/sunset.
    return ActivityResponder(
        tmp_path, FakeClient(), "gemma3:4b", lambda: KNOWN,
        notify=lambda c, t: None, now=lambda: now,
        latitude=14.5995, longitude=120.9842, utc_offset_hours=8.0,
    )


def test_sunrise_window_brackets_actual_sunrise(tmp_path) -> None:
    from datetime import datetime as _dt
    from lib.suntimes import sun_times
    now = datetime(2026, 6, 25, 15, 0)
    r = _located_responder(tmp_path, now)
    since, until, phrase = r._window("what did they do when the sun rose", "", now, True)
    assert phrase == "around sunrise"
    sunrise, _ = sun_times(now.date(), 14.5995, 120.9842, 8.0)
    anchor = _dt.combine(now.date(), sunrise)
    # The window straddles the real sunrise moment (not a fixed clock guess).
    assert since < anchor < until
    assert until <= now  # never runs past "now"


def test_sunset_window_brackets_actual_sunset(tmp_path) -> None:
    from datetime import datetime as _dt
    from lib.suntimes import sun_times
    now = datetime(2026, 6, 25, 22, 0)  # after sunset so the window isn't clamped away
    r = _located_responder(tmp_path, now)
    since, until, phrase = r._window("what were they doing at sunset", "", now, True)
    assert phrase == "around sunset"
    _, sunset = sun_times(now.date(), 14.5995, 120.9842, 8.0)
    anchor = _dt.combine(now.date(), sunset)
    assert since < anchor < until


def test_sun_window_falls_back_to_clock_when_location_unset(tmp_path) -> None:
    # No location (0,0): sun_times may be None, so the anchor is a 6am clock guess
    # and the query still resolves to a sensible "around sunrise" window.
    r = _responder(tmp_path, lambda c, t: None, now=datetime(2026, 6, 25, 15, 0))
    since, until, phrase = r._window("what did percy do at sunrise", "", datetime(2026, 6, 25, 15, 0), True)
    assert phrase == "around sunrise"
    assert since < until


def test_recall_name_resolves_species_labels() -> None:
    from lib.activity_qa import _recall_name
    # A single-member species resolves to the individual's name.
    assert _recall_name("budgie") == "Bambi"
    # A multi-member species groups without leaking a bare species word.
    assert _recall_name("cockatiel") == "one of the cockatiels"
    assert _recall_name("lovebird") == "one of the lovebirds"
    # Unknown/generic detections become an honest "unidentified bird".
    assert _recall_name("unknown_bird") == "an unidentified bird"
    assert _recall_name("bird") == "an unidentified bird"
    # Named individuals pass straight through, capitalised.
    assert _recall_name("percy") == "Percy"
    assert _recall_name("draft") == "Draft"


def test_bird_activity_list_resolves_species(tmp_path) -> None:
    from lib.activity_qa import _bird_activity_list
    from lib.journal import MemoryDetection, MemoryObservation
    obs = MemoryObservation(
        birds=["bambi", "cockatiel"],
        note="",
        detections=[
            MemoryDetection(label="budgie", confidence=0.9, activity="playing"),
            MemoryDetection(label="cockatiel", confidence=0.8, activity="resting"),
        ],
    )
    rendered = _bird_activity_list(obs)
    assert "Bambi: playing" in rendered
    assert "one of the cockatiels: resting" in rendered
    # No bare species word leaks through.
    assert "Budgie" not in rendered and "Cockatiel:" not in rendered


def test_wants_live_is_word_boundary() -> None:
    from lib.activity_qa import _wants_live
    assert _wants_live("what is percy doing now") is True
    assert _wants_live("do you know what percy did today") is False  # 'know', not 'now'


class RecordingClient:
    """Records which models chat() was asked for; answers like FakeClient."""

    def __init__(self) -> None:
        self.models: list[str] = []

    def chat(self, model, messages, **kwargs):
        self.models.append(model)
        return "Percy preened on the perch."


class DownClient:
    """Simulates the whole Ollama cluster being unreachable for every model."""

    def __init__(self) -> None:
        self.models: list[str] = []

    def chat(self, model, messages, **kwargs):
        from lib.ai.client import OllamaUnavailableError

        self.models.append(model)
        raise OllamaUnavailableError("cluster down")


class FakeHealth:
    """Stand-in for lib.ai.health.OllamaHealth: a fixed set of served models."""

    def __init__(self, served: set[str]) -> None:
        self._served = served

    def has_model(self, model: str) -> bool:
        return model in self._served


def _failover_responder(memories_dir, notify, *, client, health=None, hook=None, send_album=None):
    # Big recall model + small fallback, like production wiring.
    responder = ActivityResponder(
        memories_dir,
        client,
        "gemma3:12b",
        lambda: KNOWN,
        notify=notify,
        send_album=send_album,
        fallback_model="gemma3:4b",
        health=health,
        now=lambda: datetime(2026, 6, 25, 15, 0),
    )
    if hook is not None:
        responder.set_on_unavailable(hook)
    return responder


def test_health_skips_unserved_model_and_asks_only_fallback(tmp_path) -> None:
    # When the cluster doesn't serve the big recall model, respond() must not
    # burn a doomed request on it — the fallback model is asked directly.
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 14, 0), ["percy"], "Percy napped on the perch."))
    client = RecordingClient()
    _failover_responder(
        tmp_path, lambda c, t: None, client=client,
        health=FakeHealth({"gemma3:4b"}),
    ).respond(7, "what did percy do today?", "percy")
    assert client.models == ["gemma3:4b"]


def test_cluster_down_routes_question_to_unavailable_hook(tmp_path) -> None:
    # Every model raises OllamaUnavailableError and a replay hook is set: the
    # question is queued via the hook, and NO degraded raw-note answer is sent.
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 14, 0), ["percy"], "Percy napped on the perch."))
    sent: list = []
    queued: list = []
    client = DownClient()
    _failover_responder(
        tmp_path, lambda c, t: sent.append(t), client=client,
        hook=lambda cid, text: queued.append((cid, text)),
    ).respond(7, "what did percy do today?", "percy")
    assert queued == [(7, "what did percy do today?")]  # (chat_id, original text)
    assert sent == []  # no degraded raw journal line reached the user
    assert client.models == ["gemma3:12b", "gemma3:4b"]  # both models were tried


def test_cluster_down_without_hook_keeps_degraded_raw_note(tmp_path) -> None:
    # Old behavior pinned: with no replay hook, the last raw note is still sent
    # so the user gets *something* rather than silence.
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 14, 0), ["percy"], "Percy napped on the perch."))
    sent: list = []
    _failover_responder(
        tmp_path, lambda c, t: sent.append(t), client=DownClient(),
    ).respond(7, "what did percy do today?", "percy")
    assert len(sent) == 1
    assert "Percy napped on the perch." in sent[0]  # the raw journal note


def test_pure_photo_request_never_routes_to_unavailable_hook(tmp_path) -> None:
    # A pure photo request needs no LLM, so even with the cluster down and a
    # hook set, the photos are sent and the hook is never invoked.
    photo = tmp_path / "percy.jpg"
    photo.write_bytes(b"percy-bytes")
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 14, 0), ["percy"], "Percy napped.", [str(photo)]))
    queued: list = []
    albums: list = []
    _failover_responder(
        tmp_path, lambda c, t: None, client=DownClient(),
        hook=lambda cid, text: queued.append((cid, text)),
        send_album=lambda c, items: albums.append(items),
    ).respond(7, "show me picture of percy", "")
    assert queued == []
    assert len(albums) == 1 and albums[0][0][0] == b"percy-bytes"


def test_both_models_unserved_routes_to_unavailable_hook(tmp_path) -> None:
    # A partially-recovered cluster serving NEITHER model: has_model skips both,
    # so nothing was ever asked — with a hook set that must queue the question,
    # not degrade to a raw journal line (the same treatment as a full outage).
    append_entry(tmp_path, MemoryEntry(datetime(2026, 6, 25, 14, 0), ["percy"], "Percy napped on the perch."))
    sent: list = []
    queued: list = []
    client = RecordingClient()
    _failover_responder(
        tmp_path, lambda c, t: sent.append(t), client=client,
        health=FakeHealth(set()),  # serves neither gemma3:12b nor gemma3:4b
        hook=lambda cid, text: queued.append((cid, text)),
    ).respond(7, "what did percy do today?", "percy")
    assert client.models == []  # no doomed request was attempted
    assert queued == [(7, "what did percy do today?")]
    assert sent == []


def test_replayed_question_never_fires_live_camera_find(tmp_path) -> None:
    # "what is draft doing right now?" asked during an outage must NOT launch a
    # PTZ camera search when replayed later — same reason "find" is not a
    # replay-safe action. The plain "haven't logged" answer is sent instead.
    found: list = []
    sent: list = []
    responder = _responder(
        tmp_path, lambda c, t: sent.append(t), find=lambda cid, arg: found.append((cid, arg))
    )
    responder.set_replaying(lambda: True)
    responder.respond(7, "what is draft doing right now?", "draft")
    assert found == []
    assert sent and "haven't logged" in sent[0].lower()
