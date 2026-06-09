"""Unit tests for ``aviary_immich.client`` — pure logic, no network.

``ImmichClient.__init__`` only builds a ``requests.Session`` (no I/O), so instances are cheap
to construct. For methods that would talk to Immich we monkeypatch the instance's
``_request_json`` (or ``list_albums``) so no HTTP ever happens.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from aviary_immich.client import (
    ImmichClient,
    filename_from_headers,
    safe_filename,
    suffix_from_headers,
)
from fakes import make_album


def make_client() -> ImmichClient:
    """Construct a client; __init__ is network-free (just builds a Session)."""
    return ImmichClient("http://x/api", "key")


def boom(*args, **kwargs):
    """Stand-in for _request_json that must never be invoked."""
    raise AssertionError("_request_json should not have been called")


# --------------------------------------------------------------------------- __init__


def test_init_strips_trailing_slash_and_sets_headers():
    client = ImmichClient("http://x/api/", "secret")
    assert client.base_url == "http://x/api"
    assert client.session.headers["x-api-key"] == "secret"
    assert client.session.headers["Accept"] == "application/json"


def test_init_default_timeout():
    assert make_client().timeout_seconds == 60


# --------------------------------------------------------------------------- find_album


def test_find_album_no_owner_returns_first_match(monkeypatch):
    client = make_client()
    albums = [make_album(name="Birds", id="a"), make_album(name="Birds", id="b")]
    monkeypatch.setattr(client, "list_albums", lambda: albums)
    result = client.find_album("Birds")
    assert result["id"] == "a"


def test_find_album_no_name_match_returns_none(monkeypatch):
    client = make_client()
    monkeypatch.setattr(client, "list_albums", lambda: [make_album(name="Cats")])
    assert client.find_album("Birds") is None


def test_find_album_empty_albums_returns_none(monkeypatch):
    client = make_client()
    monkeypatch.setattr(client, "list_albums", lambda: [])
    assert client.find_album("Birds") is None


def test_find_album_owner_match_returned(monkeypatch):
    client = make_client()
    albums = [
        make_album(name="Birds", id="other", owner_id="someone-else"),
        make_album(name="Birds", id="mine", owner_id="owner-1"),
    ]
    monkeypatch.setattr(client, "list_albums", lambda: albums)
    result = client.find_album("Birds", owner_id="owner-1")
    assert result["id"] == "mine"


def test_find_album_owner_match_via_nested_owner(monkeypatch):
    client = make_client()
    albums = [
        make_album(name="Birds", id="mine", owner={"id": "owner-1"}),
    ]
    monkeypatch.setattr(client, "list_albums", lambda: albums)
    result = client.find_album("Birds", owner_id="owner-1")
    assert result["id"] == "mine"


def test_find_album_owner_no_match_not_owned_only_returns_first_match(monkeypatch):
    client = make_client()
    # Both have a known but mismatched owner -> no owner match, not owned_only -> first match.
    albums = [
        make_album(name="Birds", id="a", owner_id="x"),
        make_album(name="Birds", id="b", owner_id="y"),
    ]
    monkeypatch.setattr(client, "list_albums", lambda: albums)
    result = client.find_album("Birds", owner_id="owner-1")
    assert result["id"] == "a"


def test_find_album_owned_only_unknown_owner_returns_first_unknown(monkeypatch):
    client = make_client()
    # No ownerId / owner -> unknown owner. owned_only=True returns the first unknown-owner album.
    albums = [
        make_album(name="Birds", id="u1"),
        make_album(name="Birds", id="u2"),
    ]
    monkeypatch.setattr(client, "list_albums", lambda: albums)
    result = client.find_album("Birds", owner_id="owner-1", owned_only=True)
    assert result["id"] == "u1"


def test_find_album_owned_only_no_owned_no_unknown_returns_none(monkeypatch):
    client = make_client()
    # All matches have a known, mismatched owner -> nothing owned, nothing unknown -> None.
    albums = [
        make_album(name="Birds", id="a", owner_id="x"),
        make_album(name="Birds", id="b", owner_id="y"),
    ]
    monkeypatch.setattr(client, "list_albums", lambda: albums)
    assert client.find_album("Birds", owner_id="owner-1", owned_only=True) is None


def test_find_album_owned_only_prefers_exact_owner_over_unknown(monkeypatch):
    client = make_client()
    albums = [
        make_album(name="Birds", id="unknown"),
        make_album(name="Birds", id="mine", owner_id="owner-1"),
    ]
    monkeypatch.setattr(client, "list_albums", lambda: albums)
    result = client.find_album("Birds", owner_id="owner-1", owned_only=True)
    assert result["id"] == "mine"


# --------------------------------------------------------------------------- _album_owner_id


def test_album_owner_id_reads_owner_id():
    assert ImmichClient._album_owner_id({"ownerId": "abc"}) == "abc"


def test_album_owner_id_coerces_to_str():
    assert ImmichClient._album_owner_id({"ownerId": 123}) == "123"


def test_album_owner_id_falls_back_to_nested_owner():
    assert ImmichClient._album_owner_id({"owner": {"id": "nested"}}) == "nested"


def test_album_owner_id_nested_owner_coerced_to_str():
    assert ImmichClient._album_owner_id({"owner": {"id": 7}}) == "7"


def test_album_owner_id_none_when_absent():
    assert ImmichClient._album_owner_id({}) is None


def test_album_owner_id_none_when_owner_not_dict():
    assert ImmichClient._album_owner_id({"owner": "not-a-dict"}) is None


def test_album_owner_id_none_when_owner_dict_without_id():
    assert ImmichClient._album_owner_id({"owner": {}}) is None


def test_album_owner_id_prefers_owner_id_over_nested():
    assert ImmichClient._album_owner_id({"ownerId": "top", "owner": {"id": "nested"}}) == "top"


# --------------------------------------------------------------------------- pagination


def make_page(items, next_page=None):
    return {"assets": {"items": items, "nextPage": next_page}}


def scripted_request_json(pages):
    """Return a _request_json replacement that pops scripted page dicts in order."""
    queue = list(pages)

    def _request_json(method, path, **kwargs):
        assert method == "POST"
        assert path == "/search/metadata"
        return queue.pop(0)

    return _request_json


def test_iter_image_assets_single_short_page_stops(monkeypatch):
    client = make_client()
    # 2 items, page_size 250, no nextPage -> short page -> stop after one request.
    pages = [make_page([{"id": "1"}, {"id": "2"}])]
    monkeypatch.setattr(client, "_request_json", scripted_request_json(pages))
    result = list(client.iter_image_assets(page_size=250))
    assert [a["id"] for a in result] == ["1", "2"]


def test_iter_image_assets_follows_next_page(monkeypatch):
    client = make_client()
    pages = [
        make_page([{"id": "1"}, {"id": "2"}], next_page=2),
        make_page([{"id": "3"}]),  # short, no nextPage -> stop
    ]
    monkeypatch.setattr(client, "_request_json", scripted_request_json(pages))
    result = list(client.iter_image_assets(page_size=2))
    assert [a["id"] for a in result] == ["1", "2", "3"]


def test_iter_image_assets_stops_on_full_page_without_next_page(monkeypatch):
    client = make_client()
    # First page is full (==page_size) but has no nextPage -> client advances page; second
    # page is short -> stop. Verifies the "len(items) < page_size" stop condition.
    pages = [
        make_page([{"id": "1"}, {"id": "2"}]),  # full page (size 2), no nextPage
        make_page([{"id": "3"}]),  # short -> stop
    ]
    monkeypatch.setattr(client, "_request_json", scripted_request_json(pages))
    result = list(client.iter_image_assets(page_size=2))
    assert [a["id"] for a in result] == ["1", "2", "3"]


def test_iter_image_assets_full_page_then_empty_page_stops(monkeypatch):
    client = make_client()
    # Full page, no nextPage -> advance; empty page (len 0 < page_size) -> stop.
    pages = [
        make_page([{"id": "1"}, {"id": "2"}]),
        make_page([]),
    ]
    monkeypatch.setattr(client, "_request_json", scripted_request_json(pages))
    result = list(client.iter_image_assets(page_size=2))
    assert [a["id"] for a in result] == ["1", "2"]


def test_iter_image_assets_limit_cuts_off_mid_page(monkeypatch):
    client = make_client()
    pages = [make_page([{"id": "1"}, {"id": "2"}, {"id": "3"}], next_page=2)]
    monkeypatch.setattr(client, "_request_json", scripted_request_json(pages))
    result = list(client.iter_image_assets(page_size=250, limit=2))
    assert [a["id"] for a in result] == ["1", "2"]


def test_iter_image_assets_limit_zero_yields_nothing_after_first_item(monkeypatch):
    client = make_client()
    pages = [make_page([{"id": "1"}, {"id": "2"}])]
    monkeypatch.setattr(client, "_request_json", scripted_request_json(pages))
    # limit None is the default; explicit limit None yields everything in a short page.
    result = list(client.iter_image_assets(page_size=250, limit=None))
    assert [a["id"] for a in result] == ["1", "2"]


def test_iter_image_assets_sends_image_search_body(monkeypatch):
    client = make_client()
    captured = {}

    def _request_json(method, path, **kwargs):
        captured["body"] = kwargs["json"]
        return make_page([])

    monkeypatch.setattr(client, "_request_json", _request_json)
    list(client.iter_image_assets(page_size=10))
    body = captured["body"]
    assert body["type"] == "IMAGE"
    assert body["withExif"] is True
    assert body["page"] == 1
    assert body["size"] == 10
    assert "albumIds" not in body


def test_iter_video_assets_sends_video_search_body(monkeypatch):
    client = make_client()
    captured = {}

    def _request_json(method, path, **kwargs):
        captured["body"] = kwargs["json"]
        return make_page([])

    monkeypatch.setattr(client, "_request_json", _request_json)
    list(client.iter_video_assets(page_size=10))
    body = captured["body"]
    assert body["type"] == "VIDEO"
    assert body["withExif"] is True
    assert body["page"] == 1
    assert body["size"] == 10
    assert "albumIds" not in body


def test_iter_video_assets_yields_items(monkeypatch):
    client = make_client()
    pages = [make_page([{"id": "v1"}, {"id": "v2"}])]
    monkeypatch.setattr(client, "_request_json", scripted_request_json(pages))
    result = list(client.iter_video_assets(page_size=250))
    assert [a["id"] for a in result] == ["v1", "v2"]


def test_download_video_transcoded_uses_playback_endpoint(monkeypatch):
    client = make_client()
    captured = {}

    def _download(path, destination, params=None):
        captured["path"] = path
        captured["destination"] = destination
        return requests.structures.CaseInsensitiveDict({"content-type": "video/mp4"})

    monkeypatch.setattr(client, "_download", _download)
    client.download_video("v1", Path("/tmp/v1.mp4"))
    assert captured["path"] == "/assets/v1/video/playback"
    assert captured["destination"] == Path("/tmp/v1.mp4")


def test_download_video_original_uses_original_endpoint(monkeypatch):
    client = make_client()
    captured = {}

    def _download(path, destination, params=None):
        captured["path"] = path
        return requests.structures.CaseInsensitiveDict()

    monkeypatch.setattr(client, "_download", _download)
    client.download_video("v1", Path("/tmp/v1.mp4"), transcoded=False)
    assert captured["path"] == "/assets/v1/original"


def test_iter_album_assets_sends_album_ids(monkeypatch):
    client = make_client()
    captured = {}

    def _request_json(method, path, **kwargs):
        captured["body"] = kwargs["json"]
        return make_page([])

    monkeypatch.setattr(client, "_request_json", _request_json)
    list(client.iter_album_assets("alb-9", page_size=10))
    body = captured["body"]
    assert body["type"] == "IMAGE"
    assert body["albumIds"] == ["alb-9"]


def test_iter_album_assets_yields_items(monkeypatch):
    client = make_client()
    pages = [make_page([{"id": "x"}])]
    monkeypatch.setattr(client, "_request_json", scripted_request_json(pages))
    result = list(client.iter_album_assets("alb-1", page_size=250))
    assert [a["id"] for a in result] == ["x"]


# --------------------------------------------------------------------------- filename_from_headers


def make_headers(**kwargs) -> requests.structures.CaseInsensitiveDict:
    return requests.structures.CaseInsensitiveDict(kwargs)


def test_filename_from_headers_plain_quoted():
    headers = make_headers(**{"content-disposition": 'attachment; filename="photo.jpg"'})
    assert filename_from_headers(headers) == "photo.jpg"


def test_filename_from_headers_unquoted():
    headers = make_headers(**{"content-disposition": "attachment; filename=photo.jpg"})
    assert filename_from_headers(headers) == "photo.jpg"


def test_filename_from_headers_extended_utf8_percent_decoded():
    headers = make_headers(**{"content-disposition": "attachment; filename*=UTF-8''my%20file.jpg"})
    assert filename_from_headers(headers) == "my file.jpg"


def test_filename_from_headers_extended_utf8_quoted_percent_decoded():
    headers = make_headers(**{"content-disposition": "attachment; filename*=UTF-8''\"my%20file.jpg\""})
    assert filename_from_headers(headers) == "my file.jpg"


def test_filename_from_headers_case_insensitive_header_key():
    headers = make_headers(**{"Content-Disposition": 'attachment; filename="x.png"'})
    assert filename_from_headers(headers) == "x.png"


def test_filename_from_headers_none_when_absent():
    assert filename_from_headers(make_headers()) is None


def test_filename_from_headers_none_when_no_filename_param():
    headers = make_headers(**{"content-disposition": "attachment"})
    assert filename_from_headers(headers) is None


# --------------------------------------------------------------------------- suffix_from_headers


@pytest.mark.parametrize(
    "content_type,expected",
    [
        ("image/jpeg", ".jpg"),
        ("image/jpg", ".jpg"),
        ("image/png", ".png"),
        ("image/webp", ".webp"),
        ("image/heic", ".heic"),
        ("image/heif", ".heif"),
    ],
)
def test_suffix_from_headers_mapped_types(content_type, expected):
    headers = make_headers(**{"content-type": content_type})
    assert suffix_from_headers(headers) == expected


def test_suffix_from_headers_unknown_uses_default():
    headers = make_headers(**{"content-type": "application/octet-stream"})
    assert suffix_from_headers(headers) == ".jpg"


def test_suffix_from_headers_custom_default():
    headers = make_headers(**{"content-type": "application/octet-stream"})
    assert suffix_from_headers(headers, default=".bin") == ".bin"


def test_suffix_from_headers_missing_uses_default():
    assert suffix_from_headers(make_headers()) == ".jpg"


def test_suffix_from_headers_strips_parameters():
    headers = make_headers(**{"content-type": "image/jpeg; charset=binary"})
    assert suffix_from_headers(headers) == ".jpg"


def test_suffix_from_headers_case_insensitive_value():
    headers = make_headers(**{"content-type": "IMAGE/PNG"})
    assert suffix_from_headers(headers) == ".png"


# --------------------------------------------------------------------------- safe_filename


def test_safe_filename_forward_slash_to_underscore():
    assert safe_filename("a/b", "fb") == "a_b"


def test_safe_filename_backslash_to_underscore():
    assert safe_filename("a\\b", "fb") == "a_b"


def test_safe_filename_keeps_allowed_chars():
    assert safe_filename("photo 1.jpg", "fb") == "photo 1.jpg"
    assert safe_filename("a-b_c.d", "fb") == "a-b_c.d"


def test_safe_filename_sanitizes_other_non_alnum():
    # ":" "*" "?" all map to "_". The trailing "_" is NOT stripped (only spaces/dots are).
    assert safe_filename("a:b*c?", "fb") == "a_b_c_"


def test_safe_filename_trailing_underscore_not_stripped():
    assert safe_filename("name_", "fb") == "name_"


def test_safe_filename_strips_leading_trailing_spaces_and_dots():
    assert safe_filename("  .name.  ", "fb") == "name"


def test_safe_filename_empty_value_uses_fallback():
    assert safe_filename("", "fallback") == "fallback"


def test_safe_filename_none_value_uses_fallback():
    # ``value or fallback`` makes None fall through to the fallback.
    assert safe_filename(None, "fallback") == "fallback"


def test_safe_filename_all_dots_uses_fallback():
    assert safe_filename("...", "fallback") == "fallback"


def test_safe_filename_all_sanitized_then_stripped_uses_fallback():
    # "/" -> "_", but underscores are kept, so result is "___" which does NOT strip away.
    assert safe_filename("///", "fallback") == "___"


def test_safe_filename_only_spaces_uses_fallback():
    assert safe_filename("   ", "fallback") == "fallback"


# --------------------------------------------------------------------------- add_assets_to_album


def test_add_assets_to_album_empty_returns_empty_without_request(monkeypatch):
    client = make_client()
    monkeypatch.setattr(client, "_request_json", boom)
    assert client.add_assets_to_album("alb-1", []) == []


def test_add_assets_to_album_calls_request_with_ids(monkeypatch):
    client = make_client()
    captured = {}

    def _request_json(method, path, **kwargs):
        captured["method"] = method
        captured["path"] = path
        captured["json"] = kwargs["json"]
        return {"ok": True}

    monkeypatch.setattr(client, "_request_json", _request_json)
    result = client.add_assets_to_album("alb-1", ["x", "y"])
    assert result == {"ok": True}
    assert captured["method"] == "PUT"
    assert captured["path"] == "/albums/alb-1/assets"
    assert captured["json"] == {"ids": ["x", "y"]}
