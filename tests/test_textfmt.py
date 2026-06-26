from __future__ import annotations

from lib.textfmt import (
    escape_html,
    flatten_tables,
    render_telegram_html,
    to_plain,
)


def test_escape_html_escapes_the_three_specials() -> None:
    assert escape_html("Diet & feeding") == "Diet &amp; feeding"
    assert escape_html("/find <bird>") == "/find &lt;bird&gt;"


def test_render_escapes_then_keeps_a_care_header_safe() -> None:
    # "&" and "<bird>" must survive as visible text, not break the HTML parse.
    out = render_telegram_html("Diet & feeding — try /find <bird>")
    assert "&amp;" in out and "&lt;bird&gt;" in out
    assert "<bird>" not in out  # the raw angle brackets are gone


def test_render_converts_bold_markers() -> None:
    assert render_telegram_html("**Last night** was calm") == "<b>Last night</b> was calm"
    assert render_telegram_html("__hi__") == "<b>hi</b>"


def test_render_converts_headers_and_bullets() -> None:
    out = render_telegram_html("## Sleep\n- dark\n- quiet")
    assert "<b>Sleep</b>" in out
    assert "• dark" in out and "• quiet" in out
    assert "##" not in out and "- dark" not in out


def test_render_converts_links_and_code() -> None:
    out = render_telegram_html("see [docs](https://x.io/a) and `code`")
    assert '<a href="https://x.io/a">docs</a>' in out
    assert "<code>code</code>" in out


def test_render_flattens_a_pipe_table() -> None:
    out = render_telegram_html("| Bird | Where |\n|---|---|\n| Percy | perch |")
    assert "|" not in out
    assert "Bird: Percy, Where: perch" in out


def test_to_plain_strips_markers_without_html() -> None:
    out = to_plain("## Header\n**Bold** and `code`\n- item")
    assert "Bold" in out and "<b>" not in out and "**" not in out
    assert "Header" in out and "#" not in out
    assert "code" in out and "`" not in out
    assert "• item" in out


def test_flatten_tables_leaves_a_lone_pipe_alone() -> None:
    text = "Use /find | it locates a bird."
    assert flatten_tables(text) == text
