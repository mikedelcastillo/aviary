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


def test_render_converts_italic_strike_and_bold_italic() -> None:
    assert render_telegram_html("a *soft* word") == "a <i>soft</i> word"
    assert render_telegram_html("an _emphasised_ note") == "an <i>emphasised</i> note"
    assert render_telegram_html("~~nope~~") == "<s>nope</s>"
    assert render_telegram_html("***wow***") == "<b><i>wow</i></b>"


def test_render_italic_leaves_identifiers_and_math_alone() -> None:
    # snake_case and "2 * 3" must NOT be mangled into italics.
    assert render_telegram_html("set TELEGRAM_USER_IDS now") == "set TELEGRAM_USER_IDS now"
    assert render_telegram_html("2 * 3 = 6") == "2 * 3 = 6"


def test_render_converts_blockquote_rule_and_fence() -> None:
    assert "quoted" in render_telegram_html("> quoted") and ">" not in render_telegram_html("> quoted")
    assert render_telegram_html("---").strip() == ""  # a horizontal rule is dropped
    out = render_telegram_html("```py\nx = 1\n```")
    assert "<pre>" in out and "```" not in out


_MARKDOWN_LEAK_SAMPLES = [
    "**bold** and *italic* and _under_ and ~~strike~~",
    "# Heading\n## Sub\n- a\n* b\n+ c",
    "> a quote\n\n---\n\n`inline` and ```fenced```",
    "[link](https://e.io/x) plus ***all three***",
    "no markdown here at all, just words.",
]


def test_no_raw_markdown_markers_survive_rendering() -> None:
    import re

    for sample in _MARKDOWN_LEAK_SAMPLES:
        out = render_telegram_html(sample)
        assert "**" not in out, out
        assert "~~" not in out, out
        assert "```" not in out, out
        assert "](" not in out, out  # no leftover [text](url)
        assert not re.search(r"(?m)^\s{0,3}#{1,6}\s", out), out  # no heading marker
        assert not re.search(r"(?m)^\s{0,3}[-*+]\s", out), out  # no bullet marker
        assert not re.search(r"(?m)^\s{0,3}>\s", out), out  # no blockquote marker


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


def _tags_balanced(s: str) -> bool:
    import re

    stack: list[str] = []
    for closing, tag in re.findall(r"<(/?)([a-z]+)(?: [^<>]*)?>", s):
        if closing:
            if not stack or stack.pop() != tag:
                return False
        else:
            stack.append(tag)
    return not stack


def test_render_quote_in_url_does_not_break_the_tag() -> None:
    # A stray " inside a link URL must never escape the href attribute. Here the
    # quote stops the URL match, so no <a> is formed at all — safe, balanced text.
    out = render_telegram_html('[click](https://e.com/a"b)')
    assert _tags_balanced(out)
    assert 'a"b"' not in out  # no broken/half-open href attribute

    # A clean URL still links normally.
    ok = render_telegram_html("[docs](https://e.com/x)")
    assert ok == '<a href="https://e.com/x">docs</a>'


def test_render_overlapping_markdown_falls_back_to_plain() -> None:
    # Bold opening inside a link and closing outside it would make overlapping
    # tags; render must yield valid (balanced) HTML, not malformed tags.
    out = render_telegram_html("[link **bold](https://e.com) end**")
    assert _tags_balanced(out)
    assert "**" not in out  # markers were stripped in the plain fallback


def test_render_always_balanced_for_messy_input() -> None:
    for messy in ("## H **a`b** c`", "**unclosed bold", "[t](https://x.io) **b** `c`"):
        assert _tags_balanced(render_telegram_html(messy)), messy
