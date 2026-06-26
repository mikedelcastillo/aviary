"""Telegram-safe text formatting — the single place messages become HTML.

Outgoing Telegram text is sent with ``parse_mode=HTML`` so the bot can show real
**bold** instead of raw markdown. That has two consequences this module owns:

* every message must be HTML-escaped (``&`` ``<`` ``>``), so a care header like
  "Diet & feeding" or a help line like ``/find <bird>`` can't break the parse;
* any markdown the LLM (or a template) emits must be converted to Telegram's
  small HTML subset — and a GFM pipe table, which Telegram can't render at all,
  flattened to plain lines.

Templates author PLAIN text with simple ``**bold**`` markers; the notifier calls
:func:`render_telegram_html` centrally so no template ever hand-writes a tag.
:func:`to_plain` is the no-formatting fallback (a rejected HTML send, the
console) — same conversions, but markers removed rather than turned into tags.
"""

from __future__ import annotations

import re


def escape_html(text: str) -> str:
    """Escape the three characters Telegram's HTML parser treats specially."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# -- GFM pipe tables: Telegram shows them as raw "| --- |" noise, so flatten ----


def _is_table_sep(line: str) -> bool:
    """A separator row, e.g. ``| --- | :--: |`` — only pipes/dashes/colons/space."""
    s = line.strip()
    if "|" not in s or "-" not in s:
        return False
    return all(ch in "|-: \t" for ch in s)


def _table_cells(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [cell.strip() for cell in s.split("|")]


def _render_table(block: list[str]) -> list[str]:
    header: list[str] | None = None
    rows: list[list[str]] = []
    for line in block:
        if _is_table_sep(line):
            continue
        cells = _table_cells(line)
        if header is None:
            header = cells
        else:
            rows.append(cells)
    out: list[str] = []
    if header and not rows:  # a header with no body — just list its cells
        joined = " — ".join(c for c in header if c)
        if joined:
            out.append(joined)
        return out
    for cells in rows:
        if header and len(header) == len(cells) and any(header):
            pairs = [f"{h}: {v}" for h, v in zip(header, cells) if h or v]
            out.append(", ".join(pairs))
        else:
            joined = " — ".join(c for c in cells if c)
            if joined:
                out.append(joined)
    return out


def flatten_tables(text: str) -> str:
    """Flatten any GFM pipe tables into plain lines (header paired with rows).

    A block of consecutive pipe lines that includes a ``---`` separator row is a
    table; a stray pipe in prose (no separator) is left untouched.
    """
    if "|" not in text:
        return text
    lines = text.split("\n")
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        if "|" in lines[i]:
            j = i
            while j < n and "|" in lines[j]:
                j += 1
            block = lines[i:j]
            if len(block) >= 2 and any(_is_table_sep(b) for b in block):
                out.extend(_render_table(block))
            else:
                out.extend(block)
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


# -- markdown -> Telegram HTML --------------------------------------------------

# A URL stops at whitespace, a closing paren, or a quote/angle bracket that would
# break the href attribute — so a stray `"` can't terminate the tag early.
_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)\"<>]+)\)")
_HEADER = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*$")
_BULLET = re.compile(r"(?m)^([ \t]*)[-*+][ \t]+")
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.DOTALL)
_CODE = re.compile(r"`([^`\n]+)`")
_TAG = re.compile(r"<(/?)([a-z]+)(?: [^<>]*)?>")


def _html_balanced(s: str) -> bool:
    """True if every tag we generated is properly LIFO-nested and closed.

    Sloppy markdown (a ``**bold**`` span that opens inside a link and closes
    outside it) can produce overlapping tags Telegram rejects. We check nesting
    so :func:`render_telegram_html` can fall back to plain text BEFORE sending,
    rather than eating a rejected round-trip.
    """
    stack: list[str] = []
    for closing, tag in _TAG.findall(s):
        if closing:
            if not stack or stack.pop() != tag:
                return False
        else:
            stack.append(tag)
    return not stack


def render_telegram_html(text: str) -> str:
    """Plain text (with optional ``**markers**``) -> safe Telegram HTML.

    Always returns valid Telegram HTML: if the markdown converts to overlapping
    or unbalanced tags, it returns the escaped, marker-stripped plain form (still
    valid under parse_mode=HTML) instead of malformed tags.
    """
    s = escape_html(flatten_tables(text))
    s = _LINK.sub(lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', s)
    s = _HEADER.sub(lambda m: f"<b>{m.group(1)}</b>", s)
    s = _BULLET.sub(lambda m: f"{m.group(1)}• ", s)
    s = _BOLD.sub(lambda m: f"<b>{m.group(1) or m.group(2)}</b>", s)
    s = _CODE.sub(lambda m: f"<code>{m.group(1)}</code>", s)
    return s if _html_balanced(s) else escape_html(to_plain(text))


def to_plain(text: str) -> str:
    """Same conversions as :func:`render_telegram_html` but with markdown markers
    removed and NO HTML — for the no-parse-mode fallback and the console."""
    s = flatten_tables(text)
    s = _LINK.sub(lambda m: f"{m.group(1)} ({m.group(2)})", s)
    s = _HEADER.sub(lambda m: m.group(1), s)
    s = _BULLET.sub(lambda m: f"{m.group(1)}• ", s)
    s = _BOLD.sub(lambda m: m.group(1) or m.group(2), s)
    s = _CODE.sub(lambda m: m.group(1), s)
    return s
