"""Console and progress-bar helpers shared across the album-generation modules.

All helpers degrade gracefully when ``tqdm``/``rich`` are unavailable, mirroring the
optional-dependency pattern used elsewhere in this package.
"""

from __future__ import annotations

from typing import Any, Iterable


def progress(items: Iterable[Any], **kwargs) -> Iterable[Any]:
    try:
        from tqdm import tqdm
    except Exception:
        return items
    return tqdm(items, **kwargs)


def progress_bar(**kwargs):
    try:
        from tqdm import tqdm
    except Exception:
        return None
    return tqdm(**kwargs)


def emit(message: str) -> None:
    """Print a line that renders cleanly above an active rich progress display."""
    from aviary_immich.console import get_console

    console = get_console()
    if console is None:
        print(message)
    else:
        console.log(message)
