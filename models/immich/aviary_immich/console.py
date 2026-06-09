"""Rich-based console output helpers for the Immich album generator.

All helpers degrade gracefully to plain ``print`` if ``rich`` is unavailable, mirroring
the optional-``tqdm`` pattern used elsewhere in this package.
"""

from __future__ import annotations

from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        ProgressColumn,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text

    _RICH = True
except Exception:  # pragma: no cover - exercised only when rich is missing
    _RICH = False


_CONSOLE: Any = None


def get_console() -> Any:
    """Return a cached Console, or None when rich is unavailable."""
    global _CONSOLE
    if not _RICH:
        return None
    if _CONSOLE is None:
        _CONSOLE = Console()
    return _CONSOLE


if _RICH:

    class RateColumn(ProgressColumn):
        """Show throughput in items/second for a task."""

        def render(self, task) -> "Text":
            speed = task.finished_speed or task.speed
            if not speed:
                return Text("--/s", style="dim")
            return Text(f"{speed:.0f}/s", style="cyan")


def config_panel(
    args: Any,
    device: str,
    worker_count: int,
    chunk_size: int,
    inference_batch_size: int,
    download_workers: int,
    prefetch: int,
    cpu_workers: int = 0,
) -> None:
    rows = {
        "model": str(args.model),
        "device": device,
        "threshold": str(args.threshold),
        "workers": str(worker_count),
        "cpu workers": str(cpu_workers),
        "download workers": str(download_workers),
        "chunk size": str(chunk_size),
        "inference batch": str(inference_batch_size),
        "prefetch": str(prefetch),
    }
    console = get_console()
    if console is None:
        print("Detector config: " + " ".join(f"{key}={value}" for key, value in rows.items()))
        return

    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="bold")
    table.add_column()
    for key, value in rows.items():
        table.add_row(key, value)
    console.print(Panel(table, title="[bold]Detector config", border_style="blue", expand=False))


def account_header(
    slug: str,
    connected_as: str,
    album_name: str,
    album_id: str | None,
    already_in_album: int | None,
    dry_run: bool,
) -> None:
    console = get_console()
    album_line = f"would create/find [bold]{album_name}[/]" if dry_run else f"album [bold]{album_name}[/]"
    if album_id and not dry_run:
        album_line += f" ([dim]{album_id}[/])"
    if already_in_album is not None and not dry_run:
        album_line += f" — {already_in_album} assets already present"

    if console is None:
        print(f"\nScanning account {slug}")
        print(f"Connected as {connected_as}")
        print(album_line.replace("[bold]", "").replace("[/]", "").replace("[dim]", ""))
        return

    body = Table.grid(padding=(0, 1))
    body.add_column()
    body.add_row(f"connected as [bold]{connected_as}[/]")
    body.add_row(album_line)
    title = f"[bold cyan]{slug}[/]"
    if dry_run:
        title += "  [yellow](dry run)[/]"
    console.print(Panel(body, title=title, border_style="cyan", expand=False))


def make_scan_progress() -> Any:
    """Return a configured Progress with cache + detect tasks, or None without rich.

    Caller adds tasks via ``progress.add_task(desc, total=n, postfix="")`` and updates the
    ``postfix`` field to surface live counts (e.g. ``birds=12 err=0``).
    """
    console = get_console()
    if console is None:
        return None
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        RateColumn(),
        TimeRemainingColumn(),
        TextColumn("[green]{task.fields[postfix]}"),
        console=console,
    )


def account_summary_table(slug: str, stats: dict[str, Any]) -> None:
    console = get_console()
    rows = [
        ("scanned", stats.get("scanned", 0)),
        ("already scanned", stats.get("already", 0)),
        ("videos", stats.get("videos", 0)),
        ("birds", stats.get("birds", 0)),
        ("dogs", stats.get("dogs", 0)),
        ("cats", stats.get("cats", 0)),
        ("other", stats.get("other", 0)),
        ("errors", stats.get("errors", 0)),
        ("added to album", stats.get("added", 0)),
        ("elapsed", f"{stats.get('elapsed', 0.0):.1f}s"),
    ]
    if console is None:
        print(f"Summary {slug}: " + " · ".join(f"{label}={value}" for label, value in rows))
        return

    table = Table(title=f"[bold]{slug} summary", title_justify="left", border_style="green", expand=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for label, value in rows:
        table.add_row(label, str(value))
    console.print(table)


def grand_total_table(per_account: list[dict[str, Any]]) -> None:
    if not per_account:
        return
    console = get_console()
    keys = ("scanned", "videos", "birds", "dogs", "cats", "other", "errors", "added")
    totals = {key: sum(int(entry.get(key, 0)) for entry in per_account) for key in keys}
    totals["elapsed"] = sum(float(entry.get("elapsed", 0.0)) for entry in per_account)

    if console is None:
        print("\nGrand total: " + " · ".join(f"{key}={value}" for key, value in totals.items()))
        return

    table = Table(title="[bold]Grand total", title_justify="left", border_style="magenta", expand=False)
    table.add_column("account", style="bold cyan")
    for key in keys:
        table.add_column(key, justify="right")
    table.add_column("elapsed", justify="right")
    for entry in per_account:
        table.add_row(
            str(entry.get("slug", "")),
            *(str(entry.get(key, 0)) for key in keys),
            f"{float(entry.get('elapsed', 0.0)):.1f}s",
        )
    table.add_section()
    table.add_row(
        "[bold]total",
        *(f"[bold]{totals[key]}" for key in keys),
        f"[bold]{totals['elapsed']:.1f}s",
    )
    console.print(table)
