from __future__ import annotations

from aviary_training.roster import (
    classes_for_model,
    load_roster,
    remap_label_lines,
)


ROSTER_YAML = """
labels:
  - { name: Percy, models: [live, archive] }
  - { name: Draft, models: [archive] }
  - { name: cockatiel, models: [live] }
  - { name: unknown_bird, models: [live, archive] }
"""


def _write_roster(tmp_path):
    path = tmp_path / "roster.yaml"
    path.write_text(ROSTER_YAML, encoding="utf-8")
    return path


def test_load_roster_assigns_global_indices_in_file_order(tmp_path) -> None:
    roster = load_roster(_write_roster(tmp_path))

    assert [(label.index, label.name) for label in roster] == [
        (0, "Percy"),
        (1, "Draft"),
        (2, "cockatiel"),
        (3, "unknown_bird"),
    ]


def test_classes_for_model_filters_and_remaps_to_contiguous(tmp_path) -> None:
    roster = load_roster(_write_roster(tmp_path))

    live = classes_for_model(roster, "live")

    # Draft (archive-only, global index 1) is dropped; survivors renumber 0..N-1.
    assert live.names == {0: "Percy", 1: "cockatiel", 2: "unknown_bird"}
    assert live.remap == {0: 0, 2: 1, 3: 2}


def test_classes_for_model_archive_keeps_individuals_drops_species(tmp_path) -> None:
    roster = load_roster(_write_roster(tmp_path))

    archive = classes_for_model(roster, "archive")

    assert archive.names == {0: "Percy", 1: "Draft", 2: "unknown_bird"}
    assert archive.remap == {0: 0, 1: 1, 3: 2}


def test_remap_label_lines_rewrites_class_index() -> None:
    remap = {0: 0, 2: 1, 3: 2}

    lines = remap_label_lines(["2 0.5 0.5 0.1 0.1"], remap)

    assert lines == ["1 0.5 0.5 0.1 0.1"]


def test_remap_label_lines_drops_filtered_classes() -> None:
    remap = {0: 0, 2: 1, 3: 2}

    lines = remap_label_lines(
        ["0 0.5 0.5 0.1 0.1", "1 0.4 0.4 0.2 0.2", "", "3 0.1 0.2 0.3 0.4"],
        remap,
    )

    # Class 1 (not in remap) is dropped; blank line ignored.
    assert lines == ["0 0.5 0.5 0.1 0.1", "2 0.1 0.2 0.3 0.4"]
