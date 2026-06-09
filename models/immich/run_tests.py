#!/usr/bin/env python3
"""Entry point for ``uv run tests`` — runs the immich unit-test suite via pytest.

Kept tiny on purpose: pytest configuration (pythonpath/testpaths) lives in the root
pyproject.toml ``[tool.pytest.ini_options]`` so a bare ``uv run pytest`` behaves identically.
Extra CLI args are passed through, e.g. ``uv run --no-sync tests -k records -x``.
"""

from __future__ import annotations

import sys


def main() -> None:
    import pytest

    raise SystemExit(pytest.main(["models/immich/tests", *sys.argv[1:]]))


if __name__ == "__main__":
    main()
