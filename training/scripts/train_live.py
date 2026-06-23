#!/usr/bin/env python3
"""Build the live (real-time CCTV) bird detector end-to-end.

prepare dataset from labeled raw data -> train -> export data/models/live-NNN.pt
Extra flags pass through to training. See train_pipeline.run.
"""

from __future__ import annotations

import sys

from train_pipeline import run


def main(argv: list[str] | None = None) -> None:
    run("live", argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    main()
