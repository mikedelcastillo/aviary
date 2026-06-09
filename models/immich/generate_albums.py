#!/usr/bin/env python3
"""Generate per-account Immich Birds albums using a pretrained bird detector.

Thin entry-point shell. The implementation lives in :mod:`aviary_immich.cli`; this module exists
only because the ``generate-albums`` console script is wired to ``generate_albums:main``.
"""

from aviary_immich.cli import main

if __name__ == "__main__":
    main()
