#!/usr/bin/env python3
"""Full reset of generate-albums' local data (cache, manifests, scan state).

Thin entry-point shell. The implementation lives in :mod:`aviary_immich.clear_cache`; this module
exists only because the ``clear-cache`` console script is wired to ``clear_cache:main``.
"""

from aviary_immich.clear_cache import main

if __name__ == "__main__":
    main()
