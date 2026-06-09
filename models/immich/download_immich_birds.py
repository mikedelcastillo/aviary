#!/usr/bin/env python3
"""Download all images from each configured account's Immich Birds album.

Thin entry-point shell. The implementation lives in :mod:`aviary_immich.download`; this module
exists only because the ``download-birds`` console script is wired to ``download_immich_birds:main``.
"""

from aviary_immich.download import main

if __name__ == "__main__":
    main()
