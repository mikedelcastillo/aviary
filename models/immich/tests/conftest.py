"""Suite-wide setup for the immich tests.

``aviary_immich.config`` resolves ``IMMICH_CLIP=auto`` by probing whether ``open_clip`` is
installed, so the module-level ``MODELS``/``ALBUM_RULES``/``album_names()`` would otherwise differ
between a dev box with the CLIP stack and a CI box without it. Pin the flag to ``0`` here so those
module-level constants are a deterministic CLIP-off baseline everywhere; the CLIP-on behavior is
covered explicitly via ``album_rules(True)`` / ``model_specs(True)`` in ``test_config``.

This module is imported before the test modules (and thus before they import
``aviary_immich.config``), so the variable is set in time.
"""

import os

os.environ["IMMICH_CLIP"] = "0"
