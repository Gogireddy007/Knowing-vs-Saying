"""Unit tests for MiniGrid backend provenance."""
from __future__ import annotations

from minigrid_val.validate import minigrid_backend, _try_import_minigrid


def test_minigrid_backend_matches_importability():
    expected = "minigrid" if _try_import_minigrid() is not None \
        else "synthetic_fallback"
    assert minigrid_backend() == expected


def test_minigrid_backend_is_one_of_two_values():
    assert minigrid_backend() in ("minigrid", "synthetic_fallback")
