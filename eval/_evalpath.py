"""Restore the flat ``eval/`` import namespace after the 2026-07-29 reorganisation.

Every ``eval/`` module is imported by bare name (``import instance_io``,
``import common_metric``, ...) from the other eval modules, from ``tests/``,
from ``paper/figscripts/`` and from ``sandbox/`` tooling. Those modules now live
in subdirectories, so a bare import no longer resolves from ``eval/`` alone.

Importing this module puts every eval code subdirectory on ``sys.path``, which
keeps all of those bare imports working unchanged. Use it immediately after
adding ``eval/`` to ``sys.path``::

    sys.path.insert(0, str(ROOT / "eval"))
    import _evalpath  # noqa: F401

Subdirectory names are deliberately chosen not to collide with any stdlib module
or with any eval module name: a directory called ``statistics`` would shadow the
stdlib module of that name, and a directory called ``baselines`` would shadow
``baselines.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parent

#: Code subdirectories, in the order they should take precedence.
SUBDIRS = (
    "core",
    "baseline",
    "runners",
    "stats",
    "generators",
    "manifests",
    "diagnostics",
    "studies",
)


def install() -> None:
    """Idempotently place every eval code subdirectory on ``sys.path``."""
    for name in SUBDIRS:
        path = str(_EVAL / name)
        if path not in sys.path:
            sys.path.insert(0, path)
    root = str(_EVAL)
    if root not in sys.path:
        sys.path.insert(0, root)


install()
