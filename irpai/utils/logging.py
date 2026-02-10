"""
Logging utility for IrpAI.

Provides a consistent logger with coloured output and layer-level
granularity (e.g. ``irpai.ocr``, ``irpai.detection``).
"""

from __future__ import annotations

import logging
import sys


_LOG_FORMAT = "[%(levelname).1s %(name)s] %(message)s"
_CONFIGURED = False


def _configure_root(verbose: bool = False) -> None:
    """Set up the ``irpai`` root logger once."""
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return

    root = logging.getLogger("irpai")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str, *, verbose: bool = False) -> logging.Logger:
    """
    Return a child logger under the ``irpai`` namespace.

    Parameters
    ----------
    name:
        Dot-separated suffix, e.g. ``"ocr"`` → logger ``irpai.ocr``.
    verbose:
        If *True*, set root logger to DEBUG on first call.
    """
    _configure_root(verbose=verbose)
    return logging.getLogger(f"irpai.{name}")
