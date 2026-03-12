"""Centralised logging setup."""
from __future__ import annotations

import logging
import sys
from typing import Optional


_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring the root handler on first call."""
    _configure_root()
    return logging.getLogger(name)
