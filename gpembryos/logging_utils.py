"""Thin logging wrapper.

Usage:
    from gpembryos.logging_utils import info, init_logging
    init_logging(level="debug", filename="run.log")
    info("hello")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOGGER_NAME = "gpembryos"
_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def init_logging(level: str = "info", filename: str | Path | None = None) -> logging.Logger:
    """Initialise the project logger.

    Idempotent: calling again replaces handlers (useful when resuming).
    Writes to stdout always; also to `filename` if provided.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(_LEVELS.get(level.lower(), logging.INFO))
    logger.propagate = False

    # Clear existing handlers to make re-init safe.
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if filename is not None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(filename)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def _get() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        init_logging()
    return logger


def debug(msg: str, *args, **kwargs) -> None:
    _get().debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs) -> None:
    _get().info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    _get().warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    _get().error(msg, *args, **kwargs)
