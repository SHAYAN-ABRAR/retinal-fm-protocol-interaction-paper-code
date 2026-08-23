"""Console + file logging with a single, consistent format.

Deliberately thin: this project logs a small number of meaningful lines per
epoch, not a flood.  Named ``dr_dg`` so that noisy third-party loggers
(matplotlib font manager, PIL) can be silenced independently.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

__all__ = ["get_logger", "configure_logging"]

_LOGGER_NAME = "dr_dg"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
_DATEFMT = "%H:%M:%S"
_NOISY = ("matplotlib", "matplotlib.font_manager", "PIL", "numba", "urllib3")


def configure_logging(
    log_file: Path | str | None = None,
    *,
    level: int = logging.INFO,
    quiet_third_party: bool = True,
) -> logging.Logger:
    """Configure and return the project logger.

    Idempotent: calling it repeatedly (as happens when re-running a notebook
    cell) replaces the handlers rather than stacking duplicates.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if quiet_third_party:
        for name in _NOISY:
            logging.getLogger(name).setLevel(logging.WARNING)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the project logger, or a child of it."""
    logger = logging.getLogger(_LOGGER_NAME if name is None else f"{_LOGGER_NAME}.{name}")
    if not logging.getLogger(_LOGGER_NAME).handlers:
        configure_logging()
    return logger
