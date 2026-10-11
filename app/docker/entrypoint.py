from __future__ import annotations

import logging
import os

import uvicorn

from edge_workbench.app import create_app


def _restore_logging_levels() -> None:
    # rknnlite's native logger replaces Python's standard level-name maps with
    # single-letter RKNN levels during init_runtime(). Uvicorn uses INFO/etc.
    standard = {
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.FATAL,
        "ERROR": logging.ERROR,
        "WARN": logging.WARN,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    logging._nameToLevel.update(standard)
    logging._levelToName.update({value: name for name, value in standard.items()})


app = create_app()
_restore_logging_levels()
uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "18768")))
