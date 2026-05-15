from __future__ import annotations

import json
import logging
from typing import Any


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, **payload: Any) -> None:
    serialized = json.dumps(payload, default=str, sort_keys=True)
    logger.info("%s | %s", event, serialized)
