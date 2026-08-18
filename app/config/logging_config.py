from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from json import dumps
from typing import Any

from app.utils.request_context import get_request_id


class JsonLogFormatter(logging.Formatter):
    """Renders log records as single-line JSON for log-aggregator friendliness."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key in ("args", "msg", "exc_info", "exc_text", "stack_info") or key in payload:
                continue
            if key.startswith("_"):
                continue
            if key in logging.LogRecord.__dict__:
                continue
            payload[key] = value

        return dumps(payload, default=str)


def configure_logging(log_level: str = "INFO", json_format: bool = True) -> None:
    root = logging.getLogger()
    root.setLevel(log_level.upper())
    root.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    if json_format:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

    root.addHandler(handler)

    for noisy_logger in ("uvicorn.access",):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
