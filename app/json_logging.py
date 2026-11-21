"""Structured (JSON) logging for the prediction service.

Every log line is one JSON object written to stdout, never to a file.
Docker and Kubernetes collect stdout (`docker logs`, `kubectl logs`), and a
log tool can search JSON fields such as "prediction" or "latency_ms".

Usage:
    logger = get_logger()
    logger.info("prediction", extra={"fields": {"prediction": 21000.0}})
"""

import json
import logging
import sys


class JsonFormatter(logging.Formatter):
    """Turn a log record into one line of JSON."""

    def format(self, record):
        entry = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # extra={"fields": {...}} in a log call adds structured fields
        entry.update(getattr(record, "fields", {}))
        return json.dumps(entry)


def get_logger(name="mp203-api"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:                  # avoid duplicate lines if imported twice
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.propagate = False                 # don't also print through the root logger
    return logger
