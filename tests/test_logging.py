"""Tests for the JSON log format."""

import json
import logging

from app.json_logging import JsonFormatter, get_logger


def make_record(message, fields=None):
    record = logging.LogRecord("mp203-api", logging.INFO, __file__, 1, message, None, None)
    if fields is not None:
        record.fields = fields
    return record


def test_log_line_is_json_with_fields():
    line = JsonFormatter().format(make_record("prediction", {"prediction": 123.0}))
    parsed = json.loads(line)                # fails if it is not valid JSON
    assert parsed["message"] == "prediction"
    assert parsed["prediction"] == 123.0
    assert parsed["level"] == "INFO"


def test_log_line_without_fields():
    parsed = json.loads(JsonFormatter().format(make_record("model loaded")))
    assert parsed["message"] == "model loaded"


def test_logger_has_one_json_handler():
    get_logger()
    logger = get_logger()                    # calling twice must not duplicate lines
    ours = [h for h in logger.handlers if isinstance(h.formatter, JsonFormatter)]
    assert len(ours) == 1                    # (pytest adds its own capture handlers too)
