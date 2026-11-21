"""Tests for the data checks and cleaning, on tiny tables built here
(the real CSVs are not in git, so tests must not need them)."""

import pandas as pd

from pgrent.data_checks import (
    find_duplicates,
    find_stuck_sensor,
    find_target_typos,
    longest_run,
    normalise_label,
)


def test_normalise_label_fixes_spellings():
    levels = ["full", "semi", "unfurnished"]
    for messy in ["semi", " semi", "SEMI", "Semi ", "sem."]:
        assert normalise_label(messy, levels) == "semi"
    assert normalise_label("unf.", levels) == "unfurnished"
    assert normalise_label("ful.", levels) == "full"


def test_duplicates_count_extra_copies():
    df = pd.DataFrame({"record_id": ["a", "b", "b", "b"], "x": [1, 2, 2, 2]})
    # one original + two extra copies of "b"
    assert find_duplicates(df, "record_id") == {"present": True, "count": 2}


def test_no_duplicates():
    df = pd.DataFrame({"record_id": ["a", "b"], "x": [1, 2]})
    assert find_duplicates(df, "record_id")["present"] is False


def test_longest_run():
    assert longest_run([1, 2, 2, 2, 3, 3]) == (1, 3)


def test_stuck_sensor_dates():
    dates = pd.date_range("2026-01-01", periods=20).strftime("%Y-%m-%d")
    values = list(range(5)) + [7.0] * 10 + list(range(5))      # stuck on days 6-15
    df = pd.DataFrame({"record_id": range(20), "date": dates, "km": values})
    found = find_stuck_sensor(df, "date", "record_id", ["km"], min_days=7)
    assert found["present"] is True
    assert found["start_date"] == "2026-01-06"
    assert found["end_date"] == "2026-01-15"


def test_target_typos_found_above_gap():
    rents = pd.Series([10000, 12000, 15000, 18000, 20000, 150000, 180000])
    df = pd.DataFrame({"rent_rs": rents})
    assert find_target_typos(df, "rent_rs", gap_ratio=3) == {"present": True, "count": 2}
