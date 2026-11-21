"""Tests for the drift measure (PSI)."""

import numpy as np

from pgrent.monitor import psi


def test_psi_is_zero_for_same_data():
    data = np.arange(1000)
    assert psi(data, data) < 1e-9


def test_psi_is_large_for_shifted_data():
    rng = np.random.default_rng(0)
    before = rng.normal(250, 50, 1000)        # like room_area_sqft in train
    after = rng.normal(400, 50, 300)          # rooms got much bigger
    assert psi(before, after) > 0.25          # "big shift" by the rule of thumb


def test_psi_small_for_similar_data():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(0, 1, 2000), rng.normal(0, 1, 2000)) < 0.1
