"""Synthetic contract tests for the V4 uncertainty-agent slice."""

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.uncertainty_agent import (  # noqa: E402
    ConformalCalibrator,
    UncertaintyAgent,
    pinball_loss,
    probabilistic_metrics,
)


def test_pinball_and_metrics():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    pred = np.column_stack([y - 1.0, y, y + 1.0])
    assert pinball_loss(y, pred[:, 1], 0.5) == 0.0
    m = probabilistic_metrics(y, pred)
    assert m["coverage"] == 1.0
    assert m["interval_width"] == 2.0
    assert m["crossing_rate"] == 0.0
    assert m["crps"] >= 0.0


def test_conformal_expands_only_as_needed():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    lo = y - 0.5
    hi = y + 0.5
    c = ConformalCalibrator(alpha=0.1).fit(y, lo, hi)
    assert c.radius == 0.0
    lo2, hi2 = c.apply(np.array([10.0]), np.array([11.0]))
    assert (lo2 == np.array([10.0])).all()
    assert (hi2 == np.array([11.0])).all()


def test_agent_calibrate_interval_and_score():
    y = np.arange(10.0)
    # Deliberately under-dispersed calibration interval; conformal radius=1.
    pred = np.column_stack([y - 0.5, y, y + 0.5])
    agent = UncertaintyAgent(alpha=0.1)
    info = agent.calibrate(y, pred)
    assert info["n_calibration"] == 10.0
    assert info["radius"] >= 0.0
    lo, hi = agent.interval(pred)
    assert np.all(lo <= hi)
    scored = agent.score(y, pred)
    assert "conformal_coverage" in scored
    assert scored["conformal_coverage"] >= scored["coverage"]


if __name__ == "__main__":
    test_pinball_and_metrics()
    test_conformal_expands_only_as_needed()
    test_agent_calibrate_interval_and_score()
    print("ALL UNCERTAINTY TESTS PASSED")

