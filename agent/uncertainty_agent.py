"""Risk-aware probabilistic forecast utilities (V4 minimal slice).

This module intentionally stays model-agnostic: Wind/Price replay code can
feed quantile predictions into :class:`UncertaintyAgent` without changing the
existing point-forecast backends.  A future Quantile LightGBM backend only
needs to produce a ``(n, 3)`` array/DataFrame for q10/q50/q90.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_QUANTILES: Tuple[float, ...] = (0.1, 0.5, 0.9)


def _as_1d(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def pinball_loss(y_true: Sequence[float], y_pred: Sequence[float], quantile: float) -> float:
    """Mean pinball loss for one quantile (lower is better)."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    y = _as_1d(y_true, "y_true")
    p = _as_1d(y_pred, "y_pred")
    if y.size != p.size:
        raise ValueError("y_true and y_pred must have the same length")
    err = y - p
    return float(np.mean(np.maximum(quantile * err, (quantile - 1.0) * err)))


def _normalise_quantiles(
    predictions: Mapping[float, Sequence[float]] | Sequence[Sequence[float]] | np.ndarray,
    quantiles: Iterable[float],
) -> Dict[float, np.ndarray]:
    qs = tuple(float(q) for q in quantiles)
    if not qs or any(not 0.0 < q < 1.0 for q in qs):
        raise ValueError("quantiles must be non-empty values in (0, 1)")
    if isinstance(predictions, Mapping):
        out = {q: _as_1d(predictions[q], f"prediction q{q:g}") for q in qs}
    else:
        arr = np.asarray(predictions, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != len(qs):
            raise ValueError("predictions must have shape (n_samples, n_quantiles)")
        if not np.isfinite(arr).all():
            raise ValueError("predictions contains non-finite values")
        out = {q: arr[:, i] for i, q in enumerate(qs)}
    n = len(next(iter(out.values())))
    if any(len(v) != n for v in out.values()):
        raise ValueError("all quantile predictions must have the same length")
    return out


def probabilistic_metrics(
    y_true: Sequence[float],
    predictions: Mapping[float, Sequence[float]] | Sequence[Sequence[float]] | np.ndarray,
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
    interval: Tuple[float, float] = (0.1, 0.9),
) -> Dict[str, float]:
    """Compute V4's minimum comparable metric set.

    ``crps`` is the standard quantile approximation ``2 * mean(pinball)`` over
    supplied quantiles.  ``coverage`` and ``interval_width`` refer to the
    requested lower/upper quantiles (usually q10/q90, i.e. a 80% interval).
    """
    y = _as_1d(y_true, "y_true")
    pred = _normalise_quantiles(predictions, quantiles)
    if any(len(v) != len(y) for v in pred.values()):
        raise ValueError("y_true and predictions must have the same length")
    lo_q, hi_q = map(float, interval)
    if lo_q >= hi_q or lo_q not in pred or hi_q not in pred:
        raise ValueError("interval quantiles must be present and ordered")
    pinballs = {q: pinball_loss(y, p, q) for q, p in pred.items()}
    lo, hi = pred[lo_q], pred[hi_q]
    # Quantile crossing is reported, not silently repaired: it is a model issue.
    crossing_rate = float(np.mean(lo > hi))
    covered = (y >= lo) & (y <= hi)
    result: Dict[str, float] = {
        "nominal_coverage": float(hi_q - lo_q),
        "coverage": float(np.mean(covered)),
        "interval_width": float(np.mean(hi - lo)),
        "crossing_rate": crossing_rate,
        "crps": float(2.0 * np.mean(list(pinballs.values()))),
        "mean_pinball": float(np.mean(list(pinballs.values()))),
    }
    for q, value in pinballs.items():
        result[f"pinball_q{q:g}"] = float(value)
    return result


@dataclass
class ConformalCalibrator:
    """Split-conformal additive calibration for a central prediction interval.

    Calibration uses non-conformity ``max(lower - y, y - upper, 0)``.  The
    finite-sample quantile ``ceil((n+1)*(1-alpha))/n`` gives the usual
    marginal coverage guarantee under exchangeability.
    """

    alpha: float = 0.1
    radius: float | None = None
    n_calibration: int = 0

    def fit(
        self,
        y_true: Sequence[float],
        lower: Sequence[float],
        upper: Sequence[float],
    ) -> "ConformalCalibrator":
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        y = _as_1d(y_true, "y_true")
        lo = _as_1d(lower, "lower")
        hi = _as_1d(upper, "upper")
        if not (len(y) == len(lo) == len(hi)):
            raise ValueError("calibration arrays must have the same length")
        scores = np.maximum.reduce([lo - y, y - hi, np.zeros_like(y)])
        n = len(scores)
        # Method 'higher' implements the conservative finite-sample quantile.
        rank = min(max(int(np.ceil((n + 1) * (1.0 - self.alpha))), 1), n)
        self.radius = float(np.sort(scores)[rank - 1])
        self.n_calibration = n
        return self

    def apply(self, lower: Sequence[float], upper: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        if self.radius is None:
            raise RuntimeError("calibrator must be fit before apply")
        lo = _as_1d(lower, "lower")
        hi = _as_1d(upper, "upper")
        if len(lo) != len(hi):
            raise ValueError("lower and upper must have the same length")
        return lo - self.radius, hi + self.radius


@dataclass
class UncertaintyAgent:
    """Small orchestration facade for replay integrations.

    ``calibrate`` consumes a held-out validation window only; ``score`` can be
    called on the subsequent forecast window.  No model fitting or data access
    occurs here, keeping this class safe to add alongside existing replayers.
    """

    alpha: float = 0.1
    quantiles: Tuple[float, ...] = DEFAULT_QUANTILES
    calibrator: ConformalCalibrator | None = None

    def calibrate(self, y_true: Sequence[float], predictions) -> Dict[str, float]:
        pred = _normalise_quantiles(predictions, self.quantiles)
        lo_q, hi_q = self.quantiles[0], self.quantiles[-1]
        self.calibrator = ConformalCalibrator(alpha=self.alpha).fit(
            y_true, pred[lo_q], pred[hi_q]
        )
        return {"radius": float(self.calibrator.radius), "n_calibration": float(self.calibrator.n_calibration)}

    def score(self, y_true: Sequence[float], predictions) -> Dict[str, float]:
        metrics = probabilistic_metrics(y_true, predictions, self.quantiles)
        pred = _normalise_quantiles(predictions, self.quantiles)
        if self.calibrator is not None:
            lo, hi = self.calibrator.apply(pred[self.quantiles[0]], pred[self.quantiles[-1]])
            y = _as_1d(y_true, "y_true")
            metrics["conformal_coverage"] = float(np.mean((y >= lo) & (y <= hi)))
            metrics["conformal_interval_width"] = float(np.mean(hi - lo))
        return metrics

    def interval(self, predictions) -> Tuple[np.ndarray, np.ndarray]:
        pred = _normalise_quantiles(predictions, self.quantiles)
        lo, hi = pred[self.quantiles[0]], pred[self.quantiles[-1]]
        if self.calibrator is None:
            return lo, hi
        return self.calibrator.apply(lo, hi)
