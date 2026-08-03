"""
Counterparty credit: survival curve and marginal default probabilities.

Standard reduced-form (hazard-rate) approach, now with a piecewise-constant TERM
STRUCTURE of hazard rates rather than a single constant.  Investment-grade
credit curves are normally upward sloping, so a long-dated trade should be
charged a higher marginal default probability per unit of time than a short one.

    Lambda(t) = int_0^t lambda(u) du        (integrated hazard)
    Q(t)      = P(tau > t) = exp(-Lambda(t))                                 (2)
    PD(t_{i-1}, t_i) = Q(t_{i-1}) - Q(t_i)                                   (3)

These marginal PDs are the weights of the CVA sum in `cva_engine.py`.

Independence assumption: default is taken to be independent of all three market
factors, so no wrong-way risk is modelled.  That lets us multiply the discounted
exposure and the marginal PD term by term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass(frozen=True)
class HazardCurve:
    """Piecewise-constant hazard rates.

    `rates[k]` applies on the interval (pillars[k-1], pillars[k]], with
    pillars[-1] = infinity.  So with pillars = (1, 3) and rates = (a, b, c),
    the hazard is a on (0,1], b on (1,3] and c beyond 3 years.
    """

    rates: tuple[float, ...]
    pillars: tuple[float, ...] = config.HAZARD_PILLARS
    name: str = ""

    def __post_init__(self) -> None:
        if len(self.rates) != len(self.pillars) + 1:
            raise ValueError("need exactly one more hazard rate than pillars")

    @property
    def average_hazard(self) -> float:
        """Flat-equivalent hazard over the 5y horizon, for reporting only."""
        return float(self.integrated_hazard(config.HORIZON) / config.HORIZON)

    def integrated_hazard(self, t: np.ndarray | float) -> np.ndarray:
        """Lambda(t) = int_0^t lambda(u) du.

        Accumulated segment by segment: each segment contributes its rate times
        the length of its overlap with [0, t].
        """
        t = np.asarray(t, dtype=float)
        edges = np.concatenate([[0.0], np.asarray(self.pillars, dtype=float), [np.inf]])
        total = np.zeros_like(t)
        for k, rate in enumerate(self.rates):
            lo, hi = edges[k], edges[k + 1]
            overlap = np.clip(t, lo, hi) - lo          # length of [lo,hi] covered by [0,t]
            total = total + rate * np.maximum(overlap, 0.0)
        return total

    def survival(self, t: np.ndarray | float) -> np.ndarray:
        """Q(t) = exp(-Lambda(t)), formula (2)."""
        return np.exp(-self.integrated_hazard(t))

    def describe(self) -> str:
        segments = []
        edges = [0.0, *self.pillars]
        for k, rate in enumerate(self.rates):
            lo = edges[k]
            hi = self.pillars[k] if k < len(self.pillars) else None
            span = f"{lo:g}-{hi:g}y" if hi is not None else f">{lo:g}y"
            segments.append(f"{span}: {rate:.2%}")
        return "  ".join(segments)


def marginal_default_probabilities(times: np.ndarray, curve: HazardCurve) -> np.ndarray:
    """PD(t_{i-1}, t_i) for i = 1..n, formula (3).

    Parameters
    ----------
    times : the exposure grid *including* t_0 = 0.

    Returns
    -------
    Array of length len(times) - 1: the default probability attached to each
    exposure date from t_1 onwards.
    """
    q = curve.survival(times)
    return q[:-1] - q[1:]
