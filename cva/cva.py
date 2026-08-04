"""
Credit curves, expected exposure, the CVA formula, and the error metrics.

    Q(t)  = exp(-int_0^t lambda(u) du)                                       (2)
    PD_i  = Q(t_{i-1}) - Q(t_i)                                              (3)
    EE*_i = E[ D_d(0,t_i) * max(V(t_i), 0) ]                                 (4)
    CVA   = (1-R) * sum_i EE*_i * PD_i                                       (5)

Unilateral, uncollateralised, and default is independent of the market factors
(no wrong-way risk).  Incremental CVA is CVA(portfolio + trade) - CVA(portfolio),
which is ORDER-DEPENDENT: throughout this project the new trade is added LAST.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .products import Trade, netting_set_values


# ----------------------------------------------------------------------------
# Credit
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class HazardCurve:
    """Piecewise-constant hazard rates: rates[k] applies on (pillars[k-1], pillars[k]],
    with the last rate extending to infinity."""

    rates: tuple[float, ...]
    pillars: tuple[float, ...] = config.HAZARD_PILLARS
    name: str = ""

    def __post_init__(self):
        if len(self.rates) != len(self.pillars) + 1:
            raise ValueError("need exactly one more hazard rate than pillars")

    def integrated_hazard(self, t):
        """Lambda(t): each segment contributes its rate times its overlap with [0,t]."""
        t = np.asarray(t, dtype=float)
        edges = np.concatenate([[0.0], np.asarray(self.pillars, dtype=float), [np.inf]])
        total = np.zeros_like(t)
        for k, rate in enumerate(self.rates):
            total = total + rate * np.maximum(np.clip(t, edges[k], edges[k + 1]) - edges[k], 0.0)
        return total

    def survival(self, t):
        """Q(t) = exp(-Lambda(t)), formula (2)."""
        return np.exp(-self.integrated_hazard(t))

    @property
    def average_hazard(self) -> float:
        """Flat-equivalent hazard over the horizon, for reporting only."""
        return float(self.integrated_hazard(config.HORIZON) / config.HORIZON)

    def describe(self) -> str:
        edges = [0.0, *self.pillars]
        return "  ".join(
            f"{'%g-%g' % (edges[k], self.pillars[k]) if k < len(self.pillars) else '>%g' % edges[k]}y: {r:.2%}"
            for k, r in enumerate(self.rates)
        )


def marginal_default_probabilities(times, curve: HazardCurve) -> np.ndarray:
    """PD(t_{i-1}, t_i) for i = 1..n, formula (3).  `times` must include t_0 = 0."""
    q = curve.survival(times)
    return q[:-1] - q[1:]


# ----------------------------------------------------------------------------
# Exposure
# ----------------------------------------------------------------------------

def expected_exposure(values: np.ndarray) -> np.ndarray:
    """EE(t_i) = mean over paths of max(netted value, 0).  Undiscounted; plotted."""
    return np.maximum(values, 0.0).mean(axis=0)


def potential_future_exposure(values: np.ndarray, quantile: float = 0.95) -> np.ndarray:
    """PFE(t_i): a quantile of the positive exposure.  Not used by CVA; plotted."""
    return np.quantile(np.maximum(values, 0.0), quantile, axis=0)


def discounted_ee(values, scen, market, discounting: str = config.DISCOUNTING) -> np.ndarray:
    """EE*(t_i), formula (4).

    "pathwise" keeps the stochastic discount factor inside the expectation, which
    is required under stochastic rates because the discount factor and the
    exposure are correlated.  "curve" is the separable textbook form used by the
    single-factor base model.  The two agree exactly when rates are deterministic.
    """
    if discounting == "pathwise":
        return (scen.df_domestic * np.maximum(values, 0.0)).mean(axis=0)
    if discounting == "curve":
        return market.p_domestic_0(scen.times) * expected_exposure(values)
    raise ValueError(f"discounting must be one of {config.DISCOUNTING_MODES}")


# ----------------------------------------------------------------------------
# CVA
# ----------------------------------------------------------------------------

def cva_from_discounted_ee(ee_star, times, hazard: HazardCurve,
                           recovery: float = config.RECOVERY_RATE) -> float:
    """Formula (5).  The sum starts at i = 1: (t_0, t_1] is the first bucket in
    which the counterparty can default, and EE*(t_0) is today's known MtM."""
    pd_marginal = marginal_default_probabilities(np.asarray(times, dtype=float), hazard)
    return float((1.0 - recovery) * np.sum(ee_star[1:] * pd_marginal))


def cva_from_values(values, scen, market, hazard: HazardCurve,
                    recovery: float = config.RECOVERY_RATE,
                    discounting: str = config.DISCOUNTING) -> float:
    """CVA of a netting set from its netted scenario matrix: (4) chained into (5)."""
    return cva_from_discounted_ee(
        discounted_ee(values, scen, market, discounting), scen.times, hazard, recovery
    )


def cva_standard_error(values, scen, hazard: HazardCurve,
                       recovery: float = config.RECOVERY_RATE) -> float:
    """Monte Carlo standard error of the pathwise CVA estimate.

    CVA is the mean of a per-path quantity, so its standard error is that
    sample's standard deviation over sqrt(P).
    """
    pd_marginal = marginal_default_probabilities(np.asarray(scen.times, dtype=float), hazard)
    per_path = (1.0 - recovery) * np.sum(
        scen.df_domestic[:, 1:] * np.maximum(values[:, 1:], 0.0) * pd_marginal[None, :], axis=1
    )
    return float(per_path.std(ddof=1) / np.sqrt(per_path.size))


def true_portfolio_cva(trades: list[Trade], scen, market, hazard: HazardCurve,
                       recovery: float = config.RECOVERY_RATE,
                       discounting: str = config.DISCOUNTING) -> float:
    """Ground truth: reprice every trade on every path and date, then net."""
    return cva_from_values(netting_set_values(trades, market, scen),
                           scen, market, hazard, recovery, discounting)


def true_incremental_cva(existing_trades, new_trade, scen, market, hazard: HazardCurve,
                         recovery: float = config.RECOVERY_RATE,
                         discounting: str = config.DISCOUNTING):
    """(incremental, cva_without, cva_with), both portfolios fully simulated."""
    without = true_portfolio_cva(existing_trades, scen, market, hazard, recovery, discounting)
    with_new = true_portfolio_cva(list(existing_trades) + [new_trade], scen, market,
                                  hazard, recovery, discounting)
    return with_new - without, without, with_new


# ----------------------------------------------------------------------------
# Error metrics
# ----------------------------------------------------------------------------

def summarise_errors(proxy_cva: float, true_cva: float,
                     floor: float = 1.0, tolerance: float = 1.0) -> dict:
    """Signed, absolute and relative error, plus the over/undercharging label.

    signed_error = proxy - truth, so POSITIVE means the proxy quotes more than
    the trade really costs (overcharging: the trade is lost to a competitor) and
    NEGATIVE means undercharging (the bank absorbs the difference).  The relative
    error is NaN when the true charge is below `floor`, where a ratio carries no
    information.
    """
    error = float(proxy_cva - true_cva)
    direction = "overcharge" if error > tolerance else (
        "undercharge" if error < -tolerance else "accurate")
    return {
        "signed_error": error,
        "abs_error": abs(error),
        "rel_error": float("nan") if abs(true_cva) < floor else error / abs(true_cva),
        "direction": direction,
    }


def speedup(reference_seconds: float, proxy_seconds: float) -> float:
    """How many times faster the proxy is than the full re-simulation."""
    return float("inf") if proxy_seconds <= 0 else float(reference_seconds / proxy_seconds)
