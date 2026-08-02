"""
The CVA calculation itself, and the "true" fully-simulated benchmark.

CVA formula used throughout the thesis
--------------------------------------
        CVA = (1 - R) * sum_{i=1..n} DF(t_i) * EE(t_i) * PD(t_{i-1}, t_i)      (5)

with
    R                 recovery rate (40%),
    DF(t_i)           risk-free USD discount factor from today to t_i,
    EE(t_i)           expected positive exposure of the NETTING SET, formula (4),
    PD(t_{i-1}, t_i)  marginal default probability of the counterparty, formula (3).

Everything is unilateral (no DVA, no FVA) and uncollateralised (no CSA).

Incremental CVA
---------------
        CVA_incremental(new trade) = CVA(portfolio + new trade) - CVA(portfolio)

This is ORDER-DEPENDENT: the allocation depends on which trades are already in
the book.  Throughout this project we always define it as adding the new trade
*last* to the existing netting set, i.e. the incoming trade pays the marginal
cost given everything that is already there.  A different ordering would give a
different (equally valid) split of the same total CVA.
"""

from __future__ import annotations

import numpy as np

from . import config
from .credit import marginal_default_probabilities
from .exposure import expected_exposure
from .instruments import Trade
from .market import MarketModel
from .scenarios import netting_set_scenario_values


def cva_from_exposure(
    ee: np.ndarray,
    times: np.ndarray,
    hazard: float,
    market: MarketModel,
    recovery: float = config.RECOVERY_RATE,
) -> float:
    """Evaluate formula (5) from a ready-made EE profile.

    Parameters
    ----------
    ee    : EE(t_i) for every date in `times`, including t_0 = 0.
    times : exposure grid including t_0 = 0.

    Notes
    -----
    The sum runs from i = 1: the bucket (t_0, t_1] is the first one in which the
    counterparty can default, and EE(t_0) is today's known MtM, not an exposure
    at risk.
    """
    times = np.asarray(times, dtype=float)

    df = market.df_usd(times[1:])                            # DF(t_i), USD curve
    pd_marginal = marginal_default_probabilities(times, hazard)   # PD(t_{i-1}, t_i)

    # CVA = (1-R) * sum_i DF(t_i) * EE(t_i) * PD(t_{i-1}, t_i)
    return float((1.0 - recovery) * np.sum(df * ee[1:] * pd_marginal))


def cva_from_values(
    values: np.ndarray,
    times: np.ndarray,
    hazard: float,
    market: MarketModel,
    recovery: float = config.RECOVERY_RATE,
) -> float:
    """CVA of a netting set given its NETTED scenario matrix.

    Chains formula (4) (EE) into formula (5) (CVA).
    """
    return cva_from_exposure(expected_exposure(values), times, hazard, market, recovery)


# ----------------------------------------------------------------------------
# The "true" benchmark: full simulation of the whole netting set
# ----------------------------------------------------------------------------

def true_portfolio_cva(
    trades: list[Trade],
    times: np.ndarray,
    spots: np.ndarray,
    market: MarketModel,
    hazard: float,
    recovery: float = config.RECOVERY_RATE,
) -> float:
    """Ground truth: reprice EVERY trade on EVERY path and date, then net.

    This is the expensive calculation the two proxies are trying to avoid.
    """
    values = netting_set_scenario_values(trades, times, spots, market)
    return cva_from_values(values, times, hazard, market, recovery)


def true_incremental_cva(
    existing_trades: list[Trade],
    new_trade: Trade,
    times: np.ndarray,
    spots: np.ndarray,
    market: MarketModel,
    hazard: float,
    recovery: float = config.RECOVERY_RATE,
) -> tuple[float, float, float]:
    """True incremental CVA of `new_trade`, both portfolios fully simulated.

    Returns
    -------
    (incremental, cva_without, cva_with)

    The new trade is added LAST (see the module docstring on order dependence).
    Both portfolio CVAs are recomputed from scratch - no stored values are
    reused - so this function also serves as the timing reference for a "full
    re-simulation" pricing request.
    """
    cva_without = true_portfolio_cva(existing_trades, times, spots, market, hazard, recovery)
    cva_with = true_portfolio_cva(
        list(existing_trades) + [new_trade], times, spots, market, hazard, recovery
    )
    return cva_with - cva_without, cva_without, cva_with
