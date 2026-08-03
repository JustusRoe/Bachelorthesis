"""
The CVA calculation itself, and the "true" fully-simulated benchmark.

CVA formula
-----------
        CVA = (1 - R) * sum_{i=1..n} EE*(t_i) * PD(t_{i-1}, t_i)             (5)

with
    R                 recovery rate (40%),
    EE*(t_i)          DISCOUNTED expected positive exposure of the netting set,
    PD(t_{i-1}, t_i)  marginal default probability of the counterparty, formula (3).

Two ways of forming EE* are supported, selected by `discounting`:

    "pathwise"  EE*(t_i) = E[ D_d(0,t_i) * max(V(t_i),0) ]                   (4')
    "curve"     EE*(t_i) = P_d(0,t_i) * E[ max(V(t_i),0) ]                   (4)

Under deterministic rates the two are identical, because D_d(0,t_i) then takes
the same value P_d(0,t_i) on every path.  Under stochastic rates only the
pathwise form is correct: the discount factor and the exposure are correlated,
so E[D * max(V,0)] != E[D] * E[max(V,0)].  The "curve" form is retained because
it is the textbook expression the single-factor base model used, and the gap
between the two is a result worth reporting in its own right.

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
from .credit import HazardCurve, marginal_default_probabilities
from .exposure import discounted_expected_exposure, expected_exposure
from .instruments import Trade
from .scenarios import netting_set_scenario_values


def discounted_ee(
    values: np.ndarray,
    scen,
    market,
    discounting: str = config.DISCOUNTING,
) -> np.ndarray:
    """Form EE*(t_i) according to the chosen discounting convention."""
    if discounting == "pathwise":
        return discounted_expected_exposure(values, scen.df_domestic)
    if discounting == "curve":
        return market.p_domestic_0(scen.times) * expected_exposure(values)
    raise ValueError(f"discounting must be one of {config.DISCOUNTING_MODES}")


def cva_from_discounted_ee(
    ee_star: np.ndarray,
    times: np.ndarray,
    hazard: HazardCurve,
    recovery: float = config.RECOVERY_RATE,
) -> float:
    """Evaluate formula (5) from a ready-made discounted EE profile.

    The sum runs from i = 1: the bucket (t_0, t_1] is the first one in which the
    counterparty can default, and EE*(t_0) is today's known MtM, not an exposure
    at risk.
    """
    pd_marginal = marginal_default_probabilities(np.asarray(times, dtype=float), hazard)
    # CVA = (1-R) * sum_i EE*(t_i) * PD(t_{i-1}, t_i)
    return float((1.0 - recovery) * np.sum(ee_star[1:] * pd_marginal))


def cva_from_values(
    values: np.ndarray,
    scen,
    market,
    hazard: HazardCurve,
    recovery: float = config.RECOVERY_RATE,
    discounting: str = config.DISCOUNTING,
) -> float:
    """CVA of a netting set given its NETTED scenario matrix.

    Chains formula (4') into formula (5).
    """
    ee_star = discounted_ee(values, scen, market, discounting)
    return cva_from_discounted_ee(ee_star, scen.times, hazard, recovery)


def cva_standard_error(
    values: np.ndarray,
    scen,
    hazard: HazardCurve,
    recovery: float = config.RECOVERY_RATE,
) -> float:
    """Monte Carlo standard error of the pathwise CVA estimate.

    The CVA is a sum over dates of path averages, so it is itself the mean of a
    per-path quantity

        c_p = (1-R) * sum_i D_d(0,t_i)[p] * max(V[p,i],0) * PD_i

    and its standard error is the sample standard deviation of c_p over sqrt(P).
    Reported alongside every headline CVA so the reader can see how much of a
    difference is signal and how much is simulation noise.
    """
    times = np.asarray(scen.times, dtype=float)
    pd_marginal = marginal_default_probabilities(times, hazard)
    per_path = (1.0 - recovery) * np.sum(
        scen.df_domestic[:, 1:] * np.maximum(values[:, 1:], 0.0) * pd_marginal[None, :],
        axis=1,
    )
    return float(per_path.std(ddof=1) / np.sqrt(per_path.size))


# ----------------------------------------------------------------------------
# The "true" benchmark: full simulation of the whole netting set
# ----------------------------------------------------------------------------

def true_portfolio_cva(
    trades: list[Trade],
    scen,
    market,
    hazard: HazardCurve,
    recovery: float = config.RECOVERY_RATE,
    discounting: str = config.DISCOUNTING,
) -> float:
    """Ground truth: reprice EVERY trade on EVERY path and date, then net.

    This is the expensive calculation the two proxies are trying to avoid.
    """
    values = netting_set_scenario_values(trades, market, scen)
    return cva_from_values(values, scen, market, hazard, recovery, discounting)


def true_incremental_cva(
    existing_trades: list[Trade],
    new_trade: Trade,
    scen,
    market,
    hazard: HazardCurve,
    recovery: float = config.RECOVERY_RATE,
    discounting: str = config.DISCOUNTING,
) -> tuple[float, float, float]:
    """True incremental CVA of `new_trade`, both portfolios fully simulated.

    Returns (incremental, cva_without, cva_with).  The new trade is added LAST
    (see the module docstring on order dependence).
    """
    cva_without = true_portfolio_cva(existing_trades, scen, market, hazard, recovery, discounting)
    cva_with = true_portfolio_cva(
        list(existing_trades) + [new_trade], scen, market, hazard, recovery, discounting
    )
    return cva_with - cva_without, cva_without, cva_with
