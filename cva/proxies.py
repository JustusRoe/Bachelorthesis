"""
The two fast pricing proxies.

Both proxies answer the same business question - "what CVA should we charge for
this incoming trade?" - but they differ in how much of the existing netting set
they take into account.

STANDALONE PROXY
    Prices the new trade in ISOLATION, as if it were the only trade with the
    counterparty:
        CVA_standalone = (1-R) * sum_i DF(t_i) * E[max(V_new(t_i), 0)] * PD_i
    Because max(.,0) is applied to the single trade rather than to the netted
    portfolio, it is always >= 0 and it structurally IGNORES the netting
    benefit.  Expected to overcharge.

INCREMENTAL PROXY
    Prices the new trade against the PRE-COMPUTED scenario values of its netting
    set (the matrix a bank stores from the overnight batch):
        CVA_incremental = CVA(V_stored + V_new) - CVA(V_stored)
    Netting is handled EXACTLY here; the only approximation is that V_new comes
    from the interpolant rather than from a full repricing.  It can be negative
    for a risk-reducing trade.

Both proxies reuse the already-simulated spot paths - regenerating the risk
factors is part of the overnight batch, not of an intraday pricing request.
Only the work that a real-time request would actually have to do is timed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import config
from .cva_engine import cva_from_exposure, cva_from_values
from .exposure import expected_exposure
from .instruments import Trade
from .interpolation import ValueInterpolant
from .market import MarketModel
from .scenarios import netting_set_scenario_values as netting_set_values


@dataclass
class ProxyResult:
    """Outcome of one pricing request."""

    cva: float                  # the charge the proxy would quote, in USD
    seconds: float              # wall-clock time of the request (best of N runs)
    n_exact_evaluations: int    # exact analytic prices needed to build the proxy


def spot_range(spots: np.ndarray, buffer: float = config.SPOT_GRID_BUFFER) -> tuple[float, float]:
    """Spot interval the interpolation nodes should span.

    Taken from the simulated scenarios themselves and widened by `buffer` so
    that no path ever falls outside the node box.  Avoiding extrapolation keeps
    the linear/Chebyshev comparison about approximation quality rather than
    about extrapolation stability.
    """
    lo, hi = float(spots.min()), float(spots.max())
    return lo * (1.0 - buffer), hi * (1.0 + buffer)


def _timed(fn, repeats: int = config.N_TIMING_REPEATS):
    """Run `fn` `repeats` times and return (result, best wall-clock seconds).

    The minimum is used rather than the mean: it is the standard robust estimate
    of the intrinsic cost, least polluted by OS scheduling noise.
    """
    best = np.inf
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return result, float(best)


# ----------------------------------------------------------------------------
# Standalone proxy
# ----------------------------------------------------------------------------

def standalone_proxy(
    new_trade: Trade,
    times: np.ndarray,
    spots: np.ndarray,
    market: MarketModel,
    hazard: float,
    spot_lo: float,
    spot_hi: float,
    method: str = config.DEFAULT_INTERP_METHOD,
    n_time_nodes: int = config.N_TIME_NODES,
    n_spot_nodes: int = config.N_SPOT_NODES,
    split_legs: bool = True,
    recovery: float = config.RECOVERY_RATE,
) -> ProxyResult:
    """CVA of the incoming trade priced on its own, with no netting."""
    holder: dict[str, ValueInterpolant] = {}

    def request() -> float:
        interp = ValueInterpolant(
            new_trade, market, spot_lo, spot_hi,
            n_time_nodes=n_time_nodes, n_spot_nodes=n_spot_nodes,
            method=method, split_legs=split_legs,
        )
        holder["interp"] = interp
        v_new = interp.scenario_values(times, spots)
        # EE of the SINGLE trade: max(.,0) taken trade by trade -> no netting.
        ee = expected_exposure(v_new)
        return cva_from_exposure(ee, times, hazard, market, recovery)

    cva, seconds = _timed(request)
    return ProxyResult(cva=cva, seconds=seconds,
                       n_exact_evaluations=holder["interp"].n_exact_evaluations)


# ----------------------------------------------------------------------------
# Incremental proxy
# ----------------------------------------------------------------------------

def incremental_proxy(
    new_trade: Trade,
    stored_values: np.ndarray,
    stored_cva: float,
    times: np.ndarray,
    spots: np.ndarray,
    market: MarketModel,
    hazard: float,
    spot_lo: float,
    spot_hi: float,
    method: str = config.DEFAULT_INTERP_METHOD,
    n_time_nodes: int = config.N_TIME_NODES,
    n_spot_nodes: int = config.N_SPOT_NODES,
    split_legs: bool = True,
    recovery: float = config.RECOVERY_RATE,
) -> ProxyResult:
    """Incremental CVA against the stored scenario values of the netting set.

    Parameters
    ----------
    stored_values : (n_paths, n_times) netted MtM of the EXISTING book, produced
                    by the overnight batch and reused here for free.
    stored_cva    : the netting set's CVA implied by `stored_values`, likewise
                    known from the batch, so it is not recomputed at request time.

    The new trade is added LAST to the netting set (order-dependence convention,
    see `cva_engine`).
    """
    holder: dict[str, ValueInterpolant] = {}

    def request() -> float:
        interp = ValueInterpolant(
            new_trade, market, spot_lo, spot_hi,
            n_time_nodes=n_time_nodes, n_spot_nodes=n_spot_nodes,
            method=method, split_legs=split_legs,
        )
        holder["interp"] = interp
        v_new = interp.scenario_values(times, spots)
        # Aggregate into the stored netting-set values: netting is EXACT here.
        cva_with = cva_from_values(stored_values + v_new, times, hazard, market, recovery)
        return cva_with - stored_cva

    cva, seconds = _timed(request)
    return ProxyResult(cva=cva, seconds=seconds,
                       n_exact_evaluations=holder["interp"].n_exact_evaluations)


# ----------------------------------------------------------------------------
# The benchmark: full re-simulation
# ----------------------------------------------------------------------------

def full_resimulation_incremental(
    existing_trades: list[Trade],
    new_trade: Trade,
    stored_cva: float,
    times: np.ndarray,
    spots: np.ndarray,
    market: MarketModel,
    hazard: float,
    recovery: float = config.RECOVERY_RATE,
) -> ProxyResult:
    """TRUE incremental CVA, and the timing reference the proxies are judged against.

    This is not a proxy: every trade in the netting set is repriced from scratch
    on every path and date, the enlarged portfolio is netted, and its CVA is
    computed.  Only the subtraction of `stored_cva` is free (the standalone CVA
    of the existing book is known from the overnight batch), which makes the
    comparison against `incremental_proxy` an apples-to-apples one: both do
    exactly the same arithmetic, and differ only in how V_new and the existing
    book's scenario values are obtained.
    """
    n_trades = len(existing_trades) + 1

    def request() -> float:
        values = netting_set_values(list(existing_trades) + [new_trade], times, spots, market)
        return cva_from_values(values, times, hazard, market, recovery) - stored_cva

    cva, seconds = _timed(request)
    return ProxyResult(cva=cva, seconds=seconds, n_exact_evaluations=n_trades * spots.size)


# ----------------------------------------------------------------------------
# Reference variant: incremental WITHOUT interpolation
# ----------------------------------------------------------------------------

def incremental_exact_repricing(
    new_trade: Trade,
    stored_values: np.ndarray,
    stored_cva: float,
    times: np.ndarray,
    spots: np.ndarray,
    market: MarketModel,
    hazard: float,
    recovery: float = config.RECOVERY_RATE,
) -> ProxyResult:
    """Incremental CVA using the stored values but repricing the new trade exactly.

    Not one of the two proxies under study - it is the diagnostic that isolates
    the interpolation error from the netting treatment.  Its result equals the
    TRUE incremental CVA up to floating point, which is what proves that the
    incremental proxy's only error source is the interpolation.
    """
    def request() -> float:
        v_new = new_trade.value(times, spots, market)
        cva_with = cva_from_values(stored_values + v_new, times, hazard, market, recovery)
        return cva_with - stored_cva

    cva, seconds = _timed(request)
    return ProxyResult(cva=cva, seconds=seconds, n_exact_evaluations=spots.size)
