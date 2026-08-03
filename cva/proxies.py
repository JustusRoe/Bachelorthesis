"""
The two fast pricing proxies.

Both answer the same business question - "what CVA should we charge for this
incoming trade?" - but differ in how much of the existing netting set they see.

STANDALONE PROXY
    Prices the new trade in ISOLATION, as if it were the only trade with the
    counterparty:
        CVA_standalone = (1-R) * sum_i E[D_d(0,t_i) max(V_new(t_i), 0)] * PD_i
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

Both proxies reuse the already-simulated risk-factor paths - regenerating them
is part of the overnight batch, not of an intraday pricing request.  Only the
work a real-time request would actually have to do is timed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import config
from .credit import HazardCurve
from .cva_engine import cva_from_discounted_ee, cva_from_values, discounted_ee
from .instruments import Trade
from .interpolation import ValueInterpolant
from .scenarios import netting_set_scenario_values


# ----------------------------------------------------------------------------
# Context and results
# ----------------------------------------------------------------------------

@dataclass
class PricingContext:
    """Everything a pricing request needs that is not the trade itself.

    Bundling these keeps the proxy signatures readable now that the market model
    carries three factors and the interpolation grid spans three state
    dimensions.
    """

    market: object                              # ThreeFactorMarket
    scen: object                                # Scenarios
    state_box: dict[str, tuple[float, float]]   # interpolation node ranges
    discounting: str = config.DISCOUNTING

    @property
    def n_state_dimensions(self) -> int:
        return len(self.scen.state_names)


@dataclass
class ProxyResult:
    """Outcome of one pricing request."""

    cva: float                  # the charge the proxy would quote, in USD
    seconds: float              # wall-clock time of the request (best of N runs)
    n_exact_evaluations: int    # exact analytic prices needed to build the proxy


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
    ctx: PricingContext,
    hazard: HazardCurve,
    method: str = config.DEFAULT_INTERP_METHOD,
    n_time_nodes: int = config.N_TIME_NODES,
    n_state_nodes: int = config.N_STATE_NODES,
    split_legs: bool = True,
    recovery: float = config.RECOVERY_RATE,
    repeats: int = config.N_TIMING_REPEATS,
) -> ProxyResult:
    """CVA of the incoming trade priced on its own, with no netting."""
    holder: dict[str, ValueInterpolant] = {}

    def request() -> float:
        interp = ValueInterpolant(
            new_trade, ctx.market, ctx.state_box,
            n_time_nodes=n_time_nodes, n_state_nodes=n_state_nodes,
            method=method, split_legs=split_legs,
        )
        holder["interp"] = interp
        v_new = interp.scenario_values(ctx.scen)
        # EE* of the SINGLE trade: max(.,0) taken trade by trade -> no netting.
        ee_star = discounted_ee(v_new, ctx.scen, ctx.market, ctx.discounting)
        return cva_from_discounted_ee(ee_star, ctx.scen.times, hazard, recovery)

    cva, seconds = _timed(request, repeats)
    return ProxyResult(cva=cva, seconds=seconds,
                       n_exact_evaluations=holder["interp"].n_exact_evaluations)


# ----------------------------------------------------------------------------
# Incremental proxy
# ----------------------------------------------------------------------------

def incremental_proxy(
    new_trade: Trade,
    ctx: PricingContext,
    hazard: HazardCurve,
    stored_values: np.ndarray,
    stored_cva: float,
    method: str = config.DEFAULT_INTERP_METHOD,
    n_time_nodes: int = config.N_TIME_NODES,
    n_state_nodes: int = config.N_STATE_NODES,
    split_legs: bool = True,
    recovery: float = config.RECOVERY_RATE,
    repeats: int = config.N_TIMING_REPEATS,
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
            new_trade, ctx.market, ctx.state_box,
            n_time_nodes=n_time_nodes, n_state_nodes=n_state_nodes,
            method=method, split_legs=split_legs,
        )
        holder["interp"] = interp
        v_new = interp.scenario_values(ctx.scen)
        # Aggregate into the stored netting-set values: netting is EXACT here.
        cva_with = cva_from_values(
            stored_values + v_new, ctx.scen, ctx.market, hazard, recovery, ctx.discounting
        )
        return cva_with - stored_cva

    cva, seconds = _timed(request, repeats)
    return ProxyResult(cva=cva, seconds=seconds,
                       n_exact_evaluations=holder["interp"].n_exact_evaluations)


# ----------------------------------------------------------------------------
# The benchmark: full re-simulation
# ----------------------------------------------------------------------------

def full_resimulation_incremental(
    existing_trades: list[Trade],
    new_trade: Trade,
    ctx: PricingContext,
    hazard: HazardCurve,
    stored_cva: float,
    recovery: float = config.RECOVERY_RATE,
    repeats: int = config.N_TIMING_REPEATS,
) -> ProxyResult:
    """TRUE incremental CVA, and the timing reference the proxies are judged against.

    Not a proxy: every trade in the netting set is repriced from scratch on every
    path and date, the enlarged portfolio is netted, and its CVA is computed.
    Only the subtraction of `stored_cva` is free (the existing book's CVA is
    known from the overnight batch), which makes the comparison against
    `incremental_proxy` apples-to-apples: both do the same arithmetic and differ
    only in how V_new and the book's scenario values are obtained.
    """
    n_trades = len(existing_trades) + 1

    def request() -> float:
        values = netting_set_scenario_values(
            list(existing_trades) + [new_trade], ctx.market, ctx.scen
        )
        cva_with = cva_from_values(
            values, ctx.scen, ctx.market, hazard, recovery, ctx.discounting
        )
        return cva_with - stored_cva

    cva, seconds = _timed(request, repeats)
    return ProxyResult(cva=cva, seconds=seconds,
                       n_exact_evaluations=n_trades * ctx.scen.spots.size)


# ----------------------------------------------------------------------------
# Reference variant: incremental WITHOUT interpolation
# ----------------------------------------------------------------------------

def incremental_exact_repricing(
    new_trade: Trade,
    ctx: PricingContext,
    hazard: HazardCurve,
    stored_values: np.ndarray,
    stored_cva: float,
    recovery: float = config.RECOVERY_RATE,
    repeats: int = config.N_TIMING_REPEATS,
) -> ProxyResult:
    """Incremental CVA using the stored values but repricing the new trade exactly.

    Not one of the two proxies under study - it is the diagnostic that isolates
    the interpolation error from the netting treatment.  Its result equals the
    TRUE incremental CVA up to floating point, which is what proves the
    incremental proxy's only error source is the interpolation.
    """
    def request() -> float:
        v_new = new_trade.value(
            ctx.market, ctx.scen.times, ctx.scen.spots, ctx.scen.r_d, ctx.scen.r_f
        )
        cva_with = cva_from_values(
            stored_values + v_new, ctx.scen, ctx.market, hazard, recovery, ctx.discounting
        )
        return cva_with - stored_cva

    cva, seconds = _timed(request, repeats)
    return ProxyResult(cva=cva, seconds=seconds,
                       n_exact_evaluations=ctx.scen.spots.size)
