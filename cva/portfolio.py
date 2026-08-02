"""
Synthetic portfolio construction.

Builds the five netting sets (= five counterparties) described in
`config.NETTING_SET_SPECS`, and the "incoming" trades whose CVA charge we want
to quote.  All data is generated from a seeded RNG - nothing is downloaded and
no bank data is used anywhere in this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .credit import marginal_default_probabilities
from .instruments import (
    FXForward,
    FXSwap,
    Trade,
    make_at_market_forward,
    make_at_market_swap,
)
from .market import MarketModel


# ----------------------------------------------------------------------------
# Netting set container
# ----------------------------------------------------------------------------

@dataclass
class NettingSet:
    """A counterparty's uncollateralised netting set.

    No CSA / no collateral is modelled, so netting is the *only* risk mitigant:
    per scenario and time step we sum the trade values and take the positive
    part at counterparty level.
    """

    name: str
    trades: list[Trade]
    hazard: float
    spec: config.NettingSetSpec

    def __len__(self) -> int:
        return len(self.trades)

    @property
    def net_signed_notional(self) -> float:
        """Net directional EUR notional, summed over all *legs* (reporting only).

        An FX swap has two equal-and-opposite legs and therefore contributes
        zero, which is why this crude measure is NOT used to size the
        risk-reducing trade - see `cva_delta_weights` instead.
        """
        total = 0.0
        for trade in self.trades:
            for leg in _all_legs(trade):
                total += leg.direction * leg.notional
        return total

    def eur_delta_profile(self, times: np.ndarray, market: MarketModel) -> np.ndarray:
        """Spot sensitivity of the netted portfolio value at each exposure date.

            D(t) = dV_NS(t)/dS(t) = sum over LIVE legs of  phi * N * DF_EUR(t, T_leg)

        This follows by differentiating formula (1), whose only spot-dependent
        term is phi*N*DF_EUR(t,T)*S(t).  Note that D(t) is deterministic (it does
        not depend on the path), because the products are linear.

        Returns an array of shape (len(times),).
        """
        times = np.asarray(times, dtype=float)
        profile = np.zeros(times.size, dtype=float)
        for trade in self.trades:
            for leg in _all_legs(trade):
                tau = leg.maturity - times
                alive = tau > 1e-12
                profile += leg.direction * leg.notional * market.df_eur(np.maximum(tau, 0.0)) * alive
        return profile

    @property
    def max_tenor(self) -> float:
        """Longest maturity currently in the book."""
        return max(trade.last_date for trade in self.trades)


def _all_legs(trade: Trade) -> list[FXForward]:
    """Flatten a trade into its forward legs."""
    if isinstance(trade, FXSwap):
        return [trade.near_leg, trade.far_leg]
    return [trade]


# ----------------------------------------------------------------------------
# Existing portfolios
# ----------------------------------------------------------------------------

def _draw_tenor(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Draw a maturity from the discrete tenor menu, restricted to [lo, hi]."""
    menu = config.TENOR_MENU[(config.TENOR_MENU >= lo - 1e-9) & (config.TENOR_MENU <= hi + 1e-9)]
    if menu.size == 0:                       # degenerate range -> use the cap
        return float(hi)
    return float(rng.choice(menu))


def build_netting_set(
    spec: config.NettingSetSpec,
    market: MarketModel,
    rng: np.random.Generator,
) -> NettingSet:
    """Generate one netting set according to its specification.

    Trades are *seasoned*: their strike is the fair forward shifted by a random
    +/- 3% so the portfolio starts with a non-zero mark-to-market, as a real
    legacy book would.
    """
    n_swaps = int(round(spec.swap_fraction * spec.n_trades))
    is_swap = np.zeros(spec.n_trades, dtype=bool)
    is_swap[:n_swaps] = True
    rng.shuffle(is_swap)

    trades: list[Trade] = []
    for i in range(spec.n_trades):
        notional = float(rng.uniform(*spec.notional_range))
        notional = round(notional, -5)                      # round to EUR 0.1m
        direction = 1 if rng.random() < spec.prob_long_eur else -1
        offset = float(rng.uniform(-config.STRIKE_SEASONING, config.STRIKE_SEASONING))

        far_tenor = _draw_tenor(rng, spec.tenor_min, spec.tenor_max)

        # Can this trade be a swap?  It needs a strictly shorter near tenor.
        shorter = config.TENOR_MENU[
            (config.TENOR_MENU >= spec.tenor_min - 1e-9) & (config.TENOR_MENU < far_tenor - 1e-9)
        ]
        if is_swap[i] and shorter.size > 0:
            near_tenor = float(rng.choice(shorter))
            near_offset = float(rng.uniform(-config.STRIKE_SEASONING, config.STRIKE_SEASONING))
            trades.append(
                make_at_market_swap(
                    notional=notional,
                    near_maturity=near_tenor,
                    far_maturity=far_tenor,
                    far_direction=direction,
                    market=market,
                    label=f"{spec.name}_swap{i:02d}",
                    near_offset=near_offset,
                    far_offset=offset,
                )
            )
        else:
            trades.append(
                make_at_market_forward(
                    notional=notional,
                    maturity=far_tenor,
                    direction=direction,
                    market=market,
                    label=f"{spec.name}_fwd{i:02d}",
                    strike_offset=offset,
                )
            )

    return NettingSet(name=spec.name, trades=trades, hazard=spec.hazard, spec=spec)


def build_all_netting_sets(market: MarketModel, seed: int = config.SEED) -> list[NettingSet]:
    """Build the five netting sets.  Each set gets its own child RNG stream so
    that changing one specification does not perturb the others."""
    seed_seq = np.random.SeedSequence(seed)
    children = seed_seq.spawn(len(config.NETTING_SET_SPECS))
    return [
        build_netting_set(spec, market, np.random.default_rng(child))
        for spec, child in zip(config.NETTING_SET_SPECS, children)
    ]


# ----------------------------------------------------------------------------
# Incoming trades
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class IncomingTrade:
    """A new trade for which the desk wants a CVA charge."""

    key: str            # short machine-readable id
    description: str    # human-readable, used in plots
    trade: Trade


def cva_delta_weights(
    netting_set: NettingSet,
    times: np.ndarray,
    ns_values: np.ndarray,
    market: MarketModel,
) -> np.ndarray:
    """Weights w_i that turn a delta profile into a CVA sensitivity.

    Differentiating the CVA formula (5) with respect to the spot gives

        dCVA/dS = (1-R) * sum_i DF(t_i) * PD_i * E[ 1{V_NS(t_i) > 0} * dV/dS ]

    because d/dV max(V,0) = 1{V > 0}.  Since dV/dS is deterministic here, the
    expectation factorises into  D(t_i) * P(V_NS(t_i) > 0)  and the weights are

        w_i = (1-R) * DF(t_i) * PD_i * P(V_NS(t_i) > 0).

    Applying these weights to a delta profile answers the question "does adding
    long-EUR delta at this date increase or decrease the CVA?", which is what we
    need in order to construct a genuinely risk-reducing trade.  The naive
    value-delta dV/dS at t = 0 is NOT enough: a book can be net short EUR in
    value terms while its CVA is driven by long-EUR positions at the near dates
    where the default probability and discount weights are largest.
    """
    times = np.asarray(times, dtype=float)
    prob_positive = (ns_values > 0.0).mean(axis=0)          # P(V_NS(t_i) > 0)
    df = market.df_usd(times)
    pd_marginal = np.concatenate([[0.0], marginal_default_probabilities(times, netting_set.hazard)])
    return (1.0 - config.RECOVERY_RATE) * df * pd_marginal * prob_positive


def build_incoming_trades(
    netting_set: NettingSet,
    market: MarketModel,
    times: np.ndarray,
    ns_values: np.ndarray,
) -> list[IncomingTrade]:
    """Return the test trades priced against `netting_set`.

    Four cases, chosen to expose different behaviour of the two proxies:

    1. `new_fwd`     - plain 3y forward, long EUR.  Risk-*adding* for the mostly
                       long-EUR portfolios, so both proxies should be positive.
    2. `new_swap`    - 1y/3y FX swap.  Its near leg dies inside the interpolation
                       domain, which is the main source of interpolation error.
    3. `offset_fwd`  - forward built to cancel 50% of the netting set's CVA
                       spot delta.  This is the key showcase: its incremental
                       CVA should be negative while the standalone proxy stays
                       positive.
    4. `small_fwd`   - 6m EUR 5m forward, a small reference trade used to show
                       how the relative error scales with trade size.

    Parameters
    ----------
    times, ns_values : the exposure grid and the existing netting set's scenario
        matrix.  Needed only to size the risk-reducing trade sensibly.

    All strikes are at-market (zero MtM at t = 0), which is the correct
    convention for a trade being quoted right now.
    """
    # --- sizing of the risk-reducing trade ---------------------------------
    # The offsetting trade must be able to net against what is actually in the
    # book, so its tenor is capped at the book's longest maturity: a 3y hedge
    # against a book that is entirely inside 1y would net against nothing and
    # would simply add fresh exposure between year 1 and year 3.
    offset_tenor = float(min(config.INCOMING_OFFSET_TENOR, netting_set.max_tenor))

    times = np.asarray(times, dtype=float)
    weights = cva_delta_weights(netting_set, times, ns_values, market)

    # A = dCVA/dS of the existing book.
    portfolio_delta = float(weights @ netting_set.eur_delta_profile(times, market))

    # B = dCVA/dS contributed by ONE unit of long-EUR notional in the new trade.
    tau = offset_tenor - times
    unit_delta_profile = market.df_eur(np.maximum(tau, 0.0)) * (tau > 1e-12)
    unit_delta = float(weights @ unit_delta_profile)

    # Trade in the direction that removes delta, sized to cancel a fraction of it.
    offset_direction = -1 if portfolio_delta >= 0 else 1
    offset_notional = float(
        np.clip(
            config.OFFSET_FRACTION * abs(portfolio_delta) / max(unit_delta, 1e-30),
            config.OFFSET_MIN_NOTIONAL,
            config.OFFSET_MAX_NOTIONAL,
        )
    )

    return [
        IncomingTrade(
            key="new_fwd",
            description=f"New 3y forward, long EUR {config.INCOMING_FORWARD_NOTIONAL/1e6:.0f}m",
            trade=make_at_market_forward(
                config.INCOMING_FORWARD_NOTIONAL,
                config.INCOMING_FORWARD_TENOR,
                +1,
                market,
                label="incoming_new_fwd",
            ),
        ),
        IncomingTrade(
            key="new_swap",
            description=f"New 1y/3y FX swap, EUR {config.INCOMING_SWAP_NOTIONAL/1e6:.0f}m",
            trade=make_at_market_swap(
                config.INCOMING_SWAP_NOTIONAL,
                config.INCOMING_SWAP_NEAR,
                config.INCOMING_SWAP_FAR,
                far_direction=+1,
                market=market,
                label="incoming_new_swap",
            ),
        ),
        IncomingTrade(
            key="offset_fwd",
            description=(
                f"Risk-reducing {offset_tenor:g}y forward, "
                f"{'short' if offset_direction < 0 else 'long'} EUR {offset_notional/1e6:.0f}m"
            ),
            trade=make_at_market_forward(
                offset_notional,
                offset_tenor,
                offset_direction,
                market,
                label="incoming_offset_fwd",
            ),
        ),
        IncomingTrade(
            key="small_fwd",
            description=f"Small 6m forward, long EUR {config.INCOMING_SMALL_NOTIONAL/1e6:.0f}m",
            trade=make_at_market_forward(
                config.INCOMING_SMALL_NOTIONAL,
                config.INCOMING_SMALL_TENOR,
                +1,
                market,
                label="incoming_small_fwd",
            ),
        ),
    ]


# ----------------------------------------------------------------------------
# Reporting helper
# ----------------------------------------------------------------------------

def portfolio_to_frame(netting_sets: list[NettingSet]) -> pd.DataFrame:
    """Flatten every trade of every netting set into a DataFrame (thesis appendix)."""
    rows = []
    for ns in netting_sets:
        for trade in ns.trades:
            if isinstance(trade, FXSwap):
                for leg_name, leg in (("near", trade.near_leg), ("far", trade.far_leg)):
                    rows.append(
                        dict(
                            netting_set=ns.name, hazard=ns.hazard, label=trade.label,
                            product="FXSwap", leg=leg_name, notional_eur=leg.notional,
                            direction=leg.direction, strike=leg.strike, maturity_y=leg.maturity,
                        )
                    )
            else:
                rows.append(
                    dict(
                        netting_set=ns.name, hazard=ns.hazard, label=trade.label,
                        product="FXForward", leg="-", notional_eur=trade.notional,
                        direction=trade.direction, strike=trade.strike, maturity_y=trade.maturity,
                    )
                )
    return pd.DataFrame(rows)
