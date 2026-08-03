"""
Synthetic portfolio construction.

Builds the five netting sets (= five counterparties) described in
`config.NETTING_SET_SPECS`, and the "incoming" trades whose CVA charge we want
to quote.  All data is generated from a seeded RNG - nothing is downloaded and
no bank data is used anywhere in this project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .credit import HazardCurve, marginal_default_probabilities
from .instruments import (
    FXForward,
    FXSwap,
    Trade,
    make_at_market_forward,
    make_at_market_swap,
)


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
    hazard: HazardCurve
    spec: config.NettingSetSpec

    def __len__(self) -> int:
        return len(self.trades)

    @property
    def net_signed_notional(self) -> float:
        """Net directional EUR notional over all legs (reporting only).

        An FX swap has two equal-and-opposite legs and therefore contributes
        zero, which is why this crude measure is NOT used to size the
        risk-reducing trade - see `cva_spot_delta` instead.
        """
        return float(sum(leg.direction * leg.notional
                         for trade in self.trades for leg in trade.legs))

    @property
    def max_tenor(self) -> float:
        """Longest maturity currently in the book."""
        return max(trade.last_date for trade in self.trades)

    def spot_delta_profile(self, market, scen) -> np.ndarray:
        """dV_NS(t)/dS(t) on every path and date.

        Sum of the per-trade spot deltas from `instruments.py`:

            dV/dS = sum over LIVE legs of  phi * N * P_f(t, T_leg)

        Under stochastic rates P_f is itself random, so unlike in the
        single-factor model this profile is path-dependent.
        """
        total = np.zeros(scen.spots.shape, dtype=float)
        for trade in self.trades:
            total += trade.delta_spot(market, scen.times, scen.r_f)
        return total


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
    market,
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
        notional = round(float(rng.uniform(*spec.notional_range)), -5)   # EUR 0.1m steps
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
                    notional=notional, near_maturity=near_tenor, far_maturity=far_tenor,
                    far_direction=direction, market=market,
                    label=f"{spec.name}_swap{i:02d}",
                    near_offset=near_offset, far_offset=offset,
                )
            )
        else:
            trades.append(
                make_at_market_forward(
                    notional=notional, maturity=far_tenor, direction=direction,
                    market=market, label=f"{spec.name}_fwd{i:02d}", strike_offset=offset,
                )
            )

    return NettingSet(
        name=spec.name,
        trades=trades,
        hazard=HazardCurve(rates=spec.hazard, name=spec.name),
        spec=spec,
    )


def build_all_netting_sets(market, seed: int = config.SEED) -> list[NettingSet]:
    """Build the five netting sets.  Each set gets its own child RNG stream so
    that changing one specification does not perturb the others."""
    children = np.random.SeedSequence(seed).spawn(len(config.NETTING_SET_SPECS))
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


def cva_spot_delta(
    netting_set: NettingSet,
    market,
    scen,
    ns_values: np.ndarray,
    delta_profile: np.ndarray,
    recovery: float = config.RECOVERY_RATE,
) -> float:
    """Sensitivity of the netting set's CVA to a bump in today's FX spot.

    Differentiating the CVA formula (5) with respect to S(0):

        dCVA/dS(0) = (1-R) * sum_i PD_i
                     * E[ D_d(0,t_i) * 1{V_NS(t_i) > 0} * dV/dS(t_i) * S(t_i)/S(0) ]

    Two pieces of structure are used.  First, d/dV max(V,0) = 1{V > 0}, so only
    the in-the-money scenarios contribute.  Second, the spot enters every path
    multiplicatively, so bumping S(0) scales S(t_i) by S(t_i)/S(0) - which is
    why that factor appears.

    This is the right measure of how directional a book is for CVA purposes.
    The naive value-delta dV/dS at t = 0 is NOT enough: a book can be net short
    EUR in value terms while its CVA is driven by long-EUR positions at the near
    dates, where the default probability and discount weights are largest.
    """
    times = np.asarray(scen.times, dtype=float)
    pd_marginal = marginal_default_probabilities(times, netting_set.hazard)

    in_the_money = ns_values > 0.0
    spot_ratio = scen.spots / market.spot_0
    integrand = scen.df_domestic * in_the_money * delta_profile * spot_ratio

    # Drop t_0: the first bucket in which default can happen is (t_0, t_1].
    return float((1.0 - recovery) * np.sum(integrand[:, 1:].mean(axis=0) * pd_marginal))


def build_incoming_trades(
    netting_set: NettingSet,
    market,
    scen,
    ns_values: np.ndarray,
) -> list[IncomingTrade]:
    """Return the test trades priced against `netting_set`.

    Four cases, chosen to expose different behaviour of the two proxies:

    1. `new_fwd`     - plain 3y forward, long EUR.  Risk-*adding* for the mostly
                       long-EUR portfolios, so both proxies should be positive.
    2. `new_swap`    - 1y/3y FX swap.  Its near leg dies inside the interpolation
                       domain, and under stochastic rates it is driven mainly by
                       the rate differential rather than by the spot.
    3. `offset_fwd`  - risk-reducing forward, built to cancel 50% of the netting
                       set's CVA spot delta.  Its incremental CVA should be
                       negative while the standalone charge stays positive - the
                       key showcase.
    4. `small_fwd`   - 6m forward, EUR 5m, a small reference trade.

    All strikes are at-market (zero MtM at t = 0), the correct convention for a
    trade being quoted right now.
    """
    # --- sizing of the risk-reducing trade ---------------------------------
    # Its tenor is capped at the book's longest maturity: a 3y hedge against a
    # book that is entirely inside 1y would net against nothing and would simply
    # add fresh exposure between year 1 and year 3.
    offset_tenor = float(min(config.INCOMING_OFFSET_TENOR, netting_set.max_tenor))

    # A = dCVA/dS(0) of the existing book.
    portfolio_delta = cva_spot_delta(
        netting_set, market, scen, ns_values,
        netting_set.spot_delta_profile(market, scen),
    )

    # B = dCVA/dS(0) contributed by ONE unit of long-EUR notional in the new
    # trade, i.e. the same functional with dV/dS replaced by P_f(t, T_offset).
    unit_profile = FXForward(
        notional=1.0, strike=0.0, maturity=offset_tenor, direction=+1
    ).delta_spot(market, scen.times, scen.r_f)
    unit_delta = cva_spot_delta(netting_set, market, scen, ns_values, unit_profile)

    # Trade in the direction that removes delta, sized to cancel a fraction of it.
    offset_direction = -1 if portfolio_delta >= 0 else 1
    offset_notional = float(
        np.clip(
            config.OFFSET_FRACTION * abs(portfolio_delta) / max(abs(unit_delta), 1e-30),
            config.OFFSET_MIN_NOTIONAL,
            config.OFFSET_MAX_NOTIONAL,
        )
    )

    return [
        IncomingTrade(
            key="new_fwd",
            description=f"New 3y forward, long EUR {config.INCOMING_FORWARD_NOTIONAL/1e6:.0f}m",
            trade=make_at_market_forward(
                config.INCOMING_FORWARD_NOTIONAL, config.INCOMING_FORWARD_TENOR,
                +1, market, label="incoming_new_fwd",
            ),
        ),
        IncomingTrade(
            key="new_swap",
            description=f"New 1y/3y FX swap, EUR {config.INCOMING_SWAP_NOTIONAL/1e6:.0f}m",
            trade=make_at_market_swap(
                config.INCOMING_SWAP_NOTIONAL, config.INCOMING_SWAP_NEAR,
                config.INCOMING_SWAP_FAR, far_direction=+1, market=market,
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
                offset_notional, offset_tenor, offset_direction, market,
                label="incoming_offset_fwd",
            ),
        ),
        IncomingTrade(
            key="small_fwd",
            description=f"Small 6m forward, long EUR {config.INCOMING_SMALL_NOTIONAL/1e6:.0f}m",
            trade=make_at_market_forward(
                config.INCOMING_SMALL_NOTIONAL, config.INCOMING_SMALL_TENOR,
                +1, market, label="incoming_small_fwd",
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
            is_swap = isinstance(trade, FXSwap)
            for leg in trade.legs:
                rows.append(dict(
                    netting_set=ns.name,
                    hazard=ns.hazard.describe(),
                    label=trade.label,
                    product="FXSwap" if is_swap else "FXForward",
                    leg=("near" if leg is trade.legs[0] else "far") if is_swap else "-",
                    notional_eur=leg.notional,
                    direction=leg.direction,
                    strike=leg.strike,
                    maturity_y=leg.maturity,
                ))
    return pd.DataFrame(rows)
