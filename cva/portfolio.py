"""
Synthetic portfolios: the five netting sets and the incoming test trades.

Everything is generated from a seeded RNG - nothing is downloaded and no bank
data is used anywhere in this project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .cva import HazardCurve, marginal_default_probabilities
from .products import (
    FXForward,
    FXSwap,
    Trade,
    make_at_market_forward,
    make_at_market_swap,
)


@dataclass
class NettingSet:
    """A counterparty's uncollateralised netting set.

    No CSA is modelled, so netting is the only risk mitigant: per scenario and
    date the trade values are summed and only then floored at zero.
    """

    name: str
    trades: list[Trade]
    hazard: HazardCurve
    spec: config.NettingSetSpec

    def __len__(self) -> int:
        return len(self.trades)

    @property
    def net_signed_notional(self) -> float:
        """Net directional EUR notional over all legs.  Reporting only - a swap's
        two legs cancel here, which is why trade sizing uses the CVA delta instead."""
        return float(sum(leg.direction * leg.notional
                         for t in self.trades for leg in t.legs))

    @property
    def max_tenor(self) -> float:
        return max(t.last_date for t in self.trades)

    def spot_delta_profile(self, market, scen) -> np.ndarray:
        """dV_NS/dS on every path and date.  Path-dependent under stochastic rates,
        since P_f is itself random."""
        total = np.zeros(scen.spots.shape, dtype=float)
        for trade in self.trades:
            total += trade.delta_spot(market, scen.times, scen.r_f)
        return total


# ----------------------------------------------------------------------------
# Existing portfolios
# ----------------------------------------------------------------------------

def _draw_tenor(rng, lo: float, hi: float) -> float:
    """Draw a maturity from the discrete tenor menu, restricted to [lo, hi]."""
    menu = config.TENOR_MENU[(config.TENOR_MENU >= lo - 1e-9) & (config.TENOR_MENU <= hi + 1e-9)]
    return float(hi) if menu.size == 0 else float(rng.choice(menu))


def build_netting_set(spec: config.NettingSetSpec, market, rng) -> NettingSet:
    """Generate one netting set.  Trades are seasoned - struck +/-3% off today's
    fair forward - so the book starts with a non-zero mark-to-market."""
    is_swap = np.zeros(spec.n_trades, dtype=bool)
    is_swap[: int(round(spec.swap_fraction * spec.n_trades))] = True
    rng.shuffle(is_swap)

    trades: list[Trade] = []
    for i in range(spec.n_trades):
        notional = round(float(rng.uniform(*spec.notional_range)), -5)   # EUR 0.1m steps
        direction = 1 if rng.random() < spec.prob_long_eur else -1
        offset = float(rng.uniform(-config.STRIKE_SEASONING, config.STRIKE_SEASONING))
        far_tenor = _draw_tenor(rng, spec.tenor_min, spec.tenor_max)

        # A swap needs a strictly shorter near tenor; otherwise fall back to a forward.
        shorter = config.TENOR_MENU[(config.TENOR_MENU >= spec.tenor_min - 1e-9)
                                    & (config.TENOR_MENU < far_tenor - 1e-9)]
        if is_swap[i] and shorter.size > 0:
            trades.append(make_at_market_swap(
                notional, float(rng.choice(shorter)), far_tenor, direction, market,
                label=f"{spec.name}_swap{i:02d}",
                near_offset=float(rng.uniform(-config.STRIKE_SEASONING, config.STRIKE_SEASONING)),
                far_offset=offset))
        else:
            trades.append(make_at_market_forward(
                notional, far_tenor, direction, market,
                label=f"{spec.name}_fwd{i:02d}", strike_offset=offset))

    return NettingSet(spec.name, trades, HazardCurve(spec.hazard, name=spec.name), spec)


def build_all_netting_sets(market, seed: int = config.SEED) -> list[NettingSet]:
    """Build all five sets, each from its own child RNG stream so that changing
    one specification does not perturb the others."""
    children = np.random.SeedSequence(seed).spawn(len(config.NETTING_SET_SPECS))
    return [build_netting_set(spec, market, np.random.default_rng(child))
            for spec, child in zip(config.NETTING_SET_SPECS, children)]


# ----------------------------------------------------------------------------
# Incoming trades
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class IncomingTrade:
    key: str            # short machine-readable id
    description: str    # human-readable, used in plots
    trade: Trade


def cva_spot_delta(netting_set: NettingSet, market, scen, ns_values, delta_profile,
                   recovery: float = config.RECOVERY_RATE) -> float:
    """dCVA/dS(0) of the netting set.

    Differentiating the CVA formula with respect to today's spot:

        dCVA/dS(0) = (1-R) sum_i PD_i E[ D_d(0,t_i) 1{V>0} dV/dS(t_i) S(t_i)/S(0) ]

    using d/dV max(V,0) = 1{V>0}, and S(t_i)/S(0) because the spot enters every
    path multiplicatively.  This is the right measure of how directional a book
    is for CVA purposes: a book can be net short EUR in value terms while its CVA
    is driven by long-EUR positions at the near dates, where the default and
    discount weights are largest.
    """
    pd_marginal = marginal_default_probabilities(np.asarray(scen.times, dtype=float),
                                                 netting_set.hazard)
    integrand = (scen.df_domestic * (ns_values > 0.0) * delta_profile
                 * scen.spots / market.spot_0)
    # Drop t_0: the first bucket in which default can happen is (t_0, t_1].
    return float((1.0 - recovery) * np.sum(integrand[:, 1:].mean(axis=0) * pd_marginal))


def build_incoming_trades(netting_set: NettingSet, market, scen, ns_values) -> list[IncomingTrade]:
    """The four test trades priced against `netting_set`.

    new_fwd     plain 3y forward, long EUR - risk-adding
    new_swap    1y/3y FX swap - near leg dies inside the interpolation domain,
                and under stochastic rates it is rate-driven before settlement
    offset_fwd  risk-reducing forward sized to cancel 50% of the set's CVA spot
                delta; its incremental CVA should be negative while the
                standalone charge stays positive - the key showcase
    small_fwd   6m EUR 5m forward, a small reference trade

    All strikes are at-market, the correct convention for a trade quoted now.
    """
    # The offsetting trade's tenor is capped at the book's longest maturity: a 3y
    # hedge against a book entirely inside 1y would net against nothing.
    offset_tenor = float(min(config.INCOMING_OFFSET_TENOR, netting_set.max_tenor))

    portfolio_delta = cva_spot_delta(netting_set, market, scen, ns_values,
                                     netting_set.spot_delta_profile(market, scen))
    # Same functional applied to one unit of long-EUR notional at the offset tenor.
    unit_profile = FXForward(1.0, 0.0, offset_tenor, +1).delta_spot(
        market, scen.times, scen.r_f)
    unit_delta = cva_spot_delta(netting_set, market, scen, ns_values, unit_profile)

    offset_direction = -1 if portfolio_delta >= 0 else 1
    offset_notional = float(np.clip(
        config.OFFSET_FRACTION * abs(portfolio_delta) / max(abs(unit_delta), 1e-30),
        config.OFFSET_MIN_NOTIONAL, config.OFFSET_MAX_NOTIONAL))

    return [
        IncomingTrade(
            "new_fwd", f"New 3y forward, long EUR {config.INCOMING_FORWARD_NOTIONAL/1e6:.0f}m",
            make_at_market_forward(config.INCOMING_FORWARD_NOTIONAL,
                                   config.INCOMING_FORWARD_TENOR, +1, market,
                                   label="incoming_new_fwd")),
        IncomingTrade(
            "new_swap", f"New 1y/3y FX swap, EUR {config.INCOMING_SWAP_NOTIONAL/1e6:.0f}m",
            make_at_market_swap(config.INCOMING_SWAP_NOTIONAL, config.INCOMING_SWAP_NEAR,
                                config.INCOMING_SWAP_FAR, +1, market,
                                label="incoming_new_swap")),
        IncomingTrade(
            "offset_fwd",
            f"Risk-reducing {offset_tenor:g}y forward, "
            f"{'short' if offset_direction < 0 else 'long'} EUR {offset_notional/1e6:.0f}m",
            make_at_market_forward(offset_notional, offset_tenor, offset_direction, market,
                                   label="incoming_offset_fwd")),
        IncomingTrade(
            "small_fwd", f"Small 6m forward, long EUR {config.INCOMING_SMALL_NOTIONAL/1e6:.0f}m",
            make_at_market_forward(config.INCOMING_SMALL_NOTIONAL, config.INCOMING_SMALL_TENOR,
                                   +1, market, label="incoming_small_fwd")),
    ]


def portfolio_to_frame(netting_sets: list[NettingSet]) -> pd.DataFrame:
    """Flatten every trade of every netting set, leg by leg (thesis appendix)."""
    rows = []
    for ns in netting_sets:
        for trade in ns.trades:
            is_swap = isinstance(trade, FXSwap)
            for leg in trade.legs:
                rows.append(dict(
                    netting_set=ns.name, hazard=ns.hazard.describe(), label=trade.label,
                    product="FXSwap" if is_swap else "FXForward",
                    leg=("near" if leg is trade.legs[0] else "far") if is_swap else "-",
                    notional_eur=leg.notional, direction=leg.direction,
                    strike=leg.strike, maturity_y=leg.maturity))
    return pd.DataFrame(rows)
