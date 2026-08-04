"""
FX forwards and FX swaps: analytic valuation, analytic deltas, scenario matrices.

Both products are linear, so no nested Monte Carlo is needed: every scenario
node is priced in closed form from covered interest rate parity,

    V(t) = phi * N * [ S(t) * P_f(t,T) - K * P_d(t,T) ]                      (1)

with phi = +1 long EUR, -1 short EUR.  V stays exactly affine in S even under
stochastic rates; all the new curvature sits in the two rate factors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FXForward:
    """A EUR/USD outright forward.  `notional` is in EUR and always positive;
    the sign of the position lives in `direction` (+1 long EUR, -1 short)."""

    notional: float
    strike: float
    maturity: float
    direction: int
    label: str = ""

    @property
    def last_date(self) -> float:
        return self.maturity

    @property
    def legs(self) -> list["FXForward"]:
        return [self]

    def _alive(self, times):
        return self.maturity - np.asarray(times, dtype=float) > 1e-12

    def value(self, market, times, spots, r_d, r_f, apply_maturity: bool = True):
        """Formula (1), in USD.  All arguments broadcast, so this serves both a
        (paths x dates) scenario matrix and a tensor grid of interpolation nodes.

        `apply_maturity=False` returns the continuous pre-settlement value; the
        interpolation module uses that so it only ever fits a smooth function,
        and applies the settlement jump analytically afterwards.
        """
        p_f = market.bond_foreign(times, self.maturity, r_f)
        p_d = market.bond_domestic(times, self.maturity, r_d)
        value = self.direction * self.notional * (spots * p_f - self.strike * p_d)
        return value * self._alive(times) if apply_maturity else value

    # -- analytic sensitivities, used for risk decomposition and trade sizing --

    def delta_spot(self, market, times, r_f, alive: bool = True):
        """dV/dS = phi N P_f(t,T)."""
        d = self.direction * self.notional * market.bond_foreign(times, self.maturity, r_f)
        return d * self._alive(times) if alive else d

    def delta_x_domestic(self, market, times, r_d, alive: bool = True):
        """dV/dx_d = phi N K B_d(t,T) P_d(t,T), since dP_d/dr_d = -B_d P_d."""
        p_d = market.bond_domestic(times, self.maturity, r_d)
        b_d = market.hw_domestic.B(times, self.maturity)
        d = self.direction * self.notional * self.strike * b_d * p_d
        return d * self._alive(times) if alive else d

    def delta_x_foreign(self, market, times, spots, r_f, alive: bool = True):
        """dV/dx_f = -phi N S B_f(t,T) P_f(t,T)."""
        p_f = market.bond_foreign(times, self.maturity, r_f)
        b_f = market.hw_foreign.B(times, self.maturity)
        d = -self.direction * self.notional * spots * b_f * p_f
        return d * self._alive(times) if alive else d


@dataclass(frozen=True)
class FXSwap:
    """Near and far legs in opposite directions; valued as the sum of two forwards.

    The legs nearly cancel in the spot dimension (P_f at two nearby maturities)
    but not in the rate dimensions, because they discount over different
    horizons.  That is why the swap needs stochastic rates to have any
    pre-settlement exposure at all.
    """

    near_leg: FXForward
    far_leg: FXForward
    label: str = ""

    @property
    def notional(self) -> float:
        return self.far_leg.notional

    @property
    def direction(self) -> int:
        """Sign of the far leg - the one that survives the near leg's settlement."""
        return self.far_leg.direction

    @property
    def maturity(self) -> float:
        return self.far_leg.maturity

    @property
    def last_date(self) -> float:
        return self.far_leg.maturity

    @property
    def legs(self) -> list[FXForward]:
        return [self.near_leg, self.far_leg]

    def value(self, market, times, spots, r_d, r_f, apply_maturity: bool = True):
        return sum(l.value(market, times, spots, r_d, r_f, apply_maturity) for l in self.legs)

    def delta_spot(self, market, times, r_f, alive: bool = True):
        return sum(l.delta_spot(market, times, r_f, alive) for l in self.legs)

    def delta_x_domestic(self, market, times, r_d, alive: bool = True):
        return sum(l.delta_x_domestic(market, times, r_d, alive) for l in self.legs)

    def delta_x_foreign(self, market, times, spots, r_f, alive: bool = True):
        return sum(l.delta_x_foreign(market, times, spots, r_f, alive) for l in self.legs)


Trade = FXForward | FXSwap


# ----------------------------------------------------------------------------
# Constructors
# ----------------------------------------------------------------------------

def make_at_market_forward(notional, maturity, direction, market,
                           label: str = "", strike_offset: float = 0.0) -> FXForward:
    """Forward struck at F(0,T), optionally shifted to season a legacy trade.
    strike_offset = 0 gives zero mark-to-market today."""
    fair = float(market.forward_fx(maturity))
    return FXForward(notional, float(fair * (1.0 + strike_offset)),
                     float(maturity), int(direction), label)


def make_at_market_swap(notional, near_maturity, far_maturity, far_direction, market,
                        label: str = "", near_offset: float = 0.0,
                        far_offset: float = 0.0) -> FXSwap:
    """Swap from two at-market legs; the near leg takes the opposite direction."""
    return FXSwap(
        near_leg=make_at_market_forward(notional, near_maturity, -far_direction, market,
                                        f"{label}:near", near_offset),
        far_leg=make_at_market_forward(notional, far_maturity, far_direction, market,
                                       f"{label}:far", far_offset),
        label=label,
    )


# ----------------------------------------------------------------------------
# Scenario matrices
# ----------------------------------------------------------------------------

def netting_set_values(trades: list[Trade], market, scen) -> np.ndarray:
    """Netted MtM of a portfolio: V_NS[p,i] = sum_k V_k[p,i].

    Netting is at counterparty level: the signed trade values are summed first
    and only later floored at zero (see `cva.expected_exposure`).  Summing before
    flooring is what creates the netting benefit.  This is the matrix a bank
    stores overnight and the incremental proxy reuses.
    """
    total = np.zeros(scen.spots.shape, dtype=float)
    for trade in trades:
        total += trade.value(market, scen.times, scen.spots, scen.r_d, scen.r_f)
    return total
