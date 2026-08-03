"""
Products and their ANALYTIC valuation under the three-factor model.

Both products - the FX outright forward and the FX swap - are *linear*, so we
never need a nested Monte Carlo: at every scenario and time node we evaluate a
closed-form expression.  Moving from deterministic to stochastic rates does not
change that; it only replaces the two deterministic discount factors by the two
Hull-White bond reconstruction formulas.

Valuation formula (covered interest rate parity, seen from a USD book)
----------------------------------------------------------------------
A forward to buy N EUR against N*K USD at maturity T is worth, at time t < T:

    V(t) = phi * N * [ S(t) * P_f(t,T)  -  K * P_d(t,T) ]                    (1)

with phi = +1 for "long EUR" and phi = -1 for "short EUR".  The first term is
today's USD value of receiving N EUR at T (discount on the EUR curve, convert at
the current spot); the second is the USD value of paying N*K USD at T.

Under the single-factor base model P_d and P_f were deterministic exponentials.
Here they are stochastic:

    P_d(t,T) = A_d(t,T) exp(-B_d(t,T) r_d(t))
    P_f(t,T) = A_f(t,T) exp(-B_f(t,T) r_f(t))

Two consequences matter for the rest of the project:

* V is STILL exactly affine in the spot S.  All the new curvature sits in the
  two rate factors.
* An FX swap's two legs nearly cancel in the S-term (P_f at two nearby
  maturities), so its FX delta is only a few percent of a comparable forward -
  but its RATE deltas do not cancel, because the two legs discount over
  different horizons.  With deterministic rates such a swap has almost no risk
  driver before its near leg settles and its exposure is artificially close to
  zero.  That is the modelling gap the three-factor model closes.

After maturity the trade has settled and contributes zero exposure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------
# Forward
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class FXForward:
    """A single EUR/USD outright forward.

    notional  : EUR notional (always positive; the sign lives in `direction`)
    strike    : contracted FX rate K, USD per EUR
    maturity  : T in years from today
    direction : +1 = buy EUR / sell USD, -1 = sell EUR / buy USD
    label     : free-text identifier used in the output tables
    """

    notional: float
    strike: float
    maturity: float
    direction: int
    label: str = ""

    @property
    def last_date(self) -> float:
        """Latest date on which the trade can still carry exposure."""
        return self.maturity

    @property
    def legs(self) -> list["FXForward"]:
        return [self]

    def value(
        self,
        market,
        times: np.ndarray,
        spots: np.ndarray,
        r_d: np.ndarray,
        r_f: np.ndarray,
        apply_maturity: bool = True,
    ) -> np.ndarray:
        """Mark-to-market in USD, formula (1).

        Every argument broadcasts, so the same code serves a (paths x dates)
        scenario matrix and a small tensor grid of interpolation nodes.

        `apply_maturity=False` returns the CONTINUOUS pre-settlement value (the
        trade is not zeroed out after T).  The interpolation module uses that
        variant so that it only ever interpolates a smooth function, and applies
        the settlement jump analytically afterwards.
        """
        times = np.asarray(times, dtype=float)

        p_f = market.bond_foreign(times, self.maturity, r_f)     # P_f(t,T)
        p_d = market.bond_domestic(times, self.maturity, r_d)    # P_d(t,T)

        # V(t) = phi * N * [ S(t) P_f(t,T) - K P_d(t,T) ]
        value = self.direction * self.notional * (spots * p_f - self.strike * p_d)

        if apply_maturity:
            # A trade that has already matured settled and left the netting set.
            value = value * (self.maturity - times > 1e-12)
        return value

    # -- analytic sensitivities, used by the risk decomposition --------------

    def delta_spot(self, market, times, r_f, alive: bool = True) -> np.ndarray:
        """dV/dS = phi * N * P_f(t,T).  Differentiates the first term of (1)."""
        times = np.asarray(times, dtype=float)
        d = self.direction * self.notional * market.bond_foreign(times, self.maturity, r_f)
        return d * (self.maturity - times > 1e-12) if alive else d

    def delta_x_domestic(self, market, times, r_d, alive: bool = True) -> np.ndarray:
        """dV/dx_d.  Since dP_d/dr_d = -B_d P_d and dr_d/dx_d = 1:

            dV/dx_d = phi * N * K * B_d(t,T) * P_d(t,T)
        """
        times = np.asarray(times, dtype=float)
        p_d = market.bond_domestic(times, self.maturity, r_d)
        b_d = market.hw_domestic.B(times, self.maturity)
        d = self.direction * self.notional * self.strike * b_d * p_d
        return d * (self.maturity - times > 1e-12) if alive else d

    def delta_x_foreign(self, market, times, spots, r_f, alive: bool = True) -> np.ndarray:
        """dV/dx_f = -phi * N * S * B_f(t,T) * P_f(t,T)."""
        times = np.asarray(times, dtype=float)
        p_f = market.bond_foreign(times, self.maturity, r_f)
        b_f = market.hw_foreign.B(times, self.maturity)
        d = -self.direction * self.notional * spots * b_f * p_f
        return d * (self.maturity - times > 1e-12) if alive else d


# ----------------------------------------------------------------------------
# FX swap
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class FXSwap:
    """An FX swap = a near leg and a far leg in opposite directions.

    The bank exchanges EUR for USD on the near date and reverses the exchange on
    the far date at a different, pre-agreed rate.  Both legs are themselves
    forwards, so the swap is valued as the sum of two `FXForward` values.
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

    def value(self, market, times, spots, r_d, r_f, apply_maturity: bool = True) -> np.ndarray:
        """Swap MtM = near-leg MtM + far-leg MtM, both from formula (1)."""
        return sum(
            leg.value(market, times, spots, r_d, r_f, apply_maturity) for leg in self.legs
        )

    def delta_spot(self, market, times, r_f, alive: bool = True) -> np.ndarray:
        return sum(leg.delta_spot(market, times, r_f, alive) for leg in self.legs)

    def delta_x_domestic(self, market, times, r_d, alive: bool = True) -> np.ndarray:
        return sum(leg.delta_x_domestic(market, times, r_d, alive) for leg in self.legs)

    def delta_x_foreign(self, market, times, spots, r_f, alive: bool = True) -> np.ndarray:
        return sum(leg.delta_x_foreign(market, times, spots, r_f, alive) for leg in self.legs)


Trade = FXForward | FXSwap


# ----------------------------------------------------------------------------
# Convenience: valuing on a whole simulation
# ----------------------------------------------------------------------------

def value_on_scenarios(trade: Trade, market, scen) -> np.ndarray:
    """Exact analytic MtM of one trade on the whole (path, date) grid."""
    return trade.value(market, scen.times, scen.spots, scen.r_d, scen.r_f)


# ----------------------------------------------------------------------------
# Convenience constructors
# ----------------------------------------------------------------------------

def make_at_market_forward(
    notional: float,
    maturity: float,
    direction: int,
    market,
    label: str = "",
    strike_offset: float = 0.0,
) -> FXForward:
    """Build a forward struck at today's fair forward rate F(0,T), optionally
    shifted by `strike_offset` (a relative move used to season legacy trades).

    A trade with strike_offset = 0 has zero mark-to-market at t = 0, the natural
    convention for a *new* incoming trade.
    """
    fair = float(market.forward_fx(maturity))
    return FXForward(
        notional=notional,
        strike=float(fair * (1.0 + strike_offset)),
        maturity=float(maturity),
        direction=int(direction),
        label=label,
    )


def make_at_market_swap(
    notional: float,
    near_maturity: float,
    far_maturity: float,
    far_direction: int,
    market,
    label: str = "",
    near_offset: float = 0.0,
    far_offset: float = 0.0,
) -> FXSwap:
    """Build an FX swap from two at-market legs in opposite directions.

    `far_direction` fixes the direction of the far (long-dated) leg; the near
    leg takes the opposite sign, which is what makes the package an FX swap
    rather than two unrelated forwards.
    """
    near = make_at_market_forward(
        notional, near_maturity, -far_direction, market,
        label=f"{label}:near", strike_offset=near_offset,
    )
    far = make_at_market_forward(
        notional, far_maturity, far_direction, market,
        label=f"{label}:far", strike_offset=far_offset,
    )
    return FXSwap(near_leg=near, far_leg=far, label=label)
