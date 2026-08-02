"""
Products and their ANALYTIC valuation.

Both products in this study - the FX outright forward and the FX swap - are
*linear* in the FX spot.  That means we never need a nested Monte Carlo: at
every scenario/time node we simply evaluate a closed-form expression.  This is
the single most important computational choice in the thesis, because it makes
the "true" benchmark cheap enough to compute exactly.

Valuation formula (covered interest rate parity, seen from a USD book)
----------------------------------------------------------------------
An outright forward to buy N EUR against N*K USD at maturity T is, at time
t < T, worth the discounted difference of the two notional legs:

    V(t) = phi * N * [ DF_EUR(t,T) * S(t)  -  K * DF_USD(t,T) ]                (1)

with phi = +1 for "long EUR" and phi = -1 for "short EUR".  The first term is
today's USD value of receiving N EUR at T (discount in EUR, then convert at the
current spot); the second is the USD value of paying N*K USD at T.

After maturity the trade has settled and contributes zero exposure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .market import MarketModel


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

    def value(
        self,
        times: np.ndarray,
        spots: np.ndarray,
        market: MarketModel,
    ) -> np.ndarray:
        """Mark-to-market in USD, vectorised over scenarios and time.

        Parameters
        ----------
        times : shape (n_times,) - the exposure dates.
        spots : shape (n_paths, n_times) or (n_times,) - simulated S(t).
        market: the flat-curve model supplying DF_EUR and DF_USD.

        Returns
        -------
        Array broadcast to the shape of `spots`, implementing formula (1).
        """
        times = np.asarray(times, dtype=float)
        tau = self.maturity - times                     # time left to maturity

        # A trade that has already matured is worth nothing for exposure
        # purposes (it settled and left the netting set).
        alive = tau > 1e-12
        tau_clipped = np.maximum(tau, 0.0)

        df_eur = market.df_eur(tau_clipped)             # DF_EUR(t, T)
        df_usd = market.df_usd(tau_clipped)             # DF_USD(t, T)

        # V(t) = phi * N * [ DF_EUR(t,T) * S(t) - K * DF_USD(t,T) ]
        value = self.direction * self.notional * (df_eur * spots - self.strike * df_usd)
        return value * alive


# ----------------------------------------------------------------------------
# FX swap
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class FXSwap:
    """An FX swap = a near leg and a far leg in opposite directions.

    Economically the bank exchanges EUR for USD on the near date and reverses
    the exchange on the far date at a different, pre-agreed rate.  Because both
    legs are themselves forwards, the swap is valued as the sum of two
    `FXForward` values - no extra theory is needed.

    The two legs share the same EUR notional but have different strikes (the
    swap points) and different maturities.
    """

    near_leg: FXForward
    far_leg: FXForward
    label: str = ""

    @property
    def notional(self) -> float:
        return self.far_leg.notional

    @property
    def direction(self) -> int:
        """Directional sign of the *far* leg - the leg that dominates the
        long-dated exposure once the near leg has settled."""
        return self.far_leg.direction

    @property
    def maturity(self) -> float:
        return self.far_leg.maturity

    @property
    def last_date(self) -> float:
        return self.far_leg.maturity

    def value(
        self,
        times: np.ndarray,
        spots: np.ndarray,
        market: MarketModel,
    ) -> np.ndarray:
        """Swap MtM = near-leg MtM + far-leg MtM (both from formula (1))."""
        return (
            self.near_leg.value(times, spots, market)
            + self.far_leg.value(times, spots, market)
        )


Trade = FXForward | FXSwap


# ----------------------------------------------------------------------------
# Convenience constructors
# ----------------------------------------------------------------------------

def make_at_market_forward(
    notional: float,
    maturity: float,
    direction: int,
    market: MarketModel,
    label: str = "",
    strike_offset: float = 0.0,
) -> FXForward:
    """Build a forward whose strike is today's fair forward rate, optionally
    shifted by `strike_offset` (a relative move used to season legacy trades).

    A trade with strike_offset = 0 has zero mark-to-market at t = 0, which is
    the natural convention for a *new* incoming trade.
    """
    fair = market.forward_rate(0.0, maturity)
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
    market: MarketModel,
    label: str = "",
    near_offset: float = 0.0,
    far_offset: float = 0.0,
) -> FXSwap:
    """Build an FX swap from two at-market legs in opposite directions.

    `far_direction` fixes the direction of the far (long-dated) leg; the near
    leg automatically takes the opposite sign, which is what makes the package
    an FX swap rather than two unrelated forwards.
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
