"""
Market model: two flat risk-free curves plus a Garman-Kohlhagen spot process.

This is the *risk-factor* layer of the model.  It produces
  (a) deterministic discount factors for EUR and USD, and
  (b) a matrix of simulated EUR/USD spot scenarios,
which every downstream module consumes.

Only one risk factor is simulated (the FX spot).  Both interest-rate curves are
flat and deterministic, exactly as fixed in the thesis specification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass(frozen=True)
class MarketModel:
    """Flat-curve EUR/USD market with a lognormal spot rate.

    Parameters
    ----------
    spot_0 : S(0), USD per EUR.
    r_eur  : flat continuously-compounded EUR (foreign) zero rate.
    r_usd  : flat continuously-compounded USD (domestic) zero rate.
    vol    : annualised lognormal volatility of the spot.
    """

    spot_0: float = config.SPOT_0
    r_eur: float = config.RATE_EUR
    r_usd: float = config.RATE_USD
    vol: float = config.FX_VOL

    # -- discount factors ----------------------------------------------------

    def df_usd(self, tau: np.ndarray | float) -> np.ndarray | float:
        """DF_USD(t, t+tau) = exp(-r_USD * tau)  (flat curve)."""
        return np.exp(-self.r_usd * np.asarray(tau, dtype=float))

    def df_eur(self, tau: np.ndarray | float) -> np.ndarray | float:
        """DF_EUR(t, t+tau) = exp(-r_EUR * tau)  (flat curve)."""
        return np.exp(-self.r_eur * np.asarray(tau, dtype=float))

    # -- forward FX ----------------------------------------------------------

    def forward_rate(self, t: np.ndarray | float, maturity: float) -> np.ndarray | float:
        """Fair forward FX rate seen at time t for delivery at `maturity`.

        Covered interest rate parity:
            F(t, T) = S(t) * DF_EUR(t,T) / DF_USD(t,T) = S(t) * exp((r_USD - r_EUR)(T-t))

        Called with t = 0 it returns the *deterministic* forward implied by
        today's spot, which is how the at-market strikes are set.
        """
        tau = np.asarray(maturity, dtype=float) - np.asarray(t, dtype=float)
        return self.spot_0 * np.exp((self.r_usd - self.r_eur) * tau)

    # -- scenario generation -------------------------------------------------

    def simulate_spot(
        self,
        times: np.ndarray,
        n_paths: int,
        seed: int,
    ) -> np.ndarray:
        """Simulate the EUR/USD spot on `times` under the USD risk-neutral measure.

        The Garman-Kohlhagen model treats the foreign rate like a continuous
        dividend yield, so the spot follows a GBM with drift (r_USD - r_EUR):

            dS(t) = (r_USD - r_EUR) S(t) dt + sigma S(t) dW(t)

        which has the exact solution used below (no discretisation error):

            S(t) = S(0) * exp( (r_USD - r_EUR - 0.5*sigma^2) * t + sigma * W(t) )

        Returns
        -------
        ndarray of shape (n_paths, len(times)).  Column 0 corresponds to
        times[0]; if times[0] == 0 that column is identically S(0).
        """
        times = np.asarray(times, dtype=float)
        rng = np.random.default_rng(seed)

        # Increments of the Brownian motion: dW_k ~ N(0, dt_k), built once and
        # accumulated so that the whole path is consistent across time steps.
        dt = np.diff(times, prepend=0.0)                 # dt[0] = times[0] - 0
        dW = rng.standard_normal((n_paths, times.size)) * np.sqrt(dt)
        W = np.cumsum(dW, axis=1)                        # W(t_i) along each path

        drift = (self.r_usd - self.r_eur - 0.5 * self.vol**2) * times
        spot = self.spot_0 * np.exp(drift[None, :] + self.vol * W)
        return spot
