"""
One-factor Hull-White (extended Vasicek) short-rate model.

All the Hull-White algebra of the project lives here: the bond-price
reconstruction formula, the exact-fit to the initial curve, and the exact
simulation of the underlying Ornstein-Uhlenbeck factor.

Dynamics (under the model's OWN risk-neutral measure)
-----------------------------------------------------
        dr(t) = [theta(t) - a * r(t)] dt + sigma * dW(t)                    (H1)

theta(t) is not stored explicitly.  Instead the standard shifted decomposition
is used:

        r(t) = x(t) + alpha(t),        dx = -a x dt + sigma dW,  x(0) = 0    (H2)

        alpha(t) = f(0,t) + (sigma^2 / (2 a^2)) * (1 - exp(-a t))^2         (H3)

x is a mean-zero OU process that carries ALL the randomness; alpha is the
deterministic shift that makes the model reprice the initial curve.  This is
equivalent to specifying theta(t) but is far easier to simulate and to
differentiate, and it makes the sigma -> 0 limit exact and obvious: with
sigma = 0 we get x == 0 and r(t) == f(0,t).

Bond reconstruction
-------------------
        P(t,T) = A(t,T) * exp( -B(t,T) * r(t) )                             (H4)

        B(t,T) = (1 - exp(-a (T-t))) / a                                    (H5)

        A(t,T) = [P(0,T)/P(0,t)]
                 * exp{ B(t,T) f(0,t) - (sigma^2/(4a)) (1-exp(-2at)) B(t,T)^2 }   (H6)

(H6) is the exact-fit form (Brigo & Mercurio, "Interest Rate Models", eq. 3.39).
At t = 0 it collapses to A(0,T) = P(0,T) * exp(B(0,T) f(0,0)), which together
with r(0) = f(0,0) returns P(0,T) exactly - the model reprices today's curve by
construction, not by calibration.

Exact simulation of x
---------------------
        x(t+dt) = x(t) exp(-a dt) + sigma sqrt( (1-exp(-2 a dt)) / (2a) ) Z  (H7)

This is exact (no Euler discretisation): the OU transition density is Gaussian
with exactly these moments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curves import ZeroCurve


@dataclass(frozen=True)
class HullWhiteModel:
    """One-factor Hull-White model fitted to `curve`.

    Parameters
    ----------
    a     : mean reversion speed. Must be strictly positive (the a -> 0 limit
            would need separate formulas and is not used in this project).
    sigma : absolute (normal) volatility of the short rate. sigma = 0 makes the
            model deterministic and reduces every formula below to plain forward
            discounting off the initial curve.
    curve : the initial zero curve the model must reproduce.
    """

    a: float
    sigma: float
    curve: ZeroCurve
    name: str = ""

    def __post_init__(self) -> None:
        if self.a <= 0.0:
            raise ValueError("Hull-White mean reversion `a` must be strictly positive")

    # ------------------------------------------------------------------
    # Deterministic building blocks
    # ------------------------------------------------------------------

    def B(self, t: np.ndarray | float, T: np.ndarray | float) -> np.ndarray:
        """B(t,T) = (1 - exp(-a(T-t)))/a, formula (H5).

        Also the sensitivity of log P(t,T) to the short rate: d log P / dr = -B.
        """
        tau = np.maximum(np.asarray(T, dtype=float) - np.asarray(t, dtype=float), 0.0)
        return (1.0 - np.exp(-self.a * tau)) / self.a

    def A(self, t: np.ndarray | float, T: np.ndarray | float) -> np.ndarray:
        """A(t,T), the exact-fit coefficient of formula (H6)."""
        t = np.asarray(t, dtype=float)
        T = np.asarray(T, dtype=float)
        b = self.B(t, T)
        ratio = self.curve.discount(T) / self.curve.discount(t)
        variance_term = (self.sigma**2 / (4.0 * self.a)) * (1.0 - np.exp(-2.0 * self.a * t))
        return ratio * np.exp(b * self.curve.instantaneous_forward(t) - variance_term * b**2)

    def alpha(self, t: np.ndarray | float) -> np.ndarray:
        """alpha(t), the deterministic shift of formula (H3).

        With sigma = 0 this is exactly the initial instantaneous forward curve.
        """
        t = np.asarray(t, dtype=float)
        return self.curve.instantaneous_forward(t) + (self.sigma**2 / (2.0 * self.a**2)) * (
            1.0 - np.exp(-self.a * t)
        ) ** 2

    def integrated_alpha(self, t1: np.ndarray | float, t2: np.ndarray | float) -> np.ndarray:
        """int_{t1}^{t2} alpha(u) du, in closed form.

        Splits into the curve part, which the Nelson-Siegel curve integrates
        exactly, and the Hull-White convexity part:

            int (1-exp(-au))^2 du = (t2-t1)
                                    - 2 (exp(-a t1) - exp(-a t2)) / a
                                    + (exp(-2a t1) - exp(-2a t2)) / (2a)

        Integrating the deterministic part analytically (rather than on the time
        grid) is what makes the sigma = 0 case exact to machine precision.
        """
        t1 = np.asarray(t1, dtype=float)
        t2 = np.asarray(t2, dtype=float)
        curve_part = self.curve.integrated_forward(t1, t2)
        convexity = (
            (t2 - t1)
            - 2.0 * (np.exp(-self.a * t1) - np.exp(-self.a * t2)) / self.a
            + (np.exp(-2.0 * self.a * t1) - np.exp(-2.0 * self.a * t2)) / (2.0 * self.a)
        )
        return curve_part + (self.sigma**2 / (2.0 * self.a**2)) * convexity

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def bond_price(
        self,
        t: np.ndarray | float,
        T: np.ndarray | float,
        r: np.ndarray | float,
    ) -> np.ndarray:
        """P(t,T) = A(t,T) exp(-B(t,T) r(t)), formula (H4).

        `r` is the realised short rate on each path; `t` and `T` broadcast
        against it, so this works equally for a (paths x dates) scenario matrix
        and for a small tensor grid of interpolation nodes.
        """
        return self.A(t, T) * np.exp(-self.B(t, T) * np.asarray(r, dtype=float))

    # ------------------------------------------------------------------
    # Simulation of the mean-zero factor
    # ------------------------------------------------------------------

    def ou_decay(self, dt: np.ndarray | float) -> np.ndarray:
        """exp(-a dt): the deterministic part of the OU transition, formula (H7)."""
        return np.exp(-self.a * np.asarray(dt, dtype=float))

    def ou_step_std(self, dt: np.ndarray | float) -> np.ndarray:
        """sigma sqrt((1-exp(-2 a dt))/(2a)): the exact OU transition std, (H7)."""
        dt = np.asarray(dt, dtype=float)
        return self.sigma * np.sqrt((1.0 - np.exp(-2.0 * self.a * dt)) / (2.0 * self.a))

    # ------------------------------------------------------------------
    # Quanto drift (only used for the FOREIGN currency)
    # ------------------------------------------------------------------

    def quanto_shift(self, t: np.ndarray | float, rho_fs: float, sigma_fx: float) -> np.ndarray:
        """Deterministic drift correction q(t) for the foreign short rate.

        Under the DOMESTIC risk-neutral measure the foreign short rate picks up
        an extra drift term:

            dr_f = [theta_f(t) - a_f r_f - rho_fS sigma_f sigma_S] dt + sigma_f dW_f

        The extra term is deterministic, so it adds a deterministic shift q(t)
        on top of the foreign-measure decomposition, solving

            q'(t) = -a q(t) - rho_fS sigma_f sigma_S,     q(0) = 0

        whose solution is

            q(t) = -(rho_fS sigma_f sigma_S / a) * (1 - exp(-a t))           (H8)

        so that the full foreign short rate is  r_f = x_f + alpha_f + q_f.
        Omitting this term breaks covered interest rate parity in the simulation
        (see the quanto test in `validation.py`).
        """
        t = np.asarray(t, dtype=float)
        return -(rho_fs * self.sigma * sigma_fx / self.a) * (1.0 - np.exp(-self.a * t))

    def integrated_quanto_shift(
        self,
        t1: np.ndarray | float,
        t2: np.ndarray | float,
        rho_fs: float,
        sigma_fx: float,
    ) -> np.ndarray:
        """int_{t1}^{t2} q(u) du, in closed form.

            int (1-exp(-au)) du = (t2-t1) - (exp(-a t1) - exp(-a t2)) / a
        """
        t1 = np.asarray(t1, dtype=float)
        t2 = np.asarray(t2, dtype=float)
        base = (t2 - t1) - (np.exp(-self.a * t1) - np.exp(-self.a * t2)) / self.a
        return -(rho_fs * self.sigma * sigma_fx / self.a) * base
