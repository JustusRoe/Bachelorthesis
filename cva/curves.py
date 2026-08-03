"""
Initial (time-zero) interest rate term structures.

The Hull-White models in `hullwhite.py` are *extended* Vasicek models: their
time-dependent drift theta(t) is chosen so that the model reproduces a given
initial zero curve exactly.  This module supplies that initial curve.

Parameterisation
----------------
The curve is a Nelson-Siegel curve, the standard parametric form used by central
banks to publish zero curves.  Using a parametric form rather than interpolated
pillars matters here: the Hull-White fit needs the INSTANTANEOUS FORWARD rate
f(0,t), which is a derivative of the curve, and differentiating an interpolated
curve produces a jagged (or even discontinuous) forward curve.  Nelson-Siegel is
smooth by construction and gives both the discount factor and the forward in
closed form.

    z(t) = b0 + b1 * (1-exp(-t/tau))/(t/tau)
              + b2 * [ (1-exp(-t/tau))/(t/tau) - exp(-t/tau) ]              (N1)

    f(0,t) = b0 + b1 * exp(-t/tau) + b2 * (t/tau) * exp(-t/tau)             (N2)

    P(0,t) = exp(-t * z(t))                                                 (N3)

(N2) is consistent with (N1) and (N3): integrating (N2) from 0 to t returns
t*z(t) exactly, which is what -log P(0,t) must equal.  That identity is checked
in the validation suite.

Interpretation of the coefficients:
    b0   long-run level of the curve,
    b1   short-end spread: z(0) = b0 + b1,
    b2   curvature / hump,
    tau  where the hump sits (years).

All coefficients used in this project are SYNTHETIC.  No market data is read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ZeroCurve:
    """A Nelson-Siegel initial zero curve, continuously compounded.

    Setting b1 = b2 = 0 gives a FLAT curve at level b0, which is how the
    single-factor base model's constant rates are reproduced exactly.
    """

    b0: float
    b1: float = 0.0
    b2: float = 0.0
    tau: float = 1.5
    name: str = ""

    # -- helper --------------------------------------------------------------

    def _loading(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the two Nelson-Siegel loadings, safely at t = 0.

        The slope loading (1-exp(-x))/x tends to 1 as x -> 0; evaluating it
        naively there would divide by zero, so the limit is substituted.
        """
        t = np.asarray(t, dtype=float)
        x = np.maximum(t, 0.0) / self.tau
        exp_x = np.exp(-x)
        slope = np.where(x > 1e-12, (1.0 - exp_x) / np.where(x > 1e-12, x, 1.0), 1.0)
        curvature = slope - exp_x
        return slope, curvature

    # -- the curve -----------------------------------------------------------

    def zero_rate(self, t: np.ndarray | float) -> np.ndarray:
        """Continuously-compounded zero rate z(t), formula (N1)."""
        slope, curvature = self._loading(t)
        return self.b0 + self.b1 * slope + self.b2 * curvature

    def discount(self, t: np.ndarray | float) -> np.ndarray:
        """P(0,t) = exp(-t * z(t)), formula (N3)."""
        t = np.asarray(t, dtype=float)
        return np.exp(-t * self.zero_rate(t))

    def instantaneous_forward(self, t: np.ndarray | float) -> np.ndarray:
        """f(0,t) = -d/dt log P(0,t), formula (N2).

        This is the quantity the Hull-White exact-fit formula consumes.
        """
        t = np.asarray(t, dtype=float)
        x = np.maximum(t, 0.0) / self.tau
        exp_x = np.exp(-x)
        return self.b0 + self.b1 * exp_x + self.b2 * x * exp_x

    def integrated_forward(self, t1: np.ndarray | float, t2: np.ndarray | float) -> np.ndarray:
        """int_{t1}^{t2} f(0,u) du = log( P(0,t1) / P(0,t2) ).

        Available in closed form because of the identity above, which is what
        lets the deterministic part of the short rate be integrated EXACTLY in
        the simulation rather than approximated on the time grid.
        """
        return np.log(self.discount(t1) / self.discount(t2))

    def forward_rate(self, t1: float, t2: float) -> float:
        """Simple continuously-compounded forward rate between two dates."""
        return float(self.integrated_forward(t1, t2) / (t2 - t1))

    def describe(self, tenors=(0.25, 1.0, 2.0, 3.0, 5.0)) -> str:
        """One-line summary of the curve, for the console report."""
        parts = [f"{t:g}y {self.zero_rate(t):.3%}" for t in tenors]
        return f"{self.name or 'curve'}: " + "  ".join(parts)
