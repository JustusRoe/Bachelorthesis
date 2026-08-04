"""
Market model: Nelson-Siegel initial curves, one-factor Hull-White short rates,
and the correlated three-factor simulation.

Dynamics under the DOMESTIC (USD) risk-neutral measure:

    dr_d = [theta_d(t) - a_d r_d] dt + sigma_d dW_d
    dr_f = [theta_f(t) - a_f r_f - rho_fS sigma_f sigma_S] dt + sigma_f dW_f
    dS   = (r_d - r_f) S dt + sigma_S S dW_S

The extra term in the foreign drift is the quanto adjustment (change of measure
from the foreign to the domestic risk-neutral measure).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


# ----------------------------------------------------------------------------
# Initial zero curves
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ZeroCurve:
    """Nelson-Siegel zero curve, continuously compounded.

    A parametric form is used because the Hull-White fit needs the instantaneous
    forward f(0,t), and differentiating an interpolated pillar curve gives a
    jagged forward curve.  b1 = b2 = 0 gives a flat curve at level b0.
    """

    b0: float
    b1: float = 0.0
    b2: float = 0.0
    tau: float = 1.5
    name: str = ""

    def _loadings(self, t):
        """The two Nelson-Siegel loadings; the slope loading tends to 1 at t=0."""
        x = np.maximum(np.asarray(t, dtype=float), 0.0) / self.tau
        exp_x = np.exp(-x)
        slope = np.where(x > 1e-12, (1.0 - exp_x) / np.where(x > 1e-12, x, 1.0), 1.0)
        return slope, slope - exp_x

    def zero_rate(self, t):
        """z(t) = b0 + b1*slope + b2*curvature."""
        slope, curvature = self._loadings(t)
        return self.b0 + self.b1 * slope + self.b2 * curvature

    def discount(self, t):
        """P(0,t) = exp(-t z(t))."""
        t = np.asarray(t, dtype=float)
        return np.exp(-t * self.zero_rate(t))

    def instantaneous_forward(self, t):
        """f(0,t) = -d/dt log P(0,t) = b0 + b1 e^-x + b2 x e^-x,  x = t/tau."""
        x = np.maximum(np.asarray(t, dtype=float), 0.0) / self.tau
        exp_x = np.exp(-x)
        return self.b0 + self.b1 * exp_x + self.b2 * x * exp_x

    def integrated_forward(self, t1, t2):
        """int_{t1}^{t2} f(0,u) du = log(P(0,t1)/P(0,t2)), exact."""
        return np.log(self.discount(t1) / self.discount(t2))

    def describe(self, tenors=(0.25, 1.0, 2.0, 3.0, 5.0)) -> str:
        return f"{self.name or 'curve'}: " + "  ".join(
            f"{t:g}y {self.zero_rate(t):.3%}" for t in tenors
        )


# The default curves, assembled from the parameters in `config`.  They live here
# rather than in `config` so that `config` stays free of package imports.
CURVE_USD = ZeroCurve(**config.CURVE_USD_PARAMS)
CURVE_EUR = ZeroCurve(**config.CURVE_EUR_PARAMS)


# ----------------------------------------------------------------------------
# One-factor Hull-White
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class HullWhiteModel:
    """Extended Vasicek short rate fitted exactly to `curve`.

    theta(t) is never formed explicitly.  Instead the short rate is decomposed as
    r(t) = x(t) + alpha(t), with x a mean-zero OU factor carrying all the
    randomness and alpha the deterministic shift that reprices the initial curve.
    With sigma = 0 this collapses to r(t) = f(0,t).
    """

    a: float
    sigma: float
    curve: ZeroCurve
    name: str = ""

    def __post_init__(self):
        if self.a <= 0.0:
            raise ValueError("Hull-White mean reversion `a` must be strictly positive")

    def B(self, t, T):
        """B(t,T) = (1-exp(-a(T-t)))/a.  Also -d log P(t,T)/dr."""
        tau = np.maximum(np.asarray(T, dtype=float) - np.asarray(t, dtype=float), 0.0)
        return (1.0 - np.exp(-self.a * tau)) / self.a

    def A(self, t, T):
        """Exact-fit coefficient (Brigo & Mercurio eq. 3.39):

        A(t,T) = P(0,T)/P(0,t) * exp{B f(0,t) - sigma^2/(4a) (1-e^-2at) B^2}
        """
        t, T = np.asarray(t, dtype=float), np.asarray(T, dtype=float)
        b = self.B(t, T)
        variance = (self.sigma**2 / (4.0 * self.a)) * (1.0 - np.exp(-2.0 * self.a * t))
        return (self.curve.discount(T) / self.curve.discount(t)) * np.exp(
            b * self.curve.instantaneous_forward(t) - variance * b**2
        )

    def bond_price(self, t, T, r):
        """P(t,T) = A(t,T) exp(-B(t,T) r(t)).  All arguments broadcast."""
        return self.A(t, T) * np.exp(-self.B(t, T) * np.asarray(r, dtype=float))

    def alpha(self, t):
        """alpha(t) = f(0,t) + sigma^2/(2a^2) (1-exp(-at))^2."""
        t = np.asarray(t, dtype=float)
        return self.curve.instantaneous_forward(t) + (
            self.sigma**2 / (2.0 * self.a**2)
        ) * (1.0 - np.exp(-self.a * t)) ** 2

    def integrated_alpha(self, t1, t2):
        """int alpha du, in closed form.

        Integrating the deterministic part analytically (rather than on the time
        grid) is what makes the sigma -> 0 limit exact to machine precision.
        """
        t1, t2 = np.asarray(t1, dtype=float), np.asarray(t2, dtype=float)
        # int (1-exp(-au))^2 du
        convexity = (
            (t2 - t1)
            - 2.0 * (np.exp(-self.a * t1) - np.exp(-self.a * t2)) / self.a
            + (np.exp(-2.0 * self.a * t1) - np.exp(-2.0 * self.a * t2)) / (2.0 * self.a)
        )
        return self.curve.integrated_forward(t1, t2) + (
            self.sigma**2 / (2.0 * self.a**2)
        ) * convexity

    def ou_decay(self, dt):
        """exp(-a dt): deterministic part of the exact OU transition."""
        return np.exp(-self.a * np.asarray(dt, dtype=float))

    def ou_step_std(self, dt):
        """sigma sqrt((1-exp(-2a dt))/(2a)): exact OU transition std."""
        dt = np.asarray(dt, dtype=float)
        return self.sigma * np.sqrt((1.0 - np.exp(-2.0 * self.a * dt)) / (2.0 * self.a))

    def quanto_shift(self, t, rho_fs, sigma_fx):
        """q(t) = -(rho_fS sigma_f sigma_S / a)(1-exp(-at)).

        Solves q' = -a q - rho_fS sigma_f sigma_S with q(0) = 0, i.e. the
        deterministic shift the change of measure adds to the foreign short rate.
        """
        t = np.asarray(t, dtype=float)
        return -(rho_fs * self.sigma * sigma_fx / self.a) * (1.0 - np.exp(-self.a * t))

    def integrated_quanto_shift(self, t1, t2, rho_fs, sigma_fx):
        """int q du, in closed form."""
        t1, t2 = np.asarray(t1, dtype=float), np.asarray(t2, dtype=float)
        base = (t2 - t1) - (np.exp(-self.a * t1) - np.exp(-self.a * t2)) / self.a
        return -(rho_fs * self.sigma * sigma_fx / self.a) * base


# ----------------------------------------------------------------------------
# Simulated scenarios
# ----------------------------------------------------------------------------

@dataclass
class Scenarios:
    """State variables on the (path, date) grid.  All matrices are (n_paths, n_times)."""

    times: np.ndarray
    spots: np.ndarray
    r_d: np.ndarray
    r_f: np.ndarray
    x_d: np.ndarray                 # mean-zero OU factors: the interpolation state
    x_f: np.ndarray
    df_domestic: np.ndarray         # D_d(0,t) = exp(-int r_d du), pathwise
    state_names: tuple[str, ...]

    def state(self, name: str) -> np.ndarray:
        return {"spot": self.spots, "x_d": self.x_d, "x_f": self.x_f}[name]

    @property
    def states(self) -> list[np.ndarray]:
        return [self.state(n) for n in self.state_names]


# ----------------------------------------------------------------------------
# The three-factor market
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ThreeFactorMarket:
    """Two Hull-White curves plus a correlated lognormal FX spot.

    Build with `ThreeFactorMarket.build(...)` so the rate-model switch is applied
    consistently.
    """

    hw_domestic: HullWhiteModel
    hw_foreign: HullWhiteModel
    sigma_fx: float
    spot_0: float
    rho_ds: float
    rho_fs: float
    rho_df: float
    apply_quanto: bool
    label: str = ""

    @classmethod
    def build(
        cls,
        rate_model: str = config.RATE_MODEL,
        curve_domestic: ZeroCurve = None,
        curve_foreign: ZeroCurve = None,
        sigma_d: float = config.VOL_RATE_USD,
        sigma_f: float = config.VOL_RATE_EUR,
        sigma_fx: float = config.VOL_FX,
        a_d: float = config.MEAN_REVERSION_USD,
        a_f: float = config.MEAN_REVERSION_EUR,
        spot_0: float = config.SPOT_0,
        rho_ds: float = config.CORR_USD_FX,
        rho_fs: float = config.CORR_EUR_FX,
        rho_df: float = config.CORR_USD_EUR,
        apply_quanto: bool = config.APPLY_QUANTO_ADJUSTMENT,
    ) -> "ThreeFactorMarket":
        """`rate_model="deterministic"` simply forces sigma_d = sigma_f = 0.

        Everything else is untouched, so the two modes share curves, FX vol and
        seed and any difference is attributable to rate volatility alone.
        """
        if rate_model not in config.RATE_MODELS:
            raise ValueError(f"rate_model must be one of {config.RATE_MODELS}")
        if rate_model == "deterministic":
            sigma_d = sigma_f = 0.0
        return cls(
            hw_domestic=HullWhiteModel(a_d, sigma_d,
                                       curve_domestic or CURVE_USD, "USD"),
            hw_foreign=HullWhiteModel(a_f, sigma_f,
                                      curve_foreign or CURVE_EUR, "EUR"),
            sigma_fx=sigma_fx, spot_0=spot_0, rho_ds=rho_ds, rho_fs=rho_fs,
            rho_df=rho_df, apply_quanto=apply_quanto, label=rate_model,
        )

    # -- static properties ---------------------------------------------------

    @property
    def state_names(self) -> tuple[str, ...]:
        """State variables the pricing function depends on.

        A rate factor with zero volatility is frozen at zero and carries no
        information, so it is dropped: deterministic mode returns just ("spot",),
        reproducing the one-dimensional state of the single-factor base model.
        """
        names = ["spot"]
        if self.hw_domestic.sigma > 0.0:
            names.append("x_d")
        if self.hw_foreign.sigma > 0.0:
            names.append("x_f")
        return tuple(names)

    def correlation(self) -> np.ndarray:
        return config.correlation_matrix(self.rho_ds, self.rho_fs, self.rho_df)

    # -- curve access and bond prices ---------------------------------------

    def p_domestic_0(self, t):
        return self.hw_domestic.curve.discount(t)

    def p_foreign_0(self, t):
        return self.hw_foreign.curve.discount(t)

    def forward_fx(self, maturity):
        """Covered interest rate parity: F(0,T) = S(0) P_f(0,T) / P_d(0,T)."""
        return self.spot_0 * self.p_foreign_0(maturity) / self.p_domestic_0(maturity)

    def bond_domestic(self, t, T, r_d):
        return self.hw_domestic.bond_price(t, T, r_d)

    def bond_foreign(self, t, T, r_f):
        """P_f(t,T).  The pricing function is unchanged by the measure change;
        what changes is the distribution of r_f, which carries the quanto shift."""
        return self.hw_foreign.bond_price(t, T, r_f)

    def short_rate_domestic(self, t, x_d):
        """r_d = x_d + alpha_d(t)."""
        return np.asarray(x_d, dtype=float) + self.hw_domestic.alpha(t)

    def short_rate_foreign(self, t, x_f):
        """r_f = x_f + alpha_f(t) + q_f(t)."""
        total = np.asarray(x_f, dtype=float) + self.hw_foreign.alpha(t)
        if self.apply_quanto:
            total = total + self.hw_foreign.quanto_shift(t, self.rho_fs, self.sigma_fx)
        return total

    def rates_from_state(self, t, state: dict) -> tuple[np.ndarray, np.ndarray]:
        """Map a point of the interpolation grid to (r_d, r_f).

        The grid spans the mean-zero factors; a factor absent from it is zero.
        """
        return (self.short_rate_domestic(t, state.get("x_d", 0.0)),
                self.short_rate_foreign(t, state.get("x_f", 0.0)))

    # -- simulation ----------------------------------------------------------

    def simulate(self, times: np.ndarray, n_paths: int, seed: int) -> Scenarios:
        """Simulate all three factors on `times`.

        The FX normals are drawn first, from their own generator, with the same
        call the single-factor base model used.  The Cholesky factor of a
        correlation matrix has L[0,0] = 1, so the FX Brownian increment is that
        draw untouched and switching the rate vols off reproduces the base
        model's spot paths exactly.

        Each short rate uses its exact OU transition.  The rate integrals needed
        for the FX drift and the money-market account split into an analytic
        deterministic part and a trapezoidal stochastic part - the only
        discretisation in the model.
        """
        times = np.asarray(times, dtype=float)
        n_times = times.size
        dt = np.diff(times, prepend=0.0)
        sqrt_dt = np.sqrt(dt)

        z_fx = np.random.default_rng(seed).standard_normal((n_paths, n_times))
        z_extra = np.random.default_rng(seed + 1_000_003).standard_normal((n_paths, n_times, 2))

        chol = np.linalg.cholesky(self.correlation())
        w_s = z_fx
        w_d = chol[1, 0] * z_fx + chol[1, 1] * z_extra[:, :, 0]
        w_f = chol[2, 0] * z_fx + chol[2, 1] * z_extra[:, :, 0] + chol[2, 2] * z_extra[:, :, 1]

        x_d = np.zeros((n_paths, n_times))
        x_f = np.zeros((n_paths, n_times))
        spots = np.empty((n_paths, n_times))
        log_df = np.zeros((n_paths, n_times))
        spots[:, 0] = self.spot_0                      # times[0] == 0

        for i in range(1, n_times):
            step, t_prev, t_now = dt[i], times[i - 1], times[i]

            # exact OU transition
            x_d[:, i] = (x_d[:, i - 1] * self.hw_domestic.ou_decay(step)
                         + self.hw_domestic.ou_step_std(step) * w_d[:, i])
            x_f[:, i] = (x_f[:, i - 1] * self.hw_foreign.ou_decay(step)
                         + self.hw_foreign.ou_step_std(step) * w_f[:, i])

            # int r dt = int alpha dt (analytic) + int x dt (trapezoid)
            int_r_d = (0.5 * (x_d[:, i - 1] + x_d[:, i]) * step
                       + self.hw_domestic.integrated_alpha(t_prev, t_now))
            int_r_f = (0.5 * (x_f[:, i - 1] + x_f[:, i]) * step
                       + self.hw_foreign.integrated_alpha(t_prev, t_now))
            if self.apply_quanto:
                int_r_f = int_r_f + self.hw_foreign.integrated_quanto_shift(
                    t_prev, t_now, self.rho_fs, self.sigma_fx)

            log_df[:, i] = log_df[:, i - 1] - int_r_d       # D_d(0,t)

            # d log S = (r_d - r_f - sigma_S^2/2) dt + sigma_S dW_S
            spots[:, i] = spots[:, i - 1] * np.exp(
                int_r_d - int_r_f - 0.5 * self.sigma_fx**2 * step
                + self.sigma_fx * sqrt_dt[i] * w_s[:, i]
            )

        return Scenarios(
            times=times, spots=spots,
            r_d=self.short_rate_domestic(times[None, :], x_d),
            r_f=self.short_rate_foreign(times[None, :], x_f),
            x_d=x_d, x_f=x_f, df_domestic=np.exp(log_df),
            state_names=self.state_names,
        )

    def describe(self) -> str:
        return "\n".join([
            f"  rate model          : {self.label}",
            f"  {self.hw_domestic.curve.describe()}",
            f"  {self.hw_foreign.curve.describe()}",
            f"  spot S(0)           : {self.spot_0:.4f} USD per EUR",
            f"  a_d / a_f           : {self.hw_domestic.a:.3f} / {self.hw_foreign.a:.3f}",
            f"  sigma_d / sigma_f   : {self.hw_domestic.sigma:.4%} / {self.hw_foreign.sigma:.4%}",
            f"  sigma_S             : {self.sigma_fx:.2%}",
            f"  rho_dS/rho_fS/rho_df: {self.rho_ds:+.2f} / {self.rho_fs:+.2f} / {self.rho_df:+.2f}",
            f"  quanto adjustment   : {'ON' if self.apply_quanto else 'OFF'}",
            f"  interpolation state : {self.state_names}",
        ])
