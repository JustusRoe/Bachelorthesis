"""
The three-factor market model: two Hull-White short rates plus a lognormal FX spot.

Risk factors, all under the DOMESTIC (USD) risk-neutral measure
---------------------------------------------------------------
    dr_d = [theta_d(t) - a_d r_d] dt + sigma_d dW_d                          (M1)
    dr_f = [theta_f(t) - a_f r_f - rho_fS sigma_f sigma_S] dt + sigma_f dW_f (M2)
    dS   = (r_d - r_f) S dt + sigma_S S dW_S                                 (M3)

with a full 3x3 correlation between (W_S, W_d, W_f).

The extra term in (M2) is the QUANTO adjustment.  The foreign short rate is
specified under the foreign measure; carrying it over to the domestic measure
shifts its drift by -rho_fS sigma_f sigma_S.  Dropping that term silently breaks
covered interest rate parity in the simulation - `validation.py` demonstrates
this and measures the resulting bias.

Note that the foreign BOND PRICE formula is unchanged by the measure change:
P_f(t,T) = A_f(t,T) exp(-B_f(t,T) r_f(t)) is a deterministic function of the
state, derived under the foreign measure.  What the measure change alters is the
distribution of r_f(t), not the map from r_f to the bond price.

Simulation scheme
-----------------
Each short rate is decomposed as r = x + alpha (+ q for the foreign one), where
x is a mean-zero OU factor simulated with its EXACT Gaussian transition and
alpha, q are deterministic.  The FX spot is then advanced in log space:

    S(t_i) = S(t_{i-1}) * exp( I_d - I_f - 0.5 sigma_S^2 dt + sigma_S sqrt(dt) Z_S )

where I_d, I_f are the integrals of the two short rates across the step.  Those
integrals are split as

    int r dt  =  int alpha dt   (ANALYTIC, closed form from the curve)
               + int x dt       (trapezoidal on the simulated factor)

This split is the only discretisation in the model, and it is deliberate: the
deterministic part is integrated exactly, so switching the rate volatilities off
makes the whole scheme exact rather than merely accurate.  The remaining
O(dt^2) bias on the stochastic part is ~1e-6 in relative terms, two orders of
magnitude below the Monte Carlo error, which the curve-reproduction test
confirms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .curves import ZeroCurve
from .hullwhite import HullWhiteModel


# ----------------------------------------------------------------------------
# Simulated scenarios
# ----------------------------------------------------------------------------

@dataclass
class Scenarios:
    """The output of one simulation run: every state variable on the grid.

    All matrices have shape (n_paths, n_times) and share the same column
    ordering as `times`.
    """

    times: np.ndarray           # (n_times,) exposure dates, times[0] = 0
    spots: np.ndarray           # S(t) in USD per EUR
    r_d: np.ndarray             # realised domestic short rate
    r_f: np.ndarray             # realised foreign short rate
    x_d: np.ndarray             # mean-zero domestic OU factor (the state variable)
    x_f: np.ndarray             # mean-zero foreign OU factor
    df_domestic: np.ndarray     # D_d(0,t) = exp(-int_0^t r_d du), pathwise
    state_names: tuple[str, ...]  # which variables the interpolation grid spans

    @property
    def n_paths(self) -> int:
        return self.spots.shape[0]

    def state(self, name: str) -> np.ndarray:
        """Fetch a state variable by name (used by the interpolation grid)."""
        return {"spot": self.spots, "x_d": self.x_d, "x_f": self.x_f}[name]

    @property
    def states(self) -> list[np.ndarray]:
        """The active state variables, in the order the tensor grid uses."""
        return [self.state(name) for name in self.state_names]


# ----------------------------------------------------------------------------
# The market model
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ThreeFactorMarket:
    """Two Hull-White curves plus a correlated lognormal FX spot.

    Build it with `ThreeFactorMarket.build(...)` rather than directly, so that
    the `rate_model` switch is applied consistently.
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

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

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
        """Assemble the model.

        `rate_model="deterministic"` simply forces sigma_d = sigma_f = 0.  Every
        other input is untouched, so the deterministic and stochastic runs share
        the same curves, the same FX volatility and the same seed - which is
        what makes their difference attributable to rate volatility alone.
        """
        if rate_model not in config.RATE_MODELS:
            raise ValueError(f"rate_model must be one of {config.RATE_MODELS}")
        if rate_model == "deterministic":
            sigma_d = sigma_f = 0.0

        curve_domestic = curve_domestic if curve_domestic is not None else config.CURVE_USD
        curve_foreign = curve_foreign if curve_foreign is not None else config.CURVE_EUR

        return cls(
            hw_domestic=HullWhiteModel(a=a_d, sigma=sigma_d, curve=curve_domestic, name="USD"),
            hw_foreign=HullWhiteModel(a=a_f, sigma=sigma_f, curve=curve_foreign, name="EUR"),
            sigma_fx=sigma_fx,
            spot_0=spot_0,
            rho_ds=rho_ds,
            rho_fs=rho_fs,
            rho_df=rho_df,
            apply_quanto=apply_quanto,
            label=rate_model,
        )

    # ------------------------------------------------------------------
    # Static properties
    # ------------------------------------------------------------------

    @property
    def state_names(self) -> tuple[str, ...]:
        """State variables the pricing function actually depends on.

        A rate factor with zero volatility is frozen at zero and carries no
        information, so it is dropped from the interpolation grid.  Under
        deterministic rates this returns just ("spot",), reproducing the
        one-dimensional state of the single-factor base model.
        """
        names = ["spot"]
        if self.hw_domestic.sigma > 0.0:
            names.append("x_d")
        if self.hw_foreign.sigma > 0.0:
            names.append("x_f")
        return tuple(names)

    def correlation(self) -> np.ndarray:
        """The 3x3 correlation matrix in factor order (S, x_d, x_f)."""
        return config.correlation_matrix(self.rho_ds, self.rho_fs, self.rho_df)

    # ------------------------------------------------------------------
    # Deterministic curve access
    # ------------------------------------------------------------------

    def p_domestic_0(self, t: np.ndarray | float) -> np.ndarray:
        """Today's domestic discount factor P_d(0,t) off the initial curve."""
        return self.hw_domestic.curve.discount(t)

    def p_foreign_0(self, t: np.ndarray | float) -> np.ndarray:
        """Today's foreign discount factor P_f(0,t) off the initial curve."""
        return self.hw_foreign.curve.discount(t)

    def forward_fx(self, maturity: np.ndarray | float) -> np.ndarray:
        """Fair FX forward rate for delivery at `maturity`, seen today.

        Covered interest rate parity:  F(0,T) = S(0) P_f(0,T) / P_d(0,T).
        Used to strike the at-market trades.
        """
        return self.spot_0 * self.p_foreign_0(maturity) / self.p_domestic_0(maturity)

    # ------------------------------------------------------------------
    # Bond prices on simulated scenarios
    # ------------------------------------------------------------------

    def bond_domestic(self, t, T, r_d) -> np.ndarray:
        """P_d(t,T) from the domestic Hull-White reconstruction formula."""
        return self.hw_domestic.bond_price(t, T, r_d)

    def bond_foreign(self, t, T, r_f) -> np.ndarray:
        """P_f(t,T) from the foreign Hull-White reconstruction formula.

        Evaluated at the realised foreign short rate, which under the domestic
        measure already carries the quanto shift.
        """
        return self.hw_foreign.bond_price(t, T, r_f)

    def short_rate_domestic(self, t, x_d) -> np.ndarray:
        """r_d(t) = x_d(t) + alpha_d(t)."""
        return np.asarray(x_d, dtype=float) + self.hw_domestic.alpha(t)

    def short_rate_foreign(self, t, x_f) -> np.ndarray:
        """r_f(t) = x_f(t) + alpha_f(t) + q_f(t), including the quanto shift."""
        total = np.asarray(x_f, dtype=float) + self.hw_foreign.alpha(t)
        if self.apply_quanto:
            total = total + self.hw_foreign.quanto_shift(t, self.rho_fs, self.sigma_fx)
        return total

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        times: np.ndarray,
        n_paths: int,
        seed: int,
    ) -> Scenarios:
        """Simulate all three factors on `times`.

        Random numbers
        --------------
        The FX normals are drawn FIRST, from their own generator, with exactly
        the same call signature the single-factor base model used.  Because the
        Cholesky factor of a correlation matrix has L[0,0] = 1, the FX Brownian
        increment is that draw untouched, so switching the rate volatilities off
        reproduces the base model's spot paths.  The two extra normals needed to
        correlate the rate factors come from a separate stream, so changing the
        rate parameters never perturbs the FX paths.
        """
        times = np.asarray(times, dtype=float)
        n_times = times.size
        dt = np.diff(times, prepend=0.0)
        sqrt_dt = np.sqrt(dt)

        rng_fx = np.random.default_rng(seed)
        rng_rates = np.random.default_rng(seed + 1_000_003)   # disjoint stream

        z_fx = rng_fx.standard_normal((n_paths, n_times))
        z_extra = rng_rates.standard_normal((n_paths, n_times, 2))

        # Correlate the three drivers.  L is lower triangular with L[0,0] = 1.
        chol = np.linalg.cholesky(self.correlation())
        w_s = z_fx
        w_d = chol[1, 0] * z_fx + chol[1, 1] * z_extra[:, :, 0]
        w_f = chol[2, 0] * z_fx + chol[2, 1] * z_extra[:, :, 0] + chol[2, 2] * z_extra[:, :, 1]

        x_d = np.zeros((n_paths, n_times))
        x_f = np.zeros((n_paths, n_times))
        spots = np.empty((n_paths, n_times))
        log_df = np.zeros((n_paths, n_times))

        spots[:, 0] = self.spot_0          # times[0] == 0, so nothing has happened yet

        for i in range(1, n_times):
            step = dt[i]
            t_prev, t_now = times[i - 1], times[i]

            # --- exact OU transition, formula (H7) --------------------------
            x_d[:, i] = (
                x_d[:, i - 1] * self.hw_domestic.ou_decay(step)
                + self.hw_domestic.ou_step_std(step) * w_d[:, i]
            )
            x_f[:, i] = (
                x_f[:, i - 1] * self.hw_foreign.ou_decay(step)
                + self.hw_foreign.ou_step_std(step) * w_f[:, i]
            )

            # --- integrate the short rates across the step ------------------
            # Deterministic parts analytically, stochastic part by trapezoid.
            int_x_d = 0.5 * (x_d[:, i - 1] + x_d[:, i]) * step
            int_x_f = 0.5 * (x_f[:, i - 1] + x_f[:, i]) * step

            int_r_d = int_x_d + self.hw_domestic.integrated_alpha(t_prev, t_now)
            int_r_f = int_x_f + self.hw_foreign.integrated_alpha(t_prev, t_now)
            if self.apply_quanto:
                int_r_f = int_r_f + self.hw_foreign.integrated_quanto_shift(
                    t_prev, t_now, self.rho_fs, self.sigma_fx
                )

            # --- domestic money-market account ------------------------------
            # D_d(0,t) = exp(-int_0^t r_d du)
            log_df[:, i] = log_df[:, i - 1] - int_r_d

            # --- FX spot, formula (M3) in log space -------------------------
            # d log S = (r_d - r_f - 0.5 sigma_S^2) dt + sigma_S dW_S
            spots[:, i] = spots[:, i - 1] * np.exp(
                int_r_d - int_r_f
                - 0.5 * self.sigma_fx**2 * step
                + self.sigma_fx * sqrt_dt[i] * w_s[:, i]
            )

        return Scenarios(
            times=times,
            spots=spots,
            r_d=self.short_rate_domestic(times[None, :], x_d),
            r_f=self.short_rate_foreign(times[None, :], x_f),
            x_d=x_d,
            x_f=x_f,
            df_domestic=np.exp(log_df),
            state_names=self.state_names,
        )

    # ------------------------------------------------------------------
    # Turning interpolation-grid state values back into short rates
    # ------------------------------------------------------------------

    def rates_from_state(self, t, state: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Map a point of the interpolation grid to (r_d, r_f).

        The grid spans the mean-zero factors x_d, x_f rather than the short
        rates themselves, because their range is symmetric around zero and known
        in advance.  Any factor absent from the grid (zero volatility) is zero.
        """
        x_d = state.get("x_d", 0.0)
        x_f = state.get("x_f", 0.0)
        return self.short_rate_domestic(t, x_d), self.short_rate_foreign(t, x_f)

    def describe(self) -> str:
        """Multi-line summary for the console report."""
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
