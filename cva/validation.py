"""
Validation suite for the three-factor market model.

Five checks, run BEFORE any CVA result is produced.  A model that does not
reprice its own inputs cannot be trusted to price anything else, so this module
runs first and `main.py` refuses to continue if a mandatory test fails.

    1. CURVE REPRODUCTION      does the simulation reprice the initial curves?
    2. FX MARTINGALE           is a fair-struck forward worth zero today?
    3. QUANTO                  is the change-of-measure drift doing its job?
    4. ZERO-VOL DEGENERACY     does sigma_d = sigma_f = 0 give the base model?
    5. CONVERGENCE             how does CVA behave as paths increase?

Tests 1 and 2 are statistical: they compare a Monte Carlo estimate against a
closed-form target and pass when the gap is inside a few standard errors.  Tests
3 and 4 are deterministic comparisons run on COMMON RANDOM NUMBERS, which makes
them far sharper than an independent-runs comparison would be - the quanto bias
we are looking for is smaller than the raw Monte Carlo noise, and only cancels
because both runs see identical draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .credit import HazardCurve
from .cva_engine import cva_from_values, cva_standard_error
from .curves import ZeroCurve
from .market import ThreeFactorMarket
from .scenarios import netting_set_scenario_values


# ----------------------------------------------------------------------------

@dataclass
class Check:
    """One line of the validation report."""

    test: str
    case: str
    estimate: float
    target: float
    error: float
    tolerance: float
    passed: bool
    units: str = ""
    note: str = ""

    def as_row(self) -> dict:
        return {
            "test": self.test, "case": self.case, "estimate": self.estimate,
            "target": self.target, "error": self.error, "tolerance": self.tolerance,
            "units": self.units, "passed": self.passed, "note": self.note,
        }


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> None:
        self.checks.append(check)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.as_row() for c in self.checks])


# ----------------------------------------------------------------------------
# Test 1 - curve reproduction
# ----------------------------------------------------------------------------

def test_curve_reproduction(report: ValidationReport, n_paths: int, seed: int) -> None:
    """Does the simulated model reprice the two initial zero curves?

    DOMESTIC.  The domestic money-market account is the numeraire, so a zero
    coupon bond must satisfy

        E[ D_d(0,T) ] = P_d(0,T).                                        (V1)

    FOREIGN.  Under the domestic measure E[exp(-int r_f)] is NOT P_f(0,T) - that
    identity holds under the FOREIGN measure.  The correct domestic statement is
    that one unit of foreign currency delivered at T is worth S(0) P_f(0,T)
    today:

        E[ D_d(0,T) * S(T) ] = S(0) * P_f(0,T).                          (V2)

    (V2) is simultaneously the foreign curve check and the covered-interest-
    parity check, which is why the quanto adjustment is what makes it hold.

    A second, stronger foreign check reaches beyond the simulated date: the
    domestic value of a foreign bond bought at T and maturing at T' must be

        E[ D_d(0,T) * S(T) * P_f(T,T') ] = S(0) * P_f(0,T').             (V3)
    """
    market = ThreeFactorMarket.build(rate_model="stochastic")
    tenors = np.array(config.VALIDATION_TENORS, dtype=float)
    times = np.unique(np.concatenate([[0.0], config.TIME_GRID, tenors]))
    scen = market.simulate(times, n_paths, seed)

    index = {round(float(t), 10): i for i, t in enumerate(times)}

    for tenor in tenors:
        col = index[round(float(tenor), 10)]

        # --- (V1) domestic ---------------------------------------------------
        sample = scen.df_domestic[:, col]
        estimate, target = float(sample.mean()), float(market.p_domestic_0(tenor))
        stderr = float(sample.std(ddof=1) / np.sqrt(n_paths))
        report.add(Check(
            test="1. curve reproduction", case=f"USD P(0,{tenor:g}y)",
            estimate=estimate, target=target, error=estimate - target,
            tolerance=config.VALIDATION_TOLERANCE_SIGMAS * stderr,
            passed=abs(estimate - target) <= config.VALIDATION_TOLERANCE_SIGMAS * stderr,
            units="discount factor", note=f"MC std err {stderr:.2e}",
        ))

        # --- (V2) foreign, via covered interest parity -----------------------
        sample = scen.df_domestic[:, col] * scen.spots[:, col] / market.spot_0
        estimate, target = float(sample.mean()), float(market.p_foreign_0(tenor))
        stderr = float(sample.std(ddof=1) / np.sqrt(n_paths))
        report.add(Check(
            test="1. curve reproduction", case=f"EUR P(0,{tenor:g}y)",
            estimate=estimate, target=target, error=estimate - target,
            tolerance=config.VALIDATION_TOLERANCE_SIGMAS * stderr,
            passed=abs(estimate - target) <= config.VALIDATION_TOLERANCE_SIGMAS * stderr,
            units="discount factor", note=f"MC std err {stderr:.2e}",
        ))

        # --- (V3) foreign bond seen from a future date -----------------------
        far = float(tenor) + 2.0
        if far <= times[-1] + 5.0:
            p_f_forward = market.bond_foreign(tenor, far, scen.r_f[:, col])
            sample = scen.df_domestic[:, col] * scen.spots[:, col] / market.spot_0 * p_f_forward
            estimate, target = float(sample.mean()), float(market.p_foreign_0(far))
            stderr = float(sample.std(ddof=1) / np.sqrt(n_paths))
            report.add(Check(
                test="1. curve reproduction",
                case=f"EUR P(0,{far:g}y) via P({tenor:g}y,{far:g}y)",
                estimate=estimate, target=target, error=estimate - target,
                tolerance=config.VALIDATION_TOLERANCE_SIGMAS * stderr,
                passed=abs(estimate - target) <= config.VALIDATION_TOLERANCE_SIGMAS * stderr,
                units="discount factor", note=f"MC std err {stderr:.2e}",
            ))


# ----------------------------------------------------------------------------
# Test 2 - FX martingale / fair forward
# ----------------------------------------------------------------------------

def fair_forward_error(market: ThreeFactorMarket, scen, tenor: float, times: np.ndarray):
    """Simulated present value of a forward struck at today's fair rate.

    A forward on one EUR struck at K = F(0,T) = S(0) P_f(0,T)/P_d(0,T) must be
    worth exactly zero today.  Simulated, its value is

        E[ D_d(0,T) * (S(T) - K) ]

    Returns (value, standard error, relative error against the forward level).
    """
    col = int(np.argmin(np.abs(times - tenor)))
    strike = float(market.forward_fx(tenor))
    payoff = scen.df_domestic[:, col] * (scen.spots[:, col] - strike)
    value = float(payoff.mean())
    stderr = float(payoff.std(ddof=1) / np.sqrt(payoff.size))
    # Express as a relative error on the forward rate itself, in basis points.
    scale = strike * float(market.p_domestic_0(tenor))
    return value, stderr, value / scale


def test_fx_martingale(report: ValidationReport, n_paths: int, seed: int) -> None:
    """A fair-struck forward must have zero value at t = 0."""
    market = ThreeFactorMarket.build(rate_model="stochastic")
    tenors = np.array(config.VALIDATION_TENORS, dtype=float)
    times = np.unique(np.concatenate([[0.0], config.TIME_GRID, tenors]))
    scen = market.simulate(times, n_paths, seed)

    for tenor in tenors:
        value, stderr, relative = fair_forward_error(market, scen, float(tenor), times)
        report.add(Check(
            test="2. FX martingale", case=f"fair {tenor:g}y forward, PV per EUR",
            estimate=value, target=0.0, error=value,
            tolerance=config.VALIDATION_TOLERANCE_SIGMAS * stderr,
            passed=abs(value) <= config.VALIDATION_TOLERANCE_SIGMAS * stderr,
            units="USD per EUR of notional",
            note=f"{1e4*relative:+.2f} bp of forward; MC std err {stderr:.2e}",
        ))


# ----------------------------------------------------------------------------
# Test 3 - quanto adjustment
# ----------------------------------------------------------------------------

def analytic_quanto_bias(market: ThreeFactorMarket, tenor: float) -> float:
    """Closed-form bias on the T-forward if the quanto adjustment were omitted.

    Omitting the -rho_fS sigma_f sigma_S drift leaves the foreign short rate with
    its FOREIGN-measure dynamics under the DOMESTIC measure.  Changing measure
    with the FX density then shifts it by +rho_fS sigma_f sigma_S instead of
    leaving it alone, i.e. by

        p(t) = (rho_fS sigma_f sigma_S / a_f) * (1 - exp(-a_f t)),

    so the simulated foreign discount factor is out by exp(-int_0^T p(u) du) and
    the fair forward is overstated by the same factor:

        relative bias = exp( -(rho sigma_f sigma_S / a_f)
                             * [T - (1-exp(-a_f T))/a_f] ) - 1              (V4)
    """
    hw_f = market.hw_foreign
    a = hw_f.a
    coefficient = market.rho_fs * hw_f.sigma * market.sigma_fx / a
    integral = coefficient * (tenor - (1.0 - np.exp(-a * tenor)) / a)
    return float(np.exp(-integral) - 1.0)


def test_quanto(report: ValidationReport, n_paths: int, seed: int) -> None:
    """Is the change-of-measure drift on the foreign short rate doing its job?

    For each rho_fS the fair forward is priced twice on COMMON RANDOM NUMBERS,
    once with the adjustment and once without.  Common random numbers matter
    here: the bias being measured (tens of basis points) is comparable to the
    raw Monte Carlo noise, so an independent-runs comparison would be
    inconclusive, whereas the paired difference is essentially noiseless.

    Two things must hold.
      (a) WITH the adjustment the forward is fair at every rho_fS, including the
          strongly correlated case.
      (b) WITHOUT it the forward is biased by exactly the amount formula (V4)
          predicts - which is what shows the term is not merely cosmetic.
    """
    tenor = 5.0
    times = np.unique(np.concatenate([[0.0], config.TIME_GRID, [tenor]]))

    for rho_fs in config.QUANTO_TEST_CORRELATIONS:
        results = {}
        for applied in (True, False):
            market = ThreeFactorMarket.build(
                rate_model="stochastic", rho_fs=rho_fs, apply_quanto=applied
            )
            scen = market.simulate(times, n_paths, seed)     # same seed -> same draws
            value, stderr, relative = fair_forward_error(market, scen, tenor, times)
            results[applied] = (value, stderr, relative, market)

        # (a) the adjusted model must price the forward fairly
        value, stderr, relative, market = results[True]
        report.add(Check(
            test="3. quanto adjustment", case=f"rho_fS={rho_fs:+.2f}, adjustment ON",
            estimate=1e4 * relative, target=0.0, error=1e4 * relative,
            tolerance=1e4 * config.VALIDATION_TOLERANCE_SIGMAS * stderr
                      / (market.forward_fx(tenor) * market.p_domestic_0(tenor)),
            passed=abs(value) <= config.VALIDATION_TOLERANCE_SIGMAS * stderr,
            units="bp of the fair forward",
            note="fair forward must price to zero",
        ))

        # (b) the unadjusted model must be off by the predicted amount
        predicted = analytic_quanto_bias(market, tenor)
        observed = results[False][2] - results[True][2]
        # Tolerance: 5% of the predicted bias, or 0.5 bp when the prediction is
        # itself ~0 (the rho_fS = 0 case, where there is nothing to correct).
        tolerance = max(0.05 * abs(predicted), 0.5e-4)
        report.add(Check(
            test="3. quanto adjustment", case=f"rho_fS={rho_fs:+.2f}, adjustment OFF",
            estimate=1e4 * observed, target=1e4 * predicted,
            error=1e4 * (observed - predicted), tolerance=1e4 * tolerance,
            passed=abs(observed - predicted) <= tolerance,
            units="bp of the fair forward",
            note="paired difference vs closed form (V4)",
        ))


# ----------------------------------------------------------------------------
# Test 4 - zero-volatility degeneracy
# ----------------------------------------------------------------------------

def test_zero_vol_degeneracy(report: ValidationReport, n_paths: int, seed: int) -> None:
    """With sigma_d = sigma_f = 0 the model must collapse to the base model.

    Two statements are checked, both to machine precision rather than to within
    Monte Carlo error, because both runs use identical random draws.

      (a) DETERMINISTIC MODE.  Building with rate_model="deterministic" must be
          identical to building the stochastic model with both rate volatilities
          set to zero.

      (b) SINGLE-FACTOR EQUIVALENCE.  With FLAT curves the zero-volatility model
          must reproduce the closed-form single-factor base model exactly:

              S(t)      = S(0) exp((r_d - r_f - sigma_S^2/2) t + sigma_S W(t))
              D_d(0,t)  = exp(-r_d t)
              P_x(t,T)  = exp(-r_x (T-t))

          This is the fallback the thesis reports against, so it has to be an
          identity, not an approximation.
    """
    times = config.TIME_GRID

    # --- (a) deterministic mode == stochastic mode with zero rate vols ------
    det = ThreeFactorMarket.build(rate_model="deterministic")
    zero = ThreeFactorMarket.build(rate_model="stochastic", sigma_d=0.0, sigma_f=0.0)
    scen_det = det.simulate(times, n_paths, seed)
    scen_zero = zero.simulate(times, n_paths, seed)

    gap = float(np.max(np.abs(scen_det.spots - scen_zero.spots) / scen_det.spots))
    report.add(Check(
        test="4. zero-vol degeneracy", case="deterministic mode == sigma_rate=0",
        estimate=gap, target=0.0, error=gap, tolerance=1e-14, passed=gap <= 1e-14,
        units="max relative spot difference", note="identical random draws",
    ))

    # --- (b) flat curves + zero rate vol == the single-factor base model ----
    r_d, r_f = 0.04, 0.02
    flat = ThreeFactorMarket.build(
        rate_model="deterministic",
        curve_domestic=ZeroCurve(b0=r_d, name="USD flat"),
        curve_foreign=ZeroCurve(b0=r_f, name="EUR flat"),
    )
    scen_flat = flat.simulate(times, n_paths, seed)

    # Closed-form single-factor base model on the same Brownian increments.
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, times.size))
    dt = np.diff(times, prepend=0.0)
    brownian = np.cumsum(z * np.sqrt(dt), axis=1)
    reference_spots = config.SPOT_0 * np.exp(
        (r_d - r_f - 0.5 * config.VOL_FX**2) * times[None, :] + config.VOL_FX * brownian
    )

    gap = float(np.max(np.abs(scen_flat.spots - reference_spots) / reference_spots))
    report.add(Check(
        test="4. zero-vol degeneracy", case="flat curves == single-factor GBM spot",
        estimate=gap, target=0.0, error=gap, tolerance=1e-12, passed=gap <= 1e-12,
        units="max relative spot difference",
        note="closed-form base model, same draws",
    ))

    df_gap = float(np.max(np.abs(scen_flat.df_domestic - np.exp(-r_d * times)[None, :])))
    report.add(Check(
        test="4. zero-vol degeneracy", case="flat curves == exp(-r_d t) discounting",
        estimate=df_gap, target=0.0, error=df_gap, tolerance=1e-14, passed=df_gap <= 1e-14,
        units="max absolute DF difference", note="deterministic discounting recovered",
    ))

    # Bond prices must degenerate to plain forward discounting.
    bond_gap = float(np.max(np.abs(
        flat.bond_foreign(times[None, :], 5.0, scen_flat.r_f)
        - np.exp(-r_f * np.maximum(5.0 - times, 0.0))[None, :]
    )))
    report.add(Check(
        test="4. zero-vol degeneracy", case="flat curves == exp(-r_f (T-t)) bond",
        estimate=bond_gap, target=0.0, error=bond_gap, tolerance=1e-14,
        passed=bond_gap <= 1e-14, units="max absolute bond difference",
        note="Hull-White A(t,T) collapses correctly",
    ))


# ----------------------------------------------------------------------------
# Test 5 - Monte Carlo convergence
# ----------------------------------------------------------------------------

def test_convergence(
    report: ValidationReport,
    netting_set,
    seed: int,
) -> pd.DataFrame:
    """CVA of one netting set against the number of paths, with 95% intervals.

    Reported for the largest netting set, which carries the biggest CVA and
    therefore the biggest absolute simulation noise.  The pass criterion is
    twofold: the standard error must fall roughly like 1/sqrt(N), and the
    production estimate must sit inside the confidence interval of the largest
    run.
    """
    market = ThreeFactorMarket.build(rate_model="stochastic")
    times = config.TIME_GRID
    rows = []

    for n_paths in config.CONVERGENCE_PATHS:
        scen = market.simulate(times, n_paths, seed)
        values = netting_set_scenario_values(netting_set.trades, market, scen)
        cva = cva_from_values(values, scen, market, netting_set.hazard)
        stderr = cva_standard_error(values, scen, netting_set.hazard)
        rows.append(dict(
            n_paths=n_paths, cva=cva, std_error=stderr,
            ci_low=cva - 1.96 * stderr, ci_high=cva + 1.96 * stderr,
            rel_std_error=stderr / abs(cva) if cva else np.nan,
        ))

    frame = pd.DataFrame(rows)

    # (a) the standard error should scale like 1/sqrt(N): fit a log-log slope.
    slope = float(np.polyfit(np.log(frame["n_paths"]), np.log(frame["std_error"]), 1)[0])
    report.add(Check(
        test="5. convergence", case="std error scaling exponent",
        estimate=slope, target=-0.5, error=slope + 0.5, tolerance=0.10,
        passed=abs(slope + 0.5) <= 0.10, units="d log(se) / d log(N)",
        note="Monte Carlo theory predicts -0.5",
    ))

    # (b) the production estimate must lie inside the finest run's interval.
    production = frame.loc[frame["n_paths"] == config.N_PATHS, "cva"]
    finest = frame.iloc[-1]
    if not production.empty:
        estimate = float(production.iloc[0])
        inside = finest["ci_low"] <= estimate <= finest["ci_high"]
        report.add(Check(
            test="5. convergence",
            case=f"{config.N_PATHS:,}-path CVA inside {int(finest['n_paths']):,}-path 95% CI",
            estimate=estimate, target=float(finest["cva"]),
            error=estimate - float(finest["cva"]),
            tolerance=1.96 * float(finest["std_error"]),
            passed=bool(inside), units="USD",
            note=f"production relative std error "
                 f"{frame.loc[frame['n_paths'] == config.N_PATHS, 'rel_std_error'].iloc[0]:.3%}",
        ))

    return frame


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def run_all(netting_set, seed: int = config.SEED) -> tuple[ValidationReport, pd.DataFrame]:
    """Run the full suite and return the report plus the convergence table."""
    report = ValidationReport()
    n_paths = config.VALIDATION_PATHS

    test_curve_reproduction(report, n_paths, seed)
    test_fx_martingale(report, n_paths, seed)
    test_quanto(report, n_paths, seed)
    test_zero_vol_degeneracy(report, min(n_paths, 5_000), seed)
    convergence = test_convergence(report, netting_set, seed)

    return report, convergence


def format_report(report: ValidationReport) -> str:
    """Render the validation report as an aligned console table."""
    lines = []
    current = None
    for check in report.checks:
        if check.test != current:
            current = check.test
            lines.append("")
            lines.append(f"  {current}")
        status = "PASS" if check.passed else "FAIL"
        lines.append(
            f"    [{status}] {check.case:<46s} "
            f"estimate {check.estimate:>14.8g}  target {check.target:>14.8g}  "
            f"|error| {abs(check.error):>11.4g}  tol {check.tolerance:>11.4g}"
            + (f"   ({check.note})" if check.note else "")
        )
    return "\n".join(lines)
