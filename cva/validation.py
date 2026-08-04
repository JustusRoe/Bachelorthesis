"""
Model validation and the FX-vs-rates risk decomposition.

Five checks run before any CVA is reported; `main.py` refuses to continue if one
fails.  Tests 1-2 are statistical (a Monte Carlo estimate against a closed-form
target, passing within a few standard errors); tests 3-4 are deterministic
comparisons on COMMON RANDOM NUMBERS, which is what makes them sharp enough to
resolve effects smaller than the raw simulation noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .cva import cva_from_values, cva_standard_error
from .model import ThreeFactorMarket, ZeroCurve
from .products import Trade, netting_set_values


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


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, **kwargs):
        self.checks.append(Check(**kwargs))

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(c) for c in self.checks])


def _within_sigmas(estimate, target, stderr):
    """Pass if the estimate is inside the configured number of standard errors."""
    tol = config.VALIDATION_TOLERANCE_SIGMAS * stderr
    return abs(estimate - target) <= tol, tol


# ----------------------------------------------------------------------------
# Test 1 - curve reproduction
# ----------------------------------------------------------------------------

def test_curve_reproduction(report: ValidationReport, n_paths: int, seed: int) -> None:
    """Does the simulation reprice both initial zero curves?

    Domestic, with the domestic money-market account as numeraire:
        E[D_d(0,T)] = P_d(0,T)

    Foreign: under the domestic measure E[exp(-int r_f)] is NOT P_f(0,T) - that
    identity holds under the FOREIGN measure.  The correct domestic statement is
    that one unit of foreign currency delivered at T is worth S(0) P_f(0,T):
        E[D_d(0,T) S(T)] = S(0) P_f(0,T)
    which is simultaneously the covered-interest-parity check.  A third variant
    reaches past the simulated date via a forward-starting foreign bond:
        E[D_d(0,T) S(T) P_f(T,T')] = S(0) P_f(0,T')
    """
    market = ThreeFactorMarket.build("stochastic")
    tenors = np.array(config.VALIDATION_TENORS, dtype=float)
    times = np.unique(np.concatenate([[0.0], config.TIME_GRID, tenors]))
    scen = market.simulate(times, n_paths, seed)
    index = {round(float(t), 10): i for i, t in enumerate(times)}

    def record(case, sample, target):
        estimate = float(sample.mean())
        stderr = float(sample.std(ddof=1) / np.sqrt(n_paths))
        passed, tol = _within_sigmas(estimate, target, stderr)
        report.add(test="1. curve reproduction", case=case, estimate=estimate,
                   target=float(target), error=estimate - float(target), tolerance=tol,
                   passed=passed, units="discount factor", note=f"MC std err {stderr:.2e}")

    for tenor in tenors:
        col = index[round(float(tenor), 10)]
        fx_leg = scen.df_domestic[:, col] * scen.spots[:, col] / market.spot_0
        far = float(tenor) + 2.0
        record(f"USD P(0,{tenor:g}y)", scen.df_domestic[:, col], market.p_domestic_0(tenor))
        record(f"EUR P(0,{tenor:g}y)", fx_leg, market.p_foreign_0(tenor))
        record(f"EUR P(0,{far:g}y) via P({tenor:g}y,{far:g}y)",
               fx_leg * market.bond_foreign(tenor, far, scen.r_f[:, col]),
               market.p_foreign_0(far))


# ----------------------------------------------------------------------------
# Test 2 - FX martingale
# ----------------------------------------------------------------------------

def fair_forward_error(market, scen, tenor: float, times):
    """PV of a forward struck at today's fair rate; must be zero.

    Returns (value, standard error, value relative to the forward level).
    """
    col = int(np.argmin(np.abs(times - tenor)))
    strike = float(market.forward_fx(tenor))
    payoff = scen.df_domestic[:, col] * (scen.spots[:, col] - strike)
    value = float(payoff.mean())
    stderr = float(payoff.std(ddof=1) / np.sqrt(payoff.size))
    return value, stderr, value / (strike * float(market.p_domestic_0(tenor)))


def test_fx_martingale(report: ValidationReport, n_paths: int, seed: int) -> None:
    """A fair-struck forward must have zero value at t = 0."""
    market = ThreeFactorMarket.build("stochastic")
    tenors = np.array(config.VALIDATION_TENORS, dtype=float)
    times = np.unique(np.concatenate([[0.0], config.TIME_GRID, tenors]))
    scen = market.simulate(times, n_paths, seed)

    for tenor in tenors:
        value, stderr, relative = fair_forward_error(market, scen, float(tenor), times)
        passed, tol = _within_sigmas(value, 0.0, stderr)
        report.add(test="2. FX martingale", case=f"fair {tenor:g}y forward, PV per EUR",
                   estimate=value, target=0.0, error=value, tolerance=tol, passed=passed,
                   units="USD per EUR of notional",
                   note=f"{1e4*relative:+.2f} bp of forward; MC std err {stderr:.2e}")


# ----------------------------------------------------------------------------
# Test 3 - quanto adjustment
# ----------------------------------------------------------------------------

def analytic_quanto_bias(market, tenor: float) -> float:
    """Closed-form bias on the T-forward if the quanto adjustment were omitted.

    Without it the foreign short rate keeps its FOREIGN-measure dynamics under
    the DOMESTIC measure; changing measure with the FX density then shifts it by
    p(t) = (rho_fS sigma_f sigma_S / a_f)(1 - exp(-a_f t)) instead of leaving it
    alone, so the fair forward is overstated by

        exp( -(rho sigma_f sigma_S / a_f) [T - (1-exp(-a_f T))/a_f] ) - 1
    """
    a = market.hw_foreign.a
    coefficient = market.rho_fs * market.hw_foreign.sigma * market.sigma_fx / a
    return float(np.exp(-coefficient * (tenor - (1.0 - np.exp(-a * tenor)) / a)) - 1.0)


def test_quanto(report: ValidationReport, n_paths: int, seed: int) -> None:
    """Is the change-of-measure drift on the foreign short rate doing its job?

    For each rho_fS the fair forward is priced twice on COMMON RANDOM NUMBERS,
    with and without the adjustment.  That matters: the bias is comparable to the
    raw Monte Carlo noise, so an independent-runs comparison would be
    inconclusive, whereas the paired difference is essentially noiseless.

    (a) WITH the adjustment the forward must be fair at every rho_fS.
    (b) WITHOUT it the bias must match the closed form above.
    """
    tenor = 5.0
    times = np.unique(np.concatenate([[0.0], config.TIME_GRID, [tenor]]))

    for rho_fs in config.QUANTO_TEST_CORRELATIONS:
        results = {}
        for applied in (True, False):
            market = ThreeFactorMarket.build("stochastic", rho_fs=rho_fs, apply_quanto=applied)
            scen = market.simulate(times, n_paths, seed)      # same seed -> same draws
            results[applied] = (*fair_forward_error(market, scen, tenor, times), market)

        value, stderr, relative, market = results[True]
        scale = float(market.forward_fx(tenor) * market.p_domestic_0(tenor))
        passed, tol = _within_sigmas(value, 0.0, stderr)
        report.add(test="3. quanto adjustment", case=f"rho_fS={rho_fs:+.2f}, adjustment ON",
                   estimate=1e4 * relative, target=0.0, error=1e4 * relative,
                   tolerance=1e4 * tol / scale, passed=passed,
                   units="bp of the fair forward", note="fair forward must price to zero")

        predicted = analytic_quanto_bias(market, tenor)
        observed = results[False][2] - results[True][2]
        # 5% of the predicted bias, or 0.5bp when the prediction is itself ~0
        # (the rho_fS = 0 case, where there is nothing to correct).
        tolerance = max(0.05 * abs(predicted), 0.5e-4)
        report.add(test="3. quanto adjustment", case=f"rho_fS={rho_fs:+.2f}, adjustment OFF",
                   estimate=1e4 * observed, target=1e4 * predicted,
                   error=1e4 * (observed - predicted), tolerance=1e4 * tolerance,
                   passed=abs(observed - predicted) <= tolerance,
                   units="bp of the fair forward", note="paired difference vs closed form")


# ----------------------------------------------------------------------------
# Test 4 - zero-volatility degeneracy
# ----------------------------------------------------------------------------

def test_zero_vol_degeneracy(report: ValidationReport, n_paths: int, seed: int) -> None:
    """With sigma_d = sigma_f = 0 the model must collapse to the base model.

    Checked to machine precision rather than within Monte Carlo error, because
    both runs use identical random draws.  Two statements: deterministic mode
    equals the stochastic model with zero rate vols, and with FLAT curves the
    result equals the closed-form single-factor base model.
    """
    times = config.TIME_GRID

    def record(case, gap, tol, units, note):
        report.add(test="4. zero-vol degeneracy", case=case, estimate=gap, target=0.0,
                   error=gap, tolerance=tol, passed=gap <= tol, units=units, note=note)

    det = ThreeFactorMarket.build("deterministic").simulate(times, n_paths, seed)
    zero = ThreeFactorMarket.build("stochastic", sigma_d=0.0, sigma_f=0.0).simulate(
        times, n_paths, seed)
    record("deterministic mode == sigma_rate=0",
           float(np.max(np.abs(det.spots - zero.spots) / det.spots)), 1e-14,
           "max relative spot difference", "identical random draws")

    r_d, r_f = 0.04, 0.02
    flat = ThreeFactorMarket.build("deterministic",
                                   curve_domestic=ZeroCurve(b0=r_d, name="USD flat"),
                                   curve_foreign=ZeroCurve(b0=r_f, name="EUR flat"))
    scen = flat.simulate(times, n_paths, seed)

    # Closed-form single-factor base model on the same Brownian increments.
    z = np.random.default_rng(seed).standard_normal((n_paths, times.size))
    brownian = np.cumsum(z * np.sqrt(np.diff(times, prepend=0.0)), axis=1)
    reference = config.SPOT_0 * np.exp(
        (r_d - r_f - 0.5 * config.VOL_FX**2) * times[None, :] + config.VOL_FX * brownian)

    record("flat curves == single-factor GBM spot",
           float(np.max(np.abs(scen.spots - reference) / reference)), 1e-12,
           "max relative spot difference", "closed-form base model, same draws")
    record("flat curves == exp(-r_d t) discounting",
           float(np.max(np.abs(scen.df_domestic - np.exp(-r_d * times)[None, :]))), 1e-14,
           "max absolute DF difference", "deterministic discounting recovered")
    record("flat curves == exp(-r_f (T-t)) bond",
           float(np.max(np.abs(flat.bond_foreign(times[None, :], 5.0, scen.r_f)
                               - np.exp(-r_f * np.maximum(5.0 - times, 0.0))[None, :]))),
           1e-14, "max absolute bond difference", "Hull-White A(t,T) collapses correctly")


# ----------------------------------------------------------------------------
# Test 5 - Monte Carlo convergence
# ----------------------------------------------------------------------------

def test_convergence(report: ValidationReport, netting_set, seed: int) -> pd.DataFrame:
    """CVA against path count, with 95% intervals.

    Two criteria: the standard error must fall like 1/sqrt(N), and the production
    estimate must lie inside the finest run's confidence interval.
    """
    market = ThreeFactorMarket.build("stochastic")
    rows = []
    for n_paths in config.CONVERGENCE_PATHS:
        scen = market.simulate(config.TIME_GRID, n_paths, seed)
        values = netting_set_values(netting_set.trades, market, scen)
        cva = cva_from_values(values, scen, market, netting_set.hazard)
        stderr = cva_standard_error(values, scen, netting_set.hazard)
        rows.append(dict(n_paths=n_paths, cva=cva, std_error=stderr,
                         ci_low=cva - 1.96 * stderr, ci_high=cva + 1.96 * stderr,
                         rel_std_error=stderr / abs(cva) if cva else np.nan))
    frame = pd.DataFrame(rows)

    slope = float(np.polyfit(np.log(frame["n_paths"]), np.log(frame["std_error"]), 1)[0])
    report.add(test="5. convergence", case="std error scaling exponent", estimate=slope,
               target=-0.5, error=slope + 0.5, tolerance=0.10,
               passed=abs(slope + 0.5) <= 0.10, units="d log(se) / d log(N)",
               note="Monte Carlo theory predicts -0.5")

    production = frame.loc[frame["n_paths"] == config.N_PATHS]
    finest = frame.iloc[-1]
    if not production.empty:
        estimate = float(production["cva"].iloc[0])
        report.add(test="5. convergence",
                   case=f"{config.N_PATHS:,}-path CVA inside {int(finest['n_paths']):,}-path 95% CI",
                   estimate=estimate, target=float(finest["cva"]),
                   error=estimate - float(finest["cva"]),
                   tolerance=1.96 * float(finest["std_error"]),
                   passed=bool(finest["ci_low"] <= estimate <= finest["ci_high"]), units="USD",
                   note=f"production relative std error "
                        f"{float(production['rel_std_error'].iloc[0]):.3%}")
    return frame


# ----------------------------------------------------------------------------
# Driver and formatting
# ----------------------------------------------------------------------------

def run_all(netting_set, seed: int = config.SEED):
    """Run the full suite; returns (report, convergence table)."""
    report = ValidationReport()
    n = config.VALIDATION_PATHS
    test_curve_reproduction(report, n, seed)
    test_fx_martingale(report, n, seed)
    test_quanto(report, n, seed)
    test_zero_vol_degeneracy(report, min(n, 5_000), seed)
    return report, test_convergence(report, netting_set, seed)


def format_report(report: ValidationReport) -> str:
    """Render the report as an aligned console table."""
    lines, current = [], None
    for c in report.checks:
        if c.test != current:
            current = c.test
            lines += ["", f"  {current}"]
        lines.append(
            f"    [{'PASS' if c.passed else 'FAIL'}] {c.case:<46s} "
            f"estimate {c.estimate:>14.8g}  target {c.target:>14.8g}  "
            f"|error| {abs(c.error):>11.4g}  tol {c.tolerance:>11.4g}"
            + (f"   ({c.note})" if c.note else ""))
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# FX vs rates risk decomposition
# ----------------------------------------------------------------------------

def variance_components(trade: Trade, market, scen) -> dict:
    """Blocks of the trade's instantaneous variance, averaged over paths per date.

    With factors (S, x_d, x_f) driven by vols v = (sigma_S S, sigma_d, sigma_f),
    Var(dV)/dt = sum_ij Delta_i Delta_j v_i v_j rho_ij, which splits into

        FX    = (Delta_S sigma_S S)^2
        RATES = Delta_d^2 sigma_d^2 + Delta_f^2 sigma_f^2 + 2 rho_df ...
        CROSS = 2 Delta_S sigma_S S (rho_dS Delta_d sigma_d + rho_fS Delta_f sigma_f)

    The cross term is signed, so shares can fall outside 0-100%.
    """
    fx_move = trade.delta_spot(market, scen.times, scen.r_f) * market.sigma_fx * scen.spots
    d_move = trade.delta_x_domestic(market, scen.times, scen.r_d) * market.hw_domestic.sigma
    f_move = trade.delta_x_foreign(market, scen.times, scen.spots,
                                   scen.r_f) * market.hw_foreign.sigma

    fx = fx_move**2
    rates = d_move**2 + f_move**2 + 2.0 * market.rho_df * d_move * f_move
    cross = 2.0 * fx_move * (market.rho_ds * d_move + market.rho_fs * f_move)
    return {"fx": fx.mean(axis=0), "rates": rates.mean(axis=0),
            "cross": cross.mean(axis=0), "total": (fx + rates + cross).mean(axis=0)}


def risk_split_table(trades: dict, market, scen,
                     report_dates=(0.25, 0.75, 1.5, 2.5),
                     near_leg_date: float = 1.0) -> pd.DataFrame:
    """Tabulate the FX / rates split for several trades at several dates.

    `report_dates` should straddle `near_leg_date` so the before/after contrast
    is visible; the table flags which side of settlement each row falls on.
    """
    times = np.asarray(scen.times, dtype=float)
    rows = []
    for name, trade in trades.items():
        components = variance_components(trade, market, scen)
        for target in report_dates:
            col = int(np.argmin(np.abs(times - target)))
            fx, rates = float(components["fx"][col]), float(components["rates"][col])
            cross, total = float(components["cross"][col]), float(components["total"][col])
            rows.append(dict(
                trade=name, date_y=round(float(times[col]), 3),
                phase="before near-leg settlement" if times[col] < near_leg_date
                      else "after near-leg settlement",
                total_vol_usd=np.sqrt(max(total, 0.0)),
                fx_share=fx / total if total else np.nan,
                rates_share=rates / total if total else np.nan,
                cross_share=cross / total if total else np.nan,
                fx_share_excl_cross=fx / (fx + rates) if (fx + rates) else np.nan))
    return pd.DataFrame(rows)
