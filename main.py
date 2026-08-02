"""
End-to-end driver for the CVA proxy study.

Run with:      python main.py

Pipeline
--------
    market model & paths          -> cva.market
    analytic FX valuation         -> cva.instruments
    scenario matrix               -> cva.scenarios
    netting & expected exposure   -> cva.exposure
    survival / PD curves          -> cva.credit
    true CVA (full simulation)    -> cva.cva_engine
    standalone / incremental proxy-> cva.proxies
    interpolation (lin/chebyshev) -> cva.interpolation
    error metrics                 -> cva.metrics
    figures                       -> cva.plots

Everything is driven by `cva/config.py`; this file only orchestrates and
reports.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from cva import config, plots, proxies
from cva.cva_engine import cva_from_values
from cva.exposure import expected_exposure, potential_future_exposure
from cva.market import MarketModel
from cva.metrics import speedup, summarise_errors
from cva.portfolio import (
    build_all_netting_sets,
    build_incoming_trades,
    portfolio_to_frame,
)
from cva.scenarios import netting_set_scenario_values

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

REL_ERROR_FLOOR = 1_000.0   # USD: below this the relative error is meaningless


# ----------------------------------------------------------------------------
# Stage 1 - market, paths and the five netting sets
# ----------------------------------------------------------------------------

def build_world():
    """Simulate the risk factor once and pre-compute every netting set.

    This represents the bank's OVERNIGHT BATCH: the scenario matrices and
    portfolio CVAs produced here are stored, and the incremental proxy is
    allowed to reuse them for free during the day.
    """
    market = MarketModel()
    times = config.TIME_GRID

    t0 = time.perf_counter()
    spots = market.simulate_spot(times, config.N_PATHS, config.SEED)
    sim_seconds = time.perf_counter() - t0

    netting_sets = build_all_netting_sets(market)

    batch = {}
    for ns in netting_sets:
        values = netting_set_scenario_values(ns.trades, times, spots, market)
        batch[ns.name] = {
            "netting_set": ns,
            "values": values,                                    # stored scenario matrix
            "cva": cva_from_values(values, times, ns.hazard, market),  # stored CVA
            "ee": expected_exposure(values),
            "pfe": potential_future_exposure(values),
            "mtm_today": float(values[0, 0]),
            # The test trades are built once here (sizing the risk-reducing one
            # needs the stored values) and reused by every stage below.
            "incoming": build_incoming_trades(ns, market, times, values),
        }

    spot_lo, spot_hi = proxies.spot_range(spots)
    return market, times, spots, netting_sets, batch, (spot_lo, spot_hi), sim_seconds


# ----------------------------------------------------------------------------
# Stage 2 - the main experiment
# ----------------------------------------------------------------------------

def run_main_study(market, times, spots, netting_sets, batch, spot_box) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Price every test trade against every netting set with both proxies.

    Returns
    -------
    results  : one row per (netting set, incoming trade, interpolation method).
    scatter  : one row per (netting set, incoming trade, pricing approach), used
               for the accuracy-vs-computation-time figure.
    """
    spot_lo, spot_hi = spot_box
    rows, scatter_rows = [], []

    for ns in netting_sets:
        stored = batch[ns.name]
        stored_values, stored_cva = stored["values"], stored["cva"]

        for item in stored["incoming"]:
            # --- ground truth + timing reference: full re-simulation ---------
            truth = proxies.full_resimulation_incremental(
                ns.trades, item.trade, stored_cva, times, spots, market, ns.hazard
            )
            true_inc = truth.cva

            # --- diagnostic: exact repricing of the new trade against the
            #     stored values.  Isolates netting (exact) from interpolation.
            exact_inc = proxies.incremental_exact_repricing(
                item.trade, stored_values, stored_cva, times, spots, market, ns.hazard
            )

            scatter_rows.append(dict(
                netting_set=ns.name, trade_key=item.key, approach="full re-simulation",
                seconds=truth.seconds, abs_rel_error=0.0,
            ))

            for method in config.INTERP_METHODS:
                standalone = proxies.standalone_proxy(
                    item.trade, times, spots, market, ns.hazard, spot_lo, spot_hi,
                    method=method,
                )
                incremental = proxies.incremental_proxy(
                    item.trade, stored_values, stored_cva, times, spots, market,
                    ns.hazard, spot_lo, spot_hi, method=method,
                )

                sa_err = summarise_errors(standalone.cva, true_inc)
                inc_err = summarise_errors(incremental.cva, true_inc)

                rows.append(dict(
                    netting_set=ns.name,
                    n_trades=len(ns),
                    hazard=ns.hazard,
                    portfolio_cva=stored_cva,
                    portfolio_mtm_today=stored["mtm_today"],
                    trade_key=item.key,
                    trade_description=item.description,
                    interp_method=method,
                    # --- the three CVA numbers -------------------------------
                    true_incremental_cva=true_inc,
                    standalone_proxy_cva=standalone.cva,
                    incremental_proxy_cva=incremental.cva,
                    incremental_exact_cva=exact_inc.cva,
                    # --- errors ---------------------------------------------
                    standalone_signed_error=sa_err["signed_error"],
                    standalone_abs_error=sa_err["abs_error"],
                    standalone_rel_error=sa_err["rel_error"],
                    standalone_direction=sa_err["direction"],
                    incremental_signed_error=inc_err["signed_error"],
                    incremental_abs_error=inc_err["abs_error"],
                    incremental_rel_error=inc_err["rel_error"],
                    incremental_direction=inc_err["direction"],
                    # --- how much of the incremental error is interpolation --
                    interpolation_only_error=incremental.cva - exact_inc.cva,
                    netting_exactness_check=exact_inc.cva - true_inc,
                    # --- cost ------------------------------------------------
                    full_resim_seconds=truth.seconds,
                    standalone_seconds=standalone.seconds,
                    incremental_seconds=incremental.seconds,
                    standalone_speedup=speedup(truth.seconds, standalone.seconds),
                    incremental_speedup=speedup(truth.seconds, incremental.seconds),
                    proxy_exact_evaluations=incremental.n_exact_evaluations,
                    full_resim_evaluations=truth.n_exact_evaluations,
                ))

                for label, res, err in (
                    ("standalone proxy", standalone, sa_err),
                    (f"incremental proxy ({method})", incremental, inc_err),
                ):
                    if label == "standalone proxy" and method != "linear":
                        continue     # the standalone charge is plotted once only
                    rel = abs(err["rel_error"]) if abs(true_inc) >= REL_ERROR_FLOOR else np.nan
                    scatter_rows.append(dict(
                        netting_set=ns.name, trade_key=item.key, approach=label,
                        seconds=res.seconds, abs_rel_error=rel,
                    ))

    return pd.DataFrame(rows), pd.DataFrame(scatter_rows).dropna(subset=["abs_rel_error"])


# ----------------------------------------------------------------------------
# Stage 3 - sensitivity sweeps
# ----------------------------------------------------------------------------

def run_node_sweep(market, times, spots, netting_sets, batch, spot_box) -> pd.DataFrame:
    """Sensitivity (c): how the incremental-proxy error depends on the
    interpolation method and on the number of nodes.

    Averaged over every (netting set, trade) combination whose true incremental
    CVA is large enough for a relative error to be meaningful.
    """
    spot_lo, spot_hi = spot_box
    rows = []

    # Cache the truth once - it does not depend on the interpolation settings.
    truths = {}
    for ns in netting_sets:
        stored = batch[ns.name]
        for item in stored["incoming"]:
            truth = proxies.incremental_exact_repricing(
                item.trade, stored["values"], stored["cva"], times, spots, market, ns.hazard
            )
            truths[(ns.name, item.key)] = (item, truth.cva)

    for method in config.INTERP_METHODS:
        for n_nodes in config.NODE_SWEEP:
            errors, evaluations, seconds = [], [], []
            for ns in netting_sets:
                stored = batch[ns.name]
                for entry in stored["incoming"]:
                    item, true_inc = truths[(ns.name, entry.key)]
                    if abs(true_inc) < REL_ERROR_FLOOR:
                        continue
                    res = proxies.incremental_proxy(
                        item.trade, stored["values"], stored["cva"], times, spots, market,
                        ns.hazard, spot_lo, spot_hi, method=method,
                        n_time_nodes=n_nodes, n_spot_nodes=config.N_SPOT_NODES,
                    )
                    errors.append(abs(res.cva - true_inc) / abs(true_inc))
                    evaluations.append(res.n_exact_evaluations)
                    seconds.append(res.seconds)

            rows.append(dict(
                sweep="node_count", interp_method=method, n_time_nodes=n_nodes,
                n_spot_nodes=config.N_SPOT_NODES, split_legs=True,
                mean_abs_rel_error=float(np.mean(errors)),
                max_abs_rel_error=float(np.max(errors)),
                mean_exact_evaluations=float(np.mean(evaluations)),
                mean_seconds=float(np.mean(seconds)),
            ))

    return pd.DataFrame(rows)


def run_domain_sweep(market, times, spots, netting_sets, batch, spot_box) -> pd.DataFrame:
    """Extra sensitivity: what happens if the interpolant ignores the trade's
    maturity schedule.

    With `split_legs=False` the whole FX swap is interpolated on ONE time domain,
    so the near leg's settlement jump falls inside the domain.  This is the
    classic way to get a bad tensor proxy, and it hurts Chebyshev far more than
    the piecewise-linear interpolant because a global polynomial cannot
    represent a discontinuity.
    """
    spot_lo, spot_hi = spot_box
    rows = []

    for method in config.INTERP_METHODS:
        for split_legs in (True, False):
            errors = []
            for ns in netting_sets:
                stored = batch[ns.name]
                for item in stored["incoming"]:
                    if item.key != "new_swap":       # only the swap has two legs
                        continue
                    true_inc = proxies.incremental_exact_repricing(
                        item.trade, stored["values"], stored["cva"], times, spots,
                        market, ns.hazard,
                    ).cva
                    res = proxies.incremental_proxy(
                        item.trade, stored["values"], stored["cva"], times, spots, market,
                        ns.hazard, spot_lo, spot_hi, method=method, split_legs=split_legs,
                    )
                    errors.append(abs(res.cva - true_inc) / abs(true_inc))

            rows.append(dict(
                sweep="swap_domain_handling", interp_method=method,
                n_time_nodes=config.N_TIME_NODES, n_spot_nodes=config.N_SPOT_NODES,
                split_legs=split_legs,
                mean_abs_rel_error=float(np.mean(errors)),
                max_abs_rel_error=float(np.max(errors)),
                mean_exact_evaluations=np.nan, mean_seconds=np.nan,
            ))

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Stage 4 - reporting
# ----------------------------------------------------------------------------

def print_header(title: str) -> None:
    print()
    print("=" * 118)
    print(title)
    print("=" * 118)


def report(results: pd.DataFrame, batch: dict, netting_sets, sim_seconds: float) -> None:
    """Print the numbers that go into the thesis text."""
    print_header("CONFIGURATION")
    print(f"  seed                       : {config.SEED}")
    print(f"  Monte Carlo paths          : {config.N_PATHS:,}")
    print(f"  exposure dates             : {config.TIME_GRID.size} (0 to {config.HORIZON:g}y)")
    print(f"  spot / vol / r_EUR / r_USD : {config.SPOT_0:.2f} / {config.FX_VOL:.0%} / "
          f"{config.RATE_EUR:.2%} / {config.RATE_USD:.2%}")
    print(f"  recovery rate              : {config.RECOVERY_RATE:.0%}")
    print(f"  interpolation grid         : {config.N_TIME_NODES} time x {config.N_SPOT_NODES} spot nodes")
    print(f"  path simulation took       : {sim_seconds*1000:.0f} ms")

    print_header("1. TRUE CVA PER NETTING SET (full simulation)")
    summary = pd.DataFrame([
        dict(
            netting_set=ns.name,
            trades=len(ns),
            legs=sum(2 if type(t).__name__ == "FXSwap" else 1 for t in ns.trades),
            max_tenor_y=round(ns.max_tenor, 2),
            hazard=ns.hazard,
            net_notional_eur_m=round(ns.net_signed_notional / 1e6, 1),
            mtm_today_usd_m=round(batch[ns.name]["mtm_today"] / 1e6, 2),
            peak_EE_usd_m=round(batch[ns.name]["ee"].max() / 1e6, 2),
            true_CVA_usd=round(batch[ns.name]["cva"], 0),
        )
        for ns in netting_sets
    ])
    print(summary.to_string(index=False))

    print_header("2. THE THREE CVA NUMBERS PER INCOMING TRADE (linear interpolation)")
    lin = results[results["interp_method"] == "linear"]
    view = lin[[
        "netting_set", "trade_key", "true_incremental_cva",
        "standalone_proxy_cva", "incremental_proxy_cva", "portfolio_cva",
    ]].copy()
    view.columns = ["netting_set", "trade", "TRUE incr.", "STANDALONE", "INCREMENTAL", "portfolio CVA"]
    print(view.to_string(index=False, float_format=lambda v: f"{v:>14,.0f}"))

    print_header("3. SIGNED ERROR vs TRUE INCREMENTAL CVA  (positive = OVERCHARGE)")
    err = lin[[
        "netting_set", "trade_key",
        "standalone_signed_error", "standalone_rel_error", "standalone_direction",
        "incremental_signed_error", "incremental_rel_error", "incremental_direction",
    ]].copy()
    err.columns = ["netting_set", "trade", "SA signed", "SA rel", "SA dir",
                   "INC signed", "INC rel", "INC dir"]
    print(err.to_string(
        index=False,
        formatters={
            "SA signed": lambda v: f"{v:>13,.0f}",
            "INC signed": lambda v: f"{v:>12,.2f}",
            "SA rel": lambda v: f"{v:>8.2%}" if np.isfinite(v) else "     n/a",
            "INC rel": lambda v: f"{v:>10.5%}" if np.isfinite(v) else "       n/a",
        },
    ))

    print_header("4. DIAGNOSTIC: where does each proxy's error come from?")
    diag = lin.groupby("netting_set").agg(
        max_netting_check=("netting_exactness_check", lambda s: s.abs().max()),
        max_interp_error=("interpolation_only_error", lambda s: s.abs().max()),
        max_standalone_error=("standalone_abs_error", "max"),
    )
    print(diag.to_string(float_format=lambda v: f"{v:>16,.6f}"))
    print("\n  netting_exactness_check = (incremental with EXACT repricing) - (true incremental).")
    print("  It is ~0 by construction: the incremental proxy nets EXACTLY, so its only")
    print("  error source is the interpolation column next to it.")

    print_header("5. COMPUTATION TIME PER PRICING REQUEST")
    timing = results.groupby("interp_method").agg(
        full_resim_ms=("full_resim_seconds", lambda s: 1000 * s.mean()),
        standalone_ms=("standalone_seconds", lambda s: 1000 * s.mean()),
        incremental_ms=("incremental_seconds", lambda s: 1000 * s.mean()),
        standalone_speedup=("standalone_speedup", "mean"),
        incremental_speedup=("incremental_speedup", "mean"),
    )
    print(timing.to_string(float_format=lambda v: f"{v:>12,.2f}"))

    by_size = results[results["interp_method"] == "linear"].groupby(["netting_set", "n_trades"]).agg(
        full_resim_ms=("full_resim_seconds", lambda s: 1000 * s.mean()),
        incremental_ms=("incremental_seconds", lambda s: 1000 * s.mean()),
        speedup=("incremental_speedup", "mean"),
    )
    print("\n  Cost of a full re-simulation grows with the netting set; the proxies do not:")
    print(by_size.to_string(float_format=lambda v: f"{v:>12,.2f}"))


def report_sensitivity(node_sweep: pd.DataFrame, domain_sweep: pd.DataFrame,
                       results: pd.DataFrame) -> None:
    print_header("6. SENSITIVITY (a) NETTING-SET SIZE and (b) TRADE TYPE")
    lin = results[results["interp_method"] == "linear"]
    pivot_sa = lin.pivot_table(index="netting_set", columns="trade_key",
                               values="standalone_signed_error")
    print("  Standalone proxy - signed error in USD (positive = overcharge):")
    print(pivot_sa.to_string(float_format=lambda v: f"{v:>14,.0f}"))
    pivot_inc = lin.pivot_table(index="netting_set", columns="trade_key",
                                values="incremental_signed_error")
    print("\n  Incremental proxy - signed error in USD:")
    print(pivot_inc.to_string(float_format=lambda v: f"{v:>14,.2f}"))

    print_header("7. SENSITIVITY (c) INTERPOLATION METHOD")
    print("  Mean |relative error| of the incremental proxy over all 20 trade/netting-set pairs:")
    table = node_sweep.pivot_table(index="n_time_nodes", columns="interp_method",
                                   values="mean_abs_rel_error")
    print(table.to_string(float_format=lambda v: f"{v:>14.3e}"))
    print("\n  Naive vs leg-aware interpolation domain for the FX swap:")
    print(domain_sweep[["interp_method", "split_legs", "mean_abs_rel_error",
                        "max_abs_rel_error"]].to_string(
        index=False, float_format=lambda v: f"{v:>14.3e}"))


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main() -> None:
    t_start = time.perf_counter()

    market, times, spots, netting_sets, batch, spot_box, sim_seconds = build_world()
    results, scatter = run_main_study(market, times, spots, netting_sets, batch, spot_box)
    node_sweep = run_node_sweep(market, times, spots, netting_sets, batch, spot_box)
    domain_sweep = run_domain_sweep(market, times, spots, netting_sets, batch, spot_box)

    report(results, batch, netting_sets, sim_seconds)
    report_sensitivity(node_sweep, domain_sweep, results)

    # ---- exports -----------------------------------------------------------
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results.to_csv(os.path.join(config.RESULTS_DIR, config.MAIN_RESULTS_CSV), index=False)
    sensitivity = pd.concat([node_sweep, domain_sweep], ignore_index=True)
    sensitivity.to_csv(os.path.join(config.RESULTS_DIR, config.SENSITIVITY_CSV), index=False)
    portfolio_to_frame(netting_sets).to_csv(
        os.path.join(config.RESULTS_DIR, config.PORTFOLIO_CSV), index=False
    )

    # ---- figures -----------------------------------------------------------
    ee_by_set = {ns.name: batch[ns.name]["ee"] for ns in netting_sets}
    pfe_by_set = {ns.name: batch[ns.name]["pfe"] for ns in netting_sets}
    focus = netting_sets[-1].name

    figures = [
        plots.plot_exposure_profiles(times, ee_by_set, pfe_by_set, focus, config.RESULTS_DIR),
        plots.plot_cva_comparison(results, config.RESULTS_DIR),
        plots.plot_signed_errors(results, config.RESULTS_DIR),
        plots.plot_accuracy_vs_time(scatter, config.RESULTS_DIR),
        plots.plot_interpolation_convergence(node_sweep, domain_sweep, config.RESULTS_DIR),
        plots.plot_error_vs_netting_set_size(results, config.RESULTS_DIR),
    ]

    print_header("OUTPUT FILES")
    for path in [
        os.path.join(config.RESULTS_DIR, config.MAIN_RESULTS_CSV),
        os.path.join(config.RESULTS_DIR, config.SENSITIVITY_CSV),
        os.path.join(config.RESULTS_DIR, config.PORTFOLIO_CSV),
        *figures,
    ]:
        print(f"  {path}")
    print(f"\n  total runtime: {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    main()
