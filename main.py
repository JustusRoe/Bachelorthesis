"""
End-to-end driver for the CVA proxy study under the three-factor market model.

Run with:      python main.py

Pipeline
--------
    initial zero curves           -> cva.curves
    Hull-White algebra            -> cva.hullwhite
    three-factor simulation       -> cva.market
    MODEL VALIDATION (runs first) -> cva.validation
    analytic FX valuation         -> cva.instruments
    scenario matrix               -> cva.scenarios
    netting & expected exposure   -> cva.exposure
    survival / PD curves          -> cva.credit
    true CVA (full simulation)    -> cva.cva_engine
    standalone / incremental proxy-> cva.proxies
    interpolation (1-D or 3-D)    -> cva.interpolation
    FX vs rates risk split        -> cva.risk_split
    error metrics                 -> cva.metrics
    figures                       -> cva.plots

The whole study is run TWICE, once with RATE_MODEL="deterministic" (the
single-factor fallback) and once with "stochastic", on identical trades and
identical seeds, so that every difference is attributable to rate volatility.

Everything is driven by `cva/config.py`; this file only orchestrates and reports.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from cva import config, plots, proxies, risk_split, validation
from cva.cva_engine import cva_from_values, cva_standard_error
from cva.exposure import (
    discounted_expected_exposure,
    expected_exposure,
    potential_future_exposure,
)
from cva.interpolation import state_box
from cva.market import ThreeFactorMarket
from cva.metrics import speedup, summarise_errors
from cva.portfolio import (
    build_all_netting_sets,
    build_incoming_trades,
    portfolio_to_frame,
)
from cva.proxies import PricingContext
from cva.scenarios import netting_set_scenario_values

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)

REL_ERROR_FLOOR = 1_000.0   # USD: below this the relative error is meaningless
LINE = "=" * 122


def print_header(title: str) -> None:
    print()
    print(LINE)
    print(title)
    print(LINE)


# ============================================================================
# Stage 1 - build one world (market + paths + pre-computed netting sets)
# ============================================================================

def build_world(rate_model: str, netting_sets=None, incoming=None):
    """Simulate the risk factors once and pre-compute every netting set.

    This represents the bank's OVERNIGHT BATCH: the scenario matrices and
    portfolio CVAs produced here are stored, and the incremental proxy is
    allowed to reuse them for free during the day.

    Parameters
    ----------
    netting_sets, incoming :
        Passed in on the second call so that BOTH rate models price exactly the
        same trades.  The netting sets themselves are strike-identical either
        way (strikes come from the initial curves, which the two models share),
        but the risk-reducing trade is SIZED from the simulated state, so it
        must be built once and reused - otherwise the two studies would be
        pricing different trades and the comparison would be meaningless.
    """
    market = ThreeFactorMarket.build(rate_model=rate_model)
    times = config.TIME_GRID

    t0 = time.perf_counter()
    scen = market.simulate(times, config.N_PATHS, config.SEED)
    sim_seconds = time.perf_counter() - t0

    if netting_sets is None:
        netting_sets = build_all_netting_sets(market)

    ctx = PricingContext(
        market=market, scen=scen, state_box=state_box(scen), discounting=config.DISCOUNTING,
    )

    batch = {}
    for ns in netting_sets:
        values = netting_set_scenario_values(ns.trades, market, scen)
        batch[ns.name] = {
            "netting_set": ns,
            "values": values,                                       # stored scenario matrix
            "cva": cva_from_values(values, scen, market, ns.hazard),  # stored CVA
            "cva_se": cva_standard_error(values, scen, ns.hazard),
            "ee": expected_exposure(values),
            "ee_discounted": discounted_expected_exposure(values, scen.df_domestic),
            "pfe": potential_future_exposure(values),
            "mtm_today": float(values[0, 0]),
            "incoming": (incoming[ns.name] if incoming is not None
                         else build_incoming_trades(ns, market, scen, values)),
        }

    return dict(
        rate_model=rate_model, market=market, scen=scen, ctx=ctx,
        netting_sets=netting_sets, batch=batch, sim_seconds=sim_seconds,
    )


# ============================================================================
# Stage 2 - the main experiment
# ============================================================================

def run_main_study(world) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Price every test trade against every netting set with both proxies.

    Returns
    -------
    results  : one row per (netting set, incoming trade, interpolation method).
    scatter  : one row per (netting set, incoming trade, pricing approach), for
               the accuracy-vs-computation-time figure.
    """
    ctx, batch = world["ctx"], world["batch"]
    rows, scatter_rows = [], []

    for ns in world["netting_sets"]:
        stored = batch[ns.name]
        stored_values, stored_cva = stored["values"], stored["cva"]

        for item in stored["incoming"]:
            # --- ground truth + timing reference: full re-simulation ---------
            truth = proxies.full_resimulation_incremental(
                ns.trades, item.trade, ctx, ns.hazard, stored_cva
            )
            true_inc = truth.cva

            # --- diagnostic: exact repricing of the new trade against the
            #     stored values.  Isolates netting (exact) from interpolation.
            exact_inc = proxies.incremental_exact_repricing(
                item.trade, ctx, ns.hazard, stored_values, stored_cva
            )

            scatter_rows.append(dict(
                netting_set=ns.name, trade_key=item.key, approach="full re-simulation",
                seconds=truth.seconds, abs_rel_error=0.0,
            ))

            for method in config.INTERP_METHODS:
                standalone = proxies.standalone_proxy(
                    item.trade, ctx, ns.hazard, method=method
                )
                incremental = proxies.incremental_proxy(
                    item.trade, ctx, ns.hazard, stored_values, stored_cva, method=method
                )

                sa_err = summarise_errors(standalone.cva, true_inc)
                inc_err = summarise_errors(incremental.cva, true_inc)

                rows.append(dict(
                    rate_model=world["rate_model"],
                    netting_set=ns.name,
                    n_trades=len(ns),
                    hazard_5y_average=ns.hazard.average_hazard,
                    portfolio_cva=stored_cva,
                    portfolio_cva_std_error=stored["cva_se"],
                    portfolio_mtm_today=stored["mtm_today"],
                    trade_key=item.key,
                    trade_description=item.description,
                    interp_method=method,
                    state_dimensions=ctx.n_state_dimensions,
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
                    # --- where the incremental error comes from --------------
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

    scatter = pd.DataFrame(scatter_rows).dropna(subset=["abs_rel_error"])
    return pd.DataFrame(rows), scatter


# ============================================================================
# Stage 3 - sensitivity sweeps
# ============================================================================

def run_node_sweep(world) -> pd.DataFrame:
    """How the incremental-proxy error depends on the method and the node count.

    Run under BOTH rate models, so the 1-D (deterministic, state = spot only)
    and 3-D (stochastic, state = spot + two rate factors) curves can be compared
    directly.  `mean_exact_evaluations` is what exposes the n^3 growth.
    """
    ctx, batch = world["ctx"], world["batch"]
    rows = []

    # Cache the truth once - it does not depend on the interpolation settings.
    truths = {}
    for ns in world["netting_sets"]:
        stored = batch[ns.name]
        for item in stored["incoming"]:
            truth = proxies.incremental_exact_repricing(
                item.trade, ctx, ns.hazard, stored["values"], stored["cva"], repeats=1
            )
            truths[(ns.name, item.key)] = (item, truth.cva)

    for method in config.INTERP_METHODS:
        for n_nodes in config.NODE_SWEEP:
            errors, evaluations = [], []
            for ns in world["netting_sets"]:
                stored = batch[ns.name]
                for entry in stored["incoming"]:
                    item, true_inc = truths[(ns.name, entry.key)]
                    if abs(true_inc) < REL_ERROR_FLOOR:
                        continue
                    res = proxies.incremental_proxy(
                        item.trade, ctx, ns.hazard, stored["values"], stored["cva"],
                        method=method, n_state_nodes=n_nodes,
                        n_time_nodes=config.SWEEP_TIME_NODES, repeats=1,
                    )
                    errors.append(abs(res.cva - true_inc) / abs(true_inc))
                    evaluations.append(res.n_exact_evaluations)

            rows.append(dict(
                sweep="node_count", rate_model=world["rate_model"],
                state_dims=ctx.n_state_dimensions, interp_method=method,
                n_time_nodes=config.SWEEP_TIME_NODES, n_state_nodes=n_nodes,
                grid_nodes_per_leg=config.SWEEP_TIME_NODES * n_nodes ** ctx.n_state_dimensions,
                split_legs=True,
                mean_abs_rel_error=float(np.mean(errors)),
                max_abs_rel_error=float(np.max(errors)),
                mean_exact_evaluations=float(np.mean(evaluations)),
            ))

    return pd.DataFrame(rows)


def run_domain_sweep(world) -> pd.DataFrame:
    """What happens if the interpolant ignores the trade's maturity schedule.

    With `split_legs=False` the whole FX swap is interpolated on ONE time domain,
    so the near leg's settlement jump falls inside the domain.  This is the
    classic way to get a bad tensor proxy, and it hurts Chebyshev more than the
    piecewise-linear interpolant because a global polynomial cannot represent a
    discontinuity.
    """
    ctx, batch = world["ctx"], world["batch"]
    rows = []

    for method in config.INTERP_METHODS:
        for split_legs in (True, False):
            errors = []
            for ns in world["netting_sets"]:
                stored = batch[ns.name]
                for item in stored["incoming"]:
                    if item.key != "new_swap":       # only the swap has two legs
                        continue
                    true_inc = proxies.incremental_exact_repricing(
                        item.trade, ctx, ns.hazard, stored["values"], stored["cva"], repeats=1
                    ).cva
                    res = proxies.incremental_proxy(
                        item.trade, ctx, ns.hazard, stored["values"], stored["cva"],
                        method=method, split_legs=split_legs, repeats=1,
                    )
                    errors.append(abs(res.cva - true_inc) / abs(true_inc))

            rows.append(dict(
                sweep="swap_domain_handling", rate_model=world["rate_model"],
                state_dims=ctx.n_state_dimensions, interp_method=method,
                n_time_nodes=config.N_TIME_NODES, n_state_nodes=config.N_STATE_NODES,
                grid_nodes_per_leg=np.nan, split_legs=split_legs,
                mean_abs_rel_error=float(np.mean(errors)),
                max_abs_rel_error=float(np.max(errors)),
                mean_exact_evaluations=np.nan,
            ))

    return pd.DataFrame(rows)


# ============================================================================
# Stage 4 - deterministic vs stochastic comparison
# ============================================================================

def summarise_rate_models(worlds: dict, results: dict, method: str = "linear") -> pd.DataFrame:
    """Average the headline numbers by trade type under each rate model."""
    frames = []
    for name, frame in results.items():
        sub = frame[frame["interp_method"] == method].copy()
        grouped = sub.groupby("trade_key").agg(
            **{
                f"true_incremental_cva_{name}": ("true_incremental_cva",
                                                 lambda s: float(np.mean(np.abs(s)))),
                f"standalone_rel_error_{name}": ("standalone_rel_error",
                                                 lambda s: float(np.nanmean(np.abs(s)))),
                f"incremental_rel_error_{name}": ("incremental_rel_error",
                                                  lambda s: float(np.nanmean(np.abs(s)))),
                f"standalone_signed_error_{name}": ("standalone_signed_error", "mean"),
                f"incremental_signed_error_{name}": ("incremental_signed_error", "mean"),
            }
        )
        frames.append(grouped)

    combined = pd.concat(frames, axis=1).reset_index()
    order = ["new_fwd", "new_swap", "offset_fwd", "small_fwd"]
    combined["__order"] = combined["trade_key"].map({k: i for i, k in enumerate(order)})
    return combined.sort_values("__order").drop(columns="__order").reset_index(drop=True)


def standalone_exposure_profiles(worlds: dict, trade_keys=("new_swap", "new_fwd")) -> dict:
    """Standalone (un-netted) expected exposure of selected incoming trades.

    Computed under each rate model on the SAME trade objects, which is what
    makes the deterministic-vs-stochastic exposure chart a like-for-like
    comparison.  Uses NS1's copy of the trades; the plain forward and the swap
    are identical across netting sets (only the offsetting trade is set-specific).
    """
    profiles: dict[str, dict[str, np.ndarray]] = {}
    reference = next(iter(worlds.values()))
    first_set = reference["netting_sets"][0].name

    labels = {}
    for item in reference["batch"][first_set]["incoming"]:
        if item.key in trade_keys:
            labels[item.key] = item

    for key in trade_keys:
        item = labels[key]
        profiles[item.description] = {}
        for name, world in worlds.items():
            values = item.trade.value(
                world["market"], world["scen"].times, world["scen"].spots,
                world["scen"].r_d, world["scen"].r_f,
            )
            profiles[item.description][name] = expected_exposure(values)
    return profiles


# ============================================================================
# Stage 5 - reporting
# ============================================================================

def report_setup(world, validation_report, convergence) -> None:
    print_header("MODEL SETUP")
    print(f"  seed                       : {config.SEED}")
    print(f"  Monte Carlo paths          : {config.N_PATHS:,}")
    print(f"  exposure dates             : {config.TIME_GRID.size} (0 to {config.HORIZON:g}y)")
    print(f"  CVA discounting            : {config.DISCOUNTING}")
    print(f"  recovery rate              : {config.RECOVERY_RATE:.0%}")
    print(f"  interpolation grid         : {config.N_TIME_NODES} time x "
          f"{config.N_STATE_NODES} nodes per state dimension")
    print(f"  path simulation took       : {world['sim_seconds']*1000:.0f} ms")
    print()
    print(world["market"].describe())

    print_header("MODEL VALIDATION  (must all pass before any CVA is reported)")
    print(validation.format_report(validation_report))
    print()
    print("  Monte Carlo convergence of the NS5 netting-set CVA:")
    print(convergence.to_string(
        index=False,
        formatters={
            "n_paths": lambda v: f"{int(v):>9,}",
            "cva": lambda v: f"{v:>14,.0f}",
            "std_error": lambda v: f"{v:>12,.0f}",
            "ci_low": lambda v: f"{v:>14,.0f}",
            "ci_high": lambda v: f"{v:>14,.0f}",
            "rel_std_error": lambda v: f"{v:>9.3%}",
        },
    ))
    verdict = "ALL CHECKS PASSED" if validation_report.all_passed else "FAILURES PRESENT"
    print(f"\n  ==> {verdict}")


def report_study(world, results: pd.DataFrame) -> None:
    """Print the numbers that go into the thesis text, for one rate model."""
    batch, netting_sets = world["batch"], world["netting_sets"]
    tag = world["rate_model"].upper()

    print_header(f"[{tag}]  1. TRUE CVA PER NETTING SET (full simulation)")
    summary = pd.DataFrame([
        dict(
            netting_set=ns.name,
            trades=len(ns),
            legs=sum(len(t.legs) for t in ns.trades),
            max_tenor_y=round(ns.max_tenor, 2),
            hazard=ns.hazard.describe(),
            net_notional_eur_m=round(ns.net_signed_notional / 1e6, 1),
            mtm_today_usd_m=round(batch[ns.name]["mtm_today"] / 1e6, 2),
            peak_EE_usd_m=round(batch[ns.name]["ee"].max() / 1e6, 2),
            true_CVA_usd=round(batch[ns.name]["cva"], 0),
            CVA_std_err=round(batch[ns.name]["cva_se"], 0),
        )
        for ns in netting_sets
    ])
    print(summary.to_string(index=False))

    lin = results[results["interp_method"] == "linear"]

    print_header(f"[{tag}]  2. THE THREE CVA NUMBERS PER INCOMING TRADE (linear interpolation)")
    view = lin[["netting_set", "trade_key", "true_incremental_cva",
                "standalone_proxy_cva", "incremental_proxy_cva", "portfolio_cva"]].copy()
    view.columns = ["netting_set", "trade", "TRUE incr.", "STANDALONE",
                    "INCREMENTAL", "portfolio CVA"]
    print(view.to_string(index=False, float_format=lambda v: f"{v:>14,.0f}"))

    print_header(f"[{tag}]  3. SIGNED ERROR vs TRUE INCREMENTAL CVA  (positive = OVERCHARGE)")
    err = lin[["netting_set", "trade_key",
               "standalone_signed_error", "standalone_rel_error", "standalone_direction",
               "incremental_signed_error", "incremental_rel_error", "incremental_direction"]].copy()
    err.columns = ["netting_set", "trade", "SA signed", "SA rel", "SA dir",
                   "INC signed", "INC rel", "INC dir"]
    print(err.to_string(
        index=False,
        formatters={
            "SA signed": lambda v: f"{v:>13,.0f}",
            "INC signed": lambda v: f"{v:>12,.2f}",
            "SA rel": lambda v: f"{v:>9.2%}" if np.isfinite(v) else "      n/a",
            "INC rel": lambda v: f"{v:>10.5%}" if np.isfinite(v) else "       n/a",
        },
    ))

    print_header(f"[{tag}]  4. DIAGNOSTIC: where does each proxy's error come from?")
    diag = lin.groupby("netting_set").agg(
        max_netting_check=("netting_exactness_check", lambda s: s.abs().max()),
        max_interp_error=("interpolation_only_error", lambda s: s.abs().max()),
        max_standalone_error=("standalone_abs_error", "max"),
    )
    print(diag.to_string(float_format=lambda v: f"{v:>16,.6f}"))
    print("\n  netting_exactness_check = (incremental with EXACT repricing) - (true incremental).")
    print("  It is ~0 by construction: the incremental proxy nets EXACTLY, so its only")
    print("  error source is the interpolation column next to it.")

    print_header(f"[{tag}]  5. COMPUTATION TIME PER PRICING REQUEST")
    timing = results.groupby("interp_method").agg(
        full_resim_ms=("full_resim_seconds", lambda s: 1000 * s.mean()),
        standalone_ms=("standalone_seconds", lambda s: 1000 * s.mean()),
        incremental_ms=("incremental_seconds", lambda s: 1000 * s.mean()),
        standalone_speedup=("standalone_speedup", "mean"),
        incremental_speedup=("incremental_speedup", "mean"),
        proxy_exact_prices=("proxy_exact_evaluations", "mean"),
        resim_exact_prices=("full_resim_evaluations", "mean"),
    )
    print(timing.to_string(float_format=lambda v: f"{v:>16,.2f}"))

    by_size = lin.groupby(["netting_set", "n_trades"]).agg(
        full_resim_ms=("full_resim_seconds", lambda s: 1000 * s.mean()),
        incremental_ms=("incremental_seconds", lambda s: 1000 * s.mean()),
        speedup=("incremental_speedup", "mean"),
    )
    print("\n  Cost of a full re-simulation grows with the netting set; the proxies do not:")
    print(by_size.to_string(float_format=lambda v: f"{v:>12,.2f}"))


def report_sensitivity(node_sweep: pd.DataFrame, domain_sweep: pd.DataFrame) -> None:
    print_header("6. INTERPOLATION: 1-D (deterministic rates) vs 3-D (stochastic rates)")
    print(f"  Mean |relative error| of the incremental proxy, by nodes per state dimension")
    print(f"  (time nodes held at {config.SWEEP_TIME_NODES} so the STATE-dimension error is what is measured):")
    table = node_sweep.pivot_table(
        index="n_state_nodes", columns=["state_dims", "interp_method"],
        values="mean_abs_rel_error",
    )
    print(table.to_string(float_format=lambda v: f"{v:>14.3e}"))

    print("\n  Cost of the grid - exact analytic prices needed per pricing request:")
    cost = node_sweep.pivot_table(
        index="n_state_nodes", columns="state_dims", values="mean_exact_evaluations"
    )
    cost.columns = [f"{int(c)}-D state" for c in cost.columns]
    cost["growth factor"] = cost.iloc[:, -1] / cost.iloc[:, 0]
    print(cost.to_string(float_format=lambda v: f"{v:>14,.0f}"))

    print_header("7. FX-SWAP INTERPOLATION DOMAIN: leg-aware vs naive")
    print(domain_sweep[["rate_model", "interp_method", "split_legs",
                        "mean_abs_rel_error", "max_abs_rel_error"]].to_string(
        index=False, float_format=lambda v: f"{v:>14.3e}"))


def report_comparison(comparison: pd.DataFrame, split: pd.DataFrame) -> None:
    print_header("8. DETERMINISTIC vs STOCHASTIC RATES - side by side (linear interpolation)")
    view = comparison[[
        "trade_key",
        "true_incremental_cva_deterministic", "true_incremental_cva_stochastic",
        "standalone_rel_error_deterministic", "standalone_rel_error_stochastic",
        "incremental_rel_error_deterministic", "incremental_rel_error_stochastic",
    ]].copy()
    view.columns = ["trade", "|true incr| DET", "|true incr| STOCH",
                    "SA rel err DET", "SA rel err STOCH",
                    "INC rel err DET", "INC rel err STOCH"]
    print(view.to_string(
        index=False,
        formatters={
            "|true incr| DET": lambda v: f"{v:>15,.0f}",
            "|true incr| STOCH": lambda v: f"{v:>17,.0f}",
            "SA rel err DET": lambda v: f"{v:>15.2%}",
            "SA rel err STOCH": lambda v: f"{v:>16.2%}",
            "INC rel err DET": lambda v: f"{v:>16.5%}",
            "INC rel err STOCH": lambda v: f"{v:>17.5%}",
        },
    ))

    print_header("9. RISK DECOMPOSITION: FX spot vs interest rates (stochastic model)")
    print(split.to_string(
        index=False,
        formatters={
            "total_vol_usd": lambda v: f"{v:>14,.0f}",
            "fx_share": lambda v: f"{v:>10.1%}",
            "rates_share": lambda v: f"{v:>12.1%}",
            "cross_share": lambda v: f"{v:>12.1%}",
            "fx_share_excl_cross": lambda v: f"{v:>20.1%}",
        },
    ))
    print("\n  total_vol_usd is the one-standard-deviation annualised move in the trade's")
    print("  value implied by the three deltas.  Shares can fall outside 0-100% because the")
    print("  FX/rates cross term carries a sign; fx_share_excl_cross drops that term.")


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    t_start = time.perf_counter()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # ---- stage 1: deterministic world first, so it can define the trades ----
    world_det = build_world("deterministic")
    shared_sets = world_det["netting_sets"]
    shared_incoming = {name: entry["incoming"] for name, entry in world_det["batch"].items()}
    world_sto = build_world("stochastic", netting_sets=shared_sets, incoming=shared_incoming)
    worlds = {"deterministic": world_det, "stochastic": world_sto}

    # ---- validation, before any result is believed -------------------------
    validation_report, convergence = validation.run_all(shared_sets[-1])
    report_setup(world_sto, validation_report, convergence)
    if not validation_report.all_passed:
        for check in validation_report.failures():
            print(f"  FAILED: {check.test} / {check.case}")
        raise SystemExit("Model validation failed - refusing to report CVA results.")

    # ---- stage 2 + 3: the study, under both rate models ---------------------
    results, scatters, node_sweeps, domain_sweeps = {}, {}, [], []
    for name, world in worlds.items():
        frame, scatter = run_main_study(world)
        results[name] = frame
        scatters[name] = scatter
        node_sweeps.append(run_node_sweep(world))
        domain_sweeps.append(run_domain_sweep(world))

    node_sweep = pd.concat(node_sweeps, ignore_index=True)
    domain_sweep = pd.concat(domain_sweeps, ignore_index=True)

    # ---- stage 4: comparison and risk decomposition -------------------------
    comparison = summarise_rate_models(worlds, results)
    incoming_first = world_sto["batch"][shared_sets[0].name]["incoming"]
    representative = {
        item.key: item.trade for item in incoming_first if item.key in ("new_swap", "new_fwd")
    }
    split = risk_split.risk_split_table(
        representative, world_sto["market"], world_sto["scen"],
        near_leg_date=config.INCOMING_SWAP_NEAR,
    )

    # ---- reporting ----------------------------------------------------------
    for name in ("deterministic", "stochastic"):
        report_study(worlds[name], results[name])
    report_sensitivity(node_sweep, domain_sweep)
    report_comparison(comparison, split)

    # ---- exports ------------------------------------------------------------
    all_results = pd.concat(results.values(), ignore_index=True)
    all_results.to_csv(os.path.join(config.RESULTS_DIR, config.MAIN_RESULTS_CSV), index=False)
    pd.concat([node_sweep, domain_sweep], ignore_index=True).to_csv(
        os.path.join(config.RESULTS_DIR, config.SENSITIVITY_CSV), index=False)
    portfolio_to_frame(shared_sets).to_csv(
        os.path.join(config.RESULTS_DIR, config.PORTFOLIO_CSV), index=False)
    validation_report.to_frame().to_csv(
        os.path.join(config.RESULTS_DIR, config.VALIDATION_CSV), index=False)
    comparison.to_csv(os.path.join(config.RESULTS_DIR, config.COMPARISON_CSV), index=False)
    split.to_csv(os.path.join(config.RESULTS_DIR, config.RISK_SPLIT_CSV), index=False)
    convergence.to_csv(os.path.join(config.RESULTS_DIR, "mc_convergence.csv"), index=False)

    # ---- figures ------------------------------------------------------------
    times = config.TIME_GRID
    batch_sto = world_sto["batch"]
    figures = [
        plots.plot_exposure_profiles(
            times,
            {ns.name: batch_sto[ns.name]["ee"] for ns in shared_sets},
            {ns.name: batch_sto[ns.name]["pfe"] for ns in shared_sets},
            shared_sets[-1].name, config.RESULTS_DIR,
        ),
        plots.plot_swap_exposure_rate_models(
            times, standalone_exposure_profiles(worlds),
            config.INCOMING_SWAP_NEAR, config.RESULTS_DIR,
        ),
        plots.plot_cva_comparison(
            results["stochastic"], config.RESULTS_DIR,
            title_suffix=", stochastic rates",
        ),
        plots.plot_signed_errors(
            results["stochastic"], config.RESULTS_DIR,
            title_suffix="  -  three-factor model",
        ),
        plots.plot_accuracy_vs_time(scatters["stochastic"], config.RESULTS_DIR),
        plots.plot_interpolation_convergence(node_sweep, domain_sweep, config.RESULTS_DIR),
        plots.plot_error_vs_netting_set_size(results["stochastic"], config.RESULTS_DIR),
        plots.plot_rate_model_comparison(comparison, config.RESULTS_DIR),
        plots.plot_convergence(convergence, config.N_PATHS, config.RESULTS_DIR),
        plots.plot_risk_split(split, config.INCOMING_SWAP_NEAR, config.RESULTS_DIR),
    ]

    print_header("OUTPUT FILES")
    for path in [
        os.path.join(config.RESULTS_DIR, name) for name in (
            config.MAIN_RESULTS_CSV, config.SENSITIVITY_CSV, config.PORTFOLIO_CSV,
            config.VALIDATION_CSV, config.COMPARISON_CSV, config.RISK_SPLIT_CSV,
            "mc_convergence.csv",
        )
    ] + figures:
        print(f"  {path}")
    print(f"\n  total runtime: {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    main()
