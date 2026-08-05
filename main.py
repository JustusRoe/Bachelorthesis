"""
End-to-end driver for the CVA proxy study.

Run with:  python main.py

The study runs TWICE -- once with deterministic rates (the single-factor
fallback) and once with stochastic rates -- on the same trades and seed, so any
difference is due to rate volatility alone.  This file only orchestrates the
engines in cva/ and reports; all the maths lives there, all knobs in config.py.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from cva import config, plots, proxies, validation
from cva.cva import (
    cva_from_values,
    cva_standard_error,
    expected_exposure,
    potential_future_exposure,
    speedup,
    summarise_errors,
)
from cva.model import ThreeFactorMarket
from cva.portfolio import build_all_netting_sets, build_incoming_trades, portfolio_to_frame
from cva.products import netting_set_values
from cva.proxies import PricingContext, state_box

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)

REL_ERROR_FLOOR = 1_000.0   # USD: below this the relative error is meaningless


def header(title: str) -> None:
    print("\n" + "=" * 118 + f"\n{title}\n" + "=" * 118)


def show(df: pd.DataFrame, index: bool = False) -> None:
    """Print a DataFrame with a plain, uniform number format."""
    print(df.to_string(index=index, float_format=lambda v: f"{v:,.4g}"))


# ---------------------------------------------------------------------------
# Stage 1 - build one "world": market + simulated paths + per-counterparty batch
# ---------------------------------------------------------------------------

def build_world(rate_model, netting_sets=None, incoming=None):
    """Simulate the risk factors and pre-compute each netting set (the bank's
    overnight batch that the incremental proxy later reuses for free).

    netting_sets/incoming are passed on the second call so both rate models
    price the identical trades -- the risk-reducing trade is sized from the
    simulated state, so it must be built once and shared.
    """
    market = ThreeFactorMarket.build(rate_model=rate_model)
    t0 = time.perf_counter()
    scen = market.simulate(config.TIME_GRID, config.N_PATHS, config.SEED)
    sim_seconds = time.perf_counter() - t0

    if netting_sets is None:
        netting_sets = build_all_netting_sets(market)
    ctx = PricingContext(market, scen, state_box(scen), config.DISCOUNTING)

    batch = {}
    for ns in netting_sets:
        values = netting_set_values(ns.trades, market, scen)
        batch[ns.name] = dict(
            netting_set=ns,
            values=values,
            cva=cva_from_values(values, scen, market, ns.hazard),
            cva_se=cva_standard_error(values, scen, ns.hazard),
            ee=expected_exposure(values),
            pfe=potential_future_exposure(values),
            mtm_today=float(values[0, 0]),
            incoming=(incoming[ns.name] if incoming is not None
                      else build_incoming_trades(ns, market, scen, values)),
        )

    return dict(rate_model=rate_model, market=market, scen=scen, ctx=ctx,
                netting_sets=netting_sets, batch=batch, sim_seconds=sim_seconds)


# ---------------------------------------------------------------------------
# Stage 2 - the main experiment: price every trade with both proxies
# ---------------------------------------------------------------------------

def run_main_study(world):
    """Returns (results, scatter): one results row per (netting set, trade,
    interpolation method); scatter feeds the accuracy-vs-time figure."""
    ctx, batch = world["ctx"], world["batch"]
    rows, scatter_rows = [], []

    for ns in world["netting_sets"]:
        stored = batch[ns.name]
        stored_values, stored_cva = stored["values"], stored["cva"]

        for item in stored["incoming"]:
            # Ground truth (and timing reference): full re-simulation.
            truth = proxies.full_resimulation_incremental(
                ns.trades, item.trade, ctx, ns.hazard, stored_cva)
            true_inc = truth.cva
            # Diagnostic: exact repricing against stored values isolates netting
            # (exact) from interpolation error.
            exact_inc = proxies.incremental_exact_repricing(
                item.trade, ctx, ns.hazard, stored_values, stored_cva)

            scatter_rows.append(dict(netting_set=ns.name, trade_key=item.key,
                                     approach="full re-simulation",
                                     seconds=truth.seconds, abs_rel_error=0.0))

            for method in config.INTERP_METHODS:
                standalone = proxies.standalone_proxy(item.trade, ctx, ns.hazard, method=method)
                incremental = proxies.incremental_proxy(
                    item.trade, ctx, ns.hazard, stored_values, stored_cva, method=method)
                sa = summarise_errors(standalone.cva, true_inc)
                inc = summarise_errors(incremental.cva, true_inc)

                rows.append(dict(
                    rate_model=world["rate_model"], netting_set=ns.name, n_trades=len(ns),
                    hazard_5y_average=ns.hazard.average_hazard,
                    portfolio_cva=stored_cva, portfolio_cva_std_error=stored["cva_se"],
                    portfolio_mtm_today=stored["mtm_today"],
                    trade_key=item.key, trade_description=item.description,
                    interp_method=method, state_dimensions=ctx.n_state_dimensions,
                    # the three CVA numbers
                    true_incremental_cva=true_inc,
                    standalone_proxy_cva=standalone.cva,
                    incremental_proxy_cva=incremental.cva,
                    incremental_exact_cva=exact_inc.cva,
                    # errors
                    standalone_signed_error=sa["signed_error"],
                    standalone_abs_error=sa["abs_error"],
                    standalone_rel_error=sa["rel_error"],
                    standalone_direction=sa["direction"],
                    incremental_signed_error=inc["signed_error"],
                    incremental_abs_error=inc["abs_error"],
                    incremental_rel_error=inc["rel_error"],
                    incremental_direction=inc["direction"],
                    # where the incremental error comes from
                    interpolation_only_error=incremental.cva - exact_inc.cva,
                    netting_exactness_check=exact_inc.cva - true_inc,
                    # cost
                    full_resim_seconds=truth.seconds,
                    standalone_seconds=standalone.seconds,
                    incremental_seconds=incremental.seconds,
                    standalone_speedup=speedup(truth.seconds, standalone.seconds),
                    incremental_speedup=speedup(truth.seconds, incremental.seconds),
                    proxy_exact_evaluations=incremental.n_exact_evaluations,
                    full_resim_evaluations=truth.n_exact_evaluations,
                ))

                for label, res, err in (("standalone proxy", standalone, sa),
                                        (f"incremental proxy ({method})", incremental, inc)):
                    if label == "standalone proxy" and method != "linear":
                        continue     # standalone charge is plotted once only
                    rel = abs(err["rel_error"]) if abs(true_inc) >= REL_ERROR_FLOOR else np.nan
                    scatter_rows.append(dict(netting_set=ns.name, trade_key=item.key,
                                             approach=label, seconds=res.seconds,
                                             abs_rel_error=rel))

    scatter = pd.DataFrame(scatter_rows).dropna(subset=["abs_rel_error"])
    return pd.DataFrame(rows), scatter


# ---------------------------------------------------------------------------
# Stage 3 - sensitivity sweeps
# ---------------------------------------------------------------------------

def _incremental_truths(world, repeats=1):
    """Exact incremental CVA per (netting set, trade); independent of the grid."""
    ctx, batch = world["ctx"], world["batch"]
    truths = {}
    for ns in world["netting_sets"]:
        stored = batch[ns.name]
        for item in stored["incoming"]:
            truths[(ns.name, item.key)] = (item, proxies.incremental_exact_repricing(
                item.trade, ctx, ns.hazard, stored["values"], stored["cva"], repeats=repeats).cva)
    return truths


def run_node_sweep(world):
    """Incremental-proxy error vs node count, per method. Run under both rate
    models so 1-D (deterministic) and 3-D (stochastic) grids compare directly."""
    ctx, batch = world["ctx"], world["batch"]
    truths = _incremental_truths(world)
    rows = []
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
                        n_time_nodes=config.SWEEP_TIME_NODES, repeats=1)
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
                mean_exact_evaluations=float(np.mean(evaluations))))
    return pd.DataFrame(rows)


def run_grid_shape_sweep(world):
    """Isotropic n x n x n vs anisotropic 2 x n x n state grids. V is affine in
    S, so 2 spot nodes suffice; this measures the cost saved (and break-even
    against full re-simulation), at no accuracy cost."""
    ctx, batch = world["ctx"], world["batch"]
    truths = _incremental_truths(world)
    resim_ms = {}
    for ns in world["netting_sets"]:
        for item in batch[ns.name]["incoming"]:
            resim_ms[(ns.name, item.key)] = 1000.0 * proxies.full_resimulation_incremental(
                ns.trades, item.trade, ctx, ns.hazard, batch[ns.name]["cva"]).seconds

    shapes = {"isotropic n x n x n": lambda n: n,
              "anisotropic 2 x n x n": lambda n: config.anisotropic_nodes(n)}
    rows = []
    for shape, node_spec in shapes.items():
        for method in config.INTERP_METHODS:
            for n in config.NODE_SWEEP:
                errors, evaluations, proxy_ms, reference_ms = [], [], [], []
                for ns in world["netting_sets"]:
                    stored = batch[ns.name]
                    for item in stored["incoming"]:
                        true_inc = truths[(ns.name, item.key)][1]
                        if abs(true_inc) < REL_ERROR_FLOOR:
                            continue
                        res = proxies.incremental_proxy(
                            item.trade, ctx, ns.hazard, stored["values"], stored["cva"],
                            method=method, n_state_nodes=node_spec(n),
                            n_time_nodes=config.SWEEP_TIME_NODES)
                        errors.append(abs(res.cva - true_inc) / abs(true_inc))
                        evaluations.append(res.n_exact_evaluations)
                        proxy_ms.append(1000.0 * res.seconds)
                        reference_ms.append(resim_ms[(ns.name, item.key)])
                rows.append(dict(
                    sweep="grid_shape", rate_model=world["rate_model"],
                    state_dims=ctx.n_state_dimensions, grid_shape=shape,
                    interp_method=method, n_rate_nodes=n,
                    n_spot_nodes=config.SPOT_NODES_ANISOTROPIC if "aniso" in shape else n,
                    mean_abs_rel_error=float(np.mean(errors)),
                    max_abs_rel_error=float(np.max(errors)),
                    mean_exact_evaluations=float(np.mean(evaluations)),
                    mean_proxy_ms=float(np.mean(proxy_ms)),
                    mean_resim_ms=float(np.mean(reference_ms)),
                    speedup_vs_resim=float(np.mean(reference_ms) / np.mean(proxy_ms))))
    return pd.DataFrame(rows)


def run_domain_sweep(world):
    """Interpolating the whole FX swap on one time domain (split_legs=False)
    puts the near-leg settlement jump inside the domain -- worse for Chebyshev
    than linear, since a global polynomial can't represent a discontinuity."""
    ctx, batch = world["ctx"], world["batch"]
    rows = []
    for method in config.INTERP_METHODS:
        for split_legs in (True, False):
            errors = []
            for ns in world["netting_sets"]:
                stored = batch[ns.name]
                for item in stored["incoming"]:
                    if item.key != "new_swap":
                        continue
                    true_inc = proxies.incremental_exact_repricing(
                        item.trade, ctx, ns.hazard, stored["values"], stored["cva"], repeats=1).cva
                    res = proxies.incremental_proxy(
                        item.trade, ctx, ns.hazard, stored["values"], stored["cva"],
                        method=method, split_legs=split_legs, repeats=1)
                    errors.append(abs(res.cva - true_inc) / abs(true_inc))
            rows.append(dict(
                sweep="swap_domain_handling", rate_model=world["rate_model"],
                state_dims=ctx.n_state_dimensions, interp_method=method,
                n_time_nodes=config.N_TIME_NODES, n_state_nodes=config.N_STATE_NODES,
                grid_nodes_per_leg=np.nan, split_legs=split_legs,
                mean_abs_rel_error=float(np.mean(errors)),
                max_abs_rel_error=float(np.max(errors)),
                mean_exact_evaluations=np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 4 - deterministic vs stochastic comparison
# ---------------------------------------------------------------------------

def summarise_rate_models(results, method="linear"):
    """Average the headline numbers by trade type under each rate model."""
    frames = []
    for name, frame in results.items():
        sub = frame[frame["interp_method"] == method]
        frames.append(sub.groupby("trade_key").agg(**{
            f"true_incremental_cva_{name}": ("true_incremental_cva", lambda s: float(np.mean(np.abs(s)))),
            f"standalone_rel_error_{name}": ("standalone_rel_error", lambda s: float(np.nanmean(np.abs(s)))),
            f"incremental_rel_error_{name}": ("incremental_rel_error", lambda s: float(np.nanmean(np.abs(s)))),
            f"standalone_signed_error_{name}": ("standalone_signed_error", "mean"),
            f"incremental_signed_error_{name}": ("incremental_signed_error", "mean"),
        }))
    combined = pd.concat(frames, axis=1).reset_index()
    order = {k: i for i, k in enumerate(["new_fwd", "new_swap", "offset_fwd", "small_fwd"])}
    combined["__o"] = combined["trade_key"].map(order)
    return combined.sort_values("__o").drop(columns="__o").reset_index(drop=True)


def standalone_exposure_profiles(worlds, trade_keys=("new_swap", "new_fwd")):
    """Un-netted expected exposure of the plain forward and swap under each rate
    model, on the same trade objects (NS1's copies; these two are set-agnostic)."""
    reference = next(iter(worlds.values()))
    first_set = reference["netting_sets"][0].name
    items = {item.key: item for item in reference["batch"][first_set]["incoming"]
             if item.key in trade_keys}
    profiles = {}
    for key in trade_keys:
        item = items[key]
        profiles[item.description] = {}
        for name, world in worlds.items():
            s = world["scen"]
            values = item.trade.value(world["market"], s.times, s.spots, s.r_d, s.r_f)
            profiles[item.description][name] = expected_exposure(values)
    return profiles


# ---------------------------------------------------------------------------
# Stage 5 - reporting (console only; does not affect the exported files)
# ---------------------------------------------------------------------------

def report_setup(world, report, convergence):
    header("MODEL SETUP")
    print(f"  seed {config.SEED}   paths {config.N_PATHS:,}   "
          f"dates {config.TIME_GRID.size} (0 to {config.HORIZON:g}y)   "
          f"discounting {config.DISCOUNTING}   recovery {config.RECOVERY_RATE:.0%}")
    print(f"  interpolation grid: {config.N_TIME_NODES} time x {config.N_STATE_NODES} state nodes   "
          f"simulation took {world['sim_seconds']*1000:.0f} ms\n")
    print(world["market"].describe())

    header("MODEL VALIDATION  (must all pass before any CVA is reported)")
    print(validation.format_report(report))
    print("\n  Monte Carlo convergence of the NS5 netting-set CVA:")
    show(convergence.round(2))
    print(f"\n  ==> {'ALL CHECKS PASSED' if report.all_passed else 'FAILURES PRESENT'}")


def report_study(world, results):
    batch = world["batch"]
    tag = world["rate_model"].upper()

    header(f"[{tag}]  1. TRUE CVA PER NETTING SET (full simulation)")
    show(pd.DataFrame([dict(
        netting_set=ns.name, trades=len(ns), legs=sum(len(t.legs) for t in ns.trades),
        max_tenor_y=round(ns.max_tenor, 2), hazard=ns.hazard.describe(),
        net_notional_eur_m=round(ns.net_signed_notional / 1e6, 1),
        mtm_today_usd_m=round(batch[ns.name]["mtm_today"] / 1e6, 2),
        peak_EE_usd_m=round(batch[ns.name]["ee"].max() / 1e6, 2),
        true_CVA_usd=round(batch[ns.name]["cva"]),
        CVA_std_err=round(batch[ns.name]["cva_se"]),
    ) for ns in world["netting_sets"]]))

    lin = results[results["interp_method"] == "linear"]

    header(f"[{tag}]  2. THE THREE CVA NUMBERS PER INCOMING TRADE (linear interpolation)")
    view = lin[["netting_set", "trade_key", "true_incremental_cva",
                "standalone_proxy_cva", "incremental_proxy_cva", "portfolio_cva"]].round()
    view.columns = ["netting_set", "trade", "TRUE incr.", "STANDALONE", "INCREMENTAL", "portfolio CVA"]
    show(view)

    header(f"[{tag}]  3. SIGNED ERROR vs TRUE INCREMENTAL CVA  (positive = OVERCHARGE)")
    err = lin[["netting_set", "trade_key", "standalone_signed_error", "standalone_direction",
               "incremental_signed_error", "incremental_direction"]].copy()
    err["standalone_signed_error"] = err["standalone_signed_error"].round()
    err["incremental_signed_error"] = err["incremental_signed_error"].round(2)
    show(err)

    header(f"[{tag}]  4. DIAGNOSTIC: where does each proxy's error come from?")
    show(lin.groupby("netting_set").agg(
        max_netting_check=("netting_exactness_check", lambda s: s.abs().max()),
        max_interp_error=("interpolation_only_error", lambda s: s.abs().max()),
        max_standalone_error=("standalone_abs_error", "max")), index=True)
    print("\n  netting_exactness_check ~ 0 by construction: the incremental proxy nets EXACTLY,")
    print("  so its only error source is the interpolation column next to it.")

    header(f"[{tag}]  5. COMPUTATION TIME PER PRICING REQUEST")
    show(results.groupby("interp_method").agg(
        full_resim_ms=("full_resim_seconds", lambda s: 1000 * s.mean()),
        standalone_ms=("standalone_seconds", lambda s: 1000 * s.mean()),
        incremental_ms=("incremental_seconds", lambda s: 1000 * s.mean()),
        standalone_speedup=("standalone_speedup", "mean"),
        incremental_speedup=("incremental_speedup", "mean"),
        proxy_exact_prices=("proxy_exact_evaluations", "mean"),
        resim_exact_prices=("full_resim_evaluations", "mean")), index=True)
    print("\n  Full re-simulation cost grows with the netting set; the proxies do not:")
    show(lin.groupby(["netting_set", "n_trades"]).agg(
        full_resim_ms=("full_resim_seconds", lambda s: 1000 * s.mean()),
        incremental_ms=("incremental_seconds", lambda s: 1000 * s.mean()),
        speedup=("incremental_speedup", "mean")), index=True)


def report_sensitivity(node_sweep, domain_sweep, grid_shape):
    header("6. INTERPOLATION: 1-D (deterministic) vs 3-D (stochastic) rates")
    print(f"  Mean |rel error| of the incremental proxy by nodes per state dimension")
    print(f"  (time nodes held at {config.SWEEP_TIME_NODES} to isolate the state-dimension error):")
    show(node_sweep.pivot_table(index="n_state_nodes",
                                columns=["state_dims", "interp_method"],
                                values="mean_abs_rel_error"), index=True)
    print("\n  Exact analytic prices needed per request (the n^3 cost):")
    cost = node_sweep.pivot_table(index="n_state_nodes", columns="state_dims",
                                  values="mean_exact_evaluations")
    cost.columns = [f"{int(c)}-D state" for c in cost.columns]
    cost["growth"] = cost.iloc[:, -1] / cost.iloc[:, 0]
    show(cost, index=True)

    header("7. FX-SWAP INTERPOLATION DOMAIN: leg-aware vs naive")
    show(domain_sweep[["rate_model", "interp_method", "split_legs",
                       "mean_abs_rel_error", "max_abs_rel_error"]])

    header("7b. GRID SHAPE: isotropic n x n x n vs anisotropic 2 x n x n")
    for method in config.INTERP_METHODS:
        print(f"\n  --- {method} ---")
        show(grid_shape[grid_shape["interp_method"] == method].pivot_table(
            index="n_rate_nodes", columns="grid_shape",
            values=["mean_abs_rel_error", "mean_exact_evaluations", "mean_proxy_ms"]), index=True)
    print("\n  Break-even vs full re-simulation (speedup > 1 means the proxy wins):")
    show(grid_shape.pivot_table(index="n_rate_nodes",
                                columns=["grid_shape", "interp_method"],
                                values="speedup_vs_resim"), index=True)
    print(f"\n  full re-simulation reference: {grid_shape['mean_resim_ms'].mean():,.1f} ms/request")
    for (shape, method), sub in grid_shape.groupby(["grid_shape", "interp_method"]):
        winning = sub[sub["speedup_vs_resim"] > 1.0]["n_rate_nodes"]
        verdict = f"faster up to n = {int(winning.max())}" if len(winning) else "never faster"
        print(f"    {shape:24s} {method:10s} -> {verdict}")


def report_comparison(comparison, split):
    header("8. DETERMINISTIC vs STOCHASTIC RATES - side by side (linear interpolation)")
    view = comparison[["trade_key",
                       "true_incremental_cva_deterministic", "true_incremental_cva_stochastic",
                       "standalone_rel_error_deterministic", "standalone_rel_error_stochastic",
                       "incremental_rel_error_deterministic", "incremental_rel_error_stochastic"]].copy()
    view.columns = ["trade", "|true incr| DET", "|true incr| STOCH", "SA rel DET",
                    "SA rel STOCH", "INC rel DET", "INC rel STOCH"]
    show(view)

    header("9. RISK DECOMPOSITION: FX spot vs interest rates (stochastic model)")
    show(split)
    print("\n  total_vol_usd = one-sigma annualised move implied by the three deltas.")
    print("  Shares can fall outside 0-100% because the FX/rates cross term is signed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    t_start = time.perf_counter()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Build the deterministic world first so it defines the shared trades, then
    # the stochastic world reusing them.
    world_det = build_world("deterministic")
    shared_sets = world_det["netting_sets"]
    shared_incoming = {name: e["incoming"] for name, e in world_det["batch"].items()}
    world_sto = build_world("stochastic", netting_sets=shared_sets, incoming=shared_incoming)
    worlds = {"deterministic": world_det, "stochastic": world_sto}

    # Validate the model before believing any result.
    report, convergence = validation.run_all(shared_sets[-1])
    report_setup(world_sto, report, convergence)
    if not report.all_passed:
        raise SystemExit("Model validation failed - refusing to report CVA results.")

    # The study, under both rate models.
    results, scatters, node_sweeps, domain_sweeps = {}, {}, [], []
    for name, world in worlds.items():
        results[name], scatters[name] = run_main_study(world)
        node_sweeps.append(run_node_sweep(world))
        domain_sweeps.append(run_domain_sweep(world))
    grid_shape = run_grid_shape_sweep(world_sto)   # 3-D state only
    node_sweep = pd.concat(node_sweeps, ignore_index=True)
    domain_sweep = pd.concat(domain_sweeps, ignore_index=True)

    # Comparison and risk decomposition.
    comparison = summarise_rate_models(results)
    representative = {item.key: item.trade
                      for item in world_sto["batch"][shared_sets[0].name]["incoming"]
                      if item.key in ("new_swap", "new_fwd")}
    split = validation.risk_split_table(representative, world_sto["market"], world_sto["scen"],
                                        near_leg_date=config.INCOMING_SWAP_NEAR)

    # Reporting.
    for name in ("deterministic", "stochastic"):
        report_study(worlds[name], results[name])
    report_sensitivity(node_sweep, domain_sweep, grid_shape)
    report_comparison(comparison, split)

    # Exports.
    out = config.RESULTS_DIR
    pd.concat(results.values(), ignore_index=True).to_csv(f"{out}/{config.MAIN_RESULTS_CSV}", index=False)
    pd.concat([node_sweep, domain_sweep, grid_shape], ignore_index=True).to_csv(
        f"{out}/{config.SENSITIVITY_CSV}", index=False)
    portfolio_to_frame(shared_sets).to_csv(f"{out}/{config.PORTFOLIO_CSV}", index=False)
    report.to_frame().to_csv(f"{out}/{config.VALIDATION_CSV}", index=False)
    comparison.to_csv(f"{out}/{config.COMPARISON_CSV}", index=False)
    split.to_csv(f"{out}/{config.RISK_SPLIT_CSV}", index=False)
    convergence.to_csv(f"{out}/mc_convergence.csv", index=False)

    # Figures.
    batch = world_sto["batch"]
    figures = [
        plots.plot_exposure_profiles(
            config.TIME_GRID,
            {ns.name: batch[ns.name]["ee"] for ns in shared_sets},
            {ns.name: batch[ns.name]["pfe"] for ns in shared_sets},
            shared_sets[-1].name, out),
        plots.plot_swap_exposure_rate_models(
            config.TIME_GRID, standalone_exposure_profiles(worlds), config.INCOMING_SWAP_NEAR, out),
        plots.plot_cva_comparison(results["stochastic"], out, title_suffix=", stochastic rates"),
        plots.plot_signed_errors(results["stochastic"], out, title_suffix="  -  three-factor model"),
        plots.plot_accuracy_vs_time(scatters["stochastic"], out),
        plots.plot_interpolation_convergence(node_sweep, domain_sweep, out),
        plots.plot_error_vs_netting_set_size(results["stochastic"], out),
        plots.plot_rate_model_comparison(comparison, out),
        plots.plot_grid_shape(grid_shape, out),
        plots.plot_convergence(convergence, config.N_PATHS, out),
        plots.plot_risk_split(split, config.INCOMING_SWAP_NEAR, out),
    ]

    header("OUTPUT FILES")
    for name in (config.MAIN_RESULTS_CSV, config.SENSITIVITY_CSV, config.PORTFOLIO_CSV,
                 config.VALIDATION_CSV, config.COMPARISON_CSV, config.RISK_SPLIT_CSV,
                 "mc_convergence.csv"):
        print(f"  {out}/{name}")
    for path in figures:
        print(f"  {path}")
    print(f"\n  total runtime: {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    main()
