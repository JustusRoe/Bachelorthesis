"""
All figures produced by the study.

Each function saves one PNG into the results directory and returns its path.
Nothing here computes CVA - the plotting layer only consumes the DataFrames and
arrays produced by the rest of the package, so the numerical results and their
presentation stay separable.

Design notes
------------
The figures are meant for a printed thesis, so they commit to a single light
surface.  Categorical colours come from a validated three-slot palette (blue /
orange / aqua) that stays distinguishable under deuteranopia and tritanopia;
signed errors use a blue-red diverging pair with a neutral zero line, because
there the polarity (over- vs undercharging) IS the message.  Where a fourth
distinction is needed it is carried by line style or marker fill, never by a
fourth hue.  Every chart carries a legend, and the accompanying CSVs are the
table view.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")           # headless rendering - no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# ----------------------------------------------------------------------------
# Palette and chrome
# ----------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots (validated as a set for all-pairs CVD separation).
C_TRUE = "#2a78d6"          # slot 1, blue   - the ground truth
C_STANDALONE = "#eb6834"    # slot 2, orange - standalone proxy
C_INCREMENTAL = "#1baf7a"   # slot 3, aqua   - incremental proxy

# Diverging pair for signed errors: warm/cool poles with a neutral midpoint.
C_OVER = "#2a78d6"          # overcharging  (proxy > truth)
C_UNDER = "#d03b3b"         # undercharging (proxy < truth)

# Sequential blue ramp (steps 300-700), for the five netting sets ordered by size.
SEQ_BLUE = ["#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style_axes(ax, ygrid: bool = True) -> None:
    """Recessive chrome: hairline grid behind the marks, no box."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.8)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)


def _new_figure(*args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def format_usd(value: float, decimals: int = 1) -> str:
    """Compact money label: 1_051_765 -> '1.1m', 18_868 -> '18.9k'."""
    a = abs(value)
    if a >= 1e6:
        return f"{value/1e6:.{decimals}f}m"
    if a >= 1e3:
        return f"{value/1e3:.{decimals}f}k"
    return f"{value:.0f}"


def _tick_usd(value: float, _pos=None) -> str:
    """Axis tick label for money.

    Keeps one decimal only when it carries information, so a 2,500 tick reads
    '2.5k' rather than being rounded to '2k'.  Sub-dollar ticks are left
    unlabelled: on a symlog axis they would all render as '0' around the linear
    threshold, and a CVA charge is never quoted to that precision anyway.
    """
    if 0 < abs(value) < 1:
        return ""
    label = format_usd(value, 1)
    return label.replace(".0", "") if label.endswith((".0k", ".0m")) else label


_USD_FORMATTER = FuncFormatter(_tick_usd)


def _save(fig, results_dir: str, filename: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# 1. Expected exposure profiles
# ----------------------------------------------------------------------------

def plot_exposure_profiles(
    times: np.ndarray,
    ee_by_set: dict[str, np.ndarray],
    pfe_by_set: dict[str, np.ndarray],
    focus_set: str,
    results_dir: str,
    filename: str = "ee_profiles.png",
) -> str:
    """Left: EE(t) of every netting set.  Right: EE vs PFE for one focus set.

    The EE profile is the object the CVA formula integrates against: its hump
    shape comes from the competition between diffusion (exposure grows with the
    square root of time) and amortisation (trades mature and leave the set).
    """
    fig, (ax_all, ax_focus) = _new_figure(1, 2, figsize=(11.5, 4.2))

    for colour, (name, ee) in zip(SEQ_BLUE, ee_by_set.items()):
        ax_all.plot(times, ee / 1e6, color=colour, linewidth=2.0, label=name, zorder=3)
    _style_axes(ax_all)
    ax_all.set_title("Expected exposure by netting set", color=INK, fontsize=11, loc="left")
    ax_all.set_xlabel("time (years)", color=INK_SECONDARY, fontsize=9)
    ax_all.set_ylabel("EE (USD m)", color=INK_SECONDARY, fontsize=9)
    ax_all.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    ee, pfe = ee_by_set[focus_set], pfe_by_set[focus_set]
    ax_focus.fill_between(times, 0, pfe / 1e6, color=C_TRUE, alpha=0.12, zorder=2)
    ax_focus.plot(times, pfe / 1e6, color=C_TRUE, linewidth=2.0, linestyle="--",
                  label="PFE 95%", zorder=3)
    ax_focus.plot(times, ee / 1e6, color=C_STANDALONE, linewidth=2.0,
                  label="EE (drives CVA)", zorder=4)
    _style_axes(ax_focus)
    ax_focus.set_title(f"{focus_set}: EE vs potential future exposure",
                       color=INK, fontsize=11, loc="left")
    ax_focus.set_xlabel("time (years)", color=INK_SECONDARY, fontsize=9)
    ax_focus.set_ylabel("exposure (USD m)", color=INK_SECONDARY, fontsize=9)
    ax_focus.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    return _save(fig, results_dir, filename)


# ----------------------------------------------------------------------------
# 2. THE motivating chart: swap exposure under the two rate models
# ----------------------------------------------------------------------------

def plot_swap_exposure_rate_models(
    times: np.ndarray,
    profiles: dict[str, dict[str, np.ndarray]],
    near_leg_date: float,
    results_dir: str,
    filename: str = "swap_exposure_rate_models.png",
) -> str:
    """Exposure of an FX swap and a forward under deterministic vs stochastic rates.

    `profiles` maps a trade label to {"deterministic": EE, "stochastic": EE}.

    This is the chart that justifies the whole upgrade.  Before the near leg
    settles, the swap's two legs nearly cancel in the spot dimension, so under
    DETERMINISTIC rates it has almost no risk driver and its expected exposure
    is close to zero.  Turning the rate factors on gives it the driver it should
    always have had - the interest rate differential - and the pre-settlement
    exposure appears.  The forward is shown beside it as a control: it is driven
    by the spot in both models, so the two curves nearly coincide.
    """
    labels = list(profiles)
    fig, axes = _new_figure(1, len(labels), figsize=(5.6 * len(labels), 4.3))
    axes = np.atleast_1d(axes)

    for ax, label in zip(axes, labels):
        det = profiles[label]["deterministic"]
        stoch = profiles[label]["stochastic"]

        ax.fill_between(times, det / 1e6, stoch / 1e6,
                        where=(stoch >= det), color=C_INCREMENTAL, alpha=0.15, zorder=2,
                        label="exposure the 1-factor model misses")
        ax.plot(times, det / 1e6, color=C_STANDALONE, linewidth=2.0, linestyle="--",
                label="deterministic rates (1 factor)", zorder=3)
        ax.plot(times, stoch / 1e6, color=C_INCREMENTAL, linewidth=2.0,
                label="stochastic rates (3 factors)", zorder=4)

        ax.axvline(near_leg_date, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=2)
        ax.annotate("near leg settles", xy=(near_leg_date, ax.get_ylim()[1]),
                    xytext=(4, -10), textcoords="offset points",
                    fontsize=7.5, color=INK_MUTED, style="italic")

        _style_axes(ax)
        ax.set_title(label, color=INK, fontsize=10.5, loc="left")
        ax.set_xlabel("time (years)", color=INK_SECONDARY, fontsize=9)
        ax.set_ylabel("expected exposure (USD m)", color=INK_SECONDARY, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="upper left")

    fig.suptitle(
        "Deterministic rates understate the pre-settlement exposure of an FX swap",
        color=INK, fontsize=12, x=0.008, ha="left", y=1.02,
    )
    fig.tight_layout()
    return _save(fig, results_dir, filename)


# ----------------------------------------------------------------------------
# 3. Standalone vs incremental vs true
# ----------------------------------------------------------------------------

def plot_cva_comparison(
    results: pd.DataFrame,
    results_dir: str,
    filename: str = "cva_comparison.png",
    method: str = "linear",
    title_suffix: str = "",
) -> str:
    """Grouped bars per test trade, one small multiple per netting set.

    The showcase is the `offset_fwd` group: the true and incremental bars go
    BELOW the zero line (the trade reduces the counterparty's CVA) while the
    standalone bar stays firmly positive because it cannot see the netting set.
    """
    data = results[results["interp_method"] == method]
    sets = list(dict.fromkeys(data["netting_set"]))
    trades = list(dict.fromkeys(data["trade_key"]))

    fig, axes = _new_figure(1, len(sets), figsize=(4.0 * len(sets) / 1.35, 4.4), sharex=True)
    axes = np.atleast_1d(axes)

    x = np.arange(len(trades))
    width = 0.26
    gap = 0.015                              # 2px-equivalent surface gap

    for ax, ns_name in zip(axes, sets):
        sub = data[data["netting_set"] == ns_name].set_index("trade_key")
        true_v = sub.loc[trades, "true_incremental_cva"].to_numpy()
        sa_v = sub.loc[trades, "standalone_proxy_cva"].to_numpy()
        inc_v = sub.loc[trades, "incremental_proxy_cva"].to_numpy()

        ax.bar(x - width - gap, true_v, width, color=C_TRUE, label="true incremental", zorder=3)
        ax.bar(x, sa_v, width, color=C_STANDALONE, label="standalone proxy", zorder=3)
        ax.bar(x + width + gap, inc_v, width, color=C_INCREMENTAL,
               label="incremental proxy", zorder=3)

        lowest = min(true_v.min(), inc_v.min(), 0.0)
        highest = max(true_v.max(), sa_v.max(), inc_v.max(), 0.0)
        if lowest < 0:
            ax.set_ylim(lowest - 0.16 * (highest - lowest), highest * 1.05)
        for xi, iv in enumerate(inc_v):
            if iv < 0:
                ax.annotate(format_usd(iv), (xi + width + gap, iv),
                            textcoords="offset points", xytext=(0, -4),
                            ha="center", va="top",
                            fontsize=7.5, color=C_UNDER, fontweight="bold")

        ax.axhline(0, color=AXIS, linewidth=1.0, zorder=4)
        _style_axes(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(trades, rotation=35, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(_USD_FORMATTER)
        ax.set_title(ns_name, color=INK, fontsize=9.5, loc="left")

    axes[0].set_ylabel("CVA charge (USD)", color=INK_SECONDARY, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.06), labelcolor=INK_SECONDARY)
    fig.suptitle(
        f"CVA charge for the incoming trade  ({method} interpolation{title_suffix})  -  "
        "negative bars = the trade REDUCES counterparty CVA",
        color=INK, fontsize=11.5, x=0.008, ha="left", y=1.02,
    )
    fig.tight_layout()
    return _save(fig, results_dir, filename)


# ----------------------------------------------------------------------------
# 4. Signed error: over- vs undercharging
# ----------------------------------------------------------------------------

def plot_signed_errors(
    results: pd.DataFrame,
    results_dir: str,
    filename: str = "signed_errors.png",
    method: str = "linear",
    title_suffix: str = "",
) -> str:
    """One panel per proxy; bars above zero overcharge, below zero undercharge.

    The two panels deliberately carry their OWN vertical scale (they are two
    charts, not one chart with two axes): the standalone error is measured in
    hundreds of thousands of USD, the incremental error in tens.
    """
    data = results[results["interp_method"] == method].copy()
    data["bar_label"] = data["netting_set"].str.split("_").str[0] + " / " + data["trade_key"]

    fig, axes = _new_figure(2, 1, figsize=(12.0, 6.6), sharex=True)
    panels = [
        ("standalone_signed_error", "Standalone proxy", axes[0]),
        ("incremental_signed_error", "Incremental proxy", axes[1]),
    ]

    x = np.arange(len(data))
    for column, title, ax in panels:
        values = data[column].to_numpy()
        colours = [C_OVER if v >= 0 else C_UNDER for v in values]
        ax.bar(x, values, 0.7, color=colours, zorder=3)
        ax.axhline(0, color=INK_SECONDARY, linewidth=1.2, zorder=4)
        _style_axes(ax)

        # Symmetric-log scale: within one panel the errors span several orders of
        # magnitude.  symlog keeps the sign - the whole point of the chart -
        # while still making the small bars visible.
        worst = float(np.max(np.abs(values))) if values.size else 1.0
        ax.set_yscale("symlog", linthresh=max(worst * 1e-3, 1e-3))

        ax.set_xticks(x)
        ax.set_xticklabels(data["bar_label"], rotation=70, ha="right", fontsize=7)
        ax.yaxis.set_major_formatter(_USD_FORMATTER)
        ax.set_ylabel("signed error (USD, symlog)", color=INK_SECONDARY, fontsize=9)
        ax.set_title(
            f"{title}   -   largest absolute error {format_usd(worst)} USD",
            color=INK, fontsize=10.5, loc="left",
        )

    legend = [
        Line2D([0], [0], color=C_OVER, lw=7,
               label="signed error > 0  =  OVERCHARGING (trade lost to competitors)"),
        Line2D([0], [0], color=C_UNDER, lw=7,
               label="signed error < 0  =  UNDERCHARGING (loss borne by the bank)"),
    ]
    fig.legend(handles=legend, frameon=False, fontsize=9, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.04), labelcolor=INK_SECONDARY)
    fig.suptitle(f"Signed pricing error vs the true incremental CVA{title_suffix}",
                 color=INK, fontsize=12, x=0.008, ha="left", y=1.0)
    fig.tight_layout()
    return _save(fig, results_dir, filename)


# ----------------------------------------------------------------------------
# 5. Accuracy against computation time
# ----------------------------------------------------------------------------

def plot_accuracy_vs_time(
    scatter: pd.DataFrame,
    results_dir: str,
    filename: str = "accuracy_vs_time.png",
) -> str:
    """Cost of a pricing request against the error it buys.

    Colour carries the METHOD FAMILY (three validated slots); within the
    incremental family, filled vs open markers separate the two interpolation
    variants.  Identity therefore never rests on colour alone.
    """
    fig, ax = _new_figure(figsize=(8.2, 5.2))
    floor = 1e-16       # visual floor for "exact" and for machine-precision hits

    families = [
        ("standalone proxy", C_STANDALONE, "s", True),
        ("incremental proxy (linear)", C_INCREMENTAL, "^", False),
        ("incremental proxy (chebyshev)", C_INCREMENTAL, "D", True),
        ("full re-simulation", C_TRUE, "o", True),
    ]

    for label, colour, marker, filled in families:
        sub = scatter[scatter["approach"] == label]
        if sub.empty:
            continue
        ax.scatter(
            sub["seconds"] * 1000.0, sub["abs_rel_error"].clip(lower=floor),
            s=95, facecolor=colour if filled else "none",
            edgecolor=SURFACE if filled else colour,
            linewidth=1.8, marker=marker,
            label=label + (" - exact by construction" if "full" in label else ""),
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(floor / 10, 10)
    _style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)

    plain = FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0, subs=(2.0, 3.0, 5.0)))
    ax.xaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_formatter(plain)
    ax.tick_params(axis="x", which="minor", colors=INK_MUTED, labelsize=8)

    ax.set_xlabel("wall-clock time per pricing request (milliseconds, log scale)",
                  color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("|relative error| vs true incremental CVA (log scale)",
                  color=INK_SECONDARY, fontsize=9)
    ax.set_title("Accuracy against computation time", color=INK, fontsize=11.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, loc="center left")
    fig.tight_layout()
    return _save(fig, results_dir, filename)


# ----------------------------------------------------------------------------
# 6. Interpolation: 1-D vs 3-D, linear vs Chebyshev
# ----------------------------------------------------------------------------

def plot_interpolation_convergence(
    node_sweep: pd.DataFrame,
    domain_sweep: pd.DataFrame,
    results_dir: str,
    filename: str = "interpolation_convergence.png",
    domain_rate_model: str = "stochastic",
) -> str:
    """Three findings about the interpolation step of the incremental proxy.

    Left - convergence in the number of nodes PER STATE DIMENSION.  Colour is
    the method, line style the dimensionality of the state.

    Middle - the same errors against the number of EXACT PRICES the proxy had to
    compute.  This is where the curse of dimensionality shows: the 3-D curves are
    shifted bodily to the right, because n nodes per dimension cost n^3 prices
    instead of n.

    Right - what happens when the interpolant ignores the trade's maturity
    schedule.  With one time domain for the whole FX swap the near leg's
    settlement jump falls inside the domain; a global polynomial cannot
    represent a discontinuity, so Chebyshev loses MORE than linear.  Geometric
    convergence is a property of smooth functions, not a free upgrade.
    """
    fig, (ax_nodes, ax_cost, ax_domain) = _new_figure(1, 3, figsize=(15.5, 4.4))

    colours = {"linear": C_STANDALONE, "chebyshev": C_INCREMENTAL}
    markers = {"linear": "s", "chebyshev": "^"}
    styles = {1: "-", 3: "--"}
    floor = 1e-16

    for (dims, method), sub in node_sweep.groupby(["state_dims", "interp_method"]):
        sub = sub.sort_values("n_state_nodes")
        error = sub["mean_abs_rel_error"].clip(lower=floor)
        label = f"{method}, {int(dims)}-D state"
        kwargs = dict(color=colours[method], marker=markers[method], markersize=7,
                      linewidth=2.0, linestyle=styles[int(dims)], label=label, zorder=3)
        ax_nodes.plot(sub["n_state_nodes"], error, **kwargs)
        ax_cost.plot(sub["mean_exact_evaluations"], error, **kwargs)

    ax_nodes.set_xlabel("interpolation nodes per state dimension",
                        color=INK_SECONDARY, fontsize=9)
    ax_nodes.set_xticks(sorted(node_sweep["n_state_nodes"].unique()))   # node counts are integers
    ax_nodes.set_title("Convergence in the node count", color=INK, fontsize=10.5, loc="left")

    ax_cost.set_xscale("log")
    ax_cost.set_xlabel("exact analytic prices used to build the proxy",
                       color=INK_SECONDARY, fontsize=9)
    ax_cost.set_title("The same errors, priced in repricings",
                      color=INK, fontsize=10.5, loc="left")

    for ax in (ax_nodes, ax_cost):
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor / 4)
        ax.axhline(floor, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=2)
        _style_axes(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_ylabel("mean |relative error|", color=INK_SECONDARY, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    # -- right panel: interpolation domain for the FX swap --------------------
    # The domain sweep is run under both rate models; only one is shown, since
    # the effect is about the trade's maturity schedule rather than the rates.
    domain = domain_sweep
    if "rate_model" in domain.columns:
        domain = domain[domain["rate_model"] == domain_rate_model]

    labels = ["leg-aware domain\n(maturity applied exactly)",
              "single naive domain\n(jump inside the domain)"]
    x = np.arange(2)
    width = 0.34
    for offset, method in zip((-0.5, 0.5), ("linear", "chebyshev")):
        sub = domain[domain["interp_method"] == method].set_index("split_legs")
        values = [float(sub.loc[True, "mean_abs_rel_error"]),
                  float(sub.loc[False, "mean_abs_rel_error"])]
        bars = ax_domain.bar(x + offset * (width + 0.015), values, width,
                             color=colours[method], label=method, zorder=3)
        for rect, value in zip(bars, values):
            ax_domain.annotate(f"{value:.1e}", (rect.get_x() + rect.get_width() / 2, value),
                               textcoords="offset points", xytext=(0, 4), ha="center",
                               fontsize=7.5, color=INK_SECONDARY)

    ax_domain.set_yscale("log")
    ax_domain.set_ylim(1e-13, 3.0)
    ax_domain.set_xticks(x)
    ax_domain.set_xticklabels(labels, fontsize=8)
    ax_domain.set_xlabel("how the incoming FX swap's maturity schedule is handled",
                         color=INK_SECONDARY, fontsize=9)
    ax_domain.set_title("A discontinuity destroys the Chebyshev advantage",
                        color=INK, fontsize=10.5, loc="left")
    _style_axes(ax_domain)
    ax_domain.set_ylabel("mean |relative error|", color=INK_SECONDARY, fontsize=9)
    ax_domain.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    fig.suptitle("Incremental proxy: interpolation in 1-D (deterministic rates) "
                 "vs 3-D (stochastic rates)",
                 color=INK, fontsize=12, x=0.008, ha="left", y=1.02)
    fig.tight_layout()
    return _save(fig, results_dir, filename)


# ----------------------------------------------------------------------------
# 7. Sensitivity to netting-set size
# ----------------------------------------------------------------------------

def plot_error_vs_netting_set_size(
    results: pd.DataFrame,
    results_dir: str,
    filename: str = "error_vs_size.png",
    method: str = "linear",
) -> str:
    """Standalone overcharge against the size of the netting set it ignores.

    Left, in USD: the absolute overcharge grows with the netting set, because a
    bigger book offers more to net against and the standalone charge throws all
    of that away.

    Right, relative to the correct charge: the curves separate by TRADE TYPE,
    not by size.  What breaks the standalone shortcut is the direction of the
    incoming trade, not the size of the portfolio.
    """
    data = results[results["interp_method"] == method].copy()
    fig, (ax_abs, ax_rel) = _new_figure(1, 2, figsize=(11.2, 4.3))

    trades = list(dict.fromkeys(data["trade_key"]))
    markers = ["o", "s", "^", "D"]

    for trade_key, marker in zip(trades, markers):
        sub = data[data["trade_key"] == trade_key].sort_values("n_trades")
        ax_abs.plot(sub["n_trades"], sub["standalone_abs_error"], marker=marker,
                    markersize=8, linewidth=2.0, color=C_STANDALONE, alpha=0.9,
                    label=trade_key, zorder=3)
        ax_rel.plot(sub["n_trades"], sub["standalone_rel_error"].abs(),
                    marker=marker, markersize=8, linewidth=2.0, color=C_TRUE, alpha=0.9,
                    label=trade_key, zorder=3)

    ax_abs.set_yscale("log")
    ax_abs.yaxis.set_major_formatter(_USD_FORMATTER)
    ax_abs.set_ylabel("standalone |error| (USD, log)", color=INK_SECONDARY, fontsize=9)
    ax_abs.set_title("Absolute overcharge grows with the netting set",
                     color=INK, fontsize=10.5, loc="left")

    ax_rel.set_yscale("log")
    ax_rel.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=2)
    ax_rel.annotate("100% of the correct charge", xy=(0.03, 0.80), xycoords="axes fraction",
                    fontsize=7.5, color=INK_MUTED, style="italic")
    ax_rel.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:.0%}" if v >= 0.01 else f"{v:.2%}")
    )
    ax_rel.set_ylabel("standalone |error| / true incremental CVA",
                      color=INK_SECONDARY, fontsize=9)
    ax_rel.set_title("Relative error is set by trade type, not by size",
                     color=INK, fontsize=10.5, loc="left")

    for ax in (ax_abs, ax_rel):
        _style_axes(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_xlabel("number of trades already in the netting set",
                      color=INK_SECONDARY, fontsize=9)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    return _save(fig, results_dir, filename)


# ----------------------------------------------------------------------------
# 8. Old vs new: how much do stochastic rates change the answer?
# ----------------------------------------------------------------------------

def plot_rate_model_comparison(
    comparison: pd.DataFrame,
    results_dir: str,
    filename: str = "rate_model_comparison.png",
) -> str:
    """Deterministic vs stochastic rates, side by side.

    `comparison` must be indexed by (trade_key) with the columns produced by
    `main.summarise_rate_models`.

    Left  - the true incremental CVA itself, so the reader can see how much of
            the change is the charge moving rather than the proxy erring.
    Middle- the standalone proxy's mean relative error under each rate model.
    Right - the same for the incremental proxy, on its own much finer scale.
    """
    fig, (ax_true, ax_sa, ax_inc) = _new_figure(1, 3, figsize=(14.5, 4.4))

    trades = list(comparison["trade_key"])
    x = np.arange(len(trades))
    width = 0.36
    gap = 0.015

    panels = [
        (ax_true, "true_incremental_cva", "mean |true incremental CVA| (USD)",
         "The charge itself moves", True),
        (ax_sa, "standalone_rel_error", "mean |relative error|",
         "Standalone proxy error", False),
        (ax_inc, "incremental_rel_error", "mean |relative error|",
         "Incremental proxy error", False),
    ]

    for ax, stem, ylabel, title, is_money in panels:
        det = comparison[f"{stem}_deterministic"].to_numpy()
        sto = comparison[f"{stem}_stochastic"].to_numpy()
        ax.bar(x - 0.5 * (width + gap), det, width, color=C_STANDALONE,
               label="deterministic rates", zorder=3)
        ax.bar(x + 0.5 * (width + gap), sto, width, color=C_TRUE,
               label="stochastic rates", zorder=3)

        ax.set_yscale("log")
        _style_axes(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(trades, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left")
        if is_money:
            ax.yaxis.set_major_formatter(_USD_FORMATTER)
        else:
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"{v:.0%}" if v >= 0.01 else f"{v:.3%}")
            )

    # One figure-level legend: three identical per-axes legends would collide
    # with the bars in at least one panel.
    handles, labels = ax_true.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.08), labelcolor=INK_SECONDARY)
    fig.suptitle("What stochastic rates change, by incoming trade type "
                 "(averaged over the five netting sets)",
                 color=INK, fontsize=12, x=0.008, ha="left", y=1.02)
    fig.tight_layout()
    return _save(fig, results_dir, filename)


# ----------------------------------------------------------------------------
# 9. Monte Carlo convergence
# ----------------------------------------------------------------------------

def plot_convergence(
    convergence: pd.DataFrame,
    n_paths_production: int,
    results_dir: str,
    filename: str = "mc_convergence.png",
) -> str:
    """CVA against the number of paths, with 95% confidence intervals."""
    fig, (ax_cva, ax_se) = _new_figure(1, 2, figsize=(11.0, 4.2))

    n = convergence["n_paths"].to_numpy()
    cva = convergence["cva"].to_numpy()
    se = convergence["std_error"].to_numpy()

    ax_cva.fill_between(n, (cva - 1.96 * se) / 1e6, (cva + 1.96 * se) / 1e6,
                        color=C_TRUE, alpha=0.15, zorder=2, label="95% confidence interval")
    ax_cva.plot(n, cva / 1e6, color=C_TRUE, marker="o", markersize=7, linewidth=2.0,
                label="CVA estimate", zorder=3)
    ax_cva.axvline(n_paths_production, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=2)
    ax_cva.annotate(f"production: {n_paths_production:,} paths",
                    xy=(n_paths_production, ax_cva.get_ylim()[0]),
                    xytext=(5, 12), textcoords="offset points",
                    fontsize=7.5, color=INK_MUTED, style="italic")
    ax_cva.set_xscale("log")
    ax_cva.set_ylabel("netting-set CVA (USD m)", color=INK_SECONDARY, fontsize=9)
    ax_cva.set_title("CVA estimate against path count", color=INK, fontsize=10.5, loc="left")

    ax_se.plot(n, se, color=C_STANDALONE, marker="s", markersize=7, linewidth=2.0,
               label="observed standard error", zorder=3)
    reference = se[0] * np.sqrt(n[0] / n)
    ax_se.plot(n, reference, color=INK_MUTED, linewidth=1.5, linestyle="--",
               label=r"$1/\sqrt{N}$ reference", zorder=2)
    ax_se.set_xscale("log")
    ax_se.set_yscale("log")
    ax_se.yaxis.set_major_formatter(_USD_FORMATTER)
    ax_se.yaxis.set_minor_formatter(_USD_FORMATTER)   # else minors read "6 x 10^4"
    ax_se.set_ylabel("standard error (USD)", color=INK_SECONDARY, fontsize=9)
    ax_se.set_title("Standard error falls at the Monte Carlo rate",
                    color=INK, fontsize=10.5, loc="left")

    for ax in (ax_cva, ax_se):
        _style_axes(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_xlabel("Monte Carlo paths", color=INK_SECONDARY, fontsize=9)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    return _save(fig, results_dir, filename)


# ----------------------------------------------------------------------------
# 10. Risk decomposition
# ----------------------------------------------------------------------------

def plot_risk_split(
    split: pd.DataFrame,
    near_leg_date: float,
    results_dir: str,
    filename: str = "risk_decomposition.png",
) -> str:
    """Share of instantaneous variance coming from FX vs from rates.

    Stacked bars, one group per (trade, date).  The vertical rule marks the near
    leg's settlement: to its left the swap is a rates product, to its right it
    is an FX product.
    """
    fig, ax = _new_figure(figsize=(10.5, 4.6))

    labels = [f"{r.trade}\n{r.date_y:g}y" for r in split.itertuples()]
    x = np.arange(len(labels))

    # The cross term is SIGNED, so the segments are stacked away from zero in
    # both directions: positive contributions pile upwards, negative ones hang
    # below the axis.  Stacking a negative segment on top of the positives would
    # make a term that REDUCES the variance look like one that adds to it.  The
    # three segments always net to 100% of the total variance.
    components = [
        (split["fx_share"].to_numpy(), C_TRUE, "FX spot"),
        (split["rates_share"].to_numpy(), C_STANDALONE, "interest rates"),
        (split["cross_share"].to_numpy(), C_INCREMENTAL, "FX / rates cross term (signed)"),
    ]

    positive = np.zeros(len(x))
    negative = np.zeros(len(x))
    for values, colour, label in components:
        bottom = np.where(values >= 0, positive, negative)
        ax.bar(x, values, 0.62, bottom=bottom, color=colour, label=label, zorder=3)
        positive = positive + np.where(values >= 0, values, 0.0)
        negative = negative + np.where(values < 0, values, 0.0)

    ax.axhline(0, color=INK_SECONDARY, linewidth=1.2, zorder=4)
    ax.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=2)
    ax.annotate("segments net to 100%", xy=(0.995, 1.0), xycoords=("axes fraction", "data"),
                xytext=(0, 5), textcoords="offset points", ha="right",
                fontsize=7.5, color=INK_MUTED, style="italic")

    # Separate the "before settlement" and "after settlement" blocks per trade.
    for i in range(1, len(split)):
        if split["trade"].iloc[i] != split["trade"].iloc[i - 1]:
            ax.axvline(i - 0.5, color=AXIS, linewidth=1.0, zorder=2)
        elif (split["date_y"].iloc[i] >= near_leg_date
              > split["date_y"].iloc[i - 1]):
            ax.axvline(i - 0.5, color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=2)

    _style_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("share of instantaneous variance", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Where the risk comes from, before and after the near leg settles "
                 "(dashed rule = settlement)",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, ncol=3)

    fig.tight_layout()
    return _save(fig, results_dir, filename)


# ----------------------------------------------------------------------------
# 11. Isotropic vs anisotropic state grids
# ----------------------------------------------------------------------------

def plot_grid_shape(sweep: pd.DataFrame, results_dir: str,
                    filename: str = "grid_shape.png") -> str:
    """Isotropic n x n x n against anisotropic 2 x n x n state grids.

    V is exactly affine in the spot, so two nodes reproduce that dimension to
    machine precision; an isotropic grid spends a factor n on it for nothing.
    Colour is the interpolation method, line style the grid shape.  The right
    panel carries the break-even rule against full re-simulation: above the
    dotted line the proxy is slower than simply repricing the whole book.
    """
    fig, (ax_err, ax_cost, ax_time) = _new_figure(1, 3, figsize=(15.5, 4.4))
    colours = {"linear": C_STANDALONE, "chebyshev": C_INCREMENTAL}
    markers = {"linear": "s", "chebyshev": "^"}
    styles = {"isotropic n x n x n": "--", "anisotropic 2 x n x n": "-"}
    floor = 1e-16

    for (shape, method), sub in sweep.groupby(["grid_shape", "interp_method"]):
        sub = sub.sort_values("n_rate_nodes")
        kwargs = dict(color=colours[method], marker=markers[method], markersize=7,
                      linewidth=2.0, linestyle=styles[shape],
                      label=f"{method}, {shape.split()[0]}", zorder=3)
        ax_err.plot(sub["n_rate_nodes"], sub["mean_abs_rel_error"].clip(lower=floor), **kwargs)
        ax_time.plot(sub["n_rate_nodes"], sub["speedup_vs_resim"], **kwargs)

    ax_err.set_yscale("log")
    ax_err.set_ylim(bottom=floor / 4)
    ax_err.set_ylabel("mean |relative error|", color=INK_SECONDARY, fontsize=9)
    ax_err.set_title("Accuracy is unchanged", color=INK, fontsize=10.5, loc="left")
    # The two grid shapes agree to 7+ significant figures, so each pair of curves
    # lies on top of the other - which is exactly the finding.
    ax_err.annotate("solid and dashed coincide:\nthe spot nodes were buying nothing",
                    xy=(0.04, 0.06), xycoords="axes fraction",
                    fontsize=7.5, color=INK_MUTED, style="italic")

    # Node counts depend only on the grid shape, not on the interpolation method,
    # so this panel draws one line per shape rather than four.
    for shape, sub in sweep[sweep["interp_method"] == "linear"].groupby("grid_shape"):
        sub = sub.sort_values("n_rate_nodes")
        ax_cost.plot(sub["n_rate_nodes"], sub["mean_exact_evaluations"],
                     color=C_TRUE if "iso" in shape else INK_SECONDARY,
                     marker="o", markersize=7, linewidth=2.0, linestyle=styles[shape],
                     label=shape, zorder=3)

    ax_cost.set_yscale("log")
    ax_cost.set_ylabel("exact analytic prices per request", color=INK_SECONDARY, fontsize=9)
    ax_cost.set_title("Node count falls by a factor n/2 (both methods)",
                      color=INK, fontsize=10.5, loc="left")

    ax_time.axhline(1.0, color=INK_MUTED, linewidth=1.2, linestyle=":", zorder=2)
    ax_time.annotate("break-even with full re-simulation", xy=(0.03, 1.0),
                     xycoords=("axes fraction", "data"), xytext=(0, 5),
                     textcoords="offset points", fontsize=7.5, color=INK_MUTED, style="italic")
    ax_time.set_yscale("log")
    _speed = FuncFormatter(lambda v, _: f"{v:g}x")
    ax_time.yaxis.set_major_formatter(_speed)
    ax_time.yaxis.set_minor_formatter(_speed)   # else minors read "3 x 10^0"
    ax_time.set_ylabel("speed-up vs full re-simulation", color=INK_SECONDARY, fontsize=9)
    ax_time.set_title("Where the proxy stops paying for itself",
                      color=INK, fontsize=10.5, loc="left")

    for ax in (ax_err, ax_cost, ax_time):
        _style_axes(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_xticks(sorted(sweep["n_rate_nodes"].unique()))
        ax.set_xlabel("nodes per RATE dimension", color=INK_SECONDARY, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    fig.suptitle("Anisotropic state grid: 2 spot nodes (V is affine in S) "
                 "and n nodes per rate factor",
                 color=INK, fontsize=12, x=0.008, ha="left", y=1.02)
    fig.tight_layout()
    return _save(fig, results_dir, filename)
