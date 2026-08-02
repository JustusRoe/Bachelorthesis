"""
All figures produced by the study.

Each function saves one PNG into the results directory and returns its path.
Nothing here computes CVA - the plotting layer only consumes the DataFrames and
arrays produced by the rest of the package, so that the numerical results and
their presentation stay separable.

Design notes
------------
The figures are meant for a printed thesis, so they commit to a single light
surface.  The categorical colours are a validated three-slot palette (blue /
orange / aqua) that stays distinguishable under deuteranopia and tritanopia;
signed errors use a blue-red diverging pair with a neutral zero line, because
there the polarity (over- vs undercharging) IS the message.  Every chart carries
a legend, and the accompanying CSV is the table view.
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

# Sequential blue ramp (steps 300-700 of the blue scale), used where the five
# netting sets must be ordered by size.  Ordered data takes a one-hue ramp, not
# five categorical hues; these steps stay legible on the light surface.
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

    # -- all netting sets -----------------------------------------------------
    for colour, (name, ee) in zip(SEQ_BLUE, ee_by_set.items()):
        ax_all.plot(times, ee / 1e6, color=colour, linewidth=2.0, label=name, zorder=3)
    _style_axes(ax_all)
    ax_all.set_title("Expected exposure by netting set", color=INK, fontsize=11, loc="left")
    ax_all.set_xlabel("time (years)", color=INK_SECONDARY, fontsize=9)
    ax_all.set_ylabel("EE (USD m)", color=INK_SECONDARY, fontsize=9)
    ax_all.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    # -- focus set: EE against PFE -------------------------------------------
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
# 2. Standalone vs incremental vs true
# ----------------------------------------------------------------------------

def plot_cva_comparison(
    results: pd.DataFrame,
    results_dir: str,
    filename: str = "cva_comparison.png",
    method: str = "linear",
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

        # Direct-label the negative bars: this is the key qualitative result and
        # it also supplies the "relief" that the aqua slot needs on a light surface.
        # Reserve headroom below the lowest bar so the label cannot collide with
        # the axis.
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
        f"CVA charge for the incoming trade  ({method} interpolation)  -  "
        "negative bars = the trade REDUCES counterparty CVA",
        color=INK, fontsize=11.5, x=0.008, ha="left", y=1.02,
    )
    fig.tight_layout()
    return _save(fig, results_dir, filename)


# ----------------------------------------------------------------------------
# 3. Signed error: over- vs undercharging
# ----------------------------------------------------------------------------

def plot_signed_errors(
    results: pd.DataFrame,
    results_dir: str,
    filename: str = "signed_errors.png",
    method: str = "linear",
) -> str:
    """One panel per proxy; bars above zero overcharge, below zero undercharge.

    The two panels deliberately carry their OWN vertical scale (they are two
    charts, not one chart with two axes): the standalone error is measured in
    hundreds of thousands of USD, the incremental error in tens.  Forcing them
    onto a shared scale would render the incremental panel invisible.
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

        # Symmetric-log scale: within one panel the errors span four orders of
        # magnitude (a few USD on the small forward, millions on the offsetting
        # trade).  symlog keeps the sign - which is the whole point of the chart -
        # while still making the small bars visible; a plain linear scale would
        # flatten everything except the offsetting trades to an invisible line.
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
        Line2D([0], [0], color=C_OVER, lw=7, label="signed error > 0  =  OVERCHARGING (trade lost to competitors)"),
        Line2D([0], [0], color=C_UNDER, lw=7, label="signed error < 0  =  UNDERCHARGING (loss borne by the bank)"),
    ]
    fig.legend(handles=legend, frameon=False, fontsize=9, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.04), labelcolor=INK_SECONDARY)
    fig.suptitle("Signed pricing error vs the true incremental CVA",
                 color=INK, fontsize=12, x=0.008, ha="left", y=1.0)
    fig.tight_layout()
    return _save(fig, results_dir, filename)


# ----------------------------------------------------------------------------
# 4. Accuracy against computation time
# ----------------------------------------------------------------------------

def plot_accuracy_vs_time(
    scatter: pd.DataFrame,
    results_dir: str,
    filename: str = "accuracy_vs_time.png",
) -> str:
    """Cost of a pricing request against the error it buys.

    Colour carries the METHOD FAMILY (three validated slots); within the
    incremental family, filled vs open markers separate the two interpolation
    variants.  Identity therefore never rests on colour alone.  Both axes are
    logarithmic because the quantities span many decades.

    The full re-simulation points are exact by construction (error = 0), so they
    are drawn on a floor line rather than at a fictitious small error.
    """
    fig, ax = _new_figure(figsize=(8.2, 5.2))

    floor = 1e-16       # visual floor for "exact" and for machine-precision hits

    # (label, colour, marker, filled)
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
        face = colour if filled else "none"
        ax.scatter(
            sub["seconds"] * 1000.0, sub["abs_rel_error"].clip(lower=floor),
            s=95, facecolor=face, edgecolor=colour if not filled else SURFACE,
            linewidth=1.8, marker=marker,
            label=label + (" - exact by construction" if "full" in label else ""),
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(floor / 10, 10)
    _style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)

    # Plain millisecond ticks read better than 3x10^-2 style exponents; the
    # data spans less than two decades, so the decade ticks alone are too sparse
    # and the 2/3/5 minor ticks are labelled as well.
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
# 5. Interpolation convergence: linear vs Chebyshev
# ----------------------------------------------------------------------------

def plot_interpolation_convergence(
    node_sweep: pd.DataFrame,
    domain_sweep: pd.DataFrame,
    results_dir: str,
    filename: str = "interpolation_convergence.png",
) -> str:
    """Two findings about the interpolation step of the incremental proxy.

    Left - convergence in the number of time nodes.  Piecewise-linear
    interpolation is second order in the node spacing, so the error falls like a
    straight line on a log-log view; Chebyshev interpolation of an analytic
    function converges geometrically and reaches double precision after about
    eight nodes.  That is the cost argument: the same accuracy for far fewer
    exact repricings.

    Right - what happens when the interpolant ignores the trade's maturity
    schedule.  With one time domain for the whole FX swap, the near leg's
    settlement jump falls inside the domain.  Both methods then lose roughly ten
    percent, and Chebyshev loses MORE than linear, because a global polynomial
    cannot represent a discontinuity while a piecewise interpolant can at least
    localise it.  Geometric convergence is a property of smooth functions, not a
    free upgrade.
    """
    fig, (ax_nodes, ax_domain) = _new_figure(1, 2, figsize=(11.4, 4.4))

    styles = {"linear": (C_STANDALONE, "s"), "chebyshev": (C_INCREMENTAL, "^")}
    floor = 1e-16

    # -- left panel: convergence ---------------------------------------------
    for method, (colour, marker) in styles.items():
        sub = node_sweep[node_sweep["interp_method"] == method].sort_values("n_time_nodes")
        if sub.empty:
            continue
        ax_nodes.plot(sub["n_time_nodes"], sub["mean_abs_rel_error"].clip(lower=floor),
                      color=colour, marker=marker, markersize=8, linewidth=2.0,
                      label=method, zorder=3)

    ax_nodes.axhline(floor, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=2)
    ax_nodes.annotate("double-precision floor", xy=(0.02, 0.09), xycoords="axes fraction",
                      fontsize=7.5, color=INK_MUTED, style="italic")
    ax_nodes.set_yscale("log")
    ax_nodes.set_ylim(bottom=floor / 4)
    ax_nodes.set_xlabel("interpolation nodes in the time dimension",
                        color=INK_SECONDARY, fontsize=9)
    ax_nodes.set_title("Convergence in the number of nodes",
                       color=INK, fontsize=10.5, loc="left")

    # -- right panel: interpolation domain for the FX swap --------------------
    labels = ["leg-aware domain\n(maturity applied exactly)", "single naive domain\n(jump inside the domain)"]
    x = np.arange(2)
    width = 0.34
    for offset, (method, (colour, _)) in zip((-0.5, 0.5), styles.items()):
        sub = domain_sweep[domain_sweep["interp_method"] == method].set_index("split_legs")
        values = [sub.loc[True, "mean_abs_rel_error"], sub.loc[False, "mean_abs_rel_error"]]
        bars = ax_domain.bar(x + offset * (width + 0.015), values, width,
                             color=colour, label=method, zorder=3)
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

    for ax in (ax_nodes, ax_domain):
        _style_axes(ax)
        ax.set_ylabel("mean |relative error|", color=INK_SECONDARY, fontsize=9)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    ax_nodes.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)

    fig.suptitle("Incremental proxy: linear vs Chebyshev interpolation",
                 color=INK, fontsize=12, x=0.008, ha="left", y=1.02)
    fig.tight_layout()
    return _save(fig, results_dir, filename)


# ----------------------------------------------------------------------------
# 6. Sensitivity to netting-set size
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

    Right, relative to the charge itself: the curves separate by TRADE TYPE, not
    by size.  A risk-adding forward is mispriced by a fraction of a percent
    whatever the book looks like, while a risk-reducing trade is mispriced by
    ~200% in every single netting set.  What breaks the standalone shortcut is
    the direction of the incoming trade, not the size of the portfolio.
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
    ax_rel.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}" if v >= 0.01 else f"{v:.2%}"))
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
