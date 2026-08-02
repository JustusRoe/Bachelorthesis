"""
Central configuration for the CVA proxy study.

Every tunable number of the experiment lives here so that the methodology
chapter of the thesis can quote a single file.  Nothing in this module
performs computation - it only declares parameters.

Conventions used throughout the whole package
---------------------------------------------
* The FX pair is EUR/USD quoted as USD per EUR (S = 1.10 means 1 EUR = 1.10 USD).
* USD is the *domestic* (a.k.a. accounting / reporting) currency, EUR is the
  *foreign* currency.  All present values are expressed in USD.
* "Long EUR" (direction = +1) means the bank buys EUR forward against USD.
* Time is measured in years (ACT/365-style decimals), t = 0 is today.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------
# 1. Reproducibility and Monte Carlo size
# ----------------------------------------------------------------------------

SEED = 42               # master seed; every RNG in the project is derived from it
N_PATHS = 20_000        # number of outer Monte Carlo scenarios for the spot rate
N_TIMING_REPEATS = 3    # each pricing request is timed this often; we keep the minimum


# ----------------------------------------------------------------------------
# 2. Exposure time grid
# ----------------------------------------------------------------------------
# The grid is deliberately non-uniform: dense at the short end (many trades in
# the netting sets mature within a year) and coarse in the tail (little happens
# between year 4 and year 5).  t = 0 is included so that the EE profile starts
# at today's mark-to-market.

def build_time_grid() -> np.ndarray:
    """Return the exposure-date grid in years, starting at t = 0."""
    weekly = np.arange(1, 5) * (7.0 / 365.0)      # 1w, 2w, 3w, 4w
    monthly = np.arange(2, 25) * (1.0 / 12.0)     # 2m ... 24m
    quarterly = np.arange(9, 21) * 0.25           # 2.25y ... 5.0y
    grid = np.concatenate([[0.0], weekly, monthly, quarterly])
    return np.unique(np.round(grid, 10))


TIME_GRID = build_time_grid()          # ~40 exposure dates out to 5 years
HORIZON = float(TIME_GRID[-1])         # 5.0 years


# ----------------------------------------------------------------------------
# 3. Market model (Garman-Kohlhagen / GBM spot, two flat curves)
# ----------------------------------------------------------------------------

SPOT_0 = 1.10           # S(0), USD per EUR
RATE_EUR = 0.02         # flat continuously-compounded EUR (foreign) rate
RATE_USD = 0.04         # flat continuously-compounded USD (domestic) rate
FX_VOL = 0.10           # annualised lognormal volatility of EUR/USD

# Risk-neutral drift of the spot under the USD measure is (r_USD - r_EUR).
# With the numbers above the spot drifts *upwards* by ~2% p.a., i.e. long-EUR
# forward positions tend to move into the money.  That is what makes the
# "risk-reducing" (short-EUR) test trade genuinely exposure-reducing.


# ----------------------------------------------------------------------------
# 4. Credit parameters
# ----------------------------------------------------------------------------

RECOVERY_RATE = 0.40    # R in CVA = (1-R) * sum DF * EE * PD; identical for all names


# ----------------------------------------------------------------------------
# 5. Netting-set (counterparty) specification
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class NettingSetSpec:
    """Static description of one counterparty / netting set.

    Attributes
    ----------
    name            : label used in every table and plot
    n_trades        : how many trades the existing portfolio holds
    tenor_min/max   : maturity range (years) the trades are drawn from
    swap_fraction   : share of the trades that are FX swaps rather than forwards
    prob_long_eur   : probability that a forward is long EUR.  Values far from
                      0.5 create a *directional* netting set, which is where the
                      netting benefit (and hence the standalone-proxy error) is
                      largest.
    hazard          : constant hazard rate lambda of the counterparty
    notional_range  : uniform range for the EUR notional of each trade
    """

    name: str
    n_trades: int
    tenor_min: float
    tenor_max: float
    swap_fraction: float
    prob_long_eur: float
    hazard: float
    notional_range: tuple[float, float] = (5e6, 50e6)


NETTING_SET_SPECS: list[NettingSetSpec] = [
    NettingSetSpec("NS1_small_short",       10, 1 / 52, 1.0, 0.20, 0.90, 0.0050),
    NettingSetSpec("NS2_medium_mixed",      20, 1 / 12, 3.0, 0.30, 0.70, 0.0100),
    NettingSetSpec("NS3_balanced",          30, 1 / 52, 5.0, 0.33, 0.50, 0.0200),
    NettingSetSpec("NS4_long_dated",        40, 0.50,   5.0, 0.35, 0.75, 0.0350),
    NettingSetSpec("NS5_large_directional", 50, 1 / 52, 5.0, 0.36, 0.95, 0.0500),
]

# Discrete tenor menu the existing trades are drawn from (years).  Using a menu
# rather than a continuous draw keeps the portfolios readable in the trade
# listing that goes into the appendix of the thesis.
TENOR_MENU = np.array(
    [1 / 52, 2 / 52, 1 / 12, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
)

# Existing trades are *seasoned*: their strike was struck some time ago, so it
# differs from today's fair forward.  We emulate that with a random relative
# offset, which gives the portfolios a non-zero mark-to-market at t = 0.
STRIKE_SEASONING = 0.03     # +/- 3% around the fair forward rate


# ----------------------------------------------------------------------------
# 6. Incoming ("new") trades that we want to price
# ----------------------------------------------------------------------------

INCOMING_FORWARD_TENOR = 3.0        # plain new forward: 3y
INCOMING_FORWARD_NOTIONAL = 50e6    # EUR 50m, long EUR

INCOMING_SWAP_NEAR = 1.0            # new FX swap: sell EUR at 1y ...
INCOMING_SWAP_FAR = 3.0             # ... buy EUR back at 3y
INCOMING_SWAP_NOTIONAL = 50e6       # EUR 50m on both legs

INCOMING_SMALL_TENOR = 0.5          # small reference trade: 6m
INCOMING_SMALL_NOTIONAL = 5e6       # EUR 5m, long EUR

# The risk-reducing trade is sized *relative to the netting set it is shown to*:
# notional = OFFSET_FRACTION * |net signed EUR notional of the set|, traded in
# the opposite direction, floored so that it always has a visible effect.
INCOMING_OFFSET_TENOR = 3.0
OFFSET_FRACTION = 0.50
OFFSET_MIN_NOTIONAL = 25e6
OFFSET_MAX_NOTIONAL = 300e6


# ----------------------------------------------------------------------------
# 7. Interpolation grid used by BOTH proxies
# ----------------------------------------------------------------------------
# Both proxies avoid full repricing of the incoming trade on every path.  They
# build an interpolant of the pricing function V(t, S) on a coarse tensor grid
# of (time, spot) nodes, evaluate the *exact* analytic price only at those
# nodes, and then read the interpolant off at every (t_i, S_path) pair.
#
# NOTE on the choice of a 2-D grid.  For a linear FX product under deterministic
# rates the value is exactly affine in the spot,
#       V(t, S) = N * DF_EUR(t,T) * S  -  N * K * DF_USD(t,T),
# so interpolating in S alone would be exact to machine precision.  Coarsening
# the *time* dimension as well introduces a genuine approximation (the discount
# factors are exponential in t, and a swap's near leg drops out inside the
# domain), which is what makes the linear-vs-Chebyshev comparison meaningful.

N_TIME_NODES = 6        # interpolation nodes in the time dimension
N_SPOT_NODES = 5        # interpolation nodes in the spot dimension
SPOT_GRID_BUFFER = 0.02  # widen the spot node range by 2% beyond the simulated
                         # min/max so that no path requires extrapolation
                         # (extrapolating a Chebyshev polynomial is unstable and
                         # would make the comparison unfair rather than informative)

INTERP_METHODS = ("linear", "chebyshev")
DEFAULT_INTERP_METHOD = "linear"    # base case of the thesis

# Node-count sweep used for the convergence study (sensitivity (c)).
NODE_SWEEP = (3, 4, 6, 8, 12, 16)


# ----------------------------------------------------------------------------
# 8. Output locations
# ----------------------------------------------------------------------------

RESULTS_DIR = "results"
MAIN_RESULTS_CSV = "cva_results.csv"
SENSITIVITY_CSV = "sensitivity_results.csv"
PORTFOLIO_CSV = "portfolios.csv"
