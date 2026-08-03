"""
Central configuration for the CVA proxy study.

Every tunable number of the experiment lives here so that the methodology
chapter of the thesis can quote a single file.  Nothing in this module performs
computation - it only declares parameters.

Each parameter is labelled as one of:
    [MARKET CONVENTION]  a standard of the market, not something we invented
    [MODELLING CHOICE]   our choice; a different value would be defensible
    [SYNTHETIC]          a made-up number standing in for real data

Conventions used throughout the whole package
---------------------------------------------
* The FX pair is EUR/USD quoted as USD per EUR (S = 1.10 means 1 EUR = 1.10 USD).
* USD is the *domestic* (a.k.a. accounting / reporting) currency, EUR is the
  *foreign* currency.  All present values are expressed in USD.
* "Long EUR" (direction = +1) means the bank buys EUR forward against USD.
* Time is measured in years (ACT/365-style decimals), t = 0 is today.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curves import ZeroCurve

# ----------------------------------------------------------------------------
# 1. Reproducibility and Monte Carlo size
# ----------------------------------------------------------------------------

SEED = 42               # master seed; every RNG in the project is derived from it

# [MODELLING CHOICE] Path count. Reduced from the 20,000 of the single-factor
# model: the three-factor simulation and the 3-D interpolation grid are both
# materially more expensive, and the validation suite reports Monte Carlo
# standard errors so the resulting noise is quantified rather than assumed away.
N_PATHS = 10_000

N_TIMING_REPEATS = 3    # each pricing request is timed this often; we keep the minimum


# ----------------------------------------------------------------------------
# 2. Exposure time grid
# ----------------------------------------------------------------------------
# Deliberately non-uniform: dense at the short end (many trades in the netting
# sets mature within a year) and coarse in the tail.  t = 0 is included so that
# the EE profile starts at today's mark-to-market.

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
# 3. Which market model to run
# ----------------------------------------------------------------------------
# "stochastic"    : the full three-factor model (Hull-White USD, Hull-White EUR
#                   with quanto adjustment, lognormal FX spot).
# "deterministic" : sigma_d = sigma_f = 0.  Rates collapse to plain forward
#                   discounting off the initial curves and only the FX spot is
#                   random, which is exactly the single-factor base model.
#
# The two modes share EVERYTHING else - same curves, same seed, same trades - so
# that any difference in the results is attributable to rate volatility alone.
RATE_MODEL = "stochastic"
RATE_MODELS = ("deterministic", "stochastic")


# ----------------------------------------------------------------------------
# 4. Initial zero curves
# ----------------------------------------------------------------------------
# [SYNTHETIC] Nelson-Siegel coefficients. Shaped like a plausible late-cycle
# environment: mildly inverted at the front, flattening out further along.  The
# EUR-USD level difference is kept near 2%, matching the flat 2%/4% curves of
# the single-factor base model, so the two studies stay comparable.

CURVE_USD = ZeroCurve(b0=0.0400, b1=0.0050, b2=-0.0080, tau=1.5, name="USD")
CURVE_EUR = ZeroCurve(b0=0.0220, b1=0.0040, b2=-0.0060, tau=1.5, name="EUR")

# [MODELLING CHOICE] Flat-curve equivalents at the base model's constant rates.
# Used only by the validation suite, to prove that flat curves plus zero rate
# volatility reproduce the single-factor model exactly.
CURVE_USD_FLAT = ZeroCurve(b0=0.04, name="USD flat")
CURVE_EUR_FLAT = ZeroCurve(b0=0.02, name="EUR flat")

SPOT_0 = 1.10           # [SYNTHETIC] S(0), USD per EUR


# ----------------------------------------------------------------------------
# 5. Three-factor model parameters
# ----------------------------------------------------------------------------

# [MODELLING CHOICE] Hull-White mean reversion.  One-factor calibrations
# typically land between 0.01 and 0.10; EUR usually calibrates lower than USD.
MEAN_REVERSION_USD = 0.05       # a_d
MEAN_REVERSION_EUR = 0.03       # a_f

# [MODELLING CHOICE] Hull-White absolute (normal) rate volatilities.  The usual
# calibrated range for a one-factor model is roughly 50-120 bp.
VOL_RATE_USD = 0.0080           # sigma_d, 80 bp
VOL_RATE_EUR = 0.0060           # sigma_f, 60 bp

# [SYNTHETIC, market-plausible] Lognormal FX volatility.  Deliberately UNCHANGED
# from the single-factor base model so that old-vs-new differences come from the
# rates, not from a different FX vol.
VOL_FX = 0.10                   # sigma_S

# [MODELLING CHOICE] Correlations between the three Brownian drivers.
# Ordering of the factors everywhere in the code is (S, x_d, x_f).
#   rho_dS < 0 : higher USD rates strengthen the USD, so S (USD per EUR) falls.
#   rho_fS > 0 : higher EUR rates strengthen the EUR, so S rises.  This is the
#                economically conventional sign for a rate-differential-driven
#                pair; the quanto test deliberately also runs it at -0.40.
#   rho_df > 0 : USD and EUR rates co-move.
CORR_USD_FX = -0.15             # rho_dS
CORR_EUR_FX = 0.25              # rho_fS
CORR_USD_EUR = 0.35             # rho_df


def correlation_matrix(rho_ds: float = CORR_USD_FX,
                       rho_fs: float = CORR_EUR_FX,
                       rho_df: float = CORR_USD_EUR) -> np.ndarray:
    """The 3x3 correlation matrix in factor order (S, x_d, x_f).

    The FX factor is placed FIRST on purpose.  The Cholesky factor of a
    correlation matrix is lower triangular with L[0,0] = 1, so the FX Brownian
    increment equals its own independent normal draw untouched.  That keeps the
    FX paths bit-identical to the single-factor base model when the rate
    volatilities are switched off, which is what makes the degeneracy test
    meaningful.
    """
    return np.array([
        [1.0,     rho_ds,  rho_fs],
        [rho_ds,  1.0,     rho_df],
        [rho_fs,  rho_df,  1.0],
    ])


# [MODELLING CHOICE] Whether to apply the quanto drift adjustment to the foreign
# short rate.  Always True in production; the validation suite flips it to False
# to demonstrate what the adjustment is doing.
APPLY_QUANTO_ADJUSTMENT = True


# ----------------------------------------------------------------------------
# 6. Discounting inside the CVA integral
# ----------------------------------------------------------------------------
# "pathwise" : EE*(t_i) = E[ D_d(0,t_i) * max(V(t_i),0) ], the discount factor
#              kept INSIDE the expectation.  Correct under stochastic rates,
#              where the discount factor and the exposure are correlated.
# "curve"    : P_d(0,t_i) * E[ max(V(t_i),0) ], the textbook separable form used
#              by the single-factor base model.
#
# The two agree exactly when rates are deterministic.  Both are computed so the
# difference can be reported as a result in its own right.
DISCOUNTING = "pathwise"
DISCOUNTING_MODES = ("pathwise", "curve")


# ----------------------------------------------------------------------------
# 7. Credit parameters
# ----------------------------------------------------------------------------

# [MARKET CONVENTION] Recovery rate for senior unsecured exposure.
RECOVERY_RATE = 0.40

# [MODELLING CHOICE] Hazard-rate term structure per counterparty, piecewise
# constant with breaks at 1y and 3y.  Investment-grade credit curves are
# normally upward sloping; the averages are set close to the flat hazard rates
# used by the single-factor base model.
HAZARD_PILLARS = (1.0, 3.0)     # the rate applies on (0,1], (1,3], (3,inf)


# ----------------------------------------------------------------------------
# 8. Netting-set (counterparty) specification
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
    hazard          : piecewise-constant hazard rates on the pillars above
    notional_range  : uniform range for the EUR notional of each trade
    """

    name: str
    n_trades: int
    tenor_min: float
    tenor_max: float
    swap_fraction: float
    prob_long_eur: float
    hazard: tuple[float, ...]
    notional_range: tuple[float, float] = (5e6, 50e6)


NETTING_SET_SPECS: list[NettingSetSpec] = [
    NettingSetSpec("NS1_small_short",       10, 1 / 52, 1.0, 0.20, 0.90, (0.0035, 0.0050, 0.0065)),
    NettingSetSpec("NS2_medium_mixed",      20, 1 / 12, 3.0, 0.30, 0.70, (0.0070, 0.0100, 0.0130)),
    NettingSetSpec("NS3_balanced",          30, 1 / 52, 5.0, 0.33, 0.50, (0.0140, 0.0200, 0.0260)),
    NettingSetSpec("NS4_long_dated",        40, 0.50,   5.0, 0.35, 0.75, (0.0250, 0.0350, 0.0450)),
    NettingSetSpec("NS5_large_directional", 50, 1 / 52, 5.0, 0.36, 0.95, (0.0360, 0.0500, 0.0640)),
]

# Discrete tenor menu the existing trades are drawn from (years).
TENOR_MENU = np.array(
    [1 / 52, 2 / 52, 1 / 12, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
)

# Existing trades are *seasoned*: their strike was struck some time ago, so it
# differs from today's fair forward.
STRIKE_SEASONING = 0.03     # +/- 3% around the fair forward rate


# ----------------------------------------------------------------------------
# 9. Incoming ("new") trades that we want to price
# ----------------------------------------------------------------------------

INCOMING_FORWARD_TENOR = 3.0        # plain new forward: 3y
INCOMING_FORWARD_NOTIONAL = 50e6    # EUR 50m, long EUR

INCOMING_SWAP_NEAR = 1.0            # new FX swap: sell EUR at 1y ...
INCOMING_SWAP_FAR = 3.0             # ... buy EUR back at 3y
INCOMING_SWAP_NOTIONAL = 50e6       # EUR 50m on both legs

INCOMING_SMALL_TENOR = 0.5          # small reference trade: 6m
INCOMING_SMALL_NOTIONAL = 5e6       # EUR 5m, long EUR

INCOMING_OFFSET_TENOR = 3.0
OFFSET_FRACTION = 0.50
OFFSET_MIN_NOTIONAL = 25e6
OFFSET_MAX_NOTIONAL = 300e6


# ----------------------------------------------------------------------------
# 10. Interpolation grid used by BOTH proxies
# ----------------------------------------------------------------------------
# Both proxies avoid full repricing of the incoming trade on every path.  They
# build an interpolant of the pricing function on a coarse tensor grid over
#
#       (time)  x  (state variables)
#
# and evaluate the EXACT analytic price only at those nodes.
#
# The state is one-dimensional under deterministic rates (the FX spot alone) and
# THREE-dimensional under the three-factor model (S, x_d, x_f).  The node count
# therefore grows as n^3 rather than n - the curse of dimensionality is the
# central cost of moving to stochastic rates, and is reported explicitly.
#
# NOTE: the trade value stays EXACTLY affine in S even under stochastic rates,
#   V = N * [ S * P_f(t,T) - K * P_d(t,T) ],
# so the spot dimension is reproduced exactly by 2 nodes.  All the genuine
# curvature sits in x_d and x_f (where the bond prices are exponential) and in
# time.  The sweeps quantify this.

N_TIME_NODES = 6        # interpolation nodes in the time dimension
N_STATE_NODES = 5       # nodes PER STATE DIMENSION
STATE_GRID_BUFFER = 0.02  # widen every node range by 2% beyond the simulated
                          # min/max so that no path requires extrapolation

INTERP_METHODS = ("linear", "chebyshev")
DEFAULT_INTERP_METHOD = "linear"    # base case of the thesis

# Node-count sweep for the convergence study.  Capped at 6 because the grid
# holds N_TIME_NODES * n^3 nodes: at n = 8 that is already 3,072 exact prices
# per leg.
NODE_SWEEP = (2, 3, 4, 5, 6)

# Time nodes used DURING THE SWEEP only.  The proxy's total error is the sum of
# a time-dimension error and a state-dimension error; at the production setting
# of 6 time nodes the former dominates and would flatten the very curves the
# sweep is meant to expose.  Raising the time nodes for the sweep isolates the
# state dimensions, which is what the 1-D vs 3-D comparison is about.
SWEEP_TIME_NODES = 12

# Paths are processed in blocks when evaluating an interpolant, so that the
# intermediate tensors of a 3-D Chebyshev contraction stay bounded regardless of
# the node count.
EVAL_CHUNK_PATHS = 2_500


# ----------------------------------------------------------------------------
# 11. Validation suite
# ----------------------------------------------------------------------------

VALIDATION_PATHS = 40_000                       # more paths: these are cheap checks
VALIDATION_TENORS = (0.5, 1.0, 2.0, 3.0, 5.0)   # maturities checked by tests 1 and 2
VALIDATION_TOLERANCE_SIGMAS = 3.0               # pass if within 3 Monte Carlo std errors
QUANTO_TEST_CORRELATIONS = (0.0, -0.40)         # rho_fS values for the quanto test
CONVERGENCE_PATHS = (1_000, 2_500, 5_000, 10_000, 20_000, 40_000)


# ----------------------------------------------------------------------------
# 12. Output locations
# ----------------------------------------------------------------------------

RESULTS_DIR = "results"
MAIN_RESULTS_CSV = "cva_results.csv"
SENSITIVITY_CSV = "sensitivity_results.csv"
PORTFOLIO_CSV = "portfolios.csv"
VALIDATION_CSV = "validation_results.csv"
COMPARISON_CSV = "rate_model_comparison.csv"
RISK_SPLIT_CSV = "risk_decomposition.csv"
