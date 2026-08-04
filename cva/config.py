"""
Every tunable parameter of the study, so the methodology chapter can quote one file.

Each value is labelled [CONVENTION] (a market standard), [CHOICE] (ours; another
value would be defensible) or [SYNTHETIC] (a made-up stand-in for real data).

Conventions used throughout the package:
  * EUR/USD quoted as USD per EUR; USD is domestic, EUR foreign; PVs are in USD.
  * "Long EUR" (direction = +1) means the bank buys EUR forward against USD.
  * Time is in years, t = 0 is today.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# This module holds DATA only and deliberately imports nothing from the package,
# so it can be read top to bottom without following any references.  The curve
# objects themselves are assembled from these parameters in `model.py`.

# ---------------------------------------------------------------- simulation
SEED = 42
N_PATHS = 20_000        # [CHOICE] three factors and an n^3 grid cost more than
                        # the single-factor model, but this is the path count
                        # the thesis reports against.  MC standard errors are
                        # reported so the noise is quantified.
N_TIMING_REPEATS = 3    # each pricing request is timed this often; keep the minimum


def build_time_grid() -> np.ndarray:
    """Exposure dates in years, starting at 0.

    Non-uniform: dense at the short end where many trades mature, coarse in the tail.
    """
    return np.unique(np.round(np.concatenate([
        [0.0],
        np.arange(1, 5) * (7.0 / 365.0),    # weekly to 1m
        np.arange(2, 25) / 12.0,            # monthly to 2y
        np.arange(9, 21) * 0.25,            # quarterly to 5y
    ]), 10))


TIME_GRID = build_time_grid()            # ~40 dates
HORIZON = float(TIME_GRID[-1])           # 5.0 years

# --------------------------------------------------------------- rate model
# "stochastic"    : the full three-factor model.
# "deterministic" : sigma_d = sigma_f = 0, i.e. the single-factor base model.
# The modes share curves, FX vol, seed and trades, so any difference between
# them is attributable to rate volatility alone.
RATE_MODEL = "stochastic"
RATE_MODELS = ("deterministic", "stochastic")

# ------------------------------------------------------------- initial curves
# [SYNTHETIC] Nelson-Siegel coefficients shaped like a plausible late-cycle
# environment.  The EUR-USD level gap is kept near 2%, matching the flat 2%/4%
# curves of the base model, so the two studies stay comparable.
# `model.CURVE_USD` / `model.CURVE_EUR` are the ZeroCurve objects built from these.
CURVE_USD_PARAMS = dict(b0=0.0400, b1=0.0050, b2=-0.0080, tau=1.5, name="USD")
CURVE_EUR_PARAMS = dict(b0=0.0220, b1=0.0040, b2=-0.0060, tau=1.5, name="EUR")

SPOT_0 = 1.10           # [SYNTHETIC] S(0), USD per EUR

# ------------------------------------------------------- three-factor params
MEAN_REVERSION_USD = 0.05   # [CHOICE] a_d; 1F-HW calibrations sit in 0.01-0.10
MEAN_REVERSION_EUR = 0.03   # [CHOICE] a_f; EUR usually calibrates lower than USD
VOL_RATE_USD = 0.0080       # [CHOICE] sigma_d, 80bp; typical range 50-120bp
VOL_RATE_EUR = 0.0060       # [CHOICE] sigma_f, 60bp
VOL_FX = 0.10               # [SYNTHETIC] sigma_S; UNCHANGED from the base model
                            # so old-vs-new differences come from the rates

# [CHOICE] Correlations, factor order (S, x_d, x_f):
#   rho_dS < 0  higher USD rates strengthen the USD, so S (USD per EUR) falls
#   rho_fS > 0  higher EUR rates strengthen the EUR, so S rises
#   rho_df > 0  USD and EUR rates co-move
CORR_USD_FX = -0.15
CORR_EUR_FX = 0.25
CORR_USD_EUR = 0.35


def correlation_matrix(rho_ds=CORR_USD_FX, rho_fs=CORR_EUR_FX, rho_df=CORR_USD_EUR):
    """3x3 correlation in factor order (S, x_d, x_f).

    FX is placed FIRST on purpose: the Cholesky factor of a correlation matrix
    has L[0,0] = 1, so the FX Brownian increment equals its own draw untouched,
    which is what makes the zero-rate-vol degeneracy test exact.
    """
    return np.array([[1.0, rho_ds, rho_fs],
                     [rho_ds, 1.0, rho_df],
                     [rho_fs, rho_df, 1.0]])


# [CHOICE] Always True in production; the validation suite flips it to show what
# the quanto drift adjustment is doing.
APPLY_QUANTO_ADJUSTMENT = True

# ------------------------------------------------------------- CVA mechanics
# "pathwise" keeps the discount factor inside the expectation (correct under
# stochastic rates); "curve" is the separable textbook form of the base model.
# They agree exactly when rates are deterministic.
DISCOUNTING = "pathwise"
DISCOUNTING_MODES = ("pathwise", "curve")

RECOVERY_RATE = 0.40        # [CONVENTION] senior unsecured
HAZARD_PILLARS = (1.0, 3.0)  # hazard is constant on (0,1], (1,3], (3,inf)

# ----------------------------------------------------------------- portfolio


@dataclass(frozen=True)
class NettingSetSpec:
    """One counterparty.

    prob_long_eur far from 0.5 makes the set directional, which is where the
    netting benefit - and hence the standalone-proxy error - is largest.
    """

    name: str
    n_trades: int
    tenor_min: float
    tenor_max: float
    swap_fraction: float
    prob_long_eur: float
    hazard: tuple[float, ...]           # [CHOICE] upward sloping, as IG curves are
    notional_range: tuple[float, float] = (5e6, 50e6)


NETTING_SET_SPECS: list[NettingSetSpec] = [
    NettingSetSpec("NS1_small_short",       10, 1 / 52, 1.0, 0.20, 0.90, (0.0035, 0.0050, 0.0065)),
    NettingSetSpec("NS2_medium_mixed",      20, 1 / 12, 3.0, 0.30, 0.70, (0.0070, 0.0100, 0.0130)),
    NettingSetSpec("NS3_balanced",          30, 1 / 52, 5.0, 0.33, 0.50, (0.0140, 0.0200, 0.0260)),
    NettingSetSpec("NS4_long_dated",        40, 0.50,   5.0, 0.35, 0.75, (0.0250, 0.0350, 0.0450)),
    NettingSetSpec("NS5_large_directional", 50, 1 / 52, 5.0, 0.36, 0.95, (0.0360, 0.0500, 0.0640)),
]

TENOR_MENU = np.array([1 / 52, 2 / 52, 1 / 12, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
STRIKE_SEASONING = 0.03     # legacy trades struck +/- 3% off today's fair forward

# ------------------------------------------------------------ incoming trades
INCOMING_FORWARD_TENOR = 3.0
INCOMING_FORWARD_NOTIONAL = 50e6
INCOMING_SWAP_NEAR = 1.0            # sell EUR at 1y ...
INCOMING_SWAP_FAR = 3.0             # ... buy it back at 3y
INCOMING_SWAP_NOTIONAL = 50e6
INCOMING_SMALL_TENOR = 0.5
INCOMING_SMALL_NOTIONAL = 5e6

INCOMING_OFFSET_TENOR = 3.0         # risk-reducing trade, sized from the CVA delta
OFFSET_FRACTION = 0.50
OFFSET_MIN_NOTIONAL = 25e6
OFFSET_MAX_NOTIONAL = 300e6

# ------------------------------------------------------------- interpolation
# The grid spans (time) x (state).  The state is 1-D under deterministic rates
# and 3-D under the three-factor model, so the node count grows as n^3.
#
# Note: the trade value stays EXACTLY affine in S even under stochastic rates,
# so the spot dimension is reproduced by 2 nodes.  All genuine curvature is in
# x_d, x_f (bond prices are exponential) and in time.
N_TIME_NODES = 6
N_STATE_NODES = 5           # per state dimension
STATE_GRID_BUFFER = 0.02    # widen node ranges 2% past the simulated min/max so
                            # nothing extrapolates

INTERP_METHODS = ("linear", "chebyshev")
DEFAULT_INTERP_METHOD = "linear"

NODE_SWEEP = (2, 3, 4, 5, 6)    # capped at 6: the grid holds n_time * n^3 nodes

# ANISOTROPIC grids.  V is exactly affine in S, so two nodes reproduce the spot
# dimension to machine precision and an isotropic n^3 grid wastes a factor n on
# it.  `anisotropic_nodes(n)` spends 2 nodes on the spot and n on each rate
# factor, giving 2*n^2 state nodes instead of n^3.
SPOT_NODES_ANISOTROPIC = 2


def anisotropic_nodes(n_rate: int, n_spot: int = SPOT_NODES_ANISOTROPIC) -> dict:
    """Per-dimension node counts for a grid that is coarse in S and fine in the rates."""
    return {"spot": n_spot, "x_d": n_rate, "x_f": n_rate}
SWEEP_TIME_NODES = 12           # sweep-only.  At the production 6 the
                                # time-dimension error dominates and would
                                # flatten the very curves the sweep must expose.
EVAL_CHUNK_PATHS = 2_500        # bounds the 3-D Chebyshev intermediates

# ------------------------------------------------------------- validation
VALIDATION_PATHS = 40_000
VALIDATION_TENORS = (0.5, 1.0, 2.0, 3.0, 5.0)
VALIDATION_TOLERANCE_SIGMAS = 3.0
QUANTO_TEST_CORRELATIONS = (0.0, -0.40)
CONVERGENCE_PATHS = (1_000, 2_500, 5_000, 10_000, 20_000, 40_000)

# ----------------------------------------------------------------- outputs
RESULTS_DIR = "results"
MAIN_RESULTS_CSV = "cva_results.csv"
SENSITIVITY_CSV = "sensitivity_results.csv"
PORTFOLIO_CSV = "portfolios.csv"
VALIDATION_CSV = "validation_results.csv"
COMPARISON_CSV = "rate_model_comparison.csv"
RISK_SPLIT_CSV = "risk_decomposition.csv"
