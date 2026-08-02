"""
Interpolation of the incoming trade's pricing function.

Why this module exists
----------------------
Both proxies want the scenario values of the *incoming* trade without paying
for a full repricing on all 20,000 paths x ~40 dates.  The standard trick (and
the one studied in the Chebyshev-tensor XVA literature) is:

  1. pick a small tensor grid of (time, spot) nodes,
  2. evaluate the EXACT analytic price only at those few nodes,
  3. build an interpolant through them,
  4. read the interpolant off at every (t_i, S(t_i, path)) pair.

Step 2 is the only expensive step and it costs n_time_nodes x n_spot_nodes
evaluations instead of n_dates x n_paths.

Two interpolation methods are provided behind ONE interface:

    ValueInterpolant(..., method="linear")      <- base case of the thesis
    ValueInterpolant(..., method="chebyshev")   <- optional, swappable module

They share the grid-building and evaluation plumbing but nothing else; the
Chebyshev machinery lives in its own functions below and can be deleted without
touching the linear path.

A note on the shape of the pricing function
-------------------------------------------
For a linear FX product with deterministic rates,

    V(t, S) = phi * N * [ DF_EUR(t,T) * S  -  K * DF_USD(t,T) ],

which is *exactly affine in S*.  Any interpolation in the spot direction alone
would therefore be exact to machine precision.  The genuine approximation comes
from coarsening the TIME direction, where the discount factors are exponential
and (for a swap) a leg drops out.  That is why the grid here is two-dimensional.

Handling of maturity cut-offs
-----------------------------
A trade's value jumps to zero once it settles.  That jump is *known exactly* and
is free to apply, so by default (`split_legs=True`) we interpolate only the
smooth pre-settlement value of each leg on its own time domain [0, T_leg] and
then apply the cut-off analytically.  Setting `split_legs=False` interpolates
the whole trade on a single domain, so an FX swap's near-leg jump falls inside
the domain - a deliberately naive variant kept for the sensitivity study.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.linalg import solve

from . import config
from .instruments import FXForward, FXSwap, Trade
from .market import MarketModel


# ----------------------------------------------------------------------------
# Node placement
# ----------------------------------------------------------------------------

def uniform_nodes(lo: float, hi: float, n: int) -> np.ndarray:
    """Equally spaced nodes - the natural companion of linear interpolation."""
    return np.linspace(lo, hi, n)


def chebyshev_nodes(lo: float, hi: float, n: int) -> np.ndarray:
    """Chebyshev-Lobatto (second kind) nodes mapped from [-1,1] to [lo,hi].

        x_k = cos(pi * k / (n-1)),   k = 0..n-1

    They cluster towards the ends of the interval, which is what suppresses the
    Runge phenomenon and gives (for analytic functions) geometric convergence.
    The endpoints are included, so the interpolant is anchored at t = 0 and at
    the trade's maturity.  Returned in ascending order.
    """
    if n < 2:
        raise ValueError("need at least 2 nodes")
    k = np.arange(n)
    x = np.cos(np.pi * k / (n - 1))          # descending in [-1, 1]
    x = x[::-1]                              # ascending
    return lo + 0.5 * (hi - lo) * (x + 1.0)


def _to_unit(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Affine map [lo, hi] -> [-1, 1] used before evaluating Chebyshev series."""
    if hi <= lo:
        return np.zeros_like(np.asarray(values, dtype=float))
    return 2.0 * (np.asarray(values, dtype=float) - lo) / (hi - lo) - 1.0


# ----------------------------------------------------------------------------
# The smooth 2-D interpolant on one time domain
# ----------------------------------------------------------------------------

@dataclass
class _TensorInterpolant:
    """Interpolant of a smooth function f(t, S) on a (time, spot) tensor grid.

    Evaluation exploits a structural fact of this problem: the exposure dates
    are the SAME for every path, so `times` has only ~40 distinct values while
    `spots` has ~800,000 entries.  Fixing t therefore collapses the 2-D
    interpolant to a 1-D function of S, which is evaluated once per column.
    That is both much cheaper than a generic scattered-point lookup and closer
    to how the calculation is described in the methodology chapter.
    """

    t_nodes: np.ndarray          # (n_t,)
    s_nodes: np.ndarray          # (n_s,)
    node_values: np.ndarray      # (n_t, n_s), the exact prices at the nodes
    method: str

    def __post_init__(self) -> None:
        if self.method == "linear":
            pass                 # piecewise-bilinear needs no precomputation
        elif self.method == "chebyshev":
            # Tensor Chebyshev interpolation.  With N_t x N_s nodes and the same
            # number of coefficients the system is square, so solving it gives
            # an interpolant that matches the function exactly at every node:
            #     V = A_t  C  A_s^T   =>   C = A_t^{-1} V A_s^{-T}
            x = _to_unit(self.t_nodes, self.t_nodes[0], self.t_nodes[-1])
            y = _to_unit(self.s_nodes, self.s_nodes[0], self.s_nodes[-1])
            a_t = cheb.chebvander(x, self.t_nodes.size - 1)     # (n_t, n_t)
            a_s = cheb.chebvander(y, self.s_nodes.size - 1)     # (n_s, n_s)
            coeffs = solve(a_t, self.node_values)               # (n_t, n_s)
            self._coeffs = solve(a_s, coeffs.T).T               # (n_t, n_s)
        else:
            raise ValueError(f"unknown interpolation method: {self.method!r}")

    @property
    def n_nodes(self) -> int:
        """Number of EXACT price evaluations this interpolant required."""
        return self.t_nodes.size * self.s_nodes.size

    def _spot_bracket(self, s: np.ndarray) -> np.ndarray:
        """Index k such that s_nodes[k-1] <= s <= s_nodes[k], for every entry of s."""
        spacing = np.diff(self.s_nodes)
        if np.allclose(spacing, spacing[0]):
            k = np.floor((s - self.s_nodes[0]) / spacing[0]).astype(np.intp) + 1
        else:
            k = np.searchsorted(self.s_nodes, s)
        return np.clip(k, 1, self.s_nodes.size - 1)

    # -- step 1: collapse the time dimension ---------------------------------

    def _reduce_to_spot_curves(self, times: np.ndarray) -> np.ndarray:
        """Return, for each exposure date, the 1-D interpolant in S.

        For the linear method this is the vector of node values at that date,
        obtained by linear interpolation between the two bracketing time nodes.
        For the Chebyshev method it is the vector of Chebyshev coefficients in
        the spot direction, c_k(t) = sum_j C[j,k] T_j(x(t)).

        Shape in both cases: (n_times, n_s).
        """
        t = np.clip(np.asarray(times, dtype=float), self.t_nodes[0], self.t_nodes[-1])

        if self.method == "chebyshev":
            x = _to_unit(t, self.t_nodes[0], self.t_nodes[-1])
            return cheb.chebvander(x, self.t_nodes.size - 1) @ self._coeffs

        # Linear: locate the bracketing time nodes and blend the node rows.
        j = np.clip(np.searchsorted(self.t_nodes, t), 1, self.t_nodes.size - 1)
        lo, hi = self.t_nodes[j - 1], self.t_nodes[j]
        w = ((t - lo) / (hi - lo))[:, None]                     # (n_times, 1)
        return (1.0 - w) * self.node_values[j - 1] + w * self.node_values[j]

    # -- step 2: evaluate the 1-D interpolant on every path ------------------

    def __call__(self, times: np.ndarray, spots: np.ndarray) -> np.ndarray:
        """Evaluate at every (t_i, S[p,i]) pair.

        `times` has shape (n_times,) and `spots` shape (n_paths, n_times);
        the result has the shape of `spots`.
        """
        curves = self._reduce_to_spot_curves(times)             # (n_times, n_s)

        # Defensive clipping to the node box.  The spot node range is built with
        # a buffer around the simulated min/max, so in practice nothing is
        # clipped; the guard only prevents a Chebyshev polynomial from being
        # evaluated far outside its interval, where it would blow up.
        s = np.clip(spots, self.s_nodes[0], self.s_nodes[-1])

        if self.method == "linear":
            # Piecewise-linear in S: find each scenario's bracketing spot nodes
            # and blend the two corresponding curve values.  The linear method
            # uses equally spaced nodes, so the bracket index is arithmetic
            # rather than a binary search - a worthwhile shortcut when it has to
            # be done for every one of ~800,000 scenario points.
            k = self._spot_bracket(s)
            lo, hi = self.s_nodes[k - 1], self.s_nodes[k]
            w = (s - lo) / (hi - lo)
            cols = np.arange(curves.shape[0])[None, :]          # broadcast over paths
            return (1.0 - w) * curves[cols, k - 1] + w * curves[cols, k]

        # Chebyshev in S, evaluated by the standard three-term recurrence
        #     T_0 = 1,  T_1 = y,  T_{k+1} = 2 y T_k - T_{k-1}
        # accumulated in place so the basis is never materialised in full.
        y = _to_unit(s, self.s_nodes[0], self.s_nodes[-1])
        total = np.empty_like(y)
        total[:] = curves[None, :, 0]
        if self.s_nodes.size > 1:
            t_prev, t_cur = np.ones_like(y), y
            total += curves[None, :, 1] * t_cur
            for k in range(2, self.s_nodes.size):
                t_prev, t_cur = t_cur, 2.0 * y * t_cur - t_prev
                total += curves[None, :, k] * t_cur
        return total


# ----------------------------------------------------------------------------
# Raw (pre-settlement) pricing functions
# ----------------------------------------------------------------------------

def _legs(trade: Trade) -> list[FXForward]:
    """Flatten a trade into its forward legs."""
    if isinstance(trade, FXSwap):
        return [trade.near_leg, trade.far_leg]
    return [trade]


def _raw_leg_value(
    leg: FXForward,
    times: np.ndarray,
    spots: np.ndarray,
    market: MarketModel,
) -> np.ndarray:
    """Leg value WITHOUT the maturity cut-off - a smooth function of (t, S).

    Identical to formula (1) except that the `alive` mask is not applied, so the
    function stays continuous up to and including t = T (where it equals the
    settlement amount phi*N*(S-K)).  Keeping it continuous is what makes it
    sensible to interpolate.
    """
    times = np.asarray(times, dtype=float)
    tau = np.maximum(leg.maturity - times, 0.0)
    return leg.direction * leg.notional * (
        market.df_eur(tau) * spots - leg.strike * market.df_usd(tau)
    )


# ----------------------------------------------------------------------------
# Public interface
# ----------------------------------------------------------------------------

class ValueInterpolant:
    """Cheap approximation of a trade's scenario values.

    Parameters
    ----------
    trade          : the incoming trade to be priced.
    market         : the market model (supplies the exact prices at the nodes).
    spot_lo/spot_hi: spot range the nodes should span.
    n_time_nodes   : nodes in the time direction (per leg/domain).
    n_spot_nodes   : nodes in the spot direction.
    method         : "linear" (base case) or "chebyshev" (optional module).
    split_legs     : if True, interpolate each leg on its own [0, T_leg] domain
                     and apply maturity cut-offs analytically; if False, use one
                     domain for the whole trade (naive variant).
    """

    def __init__(
        self,
        trade: Trade,
        market: MarketModel,
        spot_lo: float,
        spot_hi: float,
        n_time_nodes: int = config.N_TIME_NODES,
        n_spot_nodes: int = config.N_SPOT_NODES,
        method: str = config.DEFAULT_INTERP_METHOD,
        split_legs: bool = True,
    ) -> None:
        self.trade = trade
        self.market = market
        self.method = method
        self.split_legs = split_legs

        node_fn = chebyshev_nodes if method == "chebyshev" else uniform_nodes
        s_nodes = node_fn(spot_lo, spot_hi, n_spot_nodes)

        # Each entry is (interpolant, cut-off date). Values are zeroed for
        # t >= cut-off, which reproduces the trade settling exactly.
        self._pieces: list[tuple[_TensorInterpolant, float]] = []

        if split_legs:
            targets = [
                (leg.maturity, lambda t, s, leg=leg: _raw_leg_value(leg, t, s, market))
                for leg in _legs(trade)
            ]
        else:
            t_end = max(leg.maturity for leg in _legs(trade))
            targets = [(t_end, lambda t, s: trade.value(t, s, market))]

        for t_end, price_fn in targets:
            t_nodes = node_fn(0.0, t_end, n_time_nodes)
            # Exact prices at the nodes: shape (n_t, n_s).
            node_values = price_fn(t_nodes[:, None], s_nodes[None, :])
            self._pieces.append(
                (_TensorInterpolant(t_nodes, s_nodes, node_values, method), t_end)
            )

    @property
    def n_exact_evaluations(self) -> int:
        """Total number of exact analytic prices needed to build the proxy.

        This is the honest "cost" of the interpolation step and is reported in
        the accuracy-vs-cost table.
        """
        return sum(piece.n_nodes for piece, _ in self._pieces)

    def scenario_values(self, times: np.ndarray, spots: np.ndarray) -> np.ndarray:
        """Approximate MtM of the trade on the full (path, date) grid."""
        times = np.asarray(times, dtype=float)
        total = np.zeros(spots.shape, dtype=float)
        for interpolant, t_end in self._pieces:
            alive = np.broadcast_to(times < t_end - 1e-12, spots.shape)
            total += interpolant(times, spots) * alive
        return total
