"""
Interpolation of the incoming trade's pricing function.

Why this module exists
----------------------
Both proxies want the scenario values of the *incoming* trade without paying for
a full repricing on all paths and dates.  The standard trick (and the one
studied in the Chebyshev-tensor XVA literature) is:

  1. pick a small tensor grid of nodes,
  2. evaluate the EXACT analytic price only at those few nodes,
  3. build an interpolant through them,
  4. read the interpolant off at every simulated point.

Step 2 is the only expensive step.

What changed with stochastic rates
----------------------------------
Under the single-factor base model the pricing function was V(t, S): one time
dimension and ONE state dimension.  Under the three-factor model it is

        V(t, S, x_d, x_f)

- one time dimension and THREE state dimensions.  With n nodes per state
dimension the grid holds

        n_time  x  n^3     nodes

instead of n_time x n.  That cubic growth is the curse of dimensionality, and it
is the central cost of the upgrade: at n = 5 the proxy needs 6 x 125 = 750 exact
prices per leg where the single-factor model needed 6 x 5 = 30.  The sweep in
`main.py` measures it.

Two interpolation methods sit behind ONE interface:

    ValueInterpolant(..., method="linear")      <- base case of the thesis
    ValueInterpolant(..., method="chebyshev")   <- optional, swappable module

Structure of the pricing function
---------------------------------
    V(t, S, x_d, x_f) = phi N [ S A_f(t,T) e^{-B_f(t,T) r_f}
                                - K A_d(t,T) e^{-B_d(t,T) r_d} ]

so it remains EXACTLY AFFINE in S even under stochastic rates: two nodes in the
spot dimension reproduce it to machine precision.  All genuine curvature lives
in x_d and x_f, where the bond prices are exponential, and in t.  This is why
the node sweep is run per-dimension-count rather than only in aggregate.

Handling of maturity cut-offs
-----------------------------
A trade's value jumps to zero once it settles.  That jump is *known exactly* and
free to apply, so by default (`split_legs=True`) we interpolate only the smooth
pre-settlement value of each leg on its own time domain [0, T_leg] and then
apply the cut-off analytically.  Setting `split_legs=False` interpolates the
whole trade on a single domain, so an FX swap's near-leg jump falls inside the
domain - a deliberately naive variant kept for the sensitivity study.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from numpy.polynomial import chebyshev as cheb

from . import config
from .instruments import Trade


# ----------------------------------------------------------------------------
# Node placement
# ----------------------------------------------------------------------------

def uniform_nodes(lo: float, hi: float, n: int) -> np.ndarray:
    """Equally spaced nodes - the natural companion of linear interpolation."""
    if n < 2:
        raise ValueError("need at least 2 nodes")
    return np.linspace(lo, hi, n)


def chebyshev_nodes(lo: float, hi: float, n: int) -> np.ndarray:
    """Chebyshev-Lobatto (second kind) nodes mapped from [-1,1] to [lo,hi].

        x_k = cos(pi * k / (n-1)),   k = 0..n-1

    They cluster towards the ends of the interval, which suppresses the Runge
    phenomenon and gives geometric convergence for analytic functions.  The
    endpoints are included, so the interpolant is anchored at t = 0 and at the
    trade's maturity.  Returned in ascending order.
    """
    if n < 2:
        raise ValueError("need at least 2 nodes")
    k = np.arange(n)
    x = np.cos(np.pi * k / (n - 1))[::-1]        # ascending in [-1, 1]
    return lo + 0.5 * (hi - lo) * (x + 1.0)


def _to_unit(values, lo: float, hi: float) -> np.ndarray:
    """Affine map [lo, hi] -> [-1, 1], used before evaluating Chebyshev series."""
    values = np.asarray(values, dtype=float)
    if hi <= lo:
        return np.zeros_like(values)
    return 2.0 * (values - lo) / (hi - lo) - 1.0


def _chebyshev_basis(y: np.ndarray, n: int) -> np.ndarray:
    """Chebyshev polynomials T_0..T_{n-1} evaluated at y.

    Built by the three-term recurrence T_0 = 1, T_1 = y, T_{j+1} = 2 y T_j - T_{j-1}.
    Returns an array of shape y.shape + (n,).
    """
    basis = np.empty(y.shape + (n,), dtype=float)
    basis[..., 0] = 1.0
    if n > 1:
        basis[..., 1] = y
    for j in range(2, n):
        basis[..., j] = 2.0 * y * basis[..., j - 1] - basis[..., j - 2]
    return basis


# ----------------------------------------------------------------------------
# The tensor interpolant over (time) x (state dimensions)
# ----------------------------------------------------------------------------

@dataclass
class _TensorInterpolant:
    """Interpolant of a smooth function f(t, s_1, ..., s_D) on a tensor grid.

    Evaluation exploits a structural fact of the problem: every path shares the
    SAME exposure dates, so `times` has only ~40 distinct values while the state
    query points number n_paths x n_times.  The time dimension is therefore
    collapsed first, cheaply, leaving a D-dimensional interpolant per exposure
    date that is then evaluated on all paths.
    """

    t_nodes: np.ndarray                     # (n_t,)
    state_nodes: list[np.ndarray]           # one node vector per state dimension
    node_values: np.ndarray                 # (n_t, n_1, ..., n_D), exact prices
    method: str
    _coeffs: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.method == "chebyshev":
            # Tensor Chebyshev interpolation.  With the same number of
            # coefficients as nodes in each dimension the system is square, so
            # solving it dimension by dimension gives an interpolant that
            # matches the function exactly at every node.
            coeffs = self.node_values
            for axis, nodes in enumerate([self.t_nodes, *self.state_nodes]):
                vander = cheb.chebvander(_to_unit(nodes, nodes[0], nodes[-1]), nodes.size - 1)
                # Contract the inverse Vandermonde along this axis, then put the
                # axis back where it was.
                coeffs = np.moveaxis(
                    np.tensordot(np.linalg.inv(vander), coeffs, axes=([1], [axis])), 0, axis
                )
            self._coeffs = coeffs
        elif self.method != "linear":
            raise ValueError(f"unknown interpolation method: {self.method!r}")

    @property
    def n_dimensions(self) -> int:
        """Number of STATE dimensions (the time dimension is not counted)."""
        return len(self.state_nodes)

    @property
    def n_nodes(self) -> int:
        """Number of EXACT price evaluations this interpolant required."""
        total = self.t_nodes.size
        for nodes in self.state_nodes:
            total *= nodes.size
        return total

    # -- step 1: collapse the time dimension ---------------------------------

    def _collapse_time(self, times: np.ndarray) -> np.ndarray:
        """Reduce the interpolant to one D-dimensional object per exposure date.

        For the linear method this blends the node hyper-slabs of the two
        bracketing time nodes.  For the Chebyshev method it contracts the
        coefficient tensor with the Chebyshev basis evaluated at each date.

        Returns an array of shape (n_times, n_1, ..., n_D).
        """
        t = np.clip(np.asarray(times, dtype=float), self.t_nodes[0], self.t_nodes[-1])

        if self.method == "chebyshev":
            basis = _chebyshev_basis(_to_unit(t, self.t_nodes[0], self.t_nodes[-1]),
                                     self.t_nodes.size)          # (n_times, n_t)
            return np.tensordot(basis, self._coeffs, axes=([1], [0]))

        j = np.clip(np.searchsorted(self.t_nodes, t), 1, self.t_nodes.size - 1)
        lo, hi = self.t_nodes[j - 1], self.t_nodes[j]
        w = (t - lo) / (hi - lo)
        w = w.reshape((-1,) + (1,) * self.n_dimensions)           # broadcast over state axes
        return (1.0 - w) * self.node_values[j - 1] + w * self.node_values[j]

    # -- step 2: evaluate over the state dimensions --------------------------

    def _evaluate_states(self, curves: np.ndarray, queries: list[np.ndarray]) -> np.ndarray:
        """Evaluate the per-date interpolants at the simulated state points.

        `curves`  : (n_times, n_1, ..., n_D) from `_collapse_time`.
        `queries` : one (n_paths, n_times) array per state dimension.
        """
        n_times = curves.shape[0]
        date_index = np.arange(n_times)[None, :]      # broadcasts against (n_paths, n_times)

        # Clip to the node box.  The ranges are built with a buffer around the
        # simulated min/max, so nothing is clipped in practice; the guard only
        # stops a Chebyshev polynomial being evaluated far outside its interval,
        # where it would blow up.
        clipped = [
            np.clip(q, nodes[0], nodes[-1])
            for q, nodes in zip(queries, self.state_nodes)
        ]

        if self.method == "chebyshev":
            # result[p,t] = sum_{a,b,c} C[t,a,b,c] T_a(y1) T_b(y2) T_c(y3)
            #
            # Contracted ONE DIMENSION AT A TIME rather than in a single einsum.
            # `p` and `t` are batch indices shared by every operand, so einsum's
            # optimiser cannot factor the expression and falls back to the naive
            # O(n^D) loop over the full query set.  Peeling the dimensions off in
            # sequence shrinks the working tensor at every step - (t,a,b,c) ->
            # (p,t,b,c) -> (p,t,c) -> (p,t) - and is several times faster.
            bases = [
                _chebyshev_basis(_to_unit(q, nodes[0], nodes[-1]), nodes.size)
                for q, nodes in zip(clipped, self.state_nodes)
            ]
            # First contraction: this is the expensive one, touching all n^D
            # coefficients for every query point.  Written as a BATCHED MATRIX
            # PRODUCT over the exposure dates so it runs through BLAS:
            #     (T, P, n_0) @ (T, n_0, R)  ->  (T, P, R),   R = n_1 * ... * n_D
            first, rest = bases[0], bases[1:]
            n_first = first.shape[-1]
            block = curves.reshape(curves.shape[0], n_first, -1)      # (T, n_0, R)
            acc = np.matmul(np.transpose(first, (1, 0, 2)), block)    # (T, P, R)
            acc = np.transpose(acc, (1, 0, 2))                        # (P, T, R)

            # Remaining contractions act on tensors that shrink by a factor n
            # each time, so a plain broadcast-and-sum is already cheap.
            trailing = [nodes.size for nodes in self.state_nodes[1:]]
            for dim, basis in enumerate(rest):
                acc = acc.reshape(basis.shape + tuple(trailing[dim + 1:]))
                acc = (basis.reshape(basis.shape + (1,) * len(trailing[dim + 1:])) * acc)
                acc = acc.sum(axis=2)

            # With a one-dimensional state there is no `rest` loop to run and the
            # matmul leaves a length-1 trailing axis; drop it so every path
            # returns the same (n_paths, n_times) shape.
            return acc.reshape(first.shape[:2])

        # Multilinear interpolation: locate each query inside its cell, then sum
        # the 2^D corner values weighted by the product of the per-dimension
        # weights.  At D = 3 that is eight lookups.
        #
        # The lookups go through a FLAT index rather than D-dimensional fancy
        # indexing.  Every corner sits at a constant element offset from the
        # lower corner, so the index arithmetic is done once and each corner is
        # then a scalar shift - which is much cheaper than letting NumPy
        # re-broadcast four index arrays eight times over.
        lower, weights = [], []
        for q, nodes in zip(clipped, self.state_nodes):
            k = np.clip(np.searchsorted(nodes, q), 1, nodes.size - 1)
            lower.append(k - 1)
            weights.append((q - nodes[k - 1]) / (nodes[k] - nodes[k - 1]))

        strides = np.ones(curves.ndim, dtype=np.intp)
        for axis in range(curves.ndim - 2, -1, -1):
            strides[axis] = strides[axis + 1] * curves.shape[axis + 1]

        base = date_index * strides[0]
        for dim, index in enumerate(lower):
            base = base + index * strides[dim + 1]
        flat = curves.reshape(-1)

        total = np.zeros(clipped[0].shape, dtype=float)
        for corner in product((0, 1), repeat=self.n_dimensions):
            weight = np.ones_like(total)
            for dim, offset in enumerate(corner):
                weight = weight * (weights[dim] if offset else 1.0 - weights[dim])
            shift = int(sum(offset * strides[dim + 1] for dim, offset in enumerate(corner)))
            total += weight * np.take(flat, base + shift)
        return total

    def __call__(self, times: np.ndarray, queries: list[np.ndarray]) -> np.ndarray:
        """Evaluate at every simulated (date, state) point.

        Paths are processed in blocks so that the intermediate tensors of a 3-D
        Chebyshev contraction stay bounded regardless of the node count.
        """
        curves = self._collapse_time(times)
        n_paths = queries[0].shape[0]
        out = np.empty(queries[0].shape, dtype=float)

        chunk = max(int(config.EVAL_CHUNK_PATHS), 1)
        for start in range(0, n_paths, chunk):
            stop = min(start + chunk, n_paths)
            out[start:stop] = self._evaluate_states(
                curves, [q[start:stop] for q in queries]
            )
        return out


# ----------------------------------------------------------------------------
# Public interface
# ----------------------------------------------------------------------------

class ValueInterpolant:
    """Cheap approximation of a trade's scenario values.

    Parameters
    ----------
    trade          : the incoming trade to be priced.
    market         : the three-factor model (supplies exact prices at the nodes).
    state_box      : {state name: (lo, hi)} for every active state variable, in
                     `market.state_names` order.
    n_time_nodes   : nodes in the time direction (per leg/domain).
    n_state_nodes  : nodes PER STATE DIMENSION.
    method         : "linear" (base case) or "chebyshev" (optional module).
    split_legs     : if True, interpolate each leg on its own [0, T_leg] domain
                     and apply maturity cut-offs analytically; if False, use one
                     domain for the whole trade (naive variant).
    """

    def __init__(
        self,
        trade: Trade,
        market,
        state_box: dict[str, tuple[float, float]],
        n_time_nodes: int = config.N_TIME_NODES,
        n_state_nodes: int = config.N_STATE_NODES,
        method: str = config.DEFAULT_INTERP_METHOD,
        split_legs: bool = True,
    ) -> None:
        self.trade = trade
        self.market = market
        self.method = method
        self.split_legs = split_legs
        self.state_names = tuple(market.state_names)

        node_fn = chebyshev_nodes if method == "chebyshev" else uniform_nodes
        state_nodes = [
            node_fn(*state_box[name], n_state_nodes) for name in self.state_names
        ]
        n_state_dims = len(self.state_names)

        # Each entry is (interpolant, cut-off date).  Values are zeroed for
        # t >= cut-off, which reproduces the trade settling exactly.
        self._pieces: list[tuple[_TensorInterpolant, float]] = []

        if split_legs:
            targets = [(leg.last_date, [leg]) for leg in trade.legs]
        else:
            targets = [(trade.last_date, list(trade.legs))]

        for t_end, legs in targets:
            t_nodes = node_fn(0.0, t_end, n_time_nodes)
            node_values = self._exact_prices_at_nodes(
                legs, t_nodes, state_nodes, n_state_dims, whole_trade=not split_legs
            )
            self._pieces.append(
                (_TensorInterpolant(t_nodes, state_nodes, node_values, method), t_end)
            )

    # ------------------------------------------------------------------

    def _exact_prices_at_nodes(
        self,
        legs,
        t_nodes: np.ndarray,
        state_nodes: list[np.ndarray],
        n_state_dims: int,
        whole_trade: bool,
    ) -> np.ndarray:
        """Evaluate the EXACT analytic price at every node of the tensor grid.

        The grid axes are (time, state_1, ..., state_D).  Each axis vector is
        reshaped so that NumPy broadcasting produces the full tensor without
        materialising any Python loop over nodes.

        `whole_trade=False` prices a single leg WITHOUT its maturity cut-off, so
        the interpolated function stays smooth; the cut-off is applied later in
        `scenario_values`.  `whole_trade=True` prices all legs WITH their
        cut-offs, which deliberately puts the near leg's settlement jump inside
        the domain (the naive variant).
        """
        ndim = 1 + n_state_dims

        def axis_shape(axis: int, vector: np.ndarray) -> np.ndarray:
            shape = [1] * ndim
            shape[axis] = vector.size
            return vector.reshape(shape)

        times = axis_shape(0, t_nodes)
        state = {
            name: axis_shape(1 + k, nodes)
            for k, (name, nodes) in enumerate(zip(self.state_names, state_nodes))
        }

        spots = state["spot"]
        r_d, r_f = self.market.rates_from_state(times, state)

        total = np.zeros([t_nodes.size] + [n.size for n in state_nodes], dtype=float)
        for leg in legs:
            total = total + leg.value(
                self.market, times, spots, r_d, r_f, apply_maturity=whole_trade
            )
        return total

    # ------------------------------------------------------------------

    @property
    def n_exact_evaluations(self) -> int:
        """Total number of exact analytic prices needed to build the proxy.

        The hardware-independent cost measure reported in the results table.
        """
        return sum(piece.n_nodes for piece, _ in self._pieces)

    @property
    def n_state_dimensions(self) -> int:
        return len(self.state_names)

    def scenario_values(self, scen) -> np.ndarray:
        """Approximate MtM of the trade on the full (path, date) grid."""
        times = np.asarray(scen.times, dtype=float)
        queries = [scen.state(name) for name in self.state_names]

        total = np.zeros(scen.spots.shape, dtype=float)
        for interpolant, t_end in self._pieces:
            alive = np.broadcast_to(times < t_end - 1e-12, scen.spots.shape)
            total += interpolant(times, queries) * alive
        return total


# ----------------------------------------------------------------------------
# Node ranges
# ----------------------------------------------------------------------------

def state_box(scen, buffer: float = config.STATE_GRID_BUFFER) -> dict[str, tuple[float, float]]:
    """Node ranges for every active state variable, taken from the simulation.

    Each range is widened by `buffer` so that no simulated point falls outside
    the node box.  Avoiding extrapolation keeps the linear/Chebyshev comparison
    about approximation quality rather than about extrapolation stability, which
    would flatter the piecewise method unfairly.

    The spot is a positive quantity, so its range is scaled multiplicatively;
    the mean-zero rate factors are centred on zero, so theirs is made symmetric.
    """
    box: dict[str, tuple[float, float]] = {}
    for name in scen.state_names:
        values = scen.state(name)
        lo, hi = float(values.min()), float(values.max())
        if name == "spot":
            box[name] = (lo * (1.0 - buffer), hi * (1.0 + buffer))
        else:
            half_width = max(abs(lo), abs(hi)) * (1.0 + buffer)
            half_width = max(half_width, 1e-8)      # never a degenerate interval
            box[name] = (-half_width, half_width)
    return box
