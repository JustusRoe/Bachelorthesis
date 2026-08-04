"""
The tensor interpolant and the two fast pricing proxies.

Both proxies avoid repricing the incoming trade on every path: they fit an
interpolant of the pricing function on a coarse tensor grid over
(time) x (state), evaluate the EXACT analytic price only at the nodes, and read
the interpolant off at every simulated point.

The state is 1-dimensional under deterministic rates (the spot alone) and
3-dimensional under the three-factor model (S, x_d, x_f), so with n nodes per
state dimension the grid holds n_time * n^3 nodes - the curse of dimensionality
is the central cost of the upgrade.

STANDALONE PROXY   prices the new trade in isolation; structurally blind to
                   netting, always >= 0, expected to overcharge.
INCREMENTAL PROXY  aggregates the new trade into the netting set's STORED
                   scenario matrix; netting is exact, interpolation is the only
                   approximation, and the result can be negative.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import product

import numpy as np
from numpy.polynomial import chebyshev as cheb

from . import config
from .cva import (
    HazardCurve,
    cva_from_discounted_ee,
    cva_from_values,
    discounted_ee,
)
from .products import Trade, netting_set_values


# ============================================================================
# Node placement
# ============================================================================

def uniform_nodes(lo: float, hi: float, n: int) -> np.ndarray:
    """Equally spaced nodes - the companion of linear interpolation."""
    if n < 2:
        raise ValueError("need at least 2 nodes")
    return np.linspace(lo, hi, n)


def chebyshev_nodes(lo: float, hi: float, n: int) -> np.ndarray:
    """Chebyshev-Lobatto nodes x_k = cos(pi k/(n-1)) mapped to [lo,hi], ascending.

    They cluster at the ends, which suppresses the Runge phenomenon and gives
    geometric convergence for analytic functions.  Endpoints are included, so the
    interpolant is anchored at t = 0 and at the trade's maturity.
    """
    if n < 2:
        raise ValueError("need at least 2 nodes")
    x = np.cos(np.pi * np.arange(n) / (n - 1))[::-1]
    return lo + 0.5 * (hi - lo) * (x + 1.0)


def _to_unit(values, lo: float, hi: float) -> np.ndarray:
    """Map [lo,hi] -> [-1,1] before evaluating a Chebyshev series."""
    values = np.asarray(values, dtype=float)
    return np.zeros_like(values) if hi <= lo else 2.0 * (values - lo) / (hi - lo) - 1.0


def _chebyshev_basis(y: np.ndarray, n: int) -> np.ndarray:
    """T_0..T_{n-1} at y by the recurrence T_{j+1} = 2 y T_j - T_{j-1}.
    Returns shape y.shape + (n,)."""
    basis = np.empty(y.shape + (n,), dtype=float)
    basis[..., 0] = 1.0
    if n > 1:
        basis[..., 1] = y
    for j in range(2, n):
        basis[..., j] = 2.0 * y * basis[..., j - 1] - basis[..., j - 2]
    return basis


def state_box(scen, buffer: float = config.STATE_GRID_BUFFER) -> dict:
    """Node ranges per state variable, taken from the simulation and widened by
    `buffer` so no simulated point needs extrapolation.

    Avoiding extrapolation keeps the linear/Chebyshev comparison about
    approximation quality rather than extrapolation stability.  The spot is
    positive so its range is scaled; the mean-zero rate factors are symmetric.
    """
    box = {}
    for name in scen.state_names:
        values = scen.state(name)
        lo, hi = float(values.min()), float(values.max())
        if name == "spot":
            box[name] = (lo * (1.0 - buffer), hi * (1.0 + buffer))
        else:
            half = max(max(abs(lo), abs(hi)) * (1.0 + buffer), 1e-8)
            box[name] = (-half, half)
    return box


# ============================================================================
# The tensor interpolant
# ============================================================================

@dataclass
class _TensorInterpolant:
    """Interpolant of f(t, s_1, ..., s_D) on a tensor grid.

    Every path shares the same exposure dates, so the time dimension is collapsed
    first (cheap, ~40 distinct values), leaving a D-dimensional interpolant per
    date to evaluate on all paths.
    """

    t_nodes: np.ndarray
    state_nodes: list[np.ndarray]
    node_values: np.ndarray              # (n_t, n_1, ..., n_D), exact prices
    method: str
    _coeffs: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.method == "chebyshev":
            # Square system per axis, so solving dimension by dimension gives an
            # interpolant that matches the function exactly at every node.
            coeffs = self.node_values
            for axis, nodes in enumerate([self.t_nodes, *self.state_nodes]):
                vander = cheb.chebvander(_to_unit(nodes, nodes[0], nodes[-1]), nodes.size - 1)
                coeffs = np.moveaxis(
                    np.tensordot(np.linalg.inv(vander), coeffs, axes=([1], [axis])), 0, axis)
            self._coeffs = coeffs
        elif self.method != "linear":
            raise ValueError(f"unknown interpolation method: {self.method!r}")

    @property
    def n_dimensions(self) -> int:
        """Number of STATE dimensions; time is not counted."""
        return len(self.state_nodes)

    @property
    def n_nodes(self) -> int:
        """Exact price evaluations this interpolant required."""
        return int(self.t_nodes.size * np.prod([n.size for n in self.state_nodes]))

    def _collapse_time(self, times) -> np.ndarray:
        """Reduce to one D-dimensional object per exposure date: (n_times, n_1, ..., n_D)."""
        t = np.clip(np.asarray(times, dtype=float), self.t_nodes[0], self.t_nodes[-1])
        if self.method == "chebyshev":
            basis = _chebyshev_basis(_to_unit(t, self.t_nodes[0], self.t_nodes[-1]),
                                     self.t_nodes.size)
            return np.tensordot(basis, self._coeffs, axes=([1], [0]))
        j = np.clip(np.searchsorted(self.t_nodes, t), 1, self.t_nodes.size - 1)
        lo, hi = self.t_nodes[j - 1], self.t_nodes[j]
        w = ((t - lo) / (hi - lo)).reshape((-1,) + (1,) * self.n_dimensions)
        return (1.0 - w) * self.node_values[j - 1] + w * self.node_values[j]

    def _evaluate_states(self, curves: np.ndarray, queries: list[np.ndarray]) -> np.ndarray:
        """Evaluate the per-date interpolants at the simulated state points."""
        # Clipping is a guard only: the node box is built with a buffer, so
        # nothing is clipped in practice.  It stops a Chebyshev polynomial being
        # evaluated far outside its interval, where it would blow up.
        clipped = [np.clip(q, n[0], n[-1]) for q, n in zip(queries, self.state_nodes)]

        if self.method == "chebyshev":
            bases = [_chebyshev_basis(_to_unit(q, n[0], n[-1]), n.size)
                     for q, n in zip(clipped, self.state_nodes)]
            # The first contraction touches all n^D coefficients for every query
            # point.  einsum cannot factor it (p and t are batch indices on every
            # operand, so it falls back to the naive O(n^D) loop), so it is
            # written as a batched matrix product and routed through BLAS:
            #     (T, P, n_0) @ (T, n_0, R) -> (T, P, R),  R = n_1 * ... * n_D
            first, rest = bases[0], bases[1:]
            block = curves.reshape(curves.shape[0], first.shape[-1], -1)
            acc = np.transpose(np.matmul(np.transpose(first, (1, 0, 2)), block), (1, 0, 2))

            # The remaining contractions shrink the tensor by a factor n each
            # time, so a broadcast-and-sum is already cheap.
            trailing = [n.size for n in self.state_nodes[1:]]
            for dim, basis in enumerate(rest):
                acc = acc.reshape(basis.shape + tuple(trailing[dim + 1:]))
                acc = (basis.reshape(basis.shape + (1,) * len(trailing[dim + 1:])) * acc)
                acc = acc.sum(axis=2)
            # With a 1-D state there is no `rest` loop and matmul leaves a
            # length-1 trailing axis; drop it.
            return acc.reshape(first.shape[:2])

        # Multilinear: locate each query in its cell, then sum the 2^D corners.
        # Every corner sits at a constant element offset from the lower corner,
        # so the index arithmetic is done once and each corner is a scalar shift
        # - much cheaper than re-broadcasting D+1 index arrays 2^D times.
        lower, weights = [], []
        for q, nodes in zip(clipped, self.state_nodes):
            k = np.clip(np.searchsorted(nodes, q), 1, nodes.size - 1)
            lower.append(k - 1)
            weights.append((q - nodes[k - 1]) / (nodes[k] - nodes[k - 1]))

        strides = np.ones(curves.ndim, dtype=np.intp)
        for axis in range(curves.ndim - 2, -1, -1):
            strides[axis] = strides[axis + 1] * curves.shape[axis + 1]

        base = np.arange(curves.shape[0])[None, :] * strides[0]
        for dim, index in enumerate(lower):
            base = base + index * strides[dim + 1]
        flat = curves.reshape(-1)

        total = np.zeros(clipped[0].shape, dtype=float)
        for corner in product((0, 1), repeat=self.n_dimensions):
            weight = np.ones_like(total)
            for dim, offset in enumerate(corner):
                weight = weight * (weights[dim] if offset else 1.0 - weights[dim])
            shift = int(sum(o * strides[d + 1] for d, o in enumerate(corner)))
            total += weight * np.take(flat, base + shift)
        return total

    def __call__(self, times, queries: list[np.ndarray]) -> np.ndarray:
        """Evaluate at every simulated (date, state) point.

        Paths are processed in blocks so the intermediate tensors of a 3-D
        Chebyshev contraction stay bounded regardless of the node count.
        """
        curves = self._collapse_time(times)
        out = np.empty(queries[0].shape, dtype=float)
        chunk = max(int(config.EVAL_CHUNK_PATHS), 1)
        for start in range(0, queries[0].shape[0], chunk):
            stop = start + chunk
            out[start:stop] = self._evaluate_states(curves, [q[start:stop] for q in queries])
        return out


class ValueInterpolant:
    """Cheap approximation of a trade's scenario values.

    A trade's value jumps to zero once it settles.  That jump is known exactly
    and free to apply, so by default each leg is interpolated on its own smooth
    domain [0, T_leg] and the cut-off is applied analytically.  `split_legs=False`
    puts the whole trade on one domain, so an FX swap's near-leg jump falls
    inside it - the naive variant kept for the sensitivity study.
    """

    def __init__(self, trade: Trade, market, state_box: dict,
                 n_time_nodes: int = config.N_TIME_NODES,
                 n_state_nodes: int | dict = config.N_STATE_NODES,
                 method: str = config.DEFAULT_INTERP_METHOD,
                 split_legs: bool = True):
        self.trade = trade
        self.market = market
        self.method = method
        self.state_names = tuple(market.state_names)

        # `n_state_nodes` is either one count for every state dimension
        # (isotropic, n^D nodes) or a per-dimension mapping (anisotropic).  The
        # anisotropic case exists because V is EXACTLY affine in S, so two nodes
        # reproduce the spot dimension to machine precision and spending more
        # there is pure waste - see `config.anisotropic_nodes`.
        counts = ([int(n_state_nodes[name]) for name in self.state_names]
                  if isinstance(n_state_nodes, dict)
                  else [int(n_state_nodes)] * len(self.state_names))
        self.nodes_per_dimension = tuple(counts)

        node_fn = chebyshev_nodes if method == "chebyshev" else uniform_nodes
        state_nodes = [node_fn(*state_box[name], n)
                       for name, n in zip(self.state_names, counts)]

        targets = ([(leg.last_date, [leg]) for leg in trade.legs] if split_legs
                   else [(trade.last_date, list(trade.legs))])

        # Each entry is (interpolant, cut-off date); values are zeroed beyond it.
        self._pieces: list[tuple[_TensorInterpolant, float]] = []
        for t_end, legs in targets:
            t_nodes = node_fn(0.0, t_end, n_time_nodes)
            values = self._exact_prices_at_nodes(legs, t_nodes, state_nodes, not split_legs)
            self._pieces.append((_TensorInterpolant(t_nodes, state_nodes, values, method), t_end))

    def _exact_prices_at_nodes(self, legs, t_nodes, state_nodes, whole_trade: bool):
        """Exact analytic price at every node, built by broadcasting the axis vectors.

        `whole_trade=False` prices one leg WITHOUT its maturity cut-off so the
        fitted function stays smooth; `True` applies the cut-offs, deliberately
        putting the near leg's jump inside the domain.
        """
        ndim = 1 + len(state_nodes)

        def axis(index, vector):
            shape = [1] * ndim
            shape[index] = vector.size
            return vector.reshape(shape)

        times = axis(0, t_nodes)
        state = {name: axis(1 + k, nodes)
                 for k, (name, nodes) in enumerate(zip(self.state_names, state_nodes))}
        r_d, r_f = self.market.rates_from_state(times, state)

        total = np.zeros([t_nodes.size] + [n.size for n in state_nodes], dtype=float)
        for leg in legs:
            total = total + leg.value(self.market, times, state["spot"], r_d, r_f,
                                      apply_maturity=whole_trade)
        return total

    @property
    def n_exact_evaluations(self) -> int:
        """Exact analytic prices needed to build the proxy - the
        hardware-independent cost measure reported in the results table."""
        return sum(piece.n_nodes for piece, _ in self._pieces)

    def scenario_values(self, scen) -> np.ndarray:
        """Approximate MtM of the trade on the full (path, date) grid."""
        times = np.asarray(scen.times, dtype=float)
        queries = [scen.state(name) for name in self.state_names]
        total = np.zeros(scen.spots.shape, dtype=float)
        for interpolant, t_end in self._pieces:
            alive = np.broadcast_to(times < t_end - 1e-12, scen.spots.shape)
            total += interpolant(times, queries) * alive
        return total


# ============================================================================
# Pricing requests
# ============================================================================

@dataclass
class PricingContext:
    """Everything a pricing request needs that is not the trade itself."""

    market: object
    scen: object
    state_box: dict
    discounting: str = config.DISCOUNTING

    @property
    def n_state_dimensions(self) -> int:
        return len(self.scen.state_names)


@dataclass
class ProxyResult:
    cva: float                  # the charge the proxy would quote, USD
    seconds: float              # wall-clock time of the request (best of N runs)
    n_exact_evaluations: int    # exact analytic prices the proxy needed


def _timed(fn, repeats: int = config.N_TIMING_REPEATS):
    """Run `fn` `repeats` times, returning (result, best seconds).

    The minimum is the standard robust estimate of intrinsic cost, least polluted
    by OS scheduling noise.
    """
    best, result = np.inf, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return result, float(best)


def standalone_proxy(new_trade: Trade, ctx: PricingContext, hazard: HazardCurve,
                     method: str = config.DEFAULT_INTERP_METHOD,
                     n_time_nodes: int = config.N_TIME_NODES,
                     n_state_nodes: int | dict = config.N_STATE_NODES,
                     split_legs: bool = True,
                     recovery: float = config.RECOVERY_RATE,
                     repeats: int = config.N_TIMING_REPEATS) -> ProxyResult:
    """CVA of the incoming trade priced on its own, with no netting."""
    holder = {}

    def request():
        interp = ValueInterpolant(new_trade, ctx.market, ctx.state_box, n_time_nodes,
                                  n_state_nodes, method, split_legs)
        holder["i"] = interp
        # max(.,0) taken trade by trade -> no netting benefit
        ee_star = discounted_ee(interp.scenario_values(ctx.scen), ctx.scen,
                                ctx.market, ctx.discounting)
        return cva_from_discounted_ee(ee_star, ctx.scen.times, hazard, recovery)

    cva, seconds = _timed(request, repeats)
    return ProxyResult(cva, seconds, holder["i"].n_exact_evaluations)


def incremental_proxy(new_trade: Trade, ctx: PricingContext, hazard: HazardCurve,
                      stored_values: np.ndarray, stored_cva: float,
                      method: str = config.DEFAULT_INTERP_METHOD,
                      n_time_nodes: int = config.N_TIME_NODES,
                      n_state_nodes: int | dict = config.N_STATE_NODES,
                      split_legs: bool = True,
                      recovery: float = config.RECOVERY_RATE,
                      repeats: int = config.N_TIMING_REPEATS) -> ProxyResult:
    """Incremental CVA against the netting set's stored scenario matrix.

    `stored_values` and `stored_cva` come from the overnight batch and are reused
    for free.  The new trade is added LAST.
    """
    holder = {}

    def request():
        interp = ValueInterpolant(new_trade, ctx.market, ctx.state_box, n_time_nodes,
                                  n_state_nodes, method, split_legs)
        holder["i"] = interp
        # Aggregate into the stored values: netting is EXACT here.
        cva_with = cva_from_values(stored_values + interp.scenario_values(ctx.scen),
                                   ctx.scen, ctx.market, hazard, recovery, ctx.discounting)
        return cva_with - stored_cva

    cva, seconds = _timed(request, repeats)
    return ProxyResult(cva, seconds, holder["i"].n_exact_evaluations)


def full_resimulation_incremental(existing_trades, new_trade: Trade, ctx: PricingContext,
                                  hazard: HazardCurve, stored_cva: float,
                                  recovery: float = config.RECOVERY_RATE,
                                  repeats: int = config.N_TIMING_REPEATS) -> ProxyResult:
    """TRUE incremental CVA, and the timing reference the proxies are judged against.

    Every trade is repriced from scratch.  Only subtracting `stored_cva` is free,
    which makes the comparison against `incremental_proxy` apples-to-apples: both
    do the same arithmetic and differ only in how the values are obtained.
    """
    n_trades = len(existing_trades) + 1

    def request():
        values = netting_set_values(list(existing_trades) + [new_trade], ctx.market, ctx.scen)
        return cva_from_values(values, ctx.scen, ctx.market, hazard, recovery,
                               ctx.discounting) - stored_cva

    cva, seconds = _timed(request, repeats)
    return ProxyResult(cva, seconds, n_trades * ctx.scen.spots.size)


def incremental_exact_repricing(new_trade: Trade, ctx: PricingContext, hazard: HazardCurve,
                                stored_values: np.ndarray, stored_cva: float,
                                recovery: float = config.RECOVERY_RATE,
                                repeats: int = config.N_TIMING_REPEATS) -> ProxyResult:
    """Diagnostic, not a proxy: stored values plus an EXACT repricing of the new trade.

    Equals the true incremental CVA up to floating point, which is what proves the
    incremental proxy's only error source is the interpolation.
    """
    def request():
        v_new = new_trade.value(ctx.market, ctx.scen.times, ctx.scen.spots,
                                ctx.scen.r_d, ctx.scen.r_f)
        return cva_from_values(stored_values + v_new, ctx.scen, ctx.market, hazard,
                               recovery, ctx.discounting) - stored_cva

    cva, seconds = _timed(request, repeats)
    return ProxyResult(cva, seconds, ctx.scen.spots.size)
