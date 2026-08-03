"""
Scenario matrix builder.

A "scenario matrix" is the mark-to-market of a trade (or of a whole netting set)
on the outer Monte Carlo grid:

        V[p, i] = value in USD on path p at exposure date t_i

Because both products are linear, every entry is produced by the closed-form
formula in `instruments.py` - there is no inner simulation anywhere, under
either rate model.

The netting-set matrix computed here is exactly the object a bank would store
overnight and reuse the next day.  The *incremental proxy* is fast precisely
because it reuses this stored matrix instead of re-simulating the book.
"""

from __future__ import annotations

import numpy as np

from .instruments import Trade


def trade_scenario_values(trade: Trade, market, scen) -> np.ndarray:
    """Exact analytic MtM of one trade on the whole (path, date) grid.

    Returns an array of shape (n_paths, n_times) in USD.
    """
    return trade.value(market, scen.times, scen.spots, scen.r_d, scen.r_f)


def netting_set_scenario_values(trades: list[Trade], market, scen) -> np.ndarray:
    """Netted MtM of a whole portfolio: V_NS[p,i] = sum_k V_k[p,i].

    Netting is applied at *counterparty level*: we sum the signed trade values
    first and only later take the positive part (see `exposure.py`).  Summing
    before flooring at zero is what creates the netting benefit.
    """
    total = np.zeros(scen.spots.shape, dtype=float)
    for trade in trades:
        total += trade.value(market, scen.times, scen.spots, scen.r_d, scen.r_f)
    return total
