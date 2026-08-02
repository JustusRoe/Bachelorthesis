"""
Exposure measures derived from a scenario matrix.

Given the netted scenario values V[p, i] of a netting set, the quantity that
drives CVA is the *expected (positive) exposure*:

        EE(t_i) = E[ max(V(t_i), 0) ]  ~=  (1/P) * sum_p max(V[p,i], 0)        (4)

The positive part is taken AFTER netting (V is already the sum over trades),
because in a default only the net claim against the counterparty is at risk.
That single `max` is where the entire netting benefit comes from, and it is the
reason a standalone (per-trade) charge overstates the true cost.
"""

from __future__ import annotations

import numpy as np


def expected_exposure(values: np.ndarray) -> np.ndarray:
    """EE(t_i): expected positive exposure = mean over paths of max(netted value, 0).

    Parameters
    ----------
    values : shape (n_paths, n_times), the *netted* MtM of the netting set.

    Returns
    -------
    Array of shape (n_times,) - formula (4).
    """
    return np.maximum(values, 0.0).mean(axis=0)


def potential_future_exposure(values: np.ndarray, quantile: float = 0.95) -> np.ndarray:
    """PFE(t_i) at the given confidence level.

    Not used by the CVA formula, but plotted alongside EE because the gap
    between EE and PFE is a convenient visual for the exposure discussion.
    """
    return np.quantile(np.maximum(values, 0.0), quantile, axis=0)
