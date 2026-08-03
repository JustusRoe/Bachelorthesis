"""
Exposure measures derived from a scenario matrix.

Given the netted scenario values V[p, i] of a netting set, the quantity that
drives CVA is the expected positive exposure:

    EE(t_i)  = E[ max(V(t_i), 0) ]                                           (4)
    EE*(t_i) = E[ D_d(0,t_i) * max(V(t_i), 0) ]                              (4')

The positive part is taken AFTER netting (V is already the sum over trades),
because in a default only the net claim against the counterparty is at risk.
That single `max` is where the entire netting benefit comes from, and it is the
reason a standalone per-trade charge overstates the true cost.

Why there are two versions
--------------------------
(4) is the textbook expected exposure and is what the exposure PLOTS show.
(4') keeps the stochastic discount factor INSIDE the expectation.  Under
deterministic rates D_d(0,t_i) is the same number on every path and the two are
related by a constant, EE* = P_d(0,t_i) * EE.  Under stochastic rates the
discount factor and the exposure are correlated - a scenario where domestic
rates fell is both a scenario with a high discount factor and (for a book that
is long foreign currency) a scenario with a particular exposure - so the product
does NOT factorise and only (4') is correct.
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


def discounted_expected_exposure(values: np.ndarray, df_domestic: np.ndarray) -> np.ndarray:
    """EE*(t_i): discounted expected positive exposure, formula (4').

    Parameters
    ----------
    values      : (n_paths, n_times) netted MtM.
    df_domestic : (n_paths, n_times) pathwise D_d(0,t_i) from the simulation.
    """
    return (df_domestic * np.maximum(values, 0.0)).mean(axis=0)


def exposure_standard_error(values: np.ndarray, df_domestic: np.ndarray) -> np.ndarray:
    """Monte Carlo standard error of EE*(t_i), one entry per exposure date.

    Reported so that the size of the simulation noise can be compared against
    the proxy errors the study is trying to measure.
    """
    sample = df_domestic * np.maximum(values, 0.0)
    return sample.std(axis=0, ddof=1) / np.sqrt(sample.shape[0])


def potential_future_exposure(values: np.ndarray, quantile: float = 0.95) -> np.ndarray:
    """PFE(t_i) at the given confidence level.

    Not used by the CVA formula, but plotted alongside EE because the gap
    between EE and PFE is a convenient visual for the exposure discussion.
    """
    return np.quantile(np.maximum(values, 0.0), quantile, axis=0)
