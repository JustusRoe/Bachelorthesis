"""
Counterparty credit: survival curve and marginal default probabilities.

We use the standard reduced-form (hazard-rate) approach.  With a constant
hazard rate lambda the survival probability to time t is

        Q(t) = P(tau > t) = exp(-lambda * t)                                   (2)

and the *marginal* probability of defaulting inside the bucket (t_{i-1}, t_i] is

        PD(t_{i-1}, t_i) = Q(t_{i-1}) - Q(t_i)                                 (3)

These marginal PDs are the weights of the CVA sum in `cva_engine.py`.

Independence assumption: default is taken to be independent of the FX spot, so
no wrong-way risk is modelled.  That lets us multiply EE and PD term by term.
"""

from __future__ import annotations

import numpy as np


def survival_probability(times: np.ndarray, hazard: float) -> np.ndarray:
    """Q(t) = exp(-lambda * t), formula (2).

    A term structure of hazard rates could be dropped in here by replacing the
    product `hazard * times` with the integrated hazard; nothing downstream
    would have to change.
    """
    return np.exp(-hazard * np.asarray(times, dtype=float))


def marginal_default_probabilities(times: np.ndarray, hazard: float) -> np.ndarray:
    """PD(t_{i-1}, t_i) for i = 1..n, formula (3).

    Parameters
    ----------
    times : the exposure grid *including* t_0 = 0.

    Returns
    -------
    Array of length len(times) - 1: the default probability attached to each
    exposure date from t_1 onwards.
    """
    q = survival_probability(times, hazard)
    return q[:-1] - q[1:]
