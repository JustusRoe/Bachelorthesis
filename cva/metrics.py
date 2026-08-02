"""
Error metrics comparing a proxy charge against the true incremental CVA.

Sign convention (this is the economically important part)
---------------------------------------------------------
        signed_error = CVA_proxy - CVA_true_incremental

    signed_error > 0  ->  the proxy quotes MORE than the trade really costs
                          -> OVERCHARGING -> the trade is likely lost to a
                             competitor who prices it correctly.
    signed_error < 0  ->  the proxy quotes LESS than the trade really costs
                          -> UNDERCHARGING -> the bank absorbs the difference.

The relative error is expressed against |CVA_true_incremental|.  It is reported
as NaN when the true incremental CVA is (near) zero, because a ratio against
zero carries no information.
"""

from __future__ import annotations

import numpy as np


def signed_error(proxy_cva: float, true_cva: float) -> float:
    """Proxy minus truth: positive = overcharge, negative = undercharge."""
    return float(proxy_cva - true_cva)


def absolute_error(proxy_cva: float, true_cva: float) -> float:
    """|proxy - truth| in USD."""
    return float(abs(proxy_cva - true_cva))


def relative_error(proxy_cva: float, true_cva: float, floor: float = 1.0) -> float:
    """Signed relative error (proxy - truth) / |truth|.

    `floor` (in USD) guards against dividing by a true incremental CVA that is
    numerically zero; below it the relative error is undefined and NaN is
    returned.
    """
    if abs(true_cva) < floor:
        return float("nan")
    return float((proxy_cva - true_cva) / abs(true_cva))


def charging_direction(err: float, tolerance: float = 1.0) -> str:
    """Human-readable label used in the results table."""
    if err > tolerance:
        return "overcharge"
    if err < -tolerance:
        return "undercharge"
    return "accurate"


def summarise_errors(proxy_cva: float, true_cva: float) -> dict[str, float | str]:
    """All three error measures plus the direction label, as one row fragment."""
    err = signed_error(proxy_cva, true_cva)
    return {
        "signed_error": err,
        "abs_error": absolute_error(proxy_cva, true_cva),
        "rel_error": relative_error(proxy_cva, true_cva),
        "direction": charging_direction(err),
    }


def speedup(reference_seconds: float, proxy_seconds: float) -> float:
    """How many times faster the proxy is than the full re-simulation."""
    if proxy_seconds <= 0:
        return float("inf")
    return float(reference_seconds / proxy_seconds)
