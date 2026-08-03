"""
Where does a trade's risk come from: the FX spot, or the rate differential?

This module answers the question that motivated the whole upgrade.  An FX swap
holds two legs in opposite directions, so its spot sensitivity is the DIFFERENCE
of two foreign discount factors at nearby maturities - almost nothing - while its
rate sensitivities do not cancel, because the two legs discount over different
horizons.  Under deterministic rates the swap therefore has essentially no risk
driver before its near leg settles, and its pre-settlement exposure collapses
towards zero.  Once the near leg has settled the swap is just an outright
forward and the spot takes over.

Method
------
The trade's instantaneous variance follows from its three analytic deltas.  With
factors ordered (S, x_d, x_f), driven by

    dS = ... + sigma_S S dW_S,   dx_d = ... + sigma_d dW_d,   dx_f = ... + sigma_f dW_f

the instantaneous variance of the trade value is

    Var(dV)/dt = sum_{i,j} Delta_i Delta_j v_i v_j rho_ij                    (R1)

where v = (sigma_S * S, sigma_d, sigma_f) and Delta is the vector of deltas from
`instruments.py`.  Splitting (R1) into its blocks gives

    FX     = (Delta_S sigma_S S)^2
    RATES  = Delta_d^2 sigma_d^2 + Delta_f^2 sigma_f^2
             + 2 rho_df Delta_d Delta_f sigma_d sigma_f
    CROSS  = 2 Delta_S sigma_S S (rho_dS Delta_d sigma_d + rho_fS Delta_f sigma_f)

The three sum to the total.  The cross term can be negative, so the reported
shares can fall outside 0-100%; a "pure" share ignoring the cross term is
therefore reported alongside, as FX / (FX + RATES).

Everything is averaged across paths at each date - the variances are averaged,
not the shares, which is the correct aggregation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .instruments import Trade


def variance_components(trade: Trade, market, scen) -> dict[str, np.ndarray]:
    """The three blocks of formula (R1), averaged over paths, per exposure date.

    Returns arrays of shape (n_times,) for "fx", "rates", "cross" and "total".
    """
    times = scen.times
    sigma_d = market.hw_domestic.sigma
    sigma_f = market.hw_foreign.sigma
    sigma_s = market.sigma_fx

    delta_s = trade.delta_spot(market, times, scen.r_f)
    delta_d = trade.delta_x_domestic(market, times, scen.r_d)
    delta_f = trade.delta_x_foreign(market, times, scen.spots, scen.r_f)

    fx_move = delta_s * sigma_s * scen.spots        # exposure to a 1-sigma FX move
    d_move = delta_d * sigma_d
    f_move = delta_f * sigma_f

    fx = fx_move**2
    rates = d_move**2 + f_move**2 + 2.0 * market.rho_df * d_move * f_move
    cross = 2.0 * fx_move * (market.rho_ds * d_move + market.rho_fs * f_move)

    return {
        "fx": fx.mean(axis=0),
        "rates": rates.mean(axis=0),
        "cross": cross.mean(axis=0),
        "total": (fx + rates + cross).mean(axis=0),
    }


def risk_split_table(
    trades: dict[str, Trade],
    market,
    scen,
    report_dates: tuple[float, ...] = (0.25, 0.75, 1.5, 2.5),
    near_leg_date: float = 1.0,
) -> pd.DataFrame:
    """Tabulate the FX / rates split for several trades at several dates.

    `report_dates` should straddle `near_leg_date` so that the before/after
    contrast is visible; the table flags which side of the settlement each row
    falls on.
    """
    times = np.asarray(scen.times, dtype=float)
    rows = []

    for name, trade in trades.items():
        components = variance_components(trade, market, scen)
        for target in report_dates:
            col = int(np.argmin(np.abs(times - target)))
            fx = float(components["fx"][col])
            rates = float(components["rates"][col])
            cross = float(components["cross"][col])
            total = float(components["total"][col])

            # Standard deviations are easier to read than variances.
            rows.append(dict(
                trade=name,
                date_y=round(float(times[col]), 3),
                phase="before near-leg settlement" if times[col] < near_leg_date
                      else "after near-leg settlement",
                total_vol_usd=np.sqrt(max(total, 0.0)),
                fx_share=fx / total if total else np.nan,
                rates_share=rates / total if total else np.nan,
                cross_share=cross / total if total else np.nan,
                fx_share_excl_cross=fx / (fx + rates) if (fx + rates) else np.nan,
            ))

    return pd.DataFrame(rows)
