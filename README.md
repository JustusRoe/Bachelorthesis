# CVA proxy pricing for incoming FX trades

Numerical model for a Bachelor's thesis in quantitative finance.  It compares two
fast **proxy** methods for quoting the CVA charge of a *new* incoming FX trade
against a fully-simulated **true** benchmark, for netting sets of FX forwards and
FX swaps.

Everything is synthetic and self-generated: no bank data, no downloads, one fixed
random seed.

---

## The three CVA quantities

| Quantity | What it is | Sign |
|---|---|---|
| **True incremental CVA** | `CVA(portfolio + new trade) − CVA(portfolio)`, both fully simulated and netted. The economically correct charge. | any |
| **Standalone proxy** | The new trade priced *in isolation*, as if it were the only trade with the counterparty. Structurally blind to netting. | always ≥ 0 |
| **Incremental proxy** | The new trade's scenario values aggregated into the netting set's *stored* scenario matrix, then the difference in total CVA. Netting is exact; the only approximation is the interpolation used to price the incoming trade cheaply. | any |

Incremental CVA is **order-dependent**. Throughout this project it is always
defined as adding the new trade **last** to the existing netting set.

---

## How to run

```bash
pip install -r requirements.txt
python main.py
```

Runtime is about one minute. Every number printed to the console is also written
to `results/`.

To change the experiment, edit **`cva/config.py`** — it holds the seed, the path
count, the time grid, the market parameters, the five netting-set specifications
and the interpolation grid. Nothing else needs touching.

---

## Model

* **Market** — one currency pair, EUR/USD quoted as USD per EUR. Two flat
  risk-free curves (EUR 2%, USD 4%) and a Garman-Kohlhagen spot process, i.e.
  geometric Brownian motion with drift `r_USD − r_EUR` under the USD risk-neutral
  measure and 10% volatility. USD is the reporting currency.
* **Products** — FX forwards and FX swaps only. Both are *linear*, so they are
  valued **analytically** at every scenario and time step from covered interest
  rate parity, `V(t) = φ·N·[DF_EUR(t,T)·S(t) − K·DF_USD(t,T)]`. An FX swap is two
  such legs in opposite directions. This is a **single, non-nested** Monte Carlo:
  outer risk-factor paths plus closed-form repricing at each node.
* **Portfolios** — 5 netting sets (5 counterparties) holding 10 / 20 / 30 / 40 /
  50 trades, maturities from 1 week to 5 years, with the tenor mix and the
  long/short balance varying across the sets so that portfolio-size and
  composition effects are visible. Trades are seasoned (strike = fair forward
  ±3%) so the books carry a non-zero mark-to-market today.
* **Collateral** — none. All netting sets are uncollateralised (no CSA), so
  netting at counterparty level is the only risk mitigant.
* **Default** — hazard-rate / survival-curve approach, `Q(t) = exp(−λt)`, with a
  constant λ per counterparty from 0.5% to 5.0%. Recovery `R = 40%`.
* **CVA** — `CVA = (1−R) · Σᵢ DF(tᵢ) · EE(tᵢ) · PD(tᵢ₋₁, tᵢ)`, where
  `EE(tᵢ) = E[max(V_netted(tᵢ), 0)]`.

Defaults: seed 42, 20,000 paths, 40 exposure dates out to 5 years.

---

## Interpolation

Both proxies avoid repricing the incoming trade on all 20,000 paths. They build
an interpolant of the pricing function `V(t, S)` on a coarse **(time, spot)
tensor grid**, evaluate the exact analytic price only at the grid nodes, and read
the interpolant off at every `(tᵢ, S_path)` pair.

Two methods sit behind one interface — `method="linear"` (the base case) and
`method="chebyshev"` (the optional module). They share only the grid plumbing.

**Why the grid is two-dimensional.** For a linear FX product under deterministic
rates the value is *exactly affine in the spot*, so interpolating in `S` alone
would be exact to machine precision and the comparison would be vacuous. The
genuine approximation comes from coarsening the **time** dimension, where the
discount factors are exponential and a swap leg drops out.

**Maturity cut-offs.** By default each leg is interpolated on its own smooth
domain `[0, T_leg]` and the settlement jump is applied analytically, because that
jump is known exactly and free to apply. `split_legs=False` puts the whole trade
on one domain — a deliberately naive variant kept for the sensitivity study.

---

## Module map

| File | Concern |
|---|---|
| `cva/config.py` | every parameter of the experiment |
| `cva/market.py` | flat curves, discount factors, GBM spot simulation |
| `cva/instruments.py` | `FXForward`, `FXSwap`, analytic valuation |
| `cva/portfolio.py` | the 5 netting sets, the incoming test trades, CVA-delta sizing |
| `cva/scenarios.py` | scenario value matrices, netting-set aggregation |
| `cva/credit.py` | survival curve, marginal default probabilities |
| `cva/exposure.py` | EE and PFE from a scenario matrix |
| `cva/cva_engine.py` | the CVA formula, true CVA, true incremental CVA |
| `cva/interpolation.py` | the (time, spot) tensor interpolant, linear and Chebyshev |
| `cva/proxies.py` | standalone proxy, incremental proxy, full-re-simulation benchmark |
| `cva/metrics.py` | signed / absolute / relative error, speed-up |
| `cva/plots.py` | all six figures |
| `main.py` | orchestration and reporting only |

---

## Incoming test trades

Priced against each of the five netting sets:

| Key | What it is |
|---|---|
| `new_fwd` | plain 3y forward, long EUR 50m — risk-adding |
| `new_swap` | 1y/3y FX swap, EUR 50m — its near leg dies inside the interpolation domain |
| `offset_fwd` | **risk-reducing** forward, built to cancel 50% of the netting set's CVA spot delta. Its incremental CVA is negative while the standalone charge stays positive — the key showcase |
| `small_fwd` | 6m forward, EUR 5m — small reference trade |

The offsetting trade is sized from the **CVA delta**
`dCVA/dS = (1−R)·Σᵢ DF(tᵢ)·PDᵢ·E[1{V>0}·dV/dS]`, not from raw notional: a book can
be net short EUR in value terms while its CVA is driven by long-EUR positions at
the near dates, where the default and discount weights are largest. Its tenor is
capped at the book's longest maturity, so it always has something to net against.

---

## Outputs

### `results/cva_results.csv`
One row per (netting set × incoming trade × interpolation method) — 40 rows.

| Column group | Meaning |
|---|---|
| `portfolio_cva`, `portfolio_mtm_today` | the existing book, before the new trade |
| `true_incremental_cva` | ground truth, full re-simulation |
| `standalone_proxy_cva`, `incremental_proxy_cva` | the two proxy quotes |
| `incremental_exact_cva` | diagnostic: incremental with the new trade repriced *exactly* |
| `*_signed_error` | proxy − truth. **Positive = overcharging** (trade lost to competitors), **negative = undercharging** (loss borne by the bank) |
| `*_abs_error`, `*_rel_error`, `*_direction` | magnitude and the over/under label |
| `interpolation_only_error` | incremental proxy − exact incremental: the pure interpolation error |
| `netting_exactness_check` | exact incremental − truth. ~0 by construction; proves netting in the incremental proxy is exact |
| `*_seconds`, `*_speedup` | wall-clock time per pricing request (best of 3) and speed-up vs full re-simulation |
| `proxy_exact_evaluations`, `full_resim_evaluations` | hardware-independent cost: number of exact analytic prices computed |

### `results/sensitivity_results.csv`
The node-count sweep (3–16 time nodes × both methods) and the swap
domain-handling comparison.

### `results/portfolios.csv`
Every trade of every netting set, leg by leg — for the thesis appendix.

### Figures

| File | Shows |
|---|---|
| `ee_profiles.png` | expected exposure of all five netting sets over time, plus EE vs PFE for one set |
| `cva_comparison.png` | standalone vs incremental vs true, per test trade, per netting set — negative bars highlighted |
| `signed_errors.png` | signed error above/below the zero line, one panel per proxy |
| `accuracy_vs_time.png` | error against wall-clock cost per pricing request |
| `interpolation_convergence.png` | linear vs Chebyshev convergence, and what a discontinuity does to it |
| `error_vs_size.png` | standalone error against netting-set size, absolute and relative |

---

## Reading the timings

The wall-clock speed-ups here (roughly 1.5× to 8×, growing with netting-set size)
**understate** what the proxies buy in practice. In this model the "expensive"
full repricing is itself only a couple of vectorised NumPy expressions per trade,
so the proxy's advantage is limited by the fixed cost of touching the
20,000 × 40 scenario array, which it still has to do. The hardware-independent
measure is the `*_evaluations` columns: the proxy builds its interpolant from
**30 exact prices** where a full re-simulation of the 50-trade netting set
evaluates **8.8 million**. With a realistic pricing library, that ratio — not the
NumPy timings — is what the speed-up would approach.

This is a didactic research prototype, not a production system.
