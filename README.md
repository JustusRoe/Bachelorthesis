# CVA proxy pricing for incoming FX trades — three-factor model

Numerical model for a Bachelor's thesis in quantitative finance.  It compares two
fast **proxy** methods for quoting the CVA charge of a *new* incoming FX trade
against a fully-simulated **true** benchmark, for netting sets of FX forwards and
FX swaps, under a three-factor market model with stochastic interest rates.

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

Runtime is about two minutes. The model validation suite runs **first** and the
script refuses to report any CVA if a check fails. Every number printed to the
console is also written to `results/`.

To change the experiment, edit **`cva/config.py`** — it holds the seed, the path
count, the time grid, the curves, all three-factor parameters, the five
netting-set specifications and the interpolation grid. Nothing else needs
touching.

---

## Market model

Three correlated risk factors under the **domestic (USD) risk-neutral measure**:

```
dr_d = [θ_d(t) − a_d·r_d] dt + σ_d dW_d                              (Hull-White, USD)
dr_f = [θ_f(t) − a_f·r_f − ρ_fS·σ_f·σ_S] dt + σ_f dW_f               (Hull-White, EUR)
dS   = (r_d − r_f)·S dt + σ_S·S dW_S                                 (FX spot)
```

* **Quanto adjustment.** The `−ρ_fS σ_f σ_S` term in the foreign drift is the
  change of measure from the foreign to the domestic risk-neutral measure.
  Omitting it breaks covered interest rate parity in the simulation — the
  validation suite measures the resulting bias and matches it against the closed
  form.
* **Exact curve fit.** `θ(t)` is never formed explicitly. The short rate is
  decomposed as `r = x + α` with `x` a mean-zero OU factor and
  `α(t) = f(0,t) + (σ²/2a²)(1−e^{−at})²`, and bonds are rebuilt from
  `P(t,T) = A(t,T)·exp(−B(t,T)·r(t))` with the exact-fit `A(t,T)`
  (Brigo–Mercurio 3.39). The model reprices today's curves by construction.
* **Initial curves** are Nelson-Siegel — a smooth parametric form, chosen because
  the Hull-White fit needs the *instantaneous forward* `f(0,t)`, and
  differentiating an interpolated pillar curve gives a jagged forward curve.
* **Valuation stays analytic.** `V = φ·N·[S·P_f(t,T) − K·P_d(t,T)]`, evaluated in
  closed form at every scenario node. This is a **single, non-nested** Monte
  Carlo under both rate models.
* **Discretisation.** The only one in the model: the FX drift needs `∫r dt` per
  step, split into `∫α dt` (analytic, closed form) plus `∫x dt` (trapezoidal).
  Integrating the deterministic part exactly is what makes the σ→0 limit exact
  to machine precision rather than merely accurate.

### The switch

`RATE_MODEL = "stochastic" | "deterministic"`. Deterministic mode simply forces
`σ_d = σ_f = 0`; everything else — curves, FX vol, seed, trades — is shared, so
any difference between the two runs is attributable to rate volatility alone.
`main.py` runs **both** and reports them side by side.

---

## Why the upgrade was needed

An FX swap holds two legs in opposite directions, so its spot delta is the
*difference* of two foreign discount factors at nearby maturities — a few percent
of a comparable forward. Its rate deltas do not cancel, because the two legs
discount over different horizons. Measured on the incoming 1y/3y swap:

| | mean pre-settlement EE | peak pre-settlement EE | post-settlement EE |
|---|---|---|---|
| deterministic rates | USD 48k | USD 88k | USD 2.80m |
| stochastic rates | USD 188k | USD 348k | USD 2.68m |
| **ratio** | **3.90×** | **3.95×** | 0.96× |

The plain 3y forward moves by 0.95× throughout — it is a spot product under both
models. The variance decomposition puts numbers on the mechanism: before
settlement the swap is **78% rates / 7% FX**; after settlement it flips to
**107% FX / 2% rates** (shares net to 100% including a signed cross term).

---

## Validation suite (`cva/validation.py`)

Runs before anything else. All five pass.

| Test | What it checks |
|---|---|
| **1. Curve reproduction** | `E[D_d(0,T)] = P_d(0,T)` for USD; `E[D_d(0,T)·S(T)] = S(0)·P_f(0,T)` for EUR (under the domestic measure the naive `E[exp(−∫r_f)]` is *not* the right identity), plus a forward-starting foreign bond check. Passes within 3 MC standard errors at every tenor. |
| **2. FX martingale** | A forward struck at `F(0,T)` prices to zero. Largest deviation 13 bp of the forward, inside the 3-σ band. |
| **3. Quanto** | Run at `ρ_fS = 0` and `−0.40`, each **with and without** the adjustment on **common random numbers**. With it, the forward is fair; without it the bias is **28.64 bp observed vs 28.60 bp predicted** by the closed form. Common random numbers are essential here — the bias is smaller than the raw MC noise (22 bp) and only the paired difference resolves it. |
| **4. Zero-vol degeneracy** | Deterministic mode ≡ `σ_rate = 0`; flat curves reproduce the closed-form single-factor GBM base model, its `exp(−r t)` discounting and its `exp(−r(T−t))` bonds. All to **≤ 3e-15**. |
| **5. Convergence** | CVA vs path count with 95% intervals. Standard-error scaling exponent **−0.510** against the theoretical −0.5; the production estimate sits inside the 40,000-path interval. |

---

## Interpolation

Both proxies avoid repricing the incoming trade on every path. They build an
interpolant of the pricing function on a coarse tensor grid over
**(time) × (state)**, evaluate the exact analytic price only at the nodes, and
read the interpolant off at every simulated point.

The state was **1-dimensional** under deterministic rates (the spot alone) and is
**3-dimensional** under the three-factor model (`S, x_d, x_f`). With `n` nodes per
state dimension the grid holds `n_time × n³` nodes:

| nodes per state dim | 1-D grid | 3-D grid | growth |
|---|---|---|---|
| 2 | 32 | 128 | 4× |
| 3 | 48 | 432 | 9× |
| 4 | 64 | 1,024 | 16× |
| 5 | 80 | 2,000 | 25× |
| 6 | 96 | 3,456 | 36× |

Two methods sit behind one interface — `method="linear"` (base case) and
`method="chebyshev"` (the optional tensor module).

**The spot dimension is free.** `V` is *exactly affine in S* even under
stochastic rates, so two nodes reproduce it to machine precision. In the 1-D
sweep both methods are perfectly flat in the node count — all their residual
error comes from the time dimension. All genuine state curvature lives in `x_d`
and `x_f`, where the bond prices are exponential.

**Anisotropic grids act on that.** `config.anisotropic_nodes(n)` spends 2 nodes
on the spot and `n` on each rate factor, giving `2n²` state nodes instead of `n³`:

| nodes per rate dim | isotropic `n³` | anisotropic `2n²` | node reduction | mean rel. error |
|---|---|---|---|---|
| 3 | 432 | 288 | 1.5× | identical |
| 4 | 1,024 | 512 | 2.0× | identical |
| 5 | 2,000 | 800 | 2.5× | identical |
| 6 | 3,456 | 1,152 | 3.0× | identical |

"Identical" means the two grids agree to 7–13 significant figures; the residual
difference is float associativity from the different contraction shape. Cutting
the spot dimension to two nodes costs nothing, exactly as the affine structure
predicts, and is the production default (`config.PRODUCTION_STATE_NODES`) for
both proxies.

**The Chebyshev evaluator was rewritten to use it.** Profiling the tensor
contraction showed the *first* state dimension going through a BLAS batched
matmul while every *remaining* dimension fell back to a Python-level
broadcast-multiply-and-sum — assumed cheap because each step shrinks the tensor
by a factor of `n`, but at 20,000 paths it was the single largest cost in the
proxy, ahead of the matmul it was meant to be cheaper than. Routing every
dimension through the same batched-matmul trick (reshaping each basis to a
`(P, T, 1, n)` batch of row vectors) cut the per-call cost roughly 4×. Combined
with the anisotropic grid, the Chebyshev proxy's speed-up against full
re-simulation went from **0.94×** (slower than simply rebuilding the book) to
**2.8×** — restoring the point of using it as a proxy at all.

**Maturity cut-offs.** By default each leg is interpolated on its own smooth
domain `[0, T_leg]` and the settlement jump is applied analytically.
`split_legs=False` puts the whole trade on one domain — the naive variant kept
for the sensitivity study, where both methods lose ~12% and Chebyshev loses
*more* than linear, because a global polynomial cannot represent a discontinuity.

---

## Module map

Nine files. `config.py` imports nothing from the package, so it reads top to
bottom as pure data.

| File | Concern |
|---|---|
| `cva/config.py` | every parameter, each labelled [CONVENTION] / [CHOICE] / [SYNTHETIC] |
| `cva/model.py` | Nelson-Siegel curves, Hull-White (`A`, `B`, `α`, OU transition, quanto shift), the three-factor simulation |
| `cva/products.py` | `FXForward`, `FXSwap`, analytic valuation, the three analytic deltas, scenario matrices |
| `cva/portfolio.py` | the 5 netting sets, the incoming test trades, CVA-delta sizing |
| `cva/cva.py` | hazard term structure, EE / discounted EE / PFE, the CVA formula, error metrics |
| `cva/proxies.py` | the N-dimensional tensor interpolant (linear + Chebyshev) and the two proxies |
| `cva/validation.py` | the five mandatory tests plus the FX-vs-rates risk decomposition |
| `cva/plots.py` | all eleven figures |
| `main.py` | orchestration and reporting only |

---

## Model parameters

| Parameter | Value | Status |
|---|---|---|
| Curve form | Nelson-Siegel | **Market convention**; coefficients synthetic |
| USD curve | β₀=4.0%, β₁=0.5%, β₂=−0.8%, τ=1.5 | Synthetic |
| EUR curve | β₀=2.2%, β₁=0.4%, β₂=−0.6%, τ=1.5 | Synthetic; keeps the ~2% differential of the base model |
| `a_d` / `a_f` | 0.05 / 0.03 | Modelling choice; 1F-HW calibrations land in 0.01–0.10 |
| `σ_d` / `σ_f` | 80bp / 60bp | Modelling choice; typical 1F-HW range 50–120bp |
| `σ_S` | 10% | Unchanged from the base model, so old-vs-new is about rates |
| `ρ_dS` | −0.15 | USD rates ↑ → USD strengthens → S falls |
| `ρ_fS` | +0.25 | EUR rates ↑ → EUR strengthens → S rises |
| `ρ_df` | +0.35 | USD/EUR rates co-move |
| Hazard | piecewise constant, upward sloping, breaks at 1y/3y | Modelling choice; IG credit curves normally slope up |
| `R` | 40% | **Market convention**, senior unsecured |
| Paths / dates | 10,000 / 40 | Reduced from 20,000: three factors and an `n³` grid cost more |

---

## Incoming test trades

| Key | What it is |
|---|---|
| `new_fwd` | plain 3y forward, long EUR 50m — risk-adding |
| `new_swap` | 1y/3y FX swap, EUR 50m — near leg dies inside the interpolation domain; rates-driven before settlement |
| `offset_fwd` | **risk-reducing** forward, built to cancel 50% of the netting set's CVA spot delta. Incremental CVA negative, standalone positive — the key showcase |
| `small_fwd` | 6m forward, EUR 5m — small reference trade |

The offsetting trade is sized from `dCVA/dS(0)`, not from raw notional: a book can
be net short EUR in value terms while its CVA is driven by long-EUR positions at
the near dates, where the default and discount weights are largest. **Both rate
models price the identical trade objects**, so the comparison is like-for-like.

---

## Outputs

### CSVs

| File | Contents |
|---|---|
| `cva_results.csv` | one row per (rate model × netting set × trade × interpolation method) — 80 rows |
| `sensitivity_results.csv` | node-count sweep (1-D and 3-D, both methods) and the swap domain-handling comparison |
| `rate_model_comparison.csv` | deterministic vs stochastic, averaged by trade type |
| `risk_decomposition.csv` | FX / rates / cross variance shares before and after settlement |
| `validation_results.csv` | every validation check with estimate, target, tolerance, pass/fail |
| `mc_convergence.csv` | CVA and confidence interval by path count |
| `portfolios.csv` | every trade of every netting set, leg by leg |

Key columns in `cva_results.csv`: `*_signed_error` (**positive = overcharging**,
negative = undercharging), `interpolation_only_error` (the pure interpolation
error), `netting_exactness_check` (~0 by construction — proves netting in the
incremental proxy is exact), `*_seconds` / `*_speedup`, and
`proxy_exact_evaluations` vs `full_resim_evaluations` (the hardware-independent
cost measure).

### Figures

| File | Shows |
|---|---|
| `swap_exposure_rate_models.png` | **the motivating chart** — swap exposure under both rate models, with the forward as a control |
| `risk_decomposition.png` | FX vs rates variance shares, before and after near-leg settlement |
| `ee_profiles.png` | expected exposure of all five netting sets, plus EE vs PFE |
| `cva_comparison.png` | standalone vs incremental vs true, per trade, per netting set |
| `signed_errors.png` | signed error above/below the zero line, one panel per proxy |
| `accuracy_vs_time.png` | error against wall-clock cost per pricing request |
| `interpolation_convergence.png` | 1-D vs 3-D convergence, the same errors priced in repricings, and the naive-domain failure |
| `grid_shape.png` | isotropic `n³` vs anisotropic `2n²` grids: accuracy, node count and break-even |
| `error_vs_size.png` | standalone error against netting-set size, absolute and relative |
| `rate_model_comparison.png` | deterministic vs stochastic, side by side |
| `mc_convergence.png` | CVA and standard error against path count |

---

## Reading the timings

At the production settings (20,000 paths, `PRODUCTION_STATE_NODES` — 2 spot / 5
rate nodes) the linear proxy runs at roughly **3.6× the speed** of a full
re-simulation and the Chebyshev proxy at roughly **2.8×**, both essentially flat
across trade types because their cost is dominated by the grid evaluation, not by
the trade being priced. The `interp_method` breakdown in `cva_results.csv`
carries the exact per-request timings.

Both routes through the tensor contraction go through BLAS batched matmuls end
to end (verified against a reference implementation, bit-identical for the
linear path). An earlier version of the Chebyshev evaluator left the later
contraction dimensions as a Python-level broadcast-sum, which looked cheap
locally (each step shrinks the tensor by a factor of `n`) but dominated the
total cost at production path counts and made the isotropic-grid proxy *slower*
than full re-simulation (0.94×) — a proxy with no speed advantage has no reason
to exist. Fixing the contraction and defaulting to the anisotropic grid recovered
the ~3× speed-up. `interpolation_convergence.png` and `grid_shape.png` still show
the isotropic sweep for the node-count comparison; `cva_results.csv` reflects the
production (anisotropic) default. Wall-clock figures carry run-to-run noise of
roughly ±15-20% on this machine; the node-count reduction is exact and is the
more reliable half of the cost argument.

The hardware-independent measure tells the other half of the story: the proxy
builds its interpolant from **~940 exact prices** where a full re-simulation of
the netting sets evaluates **12.4 million**. In this project the "expensive"
repricing is itself only a couple of vectorised NumPy expressions, so wall-clock
speed-ups understate what the proxies buy. With a realistic pricing library the
ratio of exact-price counts — not the NumPy timings — is what the speed-up would
approach.

This is a didactic research prototype, not a production system.
