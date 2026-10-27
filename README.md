# option-pricer

European option pricing and Greeks under Black-Scholes-Merton, plus a Monte
Carlo pricer benchmarked against the closed form. The Black-Scholes core is
pure standard library; the simulation modules need numpy.

## Reproducing the paper

The `paper/` folder regenerates every table and figure in the accompanying
paper on Monte Carlo convergence and moneyness. All draws are seeded, so
prices, standard errors and coverage counts match the paper exactly; only
runtimes depend on your machine.

```bash
pip install -e ".[sim]"
python paper/convergence.py        # Table 1, coverage, slopes -> paper/results/
python paper/moneyness_hedging.py  # Tables 2 and 3
python paper/figure1.py            # Figure 1 (run convergence.py first)
```

**Live demo: [root-t.vercel.app](https://root-t.vercel.app)** — the Monte Carlo
vs Black-Scholes price surface with a volatility slider, running in the browser.
Source in [`web/`](web/).

## Install

```bash
pip install -e ".[dev]"
```

## CLI

```bash
option-pricer -S 100 -K 100 -T 1 -r 5% -v 20%
```

```
Call  S=100  K=100  T=1y
  price      10.450584
  delta       0.636831
  gamma       0.018762
  vega        0.375240   per 1 vol point
  theta      -0.017573   per day
  rho         0.532325   per 1 rate point
```

Flags: `-S` spot, `-K` strike, `-T` expiry in years, `-r` risk-free rate,
`-v` volatility, `-q` dividend yield, `-k/--kind` `call|put`, `--json`.

Rates and volatility accept decimals or percentages — `-v 0.2` and `-v 20%`
are the same. Use `--json` for machine-readable output.

## Library

```python
from option_pricer import price, greeks

price(S=100, K=100, T=1.0, r=0.05, sigma=0.2, kind="call")
# 10.450583572185565

g = greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.2, kind="put")
g.delta, g.vega
# (-0.3631693488243809, 0.3752403469169379)
```

`q` handles a continuous dividend yield, which also covers Black-76 futures
options (`q = r`) and FX options (`q` = the foreign rate).

## Monte Carlo: GBM terminal prices

`option_pricer.gbm` samples terminal prices under risk-neutral geometric
Brownian motion. Requires numpy (`pip install -e ".[sim]"`); matplotlib only for
the plot.

```bash
gbm-demo --seed 42                  # 100,000 draws, writes gbm_terminal_prices.png
gbm-demo -n 1000000 --no-plot       # just the mean check
```

```
GBM terminal prices  S0=100  r=5.00%  sigma=20.00%  T=1y
  simulations                  100,000
  sample mean E[S_T]        105.054866
  theoretical S0*e^(rT)     105.127110
  error                      -0.072244  (-0.0687%)
  monte carlo std err         0.067433  (-1.07 sigma)
```

```python
import numpy as np
from option_pricer.gbm import simulate_terminal_prices

S_T = simulate_terminal_prices(S0=100, r=0.05, sigma=0.2, T=1.0, n_sims=100_000)
S_T.mean()   # ~= 100 * exp(0.05)
```

The terminal price uses the closed-form solution of the GBM SDE:

    S_T = S0 * exp((r - q - sigma^2/2) T + sigma sqrt(T) Z),   Z ~ N(0, 1)

so it takes **one normal draw per simulation, with no loop over days**. Log
returns are independent Gaussian increments, so 252 daily steps sum to a normal
with variance `sigma^2 T` -- the same distribution the single draw already
samples, at 252x the cost. Step through a calendar only for path-dependent
payoffs (Asian, barrier, lookback), where the path itself is the input.

The `- sigma^2/2` term is the Ito correction. Drop it and the *median* is still
right while the mean overshoots by `exp(sigma^2 T / 2)`; keep it and
`E[S_T] = S0 e^((r-q)T)` holds exactly. That identity is what `--seed 42` above
checks, and it is why the reported error is quoted in standard errors: at
100,000 draws the standard error is ~0.067, so a -0.07 miss is noise, whereas a
consistent +1.0 would mean the correction had been dropped.

## Interactive 3D Greek surfaces

`option_pricer.surface` writes a standalone HTML file: a Greek over
spot x time-to-expiry, rotatable by dragging, with a volatility slider that
morphs the surface. Needs numpy, scipy and plotly (`pip install -e ".[sim]"`).

```bash
greek-surface                                  # gamma, call, -> greek_surface.html
greek-surface -g vega -o vega.html
greek-surface -g theta -k put --zscale frame
```

Gamma is the striking one: its at-the-money peak sharpens as either volatility
or time to expiry falls, because gamma scales as `1/(sigma sqrt(T))`.

`plotly.js` is inlined, so the file is ~5 MB but needs no network -- open it
straight from disk, mail it, drop it in a repo.

### The z-axis choice

By default the z axis and colour range are **pinned across every frame**, so
peak heights stay comparable as the slider moves. This matters: autoscaling per
frame would make a surface whose peak drops 4x look completely unchanged, which
is the entire thing the animation exists to show.

The cost is dynamic range. Gamma at 10% vol is ~4x gamma at 40%, so with the
axis pinned to the low-vol spike the high-vol frames sit low. That is why the
default sweep is 10-40% rather than wider -- past ~4x the tall frames flatten
the rest into sheets. `--zscale frame` rescales each frame instead, and then
labels both the title and the z axis to say peaks are no longer comparable.

## Module layout

| Module | Depends on | Purpose |
| --- | --- | --- |
| `black_scholes` | stdlib only | scalar pricing + Greeks |
| `vectorized` | numpy, scipy | same formulas, broadcasting over arrays |
| `gbm` | numpy (+matplotlib) | Monte Carlo terminal prices |
| `surface` | numpy, scipy, plotly | interactive 3D Greek surfaces |
| `cli` | stdlib only | command-line front end |

`vectorized` exists because a surface is ~65,000 evaluations and an animation
re-evaluates per frame; scalar Python calls would take seconds per render. It
duplicates the formulas, so `tests/test_vectorized.py` pins the two
implementations to each other at `rel=1e-12` across a grid of 240 parameter
combinations, including every degenerate branch.

## Greek conventions

Values use market conventions rather than raw partial derivatives:

| Greek | Meaning | Unit |
| --- | --- | --- |
| `delta` | dP/dS | raw |
| `gamma` | d²P/dS² | raw |
| `vega` | dP/dσ | per 1 vol point (0.01) |
| `theta` | -dP/dT | per calendar day |
| `rho` | dP/dr | per 1 rate point (0.01) |

## Edge cases

At `T = 0` or `σ = 0` the price collapses to the discounted forward intrinsic
and the Greeks are reported as zero apart from the step-function delta, since
gamma and vega are degenerate there.

Note that a European put can trade below `K - S`: exercise cannot be brought
forward, so the strike is only received at expiry. The enforced lower bound is
the discounted one, `K·e^(-rT) - S·e^(-qT)`.

## Tests

```bash
python3 -m pytest
```

115 tests: closed-form values against Hull's reference figures, put-call parity
(with and without dividends), monotonicity, every Greek cross-checked against
central finite differences, and the degenerate expiry/zero-vol branches. The GBM tests check the
simulated mean, median, variance, and log-return skew/kurtosis against
theory, and every histogram bin against its binomial standard error.
