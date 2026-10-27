"""Convergence of the Monte Carlo call price to Black-Scholes.

Prices the at-the-money call at seven path counts, 20 seeds each, and
records the price, standard error, error against Black-Scholes and runtime.
Writes results/convergence.csv and results/convergence_summary.json.
"""

import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from option_pricer.monte_carlo import price_european_call
from option_pricer.vectorized import price as bs_price

S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
PATH_COUNTS = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
SEEDS = range(1, 21)

OUT = Path(__file__).parent / "results"


def run():
    OUT.mkdir(exist_ok=True)
    bs = float(bs_price(S0, K, T, r, sigma, kind="call"))

    runs = []
    for n in PATH_COUNTS:
        for seed in SEEDS:
            np.random.seed(seed)
            t0 = time.perf_counter()
            mc, se = price_european_call(S0, K, r, sigma, T, n)
            ms = (time.perf_counter() - t0) * 1000
            runs.append(dict(n=n, seed=seed, price=mc, se=se, error=mc - bs,
                             covered=abs(mc - bs) <= 1.96 * se, ms=ms))

    with open(OUT / "convergence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)

    rows = []
    for n in PATH_COUNTS:
        group = [x for x in runs if x["n"] == n]
        err = np.array([x["error"] for x in group])
        rows.append(dict(
            n=n,
            mean_se=float(np.mean([x["se"] for x in group])),
            rmse=float(math.sqrt(np.mean(err**2))),
            covered=int(sum(x["covered"] for x in group)),
            median_ms=float(np.median([x["ms"] for x in group])),
        ))

    log_n = np.log10([x["n"] for x in rows])
    slope = lambda key: float(np.polyfit(log_n, np.log10([x[key] for x in rows]), 1)[0])
    summary = dict(
        black_scholes=bs,
        rows=rows,
        slope_se=slope("mean_se"),
        slope_rmse=slope("rmse"),
        slope_runtime=slope("median_ms"),
        covered_total=sum(x["covered"] for x in rows),
        runs_total=len(runs),
        payoff_sd=rows[-1]["mean_se"] * math.sqrt(rows[-1]["n"]),
    )
    (OUT / "convergence_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    s = run()
    print(f"Black-Scholes price  {s['black_scholes']:.6f}")
    print(f"{'N':>10} {'mean SE':>9} {'RMSE':>9} {'in 95% CI':>10} {'ms':>8}")
    for x in s["rows"]:
        print(f"{x['n']:>10,} {x['mean_se']:>9.4f} {x['rmse']:>9.4f} {x['covered']:>7}/20 {x['median_ms']:>8.2f}")
    print(f"slope of log SE on log N   {s['slope_se']:.3f}  (theory -0.5)")
    print(f"slope of log RMSE on log N {s['slope_rmse']:.3f}")
    print(f"slope of log time on log N {s['slope_runtime']:.2f}")
    print(f"coverage {s['covered_total']}/{s['runs_total']}, payoff sd {s['payoff_sd']:.2f}")
    print("runtimes depend on your machine; everything else is exact for these seeds")
