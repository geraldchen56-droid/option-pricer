"""Convergence of the Monte Carlo call price to Black-Scholes.

Prices the at-the-money call at seven path counts, 20 seeds each, and
records the price, standard error, error against Black-Scholes and runtime.
Writes results/convergence.csv, results/convergence_summary.json and
Figure 1 (results/figure1.pdf and .png).
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


def figure1(s):
    """Figure 1: mean SE and RMSE against N on log-log axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    n = np.array([x["n"] for x in s["rows"]])
    se = np.array([x["mean_se"] for x in s["rows"]])
    rmse = np.array([x["rmse"] for x in s["rows"]])

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.0, 3.9))

    grid = np.logspace(3, 6, 50)
    ax.plot(grid, se[0] * np.sqrt(n[0]) / np.sqrt(grid), ls="--", color="0.55", lw=1.2,
            label=r"Theory: slope $-1/2$ (lies under the data)")
    ax.plot(n, se, "o-", color="#1f4e8c", lw=1.8, ms=5,
            label=f"Mean standard error (fitted slope $-${abs(s['slope_se']):.3f})")
    ax.plot(n, rmse, "s", color="0.35", ms=4.5, mfc="white", mew=1.1, label="RMSE vs Black-Scholes")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(7e2, 1.5e6)
    ax.set_ylim(0.01, 0.8)
    ax.yaxis.set_major_locator(FixedLocator([0.01, 0.03, 0.1, 0.3]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v):,}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Number of simulated paths, $N$")
    ax.set_ylabel("Pricing error (price units)")
    ax.grid(True, which="major", color="0.9", lw=0.6)
    ax.legend(frameon=False, loc="lower left", fontsize=10)
    ax.annotate(f"{se[0]:.3f}", (n[0], se[0]), xytext=(8, 4), textcoords="offset points",
                fontsize=9.5, color="#1f4e8c")
    ax.annotate(f"{se[-1]:.4f}", (n[-1], se[-1]), xytext=(-46, -4), textcoords="offset points",
                fontsize=9.5, color="#1f4e8c")

    fig.tight_layout()
    fig.savefig(OUT / "figure1.pdf")
    fig.savefig(OUT / "figure1.png", dpi=200)
    plt.close(fig)


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
    figure1(s)
    print("figure 1 written to results/figure1.pdf and results/figure1.png")
    print("runtimes depend on your machine; everything else is exact for these seeds")
