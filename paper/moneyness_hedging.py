"""Pricing error across strikes and maturities, and the delta-hedging test.

Writes results/moneyness.csv and results/hedging.csv.
"""

import csv
from pathlib import Path

import numpy as np

from option_pricer.delta_hedge import hedge_short_call, simulate_paths
from option_pricer.surface_mc import monte_carlo_surface
from option_pricer.vectorized import price as bs_price

S0, r, sigma = 100.0, 0.05, 0.2
OUT = Path(__file__).parent / "results"

BANDS = [
    ("Deep in the money", 60, 80),
    ("In the money", 80, 95),
    ("At the money", 95, 105),
    ("Out of the money", 105, 120),
    ("Deep out of the money", 120, 140.1),
]


def moneyness():
    strikes = np.linspace(60.0, 140.0, 40)
    maturities = np.linspace(0.1, 2.0, 20)
    np.random.seed(42)
    mc = monte_carlo_surface(S0, r, sigma, strikes, maturities, 100_000)
    bs = bs_price(S0, strikes[None, :], maturities[:, None], r, sigma, kind="call")
    err = mc - bs
    rel = np.abs(err) / bs

    rows = []
    for name, lo, hi in BANDS:
        m = (strikes >= lo) & (strikes < hi)
        rows.append(dict(
            band=name,
            mean_bs=float(bs[:, m].mean()),
            mean_abs_error=float(np.abs(err[:, m]).mean()),
            mean_rel_error=float(rel[:, m].mean()),
            worst_rel_error=float(rel[:, m].max()),
        ))
    return rows, float(err.mean())


def hedging():
    np.random.seed(42)
    paths = simulate_paths(S0, r, sigma, 1.0, 252, 10_000)
    rows = []
    for every in (252, 36, 12, 4, 1):
        pnl, premium = hedge_short_call(paths, 100.0, r, sigma, 1.0, hedge_every=every)
        rows.append(dict(rebalance_every_days=every, trades=len(range(0, 252, every)),
                         mean_pnl=float(pnl.mean()), sd_pnl=float(pnl.std(ddof=1)),
                         worst_pnl=float(pnl.min())))
    return rows, premium


def write(name, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    m_rows, mean_err = moneyness()
    write("moneyness.csv", m_rows)
    for x in m_rows:
        print(f"{x['band']:<22} BS {x['mean_bs']:7.2f}  |err| {x['mean_abs_error']:.4f}  "
              f"rel {x['mean_rel_error']:.2%}  worst {x['worst_rel_error']:.1%}")
    print(f"mean error over the surface {mean_err:+.4f}\n")

    h_rows, premium = hedging()
    write("hedging.csv", h_rows)
    for x in h_rows:
        print(f"every {x['rebalance_every_days']:>3}d  trades {x['trades']:>3}  mean {x['mean_pnl']:+.4f}  "
              f"sd {x['sd_pnl']:.3f}  worst {x['worst_pnl']:+.2f}")
    print(f"premium {premium:.4f}")
