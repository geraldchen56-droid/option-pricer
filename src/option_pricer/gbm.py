"""Simulating the price of a stock at a future date."""

from __future__ import annotations

import argparse
import math

import numpy as np


def simulate_terminal_prices(S0, r, sigma, T, n_sims, Z=None):
    """Simulate n_sims possible prices of the stock at time T.

    S_T = S0 * exp((r - sigma^2/2) * T + sigma * sqrt(T) * Z)

    Z is a standard normal random number: average 0, standard deviation 1.
    One Z per simulation gives one possible future price.

    Pass Z to supply your own random numbers instead of drawing fresh ones.
    Reusing the same Z across several parameter settings means the differences
    between the results come from the parameters, not from new randomness.
    """
    if Z is None:
        Z = np.random.standard_normal(n_sims)

    drift = (r - 0.5 * sigma**2) * T  # the predictable part
    shock = sigma * math.sqrt(T) * Z  # the random part

    return S0 * np.exp(drift + shock)


def lognormal_pdf(x, S0, r, sigma, T):
    """Density of S_T at the prices x (zero for x <= 0).

    log S_T is normal with mean log S0 + (r - sigma^2/2) T and standard
    deviation sigma sqrt(T), which makes S_T lognormal.
    """
    x = np.asarray(x, dtype=float)
    mu = math.log(S0) + (r - 0.5 * sigma**2) * T
    s = sigma * math.sqrt(T)

    out = np.zeros_like(x)
    pos = x > 0.0
    lx = np.log(x[pos])
    out[pos] = np.exp(-((lx - mu) ** 2) / (2.0 * s * s)) / (x[pos] * s * math.sqrt(2.0 * math.pi))
    return out


def summarise_mean(samples, S0, r, T):
    """Compare the simulated mean of S_T with the exact value S0 e^(rT).

    The error is also given in standard errors (z), so a small miss from noise
    can be told apart from a real bug such as a missing -sigma^2/2 term.
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    sample_mean = float(samples.mean())
    theoretical = S0 * math.exp(r * T)
    error = sample_mean - theoretical
    std_err = float(samples.std(ddof=1) / math.sqrt(n))

    return {
        "n_sims": n,
        "sample_mean": sample_mean,
        "theoretical_mean": theoretical,
        "error": error,
        "std_err": std_err,
        "z": error / std_err if std_err > 0 else 0.0,
        "rel_error": error / theoretical,
    }


def plot_terminal_distribution(samples, S0, r, sigma, T, out_path="gbm_terminal_prices.png", bins=100):
    """Histogram of the simulated prices with the exact density on top."""
    import matplotlib
    matplotlib.use("Agg")  # write to a file, no window
    import matplotlib.pyplot as plt

    samples = np.asarray(samples, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(samples, bins=bins, density=True, alpha=0.6, label="simulated")

    x = np.linspace(samples.min(), samples.max(), 600)
    ax.plot(x, lognormal_pdf(x, S0, r, sigma, T), lw=2, label="lognormal density")

    ax.set_xlabel("terminal price S_T")
    ax.set_ylabel("density")
    ax.set_title(f"{samples.size:,} simulated terminal prices")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="gbm-demo",
        description="Simulate GBM terminal prices and check the mean against S0 e^(rT).",
    )
    parser.add_argument("-n", "--n-sims", type=int, default=100_000, help="number of simulations")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--S0", type=float, default=100.0, help="spot price")
    parser.add_argument("-r", "--rate", type=float, default=0.05, help="risk-free rate")
    parser.add_argument("-v", "--vol", type=float, default=0.2, help="volatility")
    parser.add_argument("-T", "--expiry", type=float, default=1.0, help="years to expiry")
    parser.add_argument("--no-plot", action="store_true", help="skip the histogram")
    parser.add_argument("-o", "--out", default="gbm_terminal_prices.png", help="histogram file")
    args = parser.parse_args(argv)

    Z = np.random.default_rng(args.seed).standard_normal(args.n_sims)
    prices = simulate_terminal_prices(args.S0, args.rate, args.vol, args.expiry, args.n_sims, Z=Z)
    s = summarise_mean(prices, args.S0, args.rate, args.expiry)

    print(f"GBM terminal prices  S0={args.S0:g}  r={args.rate:.2%}  sigma={args.vol:.2%}  T={args.expiry:g}y")
    print(f"  simulations           {s['n_sims']:>14,}")
    print(f"  sample mean E[S_T]    {s['sample_mean']:>14.6f}")
    print(f"  theoretical S0*e^(rT) {s['theoretical_mean']:>14.6f}")
    print(f"  error                 {s['error']:>14.6f}  ({s['rel_error']:+.4%})")
    print(f"  monte carlo std err   {s['std_err']:>14.6f}  ({s['z']:+.2f} sigma)")

    if not args.no_plot:
        path = plot_terminal_distribution(prices, args.S0, args.rate, args.vol, args.expiry, args.out)
        print(f"  histogram saved to    {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
