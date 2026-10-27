"""Pricing a European call option by simulation."""

import math

import numpy as np

from option_pricer.gbm import simulate_terminal_prices


def price_european_call(S0, K, r, sigma, T, n_sims):
    """Price a European call by averaging its payoff over simulated futures.

    Returns (price, standard_error).
    """
    S_T = simulate_terminal_prices(S0, r, sigma, T, n_sims)

    # A call gives you the right to buy at K. If the stock ends above K you
    # gain S_T - K; if it ends below, you walk away and get nothing.
    payoffs = np.maximum(S_T - K, 0.0)

    # Each payoff arrives at time T, so it is worth less today.
    discounted = math.exp(-r * T) * payoffs

    price = discounted.mean()

    # How uncertain that average is: spread of the payoffs, shrunk by sqrt(n).
    standard_error = discounted.std(ddof=1) / math.sqrt(n_sims)

    return price, standard_error


if __name__ == "__main__":
    from option_pricer.vectorized import price as bs_price

    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
    n_sims = 100_000

    mc, se = price_european_call(S0, K, r, sigma, T, n_sims)
    exact = bs_price(S0, K, T, r, sigma, kind="call")

    print(f"monte carlo:    {mc:.4f} +/- {se:.4f}")
    print(f"black-scholes:  {exact:.4f}")
    print(f"difference:     {mc - exact:+.4f}  ({(mc - exact) / se:+.2f} standard errors)")
