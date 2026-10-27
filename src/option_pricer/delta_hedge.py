"""Hedging a short call: does Black-Scholes actually replicate the option?"""

import math

import numpy as np

from option_pricer.vectorized import greeks as bs_greeks
from option_pricer.vectorized import price as bs_price


def simulate_paths(S0, r, sigma, T, n_steps, n_paths):
    """Full price paths, not just the final price.

    Everything so far only needed S_T, so one random number per simulation was
    enough. Hedging is different: we have to know the price on every day in
    order to rebalance, so we need the whole path. That means n_steps random
    numbers per path instead of one.

    Returns an array of shape (n_paths, n_steps + 1); column 0 is S0.
    """
    dt = T / n_steps
    Z = np.random.standard_normal((n_paths, n_steps))

    # Same GBM formula as before, applied over one small step at a time.
    log_steps = (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z

    # cumsum in log space = repeated multiplication in price space.
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_steps, axis=1)], axis=1
    )
    return S0 * np.exp(log_paths)


def hedge_short_call(paths, K, r, sigma, T, hedge_every=1):
    """Sell a call at the Black-Scholes price, hedge it, report the P&L.

    The strategy: take the premium, hold `delta` shares of the stock, and keep
    the rest in cash earning interest. Rebalance the share count every
    `hedge_every` steps. At expiry, pay out whatever the option owes.

    If Black-Scholes is right, the premium plus the trading gains should cover
    that payout exactly, so the profit and loss should come out near zero.
    """
    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1
    dt = T / n_steps

    # Which columns we are allowed to trade on.
    rebalance = list(range(0, n_steps, hedge_every))

    S0 = paths[0, 0]
    premium = float(bs_price(S0, K, T, r, sigma, kind="call"))

    cash = np.full(n_paths, premium)
    shares = np.zeros(n_paths)

    for step in rebalance:
        t = step * dt
        tau = T - t                      # time left on the option
        S = paths[:, step]

        target = bs_greeks(S, K, tau, r, sigma, kind="call")["delta"]

        # Buy or sell the difference, paying for it out of cash.
        cash -= (target - shares) * S
        shares = target

        # Cash earns interest until the next trade (or expiry).
        next_step = min(step + hedge_every, n_steps)
        cash *= math.exp(r * (next_step - step) * dt)

    S_T = paths[:, -1]
    payoff = np.maximum(S_T - K, 0.0)

    # Sell the shares, settle the option.
    return cash + shares * S_T - payoff, premium


if __name__ == "__main__":
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
    n_steps, n_paths = 252, 10_000

    # One set of paths, reused for every hedging frequency, so the comparison
    # below is about rebalancing and not about different random draws.
    paths = simulate_paths(S0, r, sigma, T, n_steps, n_paths)

    print(f"short call  S0={S0:g} K={K:g} r={r:.0%} sigma={sigma:.0%} T={T:g}y")
    print(f"{n_paths:,} paths, {n_steps} trading days\n")
    print(f"{'rebalance':>12}  {'trades':>7}  {'mean P&L':>10}  {'std P&L':>9}  {'worst':>9}")

    for every in (252, 36, 12, 4, 1):
        pnl, premium = hedge_short_call(paths, K, r, sigma, T, hedge_every=every)
        label = "once" if every == n_steps else f"every {every}d"
        print(
            f"{label:>12}  {len(range(0, n_steps, every)):>7}  "
            f"{pnl.mean():>+10.4f}  {pnl.std(ddof=1):>9.4f}  {pnl.min():>+9.4f}"
        )

    print(f"\nblack-scholes premium received: {premium:.4f}")
    print("mean P&L near zero => the premium is exactly the expected cost of hedging")
