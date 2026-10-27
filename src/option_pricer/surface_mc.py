"""Monte Carlo vs Black-Scholes call price surfaces over strike and maturity."""

import math

import numpy as np
import plotly.graph_objects as go

from option_pricer.gbm import simulate_terminal_prices
from option_pricer.vectorized import price as bs_price

# One hue, light to dark, for the two price surfaces (a magnitude).
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]
PRICE_SCALE = [[i / (len(BLUE) - 1), c] for i, c in enumerate(BLUE)]

# The error surface is signed, so it needs a diverging scale: two opposite
# hues with a neutral grey at zero, so "no error" reads as blank.
ERROR_SCALE = [
    [0.0, "#0d366b"], [0.25, "#3987e5"], [0.5, "#f0efec"],
    [0.75, "#e34948"], [1.0, "#7d2020"],
]

SURFACE_BG = "#fcfcfb"
INK = "#0b0b0b"
GRID = "#e6e5e1"


def monte_carlo_surface(S0, r, sigma, strikes, maturities, n_sims):
    """Monte Carlo call prices for every (maturity, strike) pair.

    For each maturity we simulate one set of terminal prices and reuse it for
    all 40 strikes. That is the whole trick: the simulation does not depend on
    K at all -- K only enters when the payoff is applied -- so simulating once
    per maturity instead of once per (K, T) cell is 40x less work.
    """
    prices = np.empty((len(maturities), len(strikes)))

    for i, T in enumerate(maturities):
        S_T = simulate_terminal_prices(S0, r, sigma, T, n_sims)

        # S_T[None, :] has shape (1, n_sims), strikes[:, None] has shape
        # (40, 1). Subtracting them broadcasts to (40, n_sims): every strike
        # against every simulated price, in one operation.
        payoffs = np.maximum(S_T[None, :] - strikes[:, None], 0.0)

        # Average across simulations (axis=1), then discount to today.
        prices[i, :] = math.exp(-r * T) * payoffs.mean(axis=1)

    return prices


def build_figure(strikes, maturities, mc, bs):
    """Three surfaces on one figure, one visible at a time via buttons."""
    diff = mc - bs

    # MC and BS share a z range so switching between them is a fair comparison.
    price_lo = float(min(mc.min(), bs.min()))
    price_hi = float(max(mc.max(), bs.max()))
    # The error scale is centred on zero so grey always means "no error".
    err = float(np.abs(diff).max())

    def surface(z, scale, lo, hi, label, unit):
        return go.Surface(
            x=strikes, y=maturities, z=z,
            colorscale=scale, cmin=lo, cmax=hi,
            colorbar=dict(title=dict(text=unit, side="right"), outlinewidth=0, len=0.7),
            hovertemplate="K %{x:.0f}<br>T %{y:.2f}y<br>" + label + " %{z:.4f}<extra></extra>",
        )

    fig = go.Figure([
        surface(mc, PRICE_SCALE, price_lo, price_hi, "MC price", "price"),
        surface(bs, PRICE_SCALE, price_lo, price_hi, "BS price", "price"),
        surface(diff, ERROR_SCALE, -err, err, "MC - BS", "MC - BS"),
    ])
    fig.data[1].visible = False
    fig.data[2].visible = False

    views = [
        ("Monte Carlo", [True, False, False], "price", [price_lo, price_hi]),
        ("Black-Scholes", [False, True, False], "price", [price_lo, price_hi]),
        ("MC - BS", [False, False, True], "MC - BS", [-err, err]),
    ]

    fig.update_layout(
        title=dict(
            text="<b>European call price surface</b> — Monte Carlo vs Black-Scholes",
            font=dict(color=INK, size=18), x=0.02, xanchor="left",
        ),
        paper_bgcolor=SURFACE_BG, height=760, margin=dict(l=0, r=0, t=110, b=0),
        scene=dict(
            xaxis=dict(title="strike K", backgroundcolor=SURFACE_BG, gridcolor=GRID),
            yaxis=dict(title="years to expiry", backgroundcolor=SURFACE_BG, gridcolor=GRID),
            zaxis=dict(title="price", range=[price_lo, price_hi],
                       backgroundcolor=SURFACE_BG, gridcolor=GRID),
            camera=dict(eye=dict(x=1.7, y=-1.7, z=0.8)),
        ),
        updatemenus=[dict(
            type="buttons", direction="left", x=0.02, y=1.02, xanchor="left",
            showactive=True,
            buttons=[
                dict(label=label, method="update",
                     args=[{"visible": vis},
                           {"scene.zaxis.title.text": ztitle, "scene.zaxis.range": zrange}])
                for label, vis, ztitle, zrange in views
            ],
        )],
    )
    return fig


if __name__ == "__main__":
    S0, r, sigma, n_sims = 100.0, 0.05, 0.2, 100_000
    strikes = np.linspace(60.0, 140.0, 40)
    maturities = np.linspace(0.1, 2.0, 20)

    mc = monte_carlo_surface(S0, r, sigma, strikes, maturities, n_sims)

    # Black-Scholes on the same grid: maturities down the rows, strikes across
    # the columns, so the two arrays line up cell for cell.
    bs = bs_price(S0, strikes[None, :], maturities[:, None], r, sigma, kind="call")

    diff = mc - bs
    print(f"grid            {len(maturities)} maturities x {len(strikes)} strikes")
    print(f"simulations     {len(maturities)} x {n_sims:,} = {len(maturities) * n_sims:,} draws")
    print(f"MC - BS  mean   {diff.mean():+.4f}")
    print(f"         worst  {diff[np.unravel_index(np.abs(diff).argmax(), diff.shape)]:+.4f}")

    fig = build_figure(strikes, maturities, mc, bs)
    fig.write_html("surface_mc.html", include_plotlyjs="inline", auto_open=False)
    print("written to      surface_mc.html")
