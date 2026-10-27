"""MC vs BS call surfaces with a volatility slider."""

import math

import numpy as np
import plotly.graph_objects as go

from option_pricer.gbm import simulate_terminal_prices
from option_pricer.surface_mc import ERROR_SCALE, GRID, INK, PRICE_SCALE, SURFACE_BG
from option_pricer.vectorized import price as bs_price


def mc_surfaces(S0, r, sigmas, strikes, maturities, n_sims):
    """Monte Carlo call prices for every (sigma, maturity, strike).

    One Z array is drawn here and reused for every sigma and every maturity.
    That means each frame of the animation sees the same 100,000 "rolls of the
    dice", so the surface changing between frames is the effect of sigma alone
    and not fresh noise. Without this the surface would jitter from frame to
    frame and hide the real trend.
    """
    Z = np.random.standard_normal(n_sims)

    out = np.empty((len(sigmas), len(maturities), len(strikes)))
    for a, sigma in enumerate(sigmas):
        for i, T in enumerate(maturities):
            S_T = simulate_terminal_prices(S0, r, sigma, T, n_sims, Z=Z)
            payoffs = np.maximum(S_T[None, :] - strikes[:, None], 0.0)
            out[a, i, :] = math.exp(-r * T) * payoffs.mean(axis=1)
    return out


def build_figure(strikes, maturities, sigmas, mc, bs):
    """Three surfaces, buttons to switch, slider to sweep sigma."""
    diff = mc - bs

    # Ranges taken over every frame, then pinned, so a surface that grows with
    # sigma actually looks like it is growing instead of being rescaled away.
    price_lo = float(min(mc.min(), bs.min()))
    price_hi = float(max(mc.max(), bs.max()))
    err = float(np.abs(diff).max())

    def surface(z, scale, lo, hi, label, unit, visible):
        return go.Surface(
            x=strikes, y=maturities, z=z,
            colorscale=scale, cmin=lo, cmax=hi, visible=visible,
            colorbar=dict(title=dict(text=unit, side="right"), outlinewidth=0, len=0.7),
            hovertemplate="K %{x:.0f}<br>T %{y:.2f}y<br>" + label + " %{z:.4f}<extra></extra>",
        )

    def traces(a, visible=(True, False, False)):
        return [
            surface(mc[a], PRICE_SCALE, price_lo, price_hi, "MC price", "price", visible[0]),
            surface(bs[a], PRICE_SCALE, price_lo, price_hi, "BS price", "price", visible[1]),
            surface(diff[a], ERROR_SCALE, -err, err, "MC - BS", "MC - BS", visible[2]),
        ]

    start = int(np.argmin(np.abs(sigmas - 0.2)))

    # Frames carry only z, never `visible`, so stepping the slider does not
    # undo whichever surface the buttons have selected.
    frames = [
        go.Frame(
            name=f"{s:.2f}",
            traces=[0, 1, 2],
            data=[
                go.Surface(z=mc[a]), go.Surface(z=bs[a]), go.Surface(z=diff[a]),
            ],
        )
        for a, s in enumerate(sigmas)
    ]

    views = [
        ("Monte Carlo", [True, False, False], "price", [price_lo, price_hi]),
        ("Black-Scholes", [False, True, False], "price", [price_lo, price_hi]),
        ("MC - BS", [False, False, True], "MC - BS", [-err, err]),
    ]

    fig = go.Figure(data=traces(start), frames=frames)
    fig.update_layout(
        title=dict(
            text="<b>European call price surface</b> — MC vs Black-Scholes"
                 "<br><span style='font-size:12px;color:#52514e'>"
                 "z-axis pinned across all volatilities · same random draws in every frame"
                 "</span>",
            font=dict(color=INK, size=18), x=0.02, xanchor="left",
        ),
        paper_bgcolor=SURFACE_BG, height=780, margin=dict(l=0, r=0, t=110, b=0),
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
                           {"scene.zaxis.title.text": zt, "scene.zaxis.range": zr}])
                for label, vis, zt, zr in views
            ],
        )],
        sliders=[dict(
            active=start,
            currentvalue=dict(prefix="volatility  ", font=dict(color=INK, size=14)),
            pad=dict(t=30), x=0.08, len=0.84,
            steps=[
                dict(method="animate", label=f"{s:.0%}",
                     args=[[f"{s:.2f}"],
                           dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0))])
                for s in sigmas
            ],
        )],
    )
    return fig


if __name__ == "__main__":
    S0, r, n_sims = 100.0, 0.05, 100_000
    strikes = np.linspace(60.0, 140.0, 40)
    maturities = np.linspace(0.1, 2.0, 20)
    sigmas = np.round(np.arange(0.10, 0.5001, 0.05), 2)

    mc = mc_surfaces(S0, r, sigmas, strikes, maturities, n_sims)
    bs = bs_price(S0, strikes[None, None, :], maturities[None, :, None],
                  r, sigmas[:, None, None], kind="call")

    print(f"sigmas          {', '.join(f'{s:.0%}' for s in sigmas)}")
    print(f"grid per frame  {len(maturities)} maturities x {len(strikes)} strikes")
    print(f"price range     {min(mc.min(), bs.min()):.4f} .. {max(mc.max(), bs.max()):.4f}")
    print(f"MC - BS  worst  {np.abs(mc - bs).max():+.4f}")

    fig = build_figure(strikes, maturities, sigmas, mc, bs)
    fig.write_html("surface_slider.html", include_plotlyjs="inline", auto_open=False)
    print("written to      surface_slider.html")
