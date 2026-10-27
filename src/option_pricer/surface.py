"""Interactive 3D Greek surfaces over spot x time-to-expiry.

Writes a self-contained HTML file: drag to rotate, slider to sweep volatility.
The surface deforms as sigma changes -- most dramatically for gamma, whose
at-the-money peak sharpens as either volatility or time to expiry falls.

Requires numpy, scipy and plotly (``pip install -e ".[sim]"``).
"""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np

from option_pricer.vectorized import GREEK_NAMES, greeks

__all__ = ["build_surface", "make_figure", "write_surface", "main"]

# Sequential single-hue blue ramp (palette steps 100 -> 700). Magnitude is one
# quantity, so it gets one hue, light to dark -- never a rainbow.
_BLUE_RAMP = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)
_COLORSCALE = [
    [i / (len(_BLUE_RAMP) - 1), hex_] for i, hex_ in enumerate(_BLUE_RAMP)
]

_SURFACE_BG = "#fcfcfb"
_INK = "#0b0b0b"
_INK_MUTED = "#52514e"
_GRID = "#e6e5e1"

# Greeks whose value does not depend on call vs put -- the kind toggle is a
# no-op for these, which is worth saying rather than letting a user wonder.
_KIND_INVARIANT = frozenset({"gamma", "vega"})

_UNITS = {
    "price": "currency",
    "delta": "per 1 spot",
    "gamma": "delta per 1 spot",
    "vega": "per 1 vol point",
    "theta": "per calendar day",
    "rho": "per 1 rate point",
}


def build_surface(
    greek: str = "gamma",
    kind: str = "call",
    K: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    spot_range: tuple[float, float] = (40.0, 180.0),
    time_range: tuple[float, float] = (0.05, 2.0),
    vols: Sequence[float] | None = None,
    n_spot: int = 64,
    n_time: int = 48,
) -> dict:
    """Evaluate ``greek`` on a spot x maturity grid, once per volatility.

    Returns a dict with the axes, a ``(n_vols, n_time, n_spot)`` stack of
    surfaces, and the global value range. The stack is built with one
    vectorised call per volatility -- the scalar pricer would need
    ``n_vols * n_time * n_spot`` Python-level calls for the same result.

    ``time_range`` deliberately starts above zero: gamma diverges as T -> 0 at
    the money, so a grid touching expiry would be dominated by one spike.
    """
    if greek not in GREEK_NAMES:
        raise ValueError(f"greek must be one of {GREEK_NAMES}, got {greek!r}")
    if vols is None:
        # 10-40% rather than a wider sweep: gamma scales as 1/(sigma sqrt(T)),
        # so the tallest frame is sigma_max/sigma_min times the shortest. Past
        # ~4x the high-vol frames read as flat sheets against a pinned z axis.
        vols = np.round(np.linspace(0.10, 0.40, 21), 4)
    vols = np.asarray(vols, dtype=float)
    if vols.size == 0:
        raise ValueError("vols must not be empty")

    spots = np.linspace(*spot_range, n_spot)
    times = np.linspace(*time_range, n_time)
    # (n_time, 1) x (1, n_spot) -> broadcast to the full grid.
    S = spots[None, :]
    T = times[:, None]

    stack = np.stack(
        [greeks(S, K, T, r, float(v), q, kind)[greek] for v in vols], axis=0
    )

    return {
        "greek": greek,
        "kind": kind,
        "K": K,
        "r": r,
        "q": q,
        "spots": spots,
        "times": times,
        "vols": vols,
        "stack": stack,
        "zmin": float(stack.min()),
        "zmax": float(stack.max()),
    }


def make_figure(
    data: dict, initial_index: int | None = None, zscale: str = "fixed"
):
    """Build the plotly figure: rotatable surface + volatility slider.

    Args:
        data: Output of :func:`build_surface`.
        initial_index: Which volatility frame to open on.
        zscale: ``"fixed"`` pins the z axis and colour range across every frame,
            so peak heights are comparable as the slider moves -- the honest
            default, since autoscaling per frame makes a shrinking peak look
            unchanged. ``"frame"`` rescales each frame to its own maximum,
            which reads better for a wide volatility sweep at the cost of that
            comparability; the axis then carries a warning in its title.
    """
    import plotly.graph_objects as go

    if zscale not in ("fixed", "frame"):
        raise ValueError(f"zscale must be 'fixed' or 'frame', got {zscale!r}")

    greek, kind = data["greek"], data["kind"]
    spots, times, vols, stack = data["spots"], data["times"], data["vols"], data["stack"]
    zmin, zmax = data["zmin"], data["zmax"]

    def frame_range(z):
        if zscale == "fixed":
            return zmin, zmax
        lo = float(min(z.min(), 0.0)) if zmin < 0.0 else float(z.min())
        return lo, float(z.max())

    if initial_index is None:
        # Open on 20% vol if it is in range, else the middle of the sweep.
        initial_index = int(np.argmin(np.abs(vols - 0.20)))

    hover = (
        "spot %{x:.1f}<br>expiry %{y:.2f}y<br>"
        + greek
        + " %{z:.5f}<extra></extra>"
    )

    def surface(z, vol):
        lo, hi = frame_range(z)
        return go.Surface(
            x=spots,
            y=times,
            z=z,
            cmin=lo,
            cmax=hi,
            colorscale=_COLORSCALE,
            colorbar=dict(
                title=dict(text=f"{greek}<br><sub>{_UNITS[greek]}</sub>", side="right"),
                tickfont=dict(color=_INK_MUTED, size=10),
                outlinewidth=0,
                len=0.7,
            ),
            contours={
                "z": {"show": True, "usecolormap": True, "width": 1, "project": {"z": False}}
            },
            hovertemplate=hover,
            name=f"sigma={vol:.0%}",
        )

    def frame_layout(z):
        """Per-frame axis rescaling, only needed in ``frame`` mode."""
        if zscale == "fixed":
            return None
        lo, hi = frame_range(z)
        return go.Layout(scene=dict(zaxis=dict(range=[lo, hi])))

    fig = go.Figure(
        data=[surface(stack[initial_index], vols[initial_index])],
        frames=[
            go.Frame(data=[surface(z, v)], layout=frame_layout(z), name=f"{v:.4f}")
            for z, v in zip(stack, vols)
        ],
    )

    invariant = (
        f"  ·  {greek} is identical for calls and puts"
        if greek in _KIND_INVARIANT
        else ""
    )
    scale_note = (
        "z-axis fixed across all frames — peak heights are comparable"
        if zscale == "fixed"
        else "<b>z-axis rescales per frame — peak heights are NOT comparable</b>"
    )
    subtitle = (
        f"K={data['K']:g}, r={data['r']:.0%}, q={data['q']:.0%}"
        f"  ·  {scale_note}{invariant}"
    )
    z_range = [zmin, zmax] if zscale == "fixed" else None
    z_title = greek if zscale == "fixed" else f"{greek} (rescaled per frame)"

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{greek.capitalize()} surface</b> — {kind} option"
                f"<br><span style='font-size:12px;color:{_INK_MUTED}'>{subtitle}</span>"
            ),
            font=dict(color=_INK, size=18),
            x=0.02,
            xanchor="left",
        ),
        paper_bgcolor=_SURFACE_BG,
        plot_bgcolor=_SURFACE_BG,
        font=dict(color=_INK_MUTED, size=12),
        margin=dict(l=0, r=0, t=90, b=10),
        height=760,
        scene=dict(
            xaxis=dict(
                title="spot", backgroundcolor=_SURFACE_BG, gridcolor=_GRID,
                zerolinecolor=_GRID, showbackground=True,
            ),
            yaxis=dict(
                title="years to expiry", backgroundcolor=_SURFACE_BG, gridcolor=_GRID,
                zerolinecolor=_GRID, showbackground=True,
            ),
            zaxis=dict(
                title=z_title, range=z_range, backgroundcolor=_SURFACE_BG,
                gridcolor=_GRID, zerolinecolor=_GRID, showbackground=True,
            ),
            camera=dict(eye=dict(x=1.65, y=-1.65, z=0.85)),
            aspectratio=dict(x=1.0, y=1.0, z=0.62),
        ),
        sliders=[
            dict(
                active=initial_index,
                currentvalue=dict(
                    prefix="volatility  ", font=dict(color=_INK, size=14)
                ),
                pad=dict(t=24, b=8),
                x=0.08,
                len=0.84,
                steps=[
                    dict(
                        method="animate",
                        label=f"{v:.0%}",
                        args=[
                            [f"{v:.4f}"],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                    )
                    for v in vols
                ],
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                showactive=False,
                x=0.02,
                y=-0.02,
                xanchor="left",
                buttons=[
                    dict(
                        label="▶ sweep",
                        method="animate",
                        args=[
                            None,
                            dict(
                                mode="immediate",
                                fromcurrent=True,
                                frame=dict(duration=110, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="❚❚ pause",
                        method="animate",
                        args=[
                            [None],
                            dict(mode="immediate", frame=dict(duration=0, redraw=False)),
                        ],
                    ),
                ],
            )
        ],
    )
    return fig


def write_surface(
    data: dict, out_path: str = "greek_surface.html", zscale: str = "fixed"
) -> str:
    """Render to a standalone HTML file (plotly.js inlined, no network needed)."""
    fig = make_figure(data, zscale=zscale)
    fig.write_html(out_path, include_plotlyjs="inline", auto_open=False)
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="greek-surface",
        description="Write an interactive 3D Greek surface with a volatility slider.",
    )
    parser.add_argument("-g", "--greek", choices=GREEK_NAMES, default="gamma")
    parser.add_argument("-k", "--kind", choices=("call", "put"), default="call")
    parser.add_argument("-K", "--strike", type=float, default=100.0)
    parser.add_argument("-r", "--rate", type=float, default=0.05)
    parser.add_argument("-q", "--dividend", type=float, default=0.0)
    parser.add_argument("--spot-min", type=float, default=40.0)
    parser.add_argument("--spot-max", type=float, default=180.0)
    parser.add_argument("--t-min", type=float, default=0.05)
    parser.add_argument("--t-max", type=float, default=2.0)
    parser.add_argument("--vol-min", type=float, default=0.10)
    parser.add_argument("--vol-max", type=float, default=0.40)
    parser.add_argument("--n-vols", type=int, default=21)
    parser.add_argument("--n-spot", type=int, default=64)
    parser.add_argument("--n-time", type=int, default=48)
    parser.add_argument(
        "--zscale",
        choices=("fixed", "frame"),
        default="fixed",
        help="fixed: comparable peak heights (default). frame: rescale each "
        "frame, better for a wide vol sweep but peaks stop being comparable",
    )
    parser.add_argument("-o", "--out", default="greek_surface.html")
    args = parser.parse_args(argv)

    data = build_surface(
        greek=args.greek,
        kind=args.kind,
        K=args.strike,
        r=args.rate,
        q=args.dividend,
        spot_range=(args.spot_min, args.spot_max),
        time_range=(args.t_min, args.t_max),
        vols=np.round(np.linspace(args.vol_min, args.vol_max, args.n_vols), 4),
        n_spot=args.n_spot,
        n_time=args.n_time,
    )
    path = write_surface(data, args.out, zscale=args.zscale)

    cells = data["stack"].size
    print(f"{args.greek} surface ({args.kind})")
    print(f"  grid        {args.n_spot} spots x {args.n_time} maturities x {args.n_vols} vols")
    print(f"  evaluations {cells:,}")
    print(f"  range       {data['zmin']:.6f} .. {data['zmax']:.6f}")
    print(f"  z scaling   {args.zscale}")
    print(f"  written to  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
