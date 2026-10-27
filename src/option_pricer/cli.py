"""Command-line front end for the Black-Scholes pricer."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Sequence

from option_pricer import __version__
from option_pricer.black_scholes import greeks


def _percent(value: str) -> float:
    """Parse a rate or volatility given either as 0.2 or as 20%."""
    text = value.strip()
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="option-pricer",
        description="Price a European option and show its Greeks.",
        epilog=(
            "Rates and volatility accept either decimals or percentages: "
            "--vol 0.2 and --vol 20%% are equivalent."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-k",
        "--kind",
        choices=("call", "put"),
        default="call",
        help="option type (default: call)",
    )
    parser.add_argument("-S", "--spot", type=float, required=True, help="spot price")
    parser.add_argument("-K", "--strike", type=float, required=True, help="strike price")
    parser.add_argument(
        "-T",
        "--expiry",
        type=float,
        required=True,
        help="time to expiry in years",
    )
    parser.add_argument(
        "-r",
        "--rate",
        type=_percent,
        required=True,
        help="continuously compounded risk-free rate",
    )
    parser.add_argument(
        "-v", "--vol", type=_percent, required=True, help="annualised volatility"
    )
    parser.add_argument(
        "-q",
        "--dividend",
        type=_percent,
        default=0.0,
        help="continuous dividend yield (default: 0)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit JSON instead of a table"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        result = greeks(
            S=args.spot,
            K=args.strike,
            T=args.expiry,
            r=args.rate,
            sigma=args.vol,
            q=args.dividend,
            kind=args.kind,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(asdict(result), indent=2))
        return 0

    print(f"{args.kind.capitalize()}  S={args.spot:g}  K={args.strike:g}  T={args.expiry:g}y")
    print(f"  price   {result.price:>12.6f}")
    print(f"  delta   {result.delta:>12.6f}")
    print(f"  gamma   {result.gamma:>12.6f}")
    print(f"  vega    {result.vega:>12.6f}   per 1 vol point")
    print(f"  theta   {result.theta:>12.6f}   per day")
    print(f"  rho     {result.rho:>12.6f}   per 1 rate point")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
