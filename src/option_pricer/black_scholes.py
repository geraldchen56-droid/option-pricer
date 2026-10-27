"""Black-Scholes-Merton pricing and Greeks for European options.

Uses the generalised form with a continuous dividend yield ``q``, which also
covers Black-76 futures options (``q = r``) and FX options (``q`` = foreign
rate) without a separate code path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

__all__ = ["Greeks", "OptionKind", "price", "greeks", "norm_cdf", "norm_pdf"]

OptionKind = Literal["call", "put"]

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)

# Below this, T or sigma is treated as exactly zero to avoid dividing by a
# vanishing sigma*sqrt(T) and returning inf/nan instead of the payoff limit.
_TINY = 1e-12


def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


@dataclass(frozen=True)
class Greeks:
    """First- and second-order sensitivities.

    Units are the market conventions rather than the raw partial derivatives:
    ``vega`` is per 1 volatility point (0.01), ``theta`` is per calendar day,
    and ``rho`` is per 1 rate point (0.01). ``delta`` and ``gamma`` are raw.
    """

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _validate(S: float, K: float, T: float, sigma: float, kind: str) -> None:
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if S < 0.0:
        raise ValueError(f"spot must be non-negative, got {S}")
    if K < 0.0:
        raise ValueError(f"strike must be non-negative, got {K}")
    if T < 0.0:
        raise ValueError(f"time to expiry must be non-negative, got {T}")
    if sigma < 0.0:
        raise ValueError(f"volatility must be non-negative, got {sigma}")


def _d1_d2(
    S: float, K: float, T: float, r: float, sigma: float, q: float
) -> tuple[float, float]:
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def _intrinsic(S: float, K: float, T: float, r: float, q: float, kind: str) -> float:
    """Discounted forward-intrinsic value, the sigma -> 0 or T -> 0 limit."""
    fwd = S * math.exp(-q * T)
    strike_pv = K * math.exp(-r * T)
    if kind == "call":
        return max(fwd - strike_pv, 0.0)
    return max(strike_pv - fwd, 0.0)


def price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: OptionKind = "call",
) -> float:
    """Price a European option.

    Args:
        S: Spot price of the underlying.
        K: Strike price.
        T: Time to expiry in years.
        r: Continuously compounded risk-free rate (e.g. ``0.05`` for 5%).
        sigma: Annualised volatility (e.g. ``0.2`` for 20%).
        q: Continuous dividend yield.
        kind: ``"call"`` or ``"put"``.

    Returns:
        The present value of the option.
    """
    _validate(S, K, T, sigma, kind)

    if T <= _TINY or sigma <= _TINY or S <= _TINY or K <= _TINY:
        return _intrinsic(S, K, T, r, q, kind)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_s = S * math.exp(-q * T)
    disc_k = K * math.exp(-r * T)

    if kind == "call":
        return disc_s * norm_cdf(d1) - disc_k * norm_cdf(d2)
    return disc_k * norm_cdf(-d2) - disc_s * norm_cdf(-d1)


def greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: OptionKind = "call",
) -> Greeks:
    """Price a European option and return it alongside its Greeks.

    Arguments match :func:`price`. See :class:`Greeks` for the units used.

    At expiry (or zero volatility) the Greeks are degenerate -- gamma and vega
    are zero or unbounded -- so they are reported as zero apart from the
    step-function delta.
    """
    _validate(S, K, T, sigma, kind)

    pv = price(S, K, T, r, sigma, q, kind)

    if T <= _TINY or sigma <= _TINY or S <= _TINY or K <= _TINY:
        in_money = (S > K) if kind == "call" else (S < K)
        delta = (1.0 if kind == "call" else -1.0) if in_money else 0.0
        return Greeks(price=pv, delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrt_t = math.sqrt(T)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    pdf_d1 = norm_pdf(d1)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * disc_q * pdf_d1 * sqrt_t
    theta_term = -S * disc_q * pdf_d1 * sigma / (2.0 * sqrt_t)

    if kind == "call":
        delta = disc_q * norm_cdf(d1)
        theta = theta_term - r * K * disc_r * norm_cdf(d2) + q * S * disc_q * norm_cdf(d1)
        rho = K * T * disc_r * norm_cdf(d2)
    else:
        delta = -disc_q * norm_cdf(-d1)
        theta = (
            theta_term + r * K * disc_r * norm_cdf(-d2) - q * S * disc_q * norm_cdf(-d1)
        )
        rho = -K * T * disc_r * norm_cdf(-d2)

    return Greeks(
        price=pv,
        delta=delta,
        gamma=gamma,
        vega=vega / 100.0,
        theta=theta / 365.0,
        rho=rho / 100.0,
    )
