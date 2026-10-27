"""Array-valued Black-Scholes, for grids and surfaces.

A numpy twin of :mod:`option_pricer.black_scholes`. Same formulas, same Greek
conventions, same degenerate-case handling -- the only difference is that every
argument broadcasts, so a whole spot x maturity grid prices in one call.
``tests/test_vectorized.py`` pins the two implementations to each other.

Kept separate rather than merged into the scalar module so the Black-Scholes
core stays dependency-free; this one needs numpy and scipy.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr  # standard normal CDF, elementwise

from option_pricer.black_scholes import OptionKind

__all__ = ["price", "greeks", "GREEK_NAMES"]

GREEK_NAMES = ("price", "delta", "gamma", "vega", "theta", "rho")

_TINY = 1e-12
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) * _INV_SQRT_2PI


def _broadcast(S, K, T, r, sigma, q):
    return np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (S, K, T, r, sigma, q))
    )


def _degenerate_mask(S, K, T, sigma):
    """True where the closed form is well defined (T, sigma, S, K all > 0)."""
    return (T > _TINY) & (sigma > _TINY) & (S > _TINY) & (K > _TINY)


def _validate(S, K, T, sigma, kind):
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    for name, arr in (("spot", S), ("strike", K), ("time to expiry", T), ("volatility", sigma)):
        if np.any(arr < 0.0):
            raise ValueError(f"{name} must be non-negative")


def price(
    S, K, T, r, sigma, q=0.0, kind: OptionKind = "call"
) -> np.ndarray:
    """Price European options over broadcast arrays. See the scalar twin."""
    S, K, T, r, sigma, q = _broadcast(S, K, T, r, sigma, q)
    _validate(S, K, T, sigma, kind)

    ok = _degenerate_mask(S, K, T, sigma)

    # Substitute harmless values inside the masked-out cells so the vectorised
    # math never evaluates log(0) or divides by zero; np.where discards them.
    S_s = np.where(ok, S, 1.0)
    K_s = np.where(ok, K, 1.0)
    T_s = np.where(ok, T, 1.0)
    sig_s = np.where(ok, sigma, 1.0)

    vol_sqrt_t = sig_s * np.sqrt(T_s)
    d1 = (np.log(S_s / K_s) + (r - q + 0.5 * sig_s**2) * T_s) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    disc_s = S * np.exp(-q * T)
    disc_k = K * np.exp(-r * T)

    if kind == "call":
        live = disc_s * ndtr(d1) - disc_k * ndtr(d2)
        dead = np.maximum(disc_s - disc_k, 0.0)
    else:
        live = disc_k * ndtr(-d2) - disc_s * ndtr(-d1)
        dead = np.maximum(disc_k - disc_s, 0.0)

    return np.where(ok, live, dead)


def greeks(
    S, K, T, r, sigma, q=0.0, kind: OptionKind = "call"
) -> dict[str, np.ndarray]:
    """Price and Greeks over broadcast arrays, as a dict of same-shaped arrays.

    Keys are :data:`GREEK_NAMES`. Units match the scalar module: vega per 1 vol
    point, theta per calendar day, rho per 1 rate point.
    """
    S, K, T, r, sigma, q = _broadcast(S, K, T, r, sigma, q)
    _validate(S, K, T, sigma, kind)

    ok = _degenerate_mask(S, K, T, sigma)
    S_s = np.where(ok, S, 1.0)
    K_s = np.where(ok, K, 1.0)
    T_s = np.where(ok, T, 1.0)
    sig_s = np.where(ok, sigma, 1.0)

    sqrt_t = np.sqrt(T_s)
    vol_sqrt_t = sig_s * sqrt_t
    d1 = (np.log(S_s / K_s) + (r - q + 0.5 * sig_s**2) * T_s) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    pdf_d1 = _norm_pdf(d1)

    gamma = disc_q * pdf_d1 / (S_s * sig_s * sqrt_t)
    vega = S * disc_q * pdf_d1 * sqrt_t
    theta_common = -S * disc_q * pdf_d1 * sig_s / (2.0 * sqrt_t)

    if kind == "call":
        delta = disc_q * ndtr(d1)
        theta = theta_common - r * K * disc_r * ndtr(d2) + q * S * disc_q * ndtr(d1)
        rho = K * T * disc_r * ndtr(d2)
        in_money = S > K
        dead_delta = np.where(in_money, 1.0, 0.0)
    else:
        delta = -disc_q * ndtr(-d1)
        theta = theta_common + r * K * disc_r * ndtr(-d2) - q * S * disc_q * ndtr(-d1)
        rho = -K * T * disc_r * ndtr(-d2)
        in_money = S < K
        dead_delta = np.where(in_money, -1.0, 0.0)

    zero = np.zeros_like(S)
    return {
        "price": price(S, K, T, r, sigma, q, kind),
        "delta": np.where(ok, delta, dead_delta),
        "gamma": np.where(ok, gamma, zero),
        "vega": np.where(ok, vega / 100.0, zero),
        "theta": np.where(ok, theta / 365.0, zero),
        "rho": np.where(ok, rho / 100.0, zero),
    }
