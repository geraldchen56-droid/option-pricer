import math

import pytest

from option_pricer import greeks, price

# Reference values from Hull, "Options, Futures, and Other Derivatives".
BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, q=0.0)


def numeric_derivative(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


class TestPrice:
    def test_atm_call_matches_reference(self):
        assert price(**BASE, kind="call") == pytest.approx(10.450583572, rel=1e-9)

    def test_atm_put_matches_reference(self):
        assert price(**BASE, kind="put") == pytest.approx(5.573526022, rel=1e-9)

    def test_put_call_parity(self):
        call = price(**BASE, kind="call")
        put = price(**BASE, kind="put")
        S, K, T, r, q = BASE["S"], BASE["K"], BASE["T"], BASE["r"], BASE["q"]
        assert call - put == pytest.approx(
            S * math.exp(-q * T) - K * math.exp(-r * T), rel=1e-12
        )

    def test_put_call_parity_with_dividend(self):
        args = {**BASE, "S": 120.0, "K": 95.0, "T": 0.75, "q": 0.03}
        call = price(**args, kind="call")
        put = price(**args, kind="put")
        expected = args["S"] * math.exp(-args["q"] * args["T"]) - args["K"] * math.exp(
            -args["r"] * args["T"]
        )
        assert call - put == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_price_is_at_least_discounted_intrinsic(self, kind):
        """European options bound the *forward* intrinsic, not the spot one.

        A European put may sit below ``K - S`` because exercise cannot be
        brought forward, so the strike is only received at expiry.
        """
        K, T, r, q = BASE["K"], BASE["T"], BASE["r"], BASE["q"]
        for S in (50.0, 90.0, 100.0, 110.0, 150.0):
            pv = price(**{**BASE, "S": S}, kind=kind)
            fwd, strike_pv = S * math.exp(-q * T), K * math.exp(-r * T)
            floor = max(fwd - strike_pv, 0.0) if kind == "call" else max(strike_pv - fwd, 0.0)
            assert pv >= floor - 1e-12

    def test_european_put_can_fall_below_spot_intrinsic(self):
        """Guards the discounting in the branch above -- deep ITM, long dated."""
        pv = price(**{**BASE, "S": 50.0}, kind="put")
        assert pv < BASE["K"] - 50.0

    def test_call_never_falls_below_spot_intrinsic(self):
        for S in (50.0, 100.0, 150.0):
            assert price(**{**BASE, "S": S}, kind="call") >= max(S - BASE["K"], 0.0) - 1e-12

    def test_call_price_monotonic_in_spot(self):
        prices = [price(**{**BASE, "S": S}, kind="call") for S in range(80, 121, 5)]
        assert prices == sorted(prices)

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_price_monotonic_in_volatility(self, kind):
        prices = [
            price(**{**BASE, "sigma": s}, kind=kind) for s in (0.05, 0.1, 0.2, 0.4, 0.8)
        ]
        assert prices == sorted(prices)

    def test_deep_otm_call_is_worthless(self):
        assert price(**{**BASE, "S": 1.0}, kind="call") == pytest.approx(0.0, abs=1e-12)

    def test_zero_vol_gives_discounted_intrinsic(self):
        pv = price(**{**BASE, "S": 110.0, "sigma": 0.0}, kind="call")
        assert pv == pytest.approx(110.0 - 100.0 * math.exp(-0.05), rel=1e-12)

    @pytest.mark.parametrize(
        "kind,S,expected", [("call", 110.0, 10.0), ("call", 90.0, 0.0), ("put", 90.0, 10.0), ("put", 110.0, 0.0)]
    )
    def test_expiry_gives_payoff(self, kind, S, expected):
        assert price(**{**BASE, "S": S, "T": 0.0}, kind=kind) == pytest.approx(expected)

    def test_zero_spot(self):
        assert price(**{**BASE, "S": 0.0}, kind="call") == pytest.approx(0.0)
        assert price(**{**BASE, "S": 0.0}, kind="put") == pytest.approx(
            100.0 * math.exp(-0.05), rel=1e-12
        )

    @pytest.mark.parametrize(
        "bad", [{"S": -1.0}, {"K": -1.0}, {"T": -1.0}, {"sigma": -0.1}]
    )
    def test_rejects_negative_inputs(self, bad):
        with pytest.raises(ValueError):
            price(**{**BASE, **bad})

    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match="call.*put"):
            price(**BASE, kind="straddle")


class TestGreeks:
    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_delta_matches_finite_difference(self, kind):
        g = greeks(**BASE, kind=kind)
        fd = numeric_derivative(
            lambda S: price(**{**BASE, "S": S}, kind=kind), BASE["S"], 1e-4
        )
        assert g.delta == pytest.approx(fd, rel=1e-6)

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_gamma_matches_finite_difference(self, kind):
        g = greeks(**BASE, kind=kind)
        h = 1e-3
        fd = numeric_derivative(
            lambda S: greeks(**{**BASE, "S": S}, kind=kind).delta, BASE["S"], h
        )
        assert g.gamma == pytest.approx(fd, rel=1e-5)

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_vega_matches_finite_difference(self, kind):
        g = greeks(**BASE, kind=kind)
        fd = numeric_derivative(
            lambda s: price(**{**BASE, "sigma": s}, kind=kind), BASE["sigma"], 1e-5
        )
        # vega is reported per volatility point, i.e. dP/dsigma / 100
        assert g.vega == pytest.approx(fd / 100.0, rel=1e-6)

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_theta_matches_finite_difference(self, kind):
        g = greeks(**BASE, kind=kind)
        # theta is -dP/dT reported per calendar day
        fd = numeric_derivative(
            lambda T: price(**{**BASE, "T": T}, kind=kind), BASE["T"], 1e-6
        )
        assert g.theta == pytest.approx(-fd / 365.0, rel=1e-5)

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_rho_matches_finite_difference(self, kind):
        g = greeks(**BASE, kind=kind)
        fd = numeric_derivative(
            lambda r: price(**{**BASE, "r": r}, kind=kind), BASE["r"], 1e-6
        )
        assert g.rho == pytest.approx(fd / 100.0, rel=1e-6)

    def test_greeks_price_agrees_with_price(self):
        for kind in ("call", "put"):
            assert greeks(**BASE, kind=kind).price == price(**BASE, kind=kind)

    def test_gamma_and_vega_are_kind_independent(self):
        call = greeks(**BASE, kind="call")
        put = greeks(**BASE, kind="put")
        assert call.gamma == pytest.approx(put.gamma, rel=1e-12)
        assert call.vega == pytest.approx(put.vega, rel=1e-12)

    def test_delta_bounds(self):
        for S in (50.0, 100.0, 200.0):
            assert 0.0 <= greeks(**{**BASE, "S": S}, kind="call").delta <= 1.0
            assert -1.0 <= greeks(**{**BASE, "S": S}, kind="put").delta <= 0.0

    def test_delta_put_call_relation(self):
        call = greeks(**BASE, kind="call")
        put = greeks(**BASE, kind="put")
        assert call.delta - put.delta == pytest.approx(1.0, rel=1e-12)

    def test_gamma_and_vega_positive(self):
        g = greeks(**BASE, kind="call")
        assert g.gamma > 0.0
        assert g.vega > 0.0

    def test_at_expiry_greeks_are_degenerate(self):
        g = greeks(**{**BASE, "S": 110.0, "T": 0.0}, kind="call")
        assert g.price == pytest.approx(10.0)
        assert g.delta == 1.0
        assert (g.gamma, g.vega, g.theta, g.rho) == (0.0, 0.0, 0.0, 0.0)

    def test_at_expiry_otm_delta_is_zero(self):
        assert greeks(**{**BASE, "S": 90.0, "T": 0.0}, kind="call").delta == 0.0
        assert greeks(**{**BASE, "S": 90.0, "T": 0.0}, kind="put").delta == -1.0
