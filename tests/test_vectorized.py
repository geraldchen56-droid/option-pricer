import itertools
import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from option_pricer import black_scholes as scalar  # noqa: E402
from option_pricer import vectorized as vec  # noqa: E402
from option_pricer.vectorized import GREEK_NAMES  # noqa: E402

KINDS = ("call", "put")


class TestAgreesWithScalar:
    """The vectorised module duplicates the scalar formulas -- pin them together."""

    CASES = list(
        itertools.product(
            (40.0, 90.0, 100.0, 110.0, 250.0),  # spot
            (0.05, 0.5, 1.0, 3.0),              # expiry
            (0.05, 0.2, 0.8),                   # vol
            (0.0, 0.05),                        # rate
            (0.0, 0.03),                        # dividend
        )
    )

    @pytest.mark.parametrize("kind", KINDS)
    def test_price_matches_everywhere(self, kind):
        for S, T, sigma, r, q in self.CASES:
            expected = scalar.price(S, 100.0, T, r, sigma, q, kind)
            got = vec.price(S, 100.0, T, r, sigma, q, kind)
            assert float(got) == pytest.approx(expected, rel=1e-12, abs=1e-14)

    @pytest.mark.parametrize("kind,greek", itertools.product(KINDS, GREEK_NAMES))
    def test_greeks_match_everywhere(self, kind, greek):
        for S, T, sigma, r, q in self.CASES:
            expected = getattr(scalar.greeks(S, 100.0, T, r, sigma, q, kind), greek)
            got = vec.greeks(S, 100.0, T, r, sigma, q, kind)[greek]
            assert float(got) == pytest.approx(expected, rel=1e-12, abs=1e-14)

    @pytest.mark.parametrize("kind", KINDS)
    def test_degenerate_cells_match_scalar(self, kind):
        """T=0, sigma=0, S=0 and K=0 must take the same branch in both."""
        for S, T, sigma, K in (
            (100.0, 0.0, 0.2, 100.0),
            (110.0, 0.0, 0.2, 100.0),
            (90.0, 0.0, 0.2, 100.0),
            (110.0, 1.0, 0.0, 100.0),
            (0.0, 1.0, 0.2, 100.0),
            (100.0, 1.0, 0.2, 0.0),
        ):
            assert float(vec.price(S, K, T, 0.05, sigma, 0.0, kind)) == pytest.approx(
                scalar.price(S, K, T, 0.05, sigma, 0.0, kind), rel=1e-12, abs=1e-14
            )
            got = vec.greeks(S, K, T, 0.05, sigma, 0.0, kind)
            want = scalar.greeks(S, K, T, 0.05, sigma, 0.0, kind)
            for greek in GREEK_NAMES:
                assert float(got[greek]) == pytest.approx(
                    getattr(want, greek), rel=1e-12, abs=1e-14
                ), f"{greek} mismatch at S={S} K={K} T={T} sigma={sigma}"

    def test_no_warnings_on_a_grid_touching_zero(self):
        """The masked substitution must not leak log(0) or 0/0 warnings."""
        S = np.array([0.0, 50.0, 100.0])[None, :]
        T = np.array([0.0, 1.0])[:, None]
        with np.errstate(all="raise"):
            out = vec.greeks(S, 100.0, T, 0.05, 0.2, 0.0, "call")
        for arr in out.values():
            assert np.isfinite(arr).all()


class TestBroadcasting:
    def test_grid_shape(self):
        S = np.linspace(50.0, 150.0, 7)[None, :]
        T = np.linspace(0.1, 2.0, 5)[:, None]
        out = vec.greeks(S, 100.0, T, 0.05, 0.2, 0.0, "call")
        for greek in GREEK_NAMES:
            assert out[greek].shape == (5, 7)

    def test_scalar_inputs_give_zero_dim_arrays(self):
        out = vec.price(100.0, 100.0, 1.0, 0.05, 0.2)
        assert out.shape == ()

    def test_vol_axis_broadcasts_too(self):
        sigma = np.linspace(0.1, 0.5, 4).reshape(4, 1, 1)
        S = np.linspace(80.0, 120.0, 6).reshape(1, 1, 6)
        T = np.linspace(0.2, 1.0, 3).reshape(1, 3, 1)
        assert vec.price(S, 100.0, T, 0.05, sigma).shape == (4, 3, 6)


class TestVectorizedProperties:
    def test_put_call_parity_on_a_grid(self):
        S = np.linspace(60.0, 140.0, 25)
        T, r, q = 0.75, 0.04, 0.02
        call = vec.price(S, 100.0, T, r, 0.25, q, "call")
        put = vec.price(S, 100.0, T, r, 0.25, q, "put")
        np.testing.assert_allclose(
            call - put, S * math.exp(-q * T) - 100.0 * math.exp(-r * T), rtol=1e-12
        )

    def test_gamma_and_vega_are_kind_invariant(self):
        S = np.linspace(60.0, 140.0, 25)
        call = vec.greeks(S, 100.0, 1.0, 0.05, 0.2, 0.0, "call")
        put = vec.greeks(S, 100.0, 1.0, 0.05, 0.2, 0.0, "put")
        for greek in ("gamma", "vega"):
            np.testing.assert_allclose(call[greek], put[greek], rtol=1e-12)

    def test_gamma_peaks_near_the_money(self):
        S = np.linspace(60.0, 140.0, 161)
        gamma = vec.greeks(S, 100.0, 0.25, 0.05, 0.2, 0.0, "call")["gamma"]
        assert 95.0 < S[np.argmax(gamma)] < 101.0

    def test_gamma_peak_sharpens_as_expiry_approaches(self):
        """The visual point of the surface: shorter expiry, taller narrower peak."""
        S = np.linspace(60.0, 140.0, 321)
        peaks = [
            vec.greeks(S, 100.0, T, 0.05, 0.2, 0.0, "call")["gamma"].max()
            for T in (2.0, 1.0, 0.5, 0.1)
        ]
        assert peaks == sorted(peaks)

    @pytest.mark.parametrize("bad", [{"S": -1.0}, {"T": -1.0}, {"sigma": -0.1}])
    def test_rejects_negative_inputs(self, bad):
        kwargs = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, **bad}
        with pytest.raises(ValueError):
            vec.price(**kwargs)

    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match="call.*put"):
            vec.price(100.0, 100.0, 1.0, 0.05, 0.2, kind="butterfly")
