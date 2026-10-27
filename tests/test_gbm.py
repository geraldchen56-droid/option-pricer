import math

import pytest

np = pytest.importorskip("numpy")

from option_pricer.gbm import (  # noqa: E402
    lognormal_pdf,
    simulate_terminal_prices,
    summarise_mean,
)

PARAMS = dict(S0=100.0, r=0.05, sigma=0.2, T=1.0)
SEED = 20260904


@pytest.fixture(scope="module")
def samples():
    """One shared 200k draw -- the statistical assertions below are seeded."""
    return simulate_terminal_prices(
        **PARAMS, n_sims=200_000, rng=np.random.default_rng(SEED)
    )


class TestSimulate:
    def test_shape_and_dtype(self, samples):
        assert samples.shape == (200_000,)
        assert samples.dtype == np.float64

    def test_all_prices_strictly_positive(self, samples):
        assert (samples > 0.0).all()

    def test_seed_is_reproducible(self):
        a = simulate_terminal_prices(**PARAMS, n_sims=1_000, rng=np.random.default_rng(7))
        b = simulate_terminal_prices(**PARAMS, n_sims=1_000, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = simulate_terminal_prices(**PARAMS, n_sims=1_000, rng=np.random.default_rng(1))
        b = simulate_terminal_prices(**PARAMS, n_sims=1_000, rng=np.random.default_rng(2))
        assert not np.array_equal(a, b)

    def test_works_without_an_explicit_rng(self):
        out = simulate_terminal_prices(**PARAMS, n_sims=100)
        assert out.shape == (100,)
        assert (out > 0.0).all()

    def test_mean_matches_forward(self, samples):
        """E[S_T] = S0 e^(rT) -- the Ito correction is what makes this hold."""
        stats = summarise_mean(samples, PARAMS["S0"], PARAMS["r"], PARAMS["T"])
        assert abs(stats["z"]) < 4.0

    def test_median_matches_theory(self, samples):
        """Median = S0 exp((r - sigma^2/2) T), strictly below the mean."""
        expected = PARAMS["S0"] * math.exp(
            (PARAMS["r"] - 0.5 * PARAMS["sigma"] ** 2) * PARAMS["T"]
        )
        assert np.median(samples) == pytest.approx(expected, rel=5e-3)
        assert np.median(samples) < samples.mean()

    def test_variance_matches_theory(self, samples):
        S0, r, sigma, T = PARAMS["S0"], PARAMS["r"], PARAMS["sigma"], PARAMS["T"]
        expected = S0**2 * math.exp(2 * r * T) * (math.exp(sigma**2 * T) - 1.0)
        assert samples.var(ddof=1) == pytest.approx(expected, rel=2e-2)

    def test_log_returns_are_normal(self, samples):
        """Skew of log S_T should vanish; the skew lives in the price, not the log."""
        logs = np.log(samples / PARAMS["S0"])
        z = (logs - logs.mean()) / logs.std()
        assert abs((z**3).mean()) < 0.05
        assert (z**4).mean() == pytest.approx(3.0, abs=0.1)  # excess kurtosis ~ 0

    def test_dividend_yield_lowers_the_mean(self):
        rng = np.random.default_rng(SEED)
        with_div = simulate_terminal_prices(
            **PARAMS, n_sims=50_000, q=0.03, rng=np.random.default_rng(SEED)
        )
        without = simulate_terminal_prices(**PARAMS, n_sims=50_000, rng=rng)
        assert with_div.mean() < without.mean()
        # Same normals, so the ratio is the deterministic drift shift exactly.
        np.testing.assert_allclose(
            with_div / simulate_terminal_prices(
                **PARAMS, n_sims=50_000, rng=np.random.default_rng(SEED)
            ),
            math.exp(-0.03 * PARAMS["T"]),
            rtol=1e-12,
        )

    def test_zero_vol_is_deterministic(self):
        out = simulate_terminal_prices(S0=100.0, r=0.05, sigma=0.0, T=1.0, n_sims=50)
        np.testing.assert_allclose(out, 100.0 * math.exp(0.05), rtol=1e-12)

    def test_zero_horizon_returns_spot(self):
        out = simulate_terminal_prices(S0=100.0, r=0.05, sigma=0.2, T=0.0, n_sims=50)
        np.testing.assert_allclose(out, 100.0, rtol=1e-12)

    @pytest.mark.parametrize(
        "bad", [{"n_sims": 0}, {"n_sims": -5}, {"T": -1.0}, {"sigma": -0.1}, {"S0": -1.0}]
    )
    def test_rejects_bad_inputs(self, bad):
        with pytest.raises(ValueError):
            simulate_terminal_prices(**{**PARAMS, "n_sims": 10, **bad})


class TestLognormalPdf:
    def test_integrates_to_one(self):
        x = np.linspace(1e-6, 600.0, 400_001)
        assert np.trapezoid(lognormal_pdf(x, **PARAMS), x) == pytest.approx(1.0, abs=1e-4)

    def test_non_positive_prices_have_zero_density(self):
        out = lognormal_pdf(np.array([-5.0, 0.0, 1.0]), **PARAMS)
        assert out[0] == 0.0 and out[1] == 0.0
        assert out[2] > 0.0

    def test_peaks_at_the_mode(self):
        """Mode = S0 exp((r - 3 sigma^2 / 2) T) for the lognormal."""
        S0, r, sigma, T = PARAMS["S0"], PARAMS["r"], PARAMS["sigma"], PARAMS["T"]
        mode = S0 * math.exp((r - 1.5 * sigma**2) * T)
        x = np.linspace(40.0, 200.0, 20_001)
        assert x[np.argmax(lognormal_pdf(x, **PARAMS))] == pytest.approx(mode, rel=1e-3)

    def test_density_matches_histogram(self, samples):
        """Observed bin counts must sit within sampling error of the density.

        The pdf is integrated *across* each bin rather than sampled at the
        centre -- a bin's average density is not its midpoint value, and that
        curvature bias is comparable to the Monte Carlo noise at this n. Each
        bin count is then binomial, so 4 standard errors is the bar.
        """
        counts, edges = np.histogram(samples, bins=40, range=(60.0, 160.0))
        n = samples.size

        lo, hi = edges[:-1], edges[1:]
        # 65-point trapezoid per bin, vectorised over all 40 bins at once.
        t = np.linspace(0.0, 1.0, 65)
        grid = lo[:, None] + (hi - lo)[:, None] * t[None, :]
        probs = np.trapezoid(lognormal_pdf(grid, **PARAMS), grid, axis=1)

        expected = probs * n
        std_err = np.sqrt(n * probs * (1.0 - probs))
        assert np.all(np.abs(counts - expected) < 4.0 * std_err)


class TestSummariseMean:
    def test_reports_expected_fields(self, samples):
        stats = summarise_mean(samples, PARAMS["S0"], PARAMS["r"], PARAMS["T"])
        assert set(stats) == {
            "n_sims", "sample_mean", "theoretical_mean",
            "error", "std_err", "z", "rel_error",
        }
        assert stats["n_sims"] == 200_000
        assert stats["theoretical_mean"] == pytest.approx(100.0 * math.exp(0.05))

    def test_std_err_shrinks_as_sqrt_n(self):
        small = summarise_mean(
            simulate_terminal_prices(**PARAMS, n_sims=2_500, rng=np.random.default_rng(3)),
            PARAMS["S0"], PARAMS["r"], PARAMS["T"],
        )
        big = summarise_mean(
            simulate_terminal_prices(**PARAMS, n_sims=250_000, rng=np.random.default_rng(3)),
            PARAMS["S0"], PARAMS["r"], PARAMS["T"],
        )
        assert big["std_err"] == pytest.approx(small["std_err"] / 10.0, rel=0.15)


class TestPlot:
    def test_writes_a_png(self, tmp_path, samples):
        pytest.importorskip("matplotlib")
        from option_pricer.gbm import plot_terminal_distribution

        out = tmp_path / "hist.png"
        path = plot_terminal_distribution(samples[:20_000], **PARAMS, out_path=str(out))
        assert path == str(out)
        assert out.stat().st_size > 10_000
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
