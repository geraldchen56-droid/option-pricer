import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from option_pricer.surface import build_surface  # noqa: E402
from option_pricer.vectorized import GREEK_NAMES  # noqa: E402

# n_spot=15 over 40..180 lands exactly on the strike (step 10). A grid that
# straddles K without hitting it misses the ATM gamma spike entirely at low
# vol, which silently breaks any test about peak height.
SMALL = dict(n_spot=15, n_time=9, vols=(0.1, 0.2, 0.4))


class TestBuildSurface:
    def test_stack_shape_is_vols_by_time_by_spot(self):
        data = build_surface(greek="gamma", **SMALL)
        assert data["stack"].shape == (3, 9, 15)
        assert data["spots"].shape == (15,)
        assert data["times"].shape == (9,)

    def test_all_values_finite(self):
        for greek in GREEK_NAMES:
            data = build_surface(greek=greek, **SMALL)
            assert np.isfinite(data["stack"]).all(), greek

    def test_reported_range_matches_the_stack(self):
        data = build_surface(greek="gamma", **SMALL)
        assert data["zmin"] == data["stack"].min()
        assert data["zmax"] == data["stack"].max()

    def test_time_grid_avoids_expiry(self):
        """A grid touching T=0 would be dominated by the gamma singularity."""
        data = build_surface(greek="gamma", **SMALL)
        assert data["times"].min() > 0.0

    def test_gamma_is_positive_away_from_the_boundary(self):
        data = build_surface(greek="gamma", **SMALL)
        assert (data["stack"] >= 0.0).all()
        assert data["stack"].max() > 0.0

    def test_gamma_surface_is_kind_invariant(self):
        call = build_surface(greek="gamma", kind="call", **SMALL)
        put = build_surface(greek="gamma", kind="put", **SMALL)
        np.testing.assert_allclose(call["stack"], put["stack"], rtol=1e-12)

    def test_delta_surface_is_not_kind_invariant(self):
        call = build_surface(greek="delta", kind="call", **SMALL)
        put = build_surface(greek="delta", kind="put", **SMALL)
        assert not np.allclose(call["stack"], put["stack"])
        # call delta - put delta = 1 with no dividend
        np.testing.assert_allclose(call["stack"] - put["stack"], 1.0, rtol=1e-12)

    def test_higher_vol_flattens_the_gamma_peak(self):
        """Raising vol spreads the same probability mass wider, so the
        at-the-money peak drops. Measured at the strike column rather than as
        a grid max, so the result does not depend on grid resolution."""
        data = build_surface(greek="gamma", **SMALL)
        atm = int(np.argmin(np.abs(data["spots"] - data["K"])))
        peaks = [data["stack"][i, :, atm].max() for i in range(len(data["vols"]))]
        assert peaks == sorted(peaks, reverse=True)

    def test_rejects_unknown_greek(self):
        with pytest.raises(ValueError, match="greek must be one of"):
            build_surface(greek="charm", **SMALL)

    def test_rejects_empty_vols(self):
        with pytest.raises(ValueError, match="vols must not be empty"):
            build_surface(greek="gamma", vols=[], n_spot=8, n_time=8)


class TestWriteSurface:
    def test_writes_standalone_html(self, tmp_path):
        pytest.importorskip("plotly")
        from option_pricer.surface import write_surface

        data = build_surface(greek="gamma", **SMALL)
        out = tmp_path / "surface.html"
        assert write_surface(data, str(out)) == str(out)

        html = out.read_text()
        assert "plotly" in html.lower()
        # plotly.js inlined, so the file works with no network access. The
        # bundle mentions cdn.plot.ly internally (topojson default), so check
        # for an actual remote <script src>, not the bare hostname.
        assert 'src="https://cdn.plot.ly' not in html
        assert out.stat().st_size > 1_000_000

    def test_z_axis_range_is_pinned_across_frames(self, tmp_path):
        """Autoscaling per frame would make every volatility look identical."""
        pytest.importorskip("plotly")
        from option_pricer.surface import make_figure

        data = build_surface(greek="gamma", **SMALL)
        fig = make_figure(data)

        assert list(fig.layout.scene.zaxis.range) == [data["zmin"], data["zmax"]]
        assert len(fig.frames) == len(data["vols"])
        for frame in fig.frames:
            assert frame.data[0].cmin == data["zmin"]
            assert frame.data[0].cmax == data["zmax"]

    def test_one_slider_step_per_volatility(self, tmp_path):
        pytest.importorskip("plotly")
        from option_pricer.surface import make_figure

        data = build_surface(greek="gamma", **SMALL)
        fig = make_figure(data)
        steps = fig.layout.sliders[0].steps
        assert len(steps) == len(data["vols"])
        assert [s.label for s in steps] == [f"{v:.0%}" for v in data["vols"]]
        # every step must target a frame that actually exists
        names = {f.name for f in fig.frames}
        assert all(s.args[0][0] in names for s in steps)

    def test_frame_zscale_rescales_each_frame(self):
        """Escape hatch for wide vol sweeps: readable, but not comparable."""
        pytest.importorskip("plotly")
        from option_pricer.surface import make_figure

        data = build_surface(greek="gamma", **SMALL)
        fig = make_figure(data, zscale="frame")

        # No global pin, and each frame carries its own axis range.
        assert fig.layout.scene.zaxis.range is None
        maxima = [float(data["stack"][i].max()) for i in range(len(data["vols"]))]
        for frame, want_max in zip(fig.frames, maxima):
            assert frame.data[0].cmax == pytest.approx(want_max)
            assert frame.layout.scene.zaxis.range[1] == pytest.approx(want_max)
        # frames genuinely differ, otherwise the mode is pointless
        assert len(set(maxima)) == len(maxima)

    def test_frame_zscale_warns_in_the_axis_title(self):
        pytest.importorskip("plotly")
        from option_pricer.surface import make_figure

        fixed = make_figure(build_surface(greek="gamma", **SMALL), zscale="fixed")
        framed = make_figure(build_surface(greek="gamma", **SMALL), zscale="frame")
        assert "rescaled per frame" in framed.layout.scene.zaxis.title.text
        assert "NOT comparable" in framed.layout.title.text
        assert "rescaled" not in fixed.layout.scene.zaxis.title.text
        assert "comparable" in fixed.layout.title.text

    def test_rejects_unknown_zscale(self):
        pytest.importorskip("plotly")
        from option_pricer.surface import make_figure

        with pytest.raises(ValueError, match="zscale must be"):
            make_figure(build_surface(greek="gamma", **SMALL), zscale="log")
