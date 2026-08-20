import numpy as np
import pytest

from frontfinder.config.manifests import MODEL_1702_MANIFEST
from frontfinder.zarrio.pyramid import (
    FrontFields,
    build_front_pyramid,
    build_level0_dataset,
    write_front_pyramid,
)


@pytest.fixture
def small_fields():
    lat = np.linspace(25.0, 56.75, 64)
    lon = np.linspace(228.0, 299.75, 64)
    rng = np.random.default_rng(0)
    probs = {
        cls: rng.random((64, 64)).astype(np.float32) for cls in MODEL_1702_MANIFEST.served_classes
    }
    return FrontFields(
        probabilities=probs,
        lat=lat,
        lon=lon,
        valid_time="2026-08-19T12:00:00",
        cycle_time="2026-08-19T12:00:00",
    )


def test_front_fields_rejects_shape_mismatch():
    lat = np.linspace(0, 1, 10)
    lon = np.linspace(0, 1, 10)
    probs = {cls: np.zeros((5, 5)) for cls in MODEL_1702_MANIFEST.served_classes}
    with pytest.raises(ValueError):
        FrontFields(probabilities=probs, lat=lat, lon=lon, valid_time="t", cycle_time="t")


def test_build_level0_dataset_has_only_served_classes(small_fields):
    ds = build_level0_dataset(small_fields, MODEL_1702_MANIFEST)
    assert set(ds.data_vars) == set(MODEL_1702_MANIFEST.served_classes)
    assert "dryline" not in ds.data_vars
    assert "background" not in ds.data_vars


def test_build_level0_dataset_rejects_missing_class(small_fields):
    small_fields.probabilities.pop("cold")
    with pytest.raises(ValueError):
        build_level0_dataset(small_fields, MODEL_1702_MANIFEST)


def test_build_level0_dataset_carries_provenance_attrs(small_fields):
    ds = build_level0_dataset(small_fields, MODEL_1702_MANIFEST)
    assert ds.attrs["model"] == "model_1702"
    assert ds.attrs["cycle_time"] == "2026-08-19T12:00:00"


def test_build_front_pyramid_has_requested_number_of_levels(small_fields):
    pyramid = build_front_pyramid(small_fields, MODEL_1702_MANIFEST, n_levels=3)
    level_groups = [g for g in pyramid.datatree.groups if g not in ("", "/")]
    assert len(level_groups) == 3


def test_build_front_pyramid_coarsens_each_level_by_half(small_fields):
    pyramid = build_front_pyramid(small_fields, MODEL_1702_MANIFEST, n_levels=3)
    lvl0 = pyramid.datatree["0"].to_dataset()
    lvl1 = pyramid.datatree["1"].to_dataset()
    assert lvl0.sizes["lat"] == 64
    assert lvl1.sizes["lat"] == 32


def test_build_level0_dataset_embeds_colormap_and_clim_style_attrs(small_fields):
    ds = build_level0_dataset(small_fields, MODEL_1702_MANIFEST)
    for cls in MODEL_1702_MANIFEST.served_classes:
        assert ds[cls].attrs["colormap"]
        assert ds[cls].attrs["clim"] == [0.0, 1.0]


def test_build_front_pyramid_rejects_zero_levels(small_fields):
    with pytest.raises(ValueError):
        build_front_pyramid(small_fields, MODEL_1702_MANIFEST, n_levels=0)


def test_write_front_pyramid_roundtrips_through_zarr(small_fields, tmp_path):
    import xarray as xr

    pyramid = build_front_pyramid(small_fields, MODEL_1702_MANIFEST, n_levels=2)
    store_path = str(tmp_path / "test_pyramid.zarr")
    write_front_pyramid(pyramid, store_path)

    reopened = xr.open_datatree(store_path, engine="zarr")
    lvl0 = reopened["0"].to_dataset()
    np.testing.assert_allclose(
        lvl0["cold"].values, small_fields.probabilities["cold"], atol=1e-5
    )
    assert lvl0["cold"].attrs["colormap"] == "blues"
