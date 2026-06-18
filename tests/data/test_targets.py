import numpy as np
import pytest
import xarray as xr

from fronts.data.targets import (
    FRONT_CLASS_MAP,
    dilate_fronts,
    encode_targets,
    one_hot_encode_to_dataarray,
    remap_fronts,
)

N_TIME = 5
N_LAT = 80
N_LON = 80
N_CLASSES = 6

_SMALL_LAT = 4
_SMALL_LON = 4


@pytest.fixture
def label_da() -> xr.DataArray:
    rng = np.random.default_rng(42)
    data = rng.integers(0, N_CLASSES, size=(N_TIME, N_LAT, N_LON)).astype(np.int32)
    return xr.DataArray(
        data,
        dims=["time", "latitude", "longitude"],
        coords={"time": np.arange(N_TIME)},
    )


class TestOneHotEncodeToDataarray:
    def test_dims(self, label_da):
        result = one_hot_encode_to_dataarray(label_da)
        assert list(result.dims) == ["time", "latitude", "longitude", "class"]

    def test_shape(self, label_da):
        result = one_hot_encode_to_dataarray(label_da)
        assert result.shape == (N_TIME, N_LAT, N_LON, N_CLASSES)

    def test_dtype(self, label_da):
        result = one_hot_encode_to_dataarray(label_da)
        assert result.dtype == np.float32

    def test_values_correct(self, label_da):
        result = one_hot_encode_to_dataarray(label_da).values
        expected = (label_da.values[..., np.newaxis] == np.arange(N_CLASSES)).astype(np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_time_coord_preserved(self, label_da):
        result = one_hot_encode_to_dataarray(label_da)
        np.testing.assert_array_equal(result.coords["time"].values, label_da.time.values)

    def test_one_hot_property(self, label_da):
        result = one_hot_encode_to_dataarray(label_da)
        row_sums = result.values.sum(axis=-1)
        np.testing.assert_array_almost_equal(row_sums, np.ones((N_TIME, N_LAT, N_LON)))


def _make_one_hot_da(data: np.ndarray) -> xr.DataArray:
    n_time, _n_lat, _n_lon, _n_classes = data.shape
    return xr.DataArray(
        data.astype(np.float32),
        dims=["time", "latitude", "longitude", "class"],
        coords={"time": np.arange(n_time)},
    )


def _single_front_pixel_da(cls: int = 1, n_classes: int = N_CLASSES) -> xr.DataArray:
    data = np.zeros((1, _SMALL_LAT, _SMALL_LON, n_classes), dtype=np.float32)
    center_lat = _SMALL_LAT // 2
    center_lon = _SMALL_LON // 2
    data[0, center_lat, center_lon, cls] = 1.0
    data[0, :, :, 0] = 1.0 - data[0, :, :, 1:].any(axis=-1)
    return _make_one_hot_da(data)


class TestDilateFronts:
    def test_no_dilation_returns_unchanged(self):
        da = _single_front_pixel_da()
        result = dilate_fronts(da, dilation=0)
        assert result is da

    def test_shape_preserved(self):
        da = _single_front_pixel_da()
        result = dilate_fronts(da, dilation=1)
        assert result.shape == da.shape

    def test_background_is_complement(self):
        da = _single_front_pixel_da()
        result = dilate_fronts(da, dilation=1).values
        any_front = result[..., 1:].any(axis=-1)
        np.testing.assert_array_equal(result[..., 0], (~any_front).astype(np.float32))

    def test_front_pixel_expands(self):
        da = _single_front_pixel_da(cls=1)
        result = dilate_fronts(da, dilation=1).values
        center_lat = _SMALL_LAT // 2
        center_lon = _SMALL_LON // 2
        front_class = result[0, :, :, 1]
        neighbor_count = (
            front_class[center_lat - 1, center_lon]
            + front_class[center_lat + 1, center_lon]
            + front_class[center_lat, center_lon - 1]
            + front_class[center_lat, center_lon + 1]
        )
        assert neighbor_count >= 4




def _make_raw_fronts(seed: int = 0, n_time: int = N_TIME, lat: int = 8, lon: int = 8) -> xr.DataArray:
    """Build a raw (time, lat, lon) fronts DataArray of original front codes (incl. background and noise)."""
    rng = np.random.default_rng(seed)
    codes = np.array([*FRONT_CLASS_MAP.keys(), 0, 99], dtype=np.int32)  # include background and an out-of-map code
    data = rng.choice(codes, size=(n_time, lat, lon)).astype(np.int32)
    return xr.DataArray(data, dims=["time", "latitude", "longitude"], coords={"time": np.arange(n_time)})


class TestEncodeTargets:
    def test_matches_inline_composition_no_dilation(self):
        raw = _make_raw_fronts()
        expected = one_hot_encode_to_dataarray(remap_fronts(raw)).values
        np.testing.assert_array_equal(encode_targets(raw, 0).values, expected)

    def test_matches_inline_composition_with_dilation(self):
        raw = _make_raw_fronts()
        expected = dilate_fronts(one_hot_encode_to_dataarray(remap_fronts(raw)), 1).values
        np.testing.assert_array_equal(encode_targets(raw, 1).values, expected)

    def test_shape_dims_and_dtype(self):
        raw = _make_raw_fronts()
        result = encode_targets(raw, 1)
        assert list(result.dims) == ["time", "latitude", "longitude", "class"]
        assert result.shape == (N_TIME, 8, 8, N_CLASSES)
        assert result.dtype == np.float32

    def test_one_hot_when_no_dilation(self):
        raw = _make_raw_fronts()
        result = encode_targets(raw, 0).values
        np.testing.assert_array_almost_equal(result.sum(axis=-1), np.ones(result.shape[:-1]))

    def test_out_of_map_codes_become_background(self):
        raw = xr.DataArray(
            np.full((1, 4, 4), 99, dtype=np.int32),
            dims=["time", "latitude", "longitude"],
            coords={"time": np.arange(1)},
        )
        result = encode_targets(raw, 0).values
        assert (result[..., 0] == 1.0).all()
        assert (result[..., 1:] == 0.0).all()
