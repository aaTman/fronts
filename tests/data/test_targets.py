import numpy as np
import pytest
import xarray as xr

from fronts.data.targets import one_hot_encode_to_dataarray

N_TIME = 5
N_LAT = 80
N_LON = 80
N_CLASSES = 6


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
