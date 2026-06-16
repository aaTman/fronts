import numpy as np
import pytest
import xarray as xr

from fronts.data import config, inputs, targets
from fronts.data.loading import TrainingData, assemble_training_data

_VARS = ["temperature", "u_component_of_wind"]
_LEVELS = [1000, 850]
_LAT = 4
_LON = 6
_ALL_CODES = list(targets.FRONT_CLASS_MAP.keys())


def _days(start: str, periods: int) -> np.ndarray:
    return np.datetime64(start, "D") + np.arange(periods)


def _hours(start: str, periods: int, step: int) -> np.ndarray:
    return np.datetime64(start, "h") + np.arange(periods) * np.timedelta64(step, "h")


def _era5_ds(times: np.ndarray, seed: int = 0, variables: list[str] = _VARS) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    data_vars = {}
    for var in variables:
        data_vars[var] = xr.DataArray(
            rng.standard_normal((len(times), len(_LEVELS), _LAT, _LON)).astype(np.float32),
            dims=["time", "level", "latitude", "longitude"],
            coords={"time": times, "level": _LEVELS},
        )
    return xr.Dataset(data_vars)


def _fronts_da(times: np.ndarray, all_codes: bool = True) -> xr.DataArray:
    data = np.zeros((len(times), _LAT, _LON), dtype=np.int32)
    if all_codes:
        for t in range(len(times)):
            for i, code in enumerate(_ALL_CODES):
                data[t, 0, i] = code
    return xr.DataArray(data, dims=["time", "latitude", "longitude"], coords={"time": times})


def _data_config(variables: list[str] = _VARS, time_resolution: str | None = None, front_dilation: int = 0):
    dummy = config.IcechunkStorageConfig(store_path="/tmp/unused", branch_name="main")
    return config.DataConfig(
        era5_icechunk_config=dummy,
        fronts_icechunk_config=dummy,
        variables=variables,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        time_resolution=time_resolution,
        front_dilation=front_dilation,
    )


def _src(variables: list[str] = _VARS) -> config.InputSourceConfig:
    dummy = config.IcechunkStorageConfig(store_path="/tmp/unused", branch_name="main")
    return config.InputSourceConfig(name="era5", icechunk_config=dummy, variables=variables)


class TestAssembleTrainingData:
    def test_times_are_intersection(self):
        era5 = _era5_ds(_days("2020-01-01", 6))  # d0..d5
        fronts = _fronts_da(_days("2020-01-03", 6))  # d2..d7
        td = assemble_training_data([(_src(), era5)], fronts, _data_config())
        np.testing.assert_array_equal(td.times.astype("datetime64[D]"), _days("2020-01-03", 4))  # d2..d5

    def test_time_resolution_subsamples(self):
        times = _hours("2020-01-01", 8, 3)  # 00,03,06,...,21
        era5 = _era5_ds(times)
        fronts = _fronts_da(times)
        td = assemble_training_data([(_src(), era5)], fronts, _data_config(time_resolution="6h"))
        hours = td.times.astype("datetime64[h]").astype(int) % 24
        np.testing.assert_array_equal(sorted(hours), [0, 6, 12, 18])

    def test_all_types_present_keeps_every_timestep(self):
        times = _days("2020-01-01", 5)
        td = assemble_training_data([(_src(), _era5_ds(times))], _fronts_da(times), _data_config())
        assert len(td.times) == 5

    def test_presence_filter_subsamples_and_matches_filter_timesteps(self):
        times = _days("2020-01-01", 12)
        fronts = _fronts_da(times, all_codes=False)  # background only -> 50% draws
        td = assemble_training_data([(_src(), _era5_ds(times))], fronts, _data_config(), seed=3)
        expected_keep = targets.filter_timesteps(fronts, np.random.default_rng(3))
        assert len(td.times) == int(expected_keep.sum())
        assert len(td.times) < len(times)

    def test_input_positions_map_to_common_times(self):
        era5 = _era5_ds(_days("2020-01-01", 6))
        fronts = _fronts_da(_days("2020-01-03", 6))
        td = assemble_training_data([(_src(), era5)], fronts, _data_config())
        source = td.input_sources[0]
        np.testing.assert_array_equal(source.array.isel(time=source.positions).time.values, td.times)

    def test_input_channels_concatenated_per_source(self):
        times = _days("2020-01-01", 4)
        sources = [
            (_src(["temperature"]), _era5_ds(times, variables=["temperature"])),
            (_src(["u_component_of_wind"]), _era5_ds(times, seed=2, variables=["u_component_of_wind"])),
        ]
        td = assemble_training_data(sources, _fronts_da(times), _data_config())
        assert td.input_aligned.sizes["channel"] == 2 * len(_LEVELS)

    def test_lazy_inputs_single_take_equivalence(self):
        era5 = _era5_ds(_days("2020-01-01", 6))
        fronts = _fronts_da(_days("2020-01-02", 6))
        td = assemble_training_data([(_src(), era5)], fronts, _data_config())
        idxs = np.array([2, 0, 1])
        source = td.input_sources[0]
        expected = inputs.era5_to_dataarray(era5, _VARS).isel(time=source.positions[idxs])
        np.testing.assert_array_equal(td.lazy_inputs(idxs).values, expected.values)

    def test_target_uses_encode_targets_with_dilation(self):
        times = _days("2020-01-01", 5)
        fronts = _fronts_da(times)
        td = assemble_training_data([(_src(), _era5_ds(times))], fronts, _data_config(front_dilation=1))
        positions = inputs.native_positions(fronts.time.values, td.times)
        expected = targets.encode_targets(fronts, 1).isel(time=positions)
        np.testing.assert_array_equal(td.target_aligned.values, expected.values)

    def test_duplicate_native_times_keep_first(self):
        times = np.array(["2020-01-01", "2020-01-01", "2020-01-02"], dtype="datetime64[D]")
        era5 = _era5_ds(times)
        fronts = _fronts_da(times)
        td = assemble_training_data([(_src(), era5)], fronts, _data_config())
        # Two unique common days; positions index the first occurrence in the raw axis.
        np.testing.assert_array_equal(td.times, np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"))
        np.testing.assert_array_equal(td.input_sources[0].positions, [0, 2])

    def test_raises_when_source_lacks_time(self):
        times = _days("2020-01-01", 4)
        static = _era5_ds(times).isel(time=0, drop=True)
        with pytest.raises(ValueError, match="no time dimension"):
            assemble_training_data([(_src(), static)], _fronts_da(times), _data_config())


class TestTrainingData:
    def _td(self, n_time: int = 6) -> TrainingData:
        times = _days("2020-01-01", n_time)
        return assemble_training_data([(_src(), _era5_ds(times))], _fronts_da(times), _data_config())

    def test_input_aligned_equals_lazy_inputs_none(self):
        td = self._td()
        np.testing.assert_array_equal(td.input_aligned.values, td.lazy_inputs(None).values)

    def test_lazy_inputs_restricts_to_idxs(self):
        td = self._td()
        assert td.lazy_inputs(np.array([0, 2, 4])).sizes["time"] == 3

    def test_target_aligned_shape_matches_times(self):
        td = self._td()
        assert td.target_aligned.sizes["time"] == len(td.times)
        assert td.target_aligned.sizes["class"] == 6
