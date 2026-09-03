import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fronts.data.targets import FRONT_CLASS_MAP, filter_timesteps
from fronts.utils import IcechunkStorageConfig, apply_time_resolution

try:
    import tensorflow as tf

    from fronts.callbacks import CallbacksConfig
    from fronts.data.datasets import DatasetConfig, FrontsPyDataset, PatchConfig, compute_patch_lon_starts
    from fronts.data.generate import write_or_append_icechunk_store
    from fronts.data.inputs import inputs_ds_to_dataarray
    from fronts.layers import losses
    from fronts.model import ModelConfig, UNet3Plus
    from fronts.train import (
        TrainConfig,
        _build_loss,
        _build_monitor_callbacks,
        _build_run_callbacks,
        _build_test_visualization_callback,
        _compile,
        _freeze_layers,
        _load_pretrained_weights,
        _optimizer_uses_ema,
        _pred_buffer_px_from_data_config,
        _should_build_test_visualization,
        _target_latitudes,
        load_data_into_dataloader,
    )

    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

_ALL_CODES = list(FRONT_CLASS_MAP.keys())  # [1, 2, 3, 4, 15]


def _make_fronts(time_codes: list[list[int]], lat: int = 4, lon: int = 8) -> xr.DataArray:
    """Build a (time, lat, lon) fronts DataArray where each timestep gets exactly the front codes listed.

    Codes are placed at the first pixels of the first row.
    """
    n_time = len(time_codes)
    data = np.zeros((n_time, lat, lon), dtype=np.int32)
    for t, codes in enumerate(time_codes):
        for i, code in enumerate(codes):
            data[t, 0, i] = code
    return xr.DataArray(data, dims=["time", "latitude", "longitude"])


N_TIME = 5
N_LAT = 32
N_LON = 64
N_CLASSES = 6


class TestFilterTimesteps:
    def test_all_types_present_always_kept(self):
        da = _make_fronts([_ALL_CODES, _ALL_CODES])
        rng = np.random.default_rng(0)
        mask = filter_timesteps(da, rng)
        assert mask.all()

    def test_incomplete_timestep_dropped_by_rng(self):
        # One code missing — outcome is purely the RNG 50% draw.
        # Seed 0: first draw ~0.64 (>= 0.5), so dropped.
        da = _make_fronts([_ALL_CODES[:-1]])
        rng = np.random.default_rng(0)
        mask = filter_timesteps(da, rng)
        assert not mask[0]

    def test_incomplete_timestep_kept_by_rng(self):
        # Seed 2: first draw ~0.26 (< 0.5), so kept.
        da = _make_fronts([_ALL_CODES[:-1]])
        rng = np.random.default_rng(2)
        mask = filter_timesteps(da, rng)
        assert mask[0]

    def test_background_only_uses_rng(self):
        # Pure background (0) has no front types; result is a 50% draw.
        da = _make_fronts([[0]])
        kept = sum(filter_timesteps(da, np.random.default_rng(s))[0] for s in range(200))
        assert 70 < kept < 130  # expect ~100 with reasonable variance

    def test_mixed_timesteps(self):
        # First timestep complete (always kept), second incomplete (RNG-dependent).
        da = _make_fronts([_ALL_CODES, _ALL_CODES[:2]])
        rng = np.random.default_rng(0)
        mask = filter_timesteps(da, rng)
        assert mask[0]  # guaranteed

    def test_return_shape(self):
        da = _make_fronts([_ALL_CODES] * 7)
        mask = filter_timesteps(da, np.random.default_rng(0))
        assert mask.shape == (7,)
        assert mask.dtype == bool


class TestApplyTimeResolution:
    def _make_times(self, freq: str, periods: int, start: str = "2020-01-01") -> np.ndarray:
        return pd.date_range(start, periods=periods, freq=freq).values

    def test_6h_from_3h_keeps_half(self):
        times = self._make_times("3h", 8)  # 00, 03, 06, 09, 12, 15, 18, 21
        result = apply_time_resolution(times, "6h")
        assert len(result) == 4  # 00, 06, 12, 18

    def test_6h_from_3h_correct_hours(self):
        times = self._make_times("3h", 8)
        result = apply_time_resolution(times, "6h")
        hours = pd.DatetimeIndex(result).hour.tolist()
        assert hours == [0, 6, 12, 18]

    def test_already_aligned_unchanged(self):
        times = self._make_times("6h", 4)
        result = apply_time_resolution(times, "6h")
        np.testing.assert_array_equal(result, times)

    def test_12h_from_3h_correct_hours(self):
        times = self._make_times("3h", 8)
        result = apply_time_resolution(times, "12h")
        hours = pd.DatetimeIndex(result).hour.tolist()
        assert hours == [0, 12]

    def test_empty_input(self):
        times = np.array([], dtype="datetime64[ns]")
        result = apply_time_resolution(times, "6h")
        assert len(result) == 0

    def test_no_aligned_timestamps(self):
        times = pd.date_range("2020-01-01 01:00", periods=4, freq="3h").values  # 01, 04, 07, 10
        result = apply_time_resolution(times, "6h")
        assert len(result) == 0

    def test_multi_day_span(self):
        times = self._make_times("3h", 16)  # 2 days of 3h data
        result = apply_time_resolution(times, "6h")
        assert len(result) == 8
        assert all(h in (0, 6, 12, 18) for h in pd.DatetimeIndex(result).hour)


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestBuildMonitorCallbacks:
    def test_both_decay_params_set_returns_reduce_lr_on_plateau(self):
        callbacks = _build_monitor_callbacks(
            monitor="val_loss", patience=5, learning_rate_decay_factor=0.2, learning_rate_minimum=1e-6
        )
        assert len(callbacks) == 1
        callback = callbacks[0]
        assert isinstance(callback, tf.keras.callbacks.ReduceLROnPlateau)
        assert callback.monitor == "val_loss"
        assert callback.factor == 0.2
        assert callback.patience == 5
        assert callback.min_lr == 1e-6

    def test_only_decay_factor_set_returns_early_stopping(self):
        callbacks = _build_monitor_callbacks(
            monitor="val_loss", patience=5, learning_rate_decay_factor=0.2, learning_rate_minimum=None
        )
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], tf.keras.callbacks.EarlyStopping)

    def test_only_decay_minimum_set_returns_early_stopping(self):
        callbacks = _build_monitor_callbacks(
            monitor="val_loss", patience=5, learning_rate_decay_factor=None, learning_rate_minimum=1e-6
        )
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], tf.keras.callbacks.EarlyStopping)

    def test_neither_set_returns_early_stopping(self):
        callbacks = _build_monitor_callbacks(
            monitor="val_loss", patience=5, learning_rate_decay_factor=None, learning_rate_minimum=None
        )
        assert len(callbacks) == 1
        callback = callbacks[0]
        assert isinstance(callback, tf.keras.callbacks.EarlyStopping)
        assert callback.monitor == "val_loss"
        assert callback.patience == 5
        assert callback.restore_best_weights is True

    def test_min_delta_defaults_to_zero_not_keras_absolute_1e4(self):
        """min_delta must default to 0, not Keras's absolute 1e-4.

        At a loss magnitude of ~1e-3, an absolute min_delta of 1e-4 reads every epoch as
        a plateau and decays the LR to its floor within a dozen epochs.
        """
        callbacks = _build_monitor_callbacks(
            monitor="val_loss", patience=3, learning_rate_decay_factor=0.2, learning_rate_minimum=1e-6
        )
        assert callbacks[0].min_delta == 0.0

    def test_min_delta_passed_through_to_both_callback_types(self):
        reduce_lr = _build_monitor_callbacks(
            monitor="val_loss",
            patience=3,
            learning_rate_decay_factor=0.2,
            learning_rate_minimum=1e-6,
            min_delta=1e-5,
        )[0]
        assert reduce_lr.min_delta == 1e-5
        early_stop = _build_monitor_callbacks(
            monitor="val_loss",
            patience=3,
            learning_rate_decay_factor=None,
            learning_rate_minimum=None,
            min_delta=1e-5,
        )[0]
        assert early_stop.min_delta == 1e-5

    def test_lr_decay_with_early_stopping_returns_both(self):
        """LR-decay mode alone has no stop condition; early_stopping_patience adds one."""
        callbacks = _build_monitor_callbacks(
            monitor="val_loss",
            patience=3,
            learning_rate_decay_factor=0.2,
            learning_rate_minimum=1e-6,
            min_delta=1e-5,
            early_stopping_patience=12,
        )
        assert len(callbacks) == 2
        reduce_lr, early_stop = callbacks
        assert isinstance(reduce_lr, tf.keras.callbacks.ReduceLROnPlateau)
        assert reduce_lr.patience == 3
        assert isinstance(early_stop, tf.keras.callbacks.EarlyStopping)
        assert early_stop.patience == 12
        assert early_stop.restore_best_weights is True
        assert early_stop.min_delta == 1e-5

    def test_early_stopping_patience_ignored_without_lr_decay(self):
        callbacks = _build_monitor_callbacks(
            monitor="val_loss",
            patience=5,
            learning_rate_decay_factor=None,
            learning_rate_minimum=None,
            early_stopping_patience=12,
        )
        assert len(callbacks) == 1
        assert callbacks[0].patience == 5


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestOptimizerUsesEma:
    def test_plain_optimizer_without_ema_is_false(self):
        assert _optimizer_uses_ema(tf.keras.optimizers.Adam(use_ema=False)) is False

    def test_plain_optimizer_with_ema_is_true(self):
        assert _optimizer_uses_ema(tf.keras.optimizers.Adam(use_ema=True)) is True

    def test_loss_scale_wrapped_optimizer_is_unwrapped(self):
        """Mixed-precision training wraps Adam in a LossScaleOptimizer; use_ema lives on the inner optimizer."""
        wrapped = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(use_ema=True))
        assert _optimizer_uses_ema(wrapped) is True

    def test_loss_scale_wrapped_optimizer_without_ema_is_false(self):
        wrapped = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(use_ema=False))
        assert _optimizer_uses_ema(wrapped) is False


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestBuildRunCallbacks:
    """Callback-order test suite.

    `_build_run_callbacks` ordering directly determines whether EMA-swapped weights make it
    into checkpoints and EarlyStopping's best-weight snapshot — see its docstring.
    """

    def _build(self, uses_ema: bool, **overrides):
        kwargs = {
            "uses_ema": uses_ema,
            "monitor": "val_loss",
            "patience": 5,
            "learning_rate_decay_factor": None,
            "learning_rate_minimum": None,
            "monitor_min_delta": 0.0,
            "early_stopping_patience": None,
            "extra_callbacks": None,
            "wandb_project": None,
            "wandb_log_freq": "epoch",
            "model_checkpoint_path": None,
        }
        kwargs.update(overrides)
        return _build_run_callbacks(**kwargs)

    def test_no_ema_omits_swap_ema_weights_callback(self):
        callbacks = self._build(uses_ema=False)
        assert not any(isinstance(cb, tf.keras.callbacks.SwapEMAWeights) for cb in callbacks)

    def test_ema_adds_swap_ema_weights_first(self):
        callbacks = self._build(uses_ema=True)
        assert isinstance(callbacks[0], tf.keras.callbacks.SwapEMAWeights)
        assert callbacks[0].swap_on_epoch is True

    def test_swap_ema_weights_precedes_early_stopping(self):
        callbacks = self._build(uses_ema=True)
        swap_idx = next(i for i, cb in enumerate(callbacks) if isinstance(cb, tf.keras.callbacks.SwapEMAWeights))
        early_stop_idx = next(i for i, cb in enumerate(callbacks) if isinstance(cb, tf.keras.callbacks.EarlyStopping))
        assert swap_idx < early_stop_idx

    def test_swap_ema_weights_precedes_model_checkpoint(self, tmp_path):
        callbacks = self._build(uses_ema=True, model_checkpoint_path=str(tmp_path / "model"))
        swap_idx = next(i for i, cb in enumerate(callbacks) if isinstance(cb, tf.keras.callbacks.SwapEMAWeights))
        ckpt_idx = next(i for i, cb in enumerate(callbacks) if isinstance(cb, tf.keras.callbacks.ModelCheckpoint))
        assert swap_idx < ckpt_idx


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestFrontsPyDatasetGather:
    def test_select_and_order_samples(self, era5_ds, front_da, data_config):
        # A non-contiguous time selection must yield exactly those timesteps in order.
        sub_era5 = era5_ds.isel(time=[4, 2])
        sub_front = front_da.isel(time=[4, 2])
        ds = FrontsPyDataset(sub_era5, sub_front, data_config, batch_size=1)
        x0, _ = ds[0]
        x1, _ = ds[1]
        expected = inputs_ds_to_dataarray(era5_ds, data_config.variables).values
        np.testing.assert_allclose(x0[0], expected[4])
        np.testing.assert_allclose(x1[0], expected[2])

    def test_gather_preserves_order_and_values(self, era5_ds, front_da, data_config):
        order = [4, 0, 3, 1, 2]
        sub_era5 = era5_ds.isel(time=order)
        sub_front = front_da.isel(time=order)
        ds = FrontsPyDataset(sub_era5, sub_front, data_config, batch_size=1)
        expected = inputs_ds_to_dataarray(era5_ds, data_config.variables).values
        for i, native in enumerate(order):
            x, _ = ds[i]
            np.testing.assert_allclose(x[0], expected[native])

    def test_input_target_length_mismatch_raises(self, era5_ds, front_da, data_config):
        with pytest.raises(ValueError, match="differ"):
            FrontsPyDataset(
                era5_ds.isel(time=[0, 1]),
                front_da.isel(time=[0]),
                data_config,
                batch_size=1,
            )


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestFrontsPyDataset:
    def _make_ds(self, era5_ds, front_da, data_config, batch_size=2, **kwargs):
        return FrontsPyDataset(era5_ds, front_da, data_config, batch_size=batch_size, **kwargs)

    def test_input_batch_shape(self, era5_ds, front_da, data_config):
        batch_size = 2
        ds = self._make_ds(era5_ds, front_da, data_config, batch_size=batch_size)
        x_batch, _ = ds[0]
        assert x_batch.shape == (batch_size, N_LAT, N_LON, len(data_config.variables))

    def test_target_batch_shape(self, era5_ds, front_da, data_config):
        batch_size = 2
        ds = self._make_ds(era5_ds, front_da, data_config, batch_size=batch_size)
        _, y_batch = ds[0]
        assert y_batch.shape == (batch_size, N_LAT, N_LON, N_CLASSES)

    def test_covers_all_timesteps(self, era5_ds, front_da, data_config):
        batch_size = 2
        ds = self._make_ds(era5_ds, front_da, data_config, batch_size=batch_size)
        total_samples = sum(ds[i][0].shape[0] for i in range(len(ds)))
        assert total_samples == N_TIME

    def test_dtypes_are_float32(self, era5_ds, front_da, data_config):
        ds = self._make_ds(era5_ds, front_da, data_config, batch_size=2)
        x_batch, y_batch = ds[0]
        assert x_batch.dtype == np.float32
        assert y_batch.dtype == np.float32

    def test_shuffle_reshuffles_on_epoch_end(self, era5_ds, front_da, data_config):
        ds = self._make_ds(era5_ds, front_da, data_config, batch_size=1, shuffle=True, seed=0)
        order_before = ds._order.copy()
        ds.on_epoch_end()
        assert not np.array_equal(order_before, ds._order)

    def test_no_shuffle_preserves_order(self, era5_ds, front_da, data_config):
        ds = self._make_ds(era5_ds, front_da, data_config, batch_size=1, shuffle=False)
        np.testing.assert_array_equal(ds._order, np.arange(N_TIME))
        ds.on_epoch_end()
        np.testing.assert_array_equal(ds._order, np.arange(N_TIME))

    def test_drop_remainder_drops_undersized_final_batch(self, era5_ds, front_da, data_config):
        """N_TIME=5 with batch_size=2 has a 1-sample remainder batch that must be dropped.

        A trailing batch smaller than batch_size splits unevenly across replicas under
        MirroredStrategy, which triggers CUDNN_STATUS_BAD_PARAM in Conv3DBackpropFilterV2
        (https://github.com/tensorflow/tensorflow/issues/60935).
        """
        batch_size = 2
        ds = self._make_ds(era5_ds, front_da, data_config, batch_size=batch_size, drop_remainder=True)
        assert len(ds) == N_TIME // batch_size
        for i in range(len(ds)):
            x_batch, y_batch = ds[i]
            assert x_batch.shape[0] == batch_size
            assert y_batch.shape[0] == batch_size

    def test_drop_remainder_false_keeps_undersized_final_batch(self, era5_ds, front_da, data_config):
        batch_size = 2
        ds = self._make_ds(era5_ds, front_da, data_config, batch_size=batch_size, drop_remainder=False)
        assert len(ds) == math.ceil(N_TIME / batch_size)
        total_samples = sum(ds[i][0].shape[0] for i in range(len(ds)))
        assert total_samples == N_TIME


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestFrontsPyDatasetPatchMode:
    """PatchConfig support: sliding longitude windows with an optional input-only buffer."""

    _N_TIME = 2
    _N_LAT_CORE = 6
    _N_LON_CORE = 12
    _BUFFER = 2
    _PATCH_WIDTH = 4
    _N_PATCHES = 3  # starts = [0, 4, 8] for a 12-wide core and a 4-wide patch

    def _make_ds(self, flip_probability=0.0, augment=False):
        n_lat_buf = self._N_LAT_CORE + 2 * self._BUFFER
        n_lon_buf = self._N_LON_CORE + 2 * self._BUFFER
        # Every pixel gets a unique value (lat_idx * 1000 + lon_idx) so mis-slicing is caught.
        buffered_vals = (np.arange(n_lat_buf)[:, None] * 1000 + np.arange(n_lon_buf)[None, :]).astype(np.float32)
        input_ds = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    np.broadcast_to(buffered_vals, (self._N_TIME, n_lat_buf, n_lon_buf)).copy(),
                    dims=["time", "latitude", "longitude"],
                    coords={"time": np.arange(self._N_TIME)},
                )
            }
        )
        core_vals = (np.arange(self._N_LAT_CORE)[:, None] + np.arange(self._N_LON_CORE)[None, :]) % 2
        target_da = xr.DataArray(
            np.broadcast_to(core_vals, (self._N_TIME, self._N_LAT_CORE, self._N_LON_CORE)).astype(np.int32).copy(),
            dims=["time", "latitude", "longitude"],
            coords={"time": np.arange(self._N_TIME)},
        )
        dummy_store = IcechunkStorageConfig(store_path="unused", branch_name="main")
        patch_config = PatchConfig(
            n_patches=self._N_PATCHES,
            patch_lon_width_px=self._PATCH_WIDTH,
            buffer_px=self._BUFFER,
            flip_probability=flip_probability,
        )
        data_config = DatasetConfig(
            inputs_icechunk_config=dummy_store,
            targets_icechunk_config=dummy_store,
            variables=["temperature"],
            test_years=[],
            val_years=[],
            patch_config=patch_config,
        )
        return FrontsPyDataset(input_ds, target_da, data_config, batch_size=1, augment=augment, seed=0)

    def test_total_samples_equals_time_times_patches(self):
        ds = self._make_ds()
        assert ds.n_samples == self._N_TIME * self._N_PATCHES

    def test_input_patch_width_includes_buffer_on_both_sides(self):
        ds = self._make_ds()
        x, _ = ds.get_at_indices(np.array([0]))
        assert x.shape == (1, self._N_LAT_CORE + 2 * self._BUFFER, self._PATCH_WIDTH + 2 * self._BUFFER, 1)

    def test_target_patch_has_no_buffer(self):
        ds = self._make_ds()
        _, y = ds.get_at_indices(np.array([0]))
        assert y.shape[1:3] == (self._N_LAT_CORE, self._PATCH_WIDTH)

    def test_input_patch_matches_manual_slice_with_buffer(self):
        ds = self._make_ds()
        starts = compute_patch_lon_starts(self._N_LON_CORE, self._PATCH_WIDTH, self._N_PATCHES)
        n_lat_buf = self._N_LAT_CORE + 2 * self._BUFFER
        for global_idx in range(ds.n_samples):
            x, _ = ds.get_at_indices(np.array([global_idx]))
            _, patch_idx = divmod(global_idx, self._N_PATCHES)
            start = starts[patch_idx]
            expected_lon = np.arange(start, start + self._PATCH_WIDTH + 2 * self._BUFFER)
            expected = (np.arange(n_lat_buf)[:, None] * 1000 + expected_lon[None, :]).astype(np.float32)
            np.testing.assert_allclose(x[0, :, :, 0], expected)

    def test_flip_probability_one_always_flips_both_axes_when_augmenting(self):
        ds_flip = self._make_ds(flip_probability=1.0, augment=True)
        ds_noflip = self._make_ds(flip_probability=0.0, augment=True)
        x_flip, y_flip = ds_flip.get_at_indices(np.array([0]))
        x_noflip, y_noflip = ds_noflip.get_at_indices(np.array([0]))
        np.testing.assert_allclose(x_flip[0], x_noflip[0][::-1, ::-1, :])
        np.testing.assert_allclose(y_flip[0], y_noflip[0][::-1, ::-1, :])

    def test_augment_false_ignores_flip_probability(self):
        ds = self._make_ds(flip_probability=1.0, augment=False)
        x, _ = ds.get_at_indices(np.array([0]))
        starts = compute_patch_lon_starts(self._N_LON_CORE, self._PATCH_WIDTH, self._N_PATCHES)
        n_lat_buf = self._N_LAT_CORE + 2 * self._BUFFER
        expected_lon = np.arange(starts[0], starts[0] + self._PATCH_WIDTH + 2 * self._BUFFER)
        expected = (np.arange(n_lat_buf)[:, None] * 1000 + expected_lon[None, :]).astype(np.float32)
        np.testing.assert_allclose(x[0, :, :, 0], expected)


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestLoadDataIntoDataloaderLongitude:
    """A wrap-crossing bounding box (lon_max > 360, e.g. configs/generate_icechunk.yaml's.

    130-369.75) leaves the longitude coordinate non-monotonic on disk, e.g.
    [330, 350, 0, 20]. xarray's pcolormesh (used by TestVisualizationCallback's
    truth-overlay plot) raises ValueError on such a coordinate, so
    load_data_into_dataloader must return data with longitude unwrapped.
    """

    _TIMES = pd.date_range("2020-01-01", periods=4, freq="6h")
    _LAT = np.array([10.0, 20.0, 30.0, 40.0])
    _LON_WRAP = np.array([330.0, 350.0, 0.0, 20.0])  # physical domain 330 -> 380

    def _write_store(self, tmp_path, name: str, var_name: str) -> IcechunkStorageConfig:
        storage_config = IcechunkStorageConfig(store_path=str(tmp_path / name), branch_name="main")
        ds = xr.Dataset(
            {
                var_name: xr.DataArray(
                    np.zeros((len(self._TIMES), len(self._LAT), len(self._LON_WRAP)), dtype=np.float32),
                    dims=["time", "latitude", "longitude"],
                    coords={"time": self._TIMES, "latitude": self._LAT, "longitude": self._LON_WRAP},
                )
            }
        )
        write_or_append_icechunk_store(storage_config, ds)
        return storage_config

    def test_longitude_is_monotonic_after_loading(self, tmp_path):
        data_config = DatasetConfig(
            inputs_icechunk_config=self._write_store(tmp_path, "inputs", "temperature"),
            targets_icechunk_config=self._write_store(tmp_path, "targets", "identifier"),
            variables=["temperature"],
            test_years=[2020],
            val_years=[],
        )
        test_dataset = load_data_into_dataloader(data_config, split="test", seed=0)
        lons = test_dataset.input_ds["longitude"].values
        assert np.all(np.diff(lons) >= 0), f"longitude not monotonic: {lons}"


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestLoadDataIntoDataloaderCoordinates:
    """data_config.coordinates must actually restrict the loaded domain.

    Regression test for a branch-divergence bug where load_data_into_dataloader silently
    ignored data_config.coordinates and always loaded the full domain regardless of the
    bounding box set in the config.
    """

    _TIMES = pd.date_range("2020-01-01", periods=4, freq="6h")
    _LAT = np.array([10.0, 20.0, 30.0, 40.0])
    _LON = np.array([100.0, 110.0, 120.0, 130.0])

    def _write_store(self, tmp_path, name: str, var_name: str) -> IcechunkStorageConfig:
        storage_config = IcechunkStorageConfig(store_path=str(tmp_path / name), branch_name="main")
        ds = xr.Dataset(
            {
                var_name: xr.DataArray(
                    np.zeros((len(self._TIMES), len(self._LAT), len(self._LON)), dtype=np.float32),
                    dims=["time", "latitude", "longitude"],
                    coords={"time": self._TIMES, "latitude": self._LAT, "longitude": self._LON},
                )
            }
        )
        write_or_append_icechunk_store(storage_config, ds)
        return storage_config

    def test_coordinates_restrict_loaded_domain(self, tmp_path):
        from fronts.utils import BoundingBox

        data_config = DatasetConfig(
            inputs_icechunk_config=self._write_store(tmp_path, "inputs", "temperature"),
            targets_icechunk_config=self._write_store(tmp_path, "targets", "identifier"),
            variables=["temperature"],
            test_years=[2020],
            val_years=[],
            coordinates=BoundingBox(lat_min=15.0, lat_max=25.0, lon_min=105.0, lon_max=115.0),
        )
        test_dataset = load_data_into_dataloader(data_config, split="test", seed=0)
        lats = test_dataset.input_ds["latitude"].values
        lons = test_dataset.input_ds["longitude"].values
        assert lats.min() >= 15.0 and lats.max() <= 25.0, f"latitude not restricted: {lats}"
        assert lons.min() >= 105.0 and lons.max() <= 115.0, f"longitude not restricted: {lons}"


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestLoadDataIntoDataloaderPressureLevels:
    """data_config.pressure_levels must restrict the loaded store to that subset of levels."""

    _TIMES = pd.date_range("2020-01-01", periods=4, freq="6h")
    _LAT = np.array([10.0, 20.0, 30.0, 40.0])
    _LON = np.array([100.0, 110.0, 120.0, 130.0])
    _LEVELS = np.array([1000, 850, 500, 300])

    def _write_store(self, tmp_path, name: str, var_name: str) -> IcechunkStorageConfig:
        storage_config = IcechunkStorageConfig(store_path=str(tmp_path / name), branch_name="main")
        ds = xr.Dataset(
            {
                var_name: xr.DataArray(
                    np.zeros(
                        (len(self._TIMES), len(self._LEVELS), len(self._LAT), len(self._LON)), dtype=np.float32
                    ),
                    dims=["time", "level", "latitude", "longitude"],
                    coords={
                        "time": self._TIMES,
                        "level": self._LEVELS,
                        "latitude": self._LAT,
                        "longitude": self._LON,
                    },
                )
            }
        )
        write_or_append_icechunk_store(storage_config, ds)
        return storage_config

    def _write_target_store(self, tmp_path, name: str, var_name: str) -> IcechunkStorageConfig:
        storage_config = IcechunkStorageConfig(store_path=str(tmp_path / name), branch_name="main")
        ds = xr.Dataset(
            {
                var_name: xr.DataArray(
                    np.zeros((len(self._TIMES), len(self._LAT), len(self._LON)), dtype=np.float32),
                    dims=["time", "latitude", "longitude"],
                    coords={"time": self._TIMES, "latitude": self._LAT, "longitude": self._LON},
                )
            }
        )
        write_or_append_icechunk_store(storage_config, ds)
        return storage_config

    def test_pressure_levels_restrict_loaded_levels(self, tmp_path):
        data_config = DatasetConfig(
            inputs_icechunk_config=self._write_store(tmp_path, "inputs", "temperature"),
            targets_icechunk_config=self._write_target_store(tmp_path, "targets", "identifier"),
            variables=["temperature"],
            test_years=[2020],
            val_years=[],
            pressure_levels=[1000, 500],
        )
        test_dataset = load_data_into_dataloader(data_config, split="test", seed=0)
        levels = test_dataset.input_ds["level"].values
        assert sorted(levels.tolist()) == [500, 1000], f"levels not restricted: {levels}"

    def test_pressure_levels_none_keeps_all_levels(self, tmp_path):
        data_config = DatasetConfig(
            inputs_icechunk_config=self._write_store(tmp_path, "inputs", "temperature"),
            targets_icechunk_config=self._write_target_store(tmp_path, "targets", "identifier"),
            variables=["temperature"],
            test_years=[2020],
            val_years=[],
        )
        test_dataset = load_data_into_dataloader(data_config, split="test", seed=0)
        levels = test_dataset.input_ds["level"].values
        assert sorted(levels.tolist()) == sorted(self._LEVELS.tolist())


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestLoadDataIntoDataloaderPatchBuffer:
    """patch_config.buffer_px must widen only inputs_ds's loaded domain, never targets_da's."""

    _TIMES = pd.date_range("2020-01-01", periods=4, freq="6h")
    _LAT = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    _LON = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0])

    def _write_store(self, tmp_path, name: str, var_name: str) -> IcechunkStorageConfig:
        storage_config = IcechunkStorageConfig(store_path=str(tmp_path / name), branch_name="main")
        ds = xr.Dataset(
            {
                var_name: xr.DataArray(
                    np.zeros((len(self._TIMES), len(self._LAT), len(self._LON)), dtype=np.float32),
                    dims=["time", "latitude", "longitude"],
                    coords={"time": self._TIMES, "latitude": self._LAT, "longitude": self._LON},
                )
            }
        )
        write_or_append_icechunk_store(storage_config, ds)
        return storage_config

    def _data_config(self, tmp_path, coordinates, buffer_px):
        return DatasetConfig(
            inputs_icechunk_config=self._write_store(tmp_path, "inputs", "temperature"),
            targets_icechunk_config=self._write_store(tmp_path, "targets", "identifier"),
            variables=["temperature"],
            test_years=[2020],
            val_years=[],
            coordinates=coordinates,
            patch_config=PatchConfig(n_patches=2, patch_lon_width_px=2, buffer_px=buffer_px)
            if buffer_px is not None
            else None,
        )

    def test_inputs_widened_by_buffer_targets_stay_core(self, tmp_path):
        from fronts.utils import BoundingBox

        data_config = self._data_config(
            tmp_path, BoundingBox(lat_min=20.0, lat_max=40.0, lon_min=120.0, lon_max=150.0), buffer_px=1
        )
        test_dataset = load_data_into_dataloader(data_config, split="test", seed=0)
        assert test_dataset.input_ds.sizes["latitude"] == 5
        assert test_dataset.input_ds.sizes["longitude"] == 6
        assert test_dataset.target_da.sizes["latitude"] == 3
        assert test_dataset.target_da.sizes["longitude"] == 4

    def test_zero_buffer_matches_unbuffered_domain(self, tmp_path):
        from fronts.utils import BoundingBox

        data_config = self._data_config(
            tmp_path, BoundingBox(lat_min=20.0, lat_max=40.0, lon_min=120.0, lon_max=150.0), buffer_px=0
        )
        test_dataset = load_data_into_dataloader(data_config, split="test", seed=0)
        assert test_dataset.input_ds.sizes["latitude"] == 3
        assert test_dataset.input_ds.sizes["longitude"] == 4

    def test_buffer_past_store_edge_raises(self, tmp_path):
        from fronts.utils import BoundingBox

        data_config = self._data_config(
            tmp_path, BoundingBox(lat_min=0.0, lat_max=40.0, lon_min=120.0, lon_max=150.0), buffer_px=1
        )
        with pytest.raises(ValueError, match="buffer_px"):
            load_data_into_dataloader(data_config, split="test", seed=0)

    def test_patch_config_without_coordinates_raises(self, tmp_path):
        data_config = DatasetConfig(
            inputs_icechunk_config=self._write_store(tmp_path, "inputs", "temperature"),
            targets_icechunk_config=self._write_store(tmp_path, "targets", "identifier"),
            variables=["temperature"],
            test_years=[2020],
            val_years=[],
            coordinates=None,
            patch_config=PatchConfig(n_patches=2, patch_lon_width_px=2, buffer_px=0),
        )
        with pytest.raises(ValueError, match="coordinates"):
            load_data_into_dataloader(data_config, split="test", seed=0)

    def test_augment_flag_threaded_to_dataset(self, tmp_path):
        from fronts.utils import BoundingBox

        data_config = self._data_config(
            tmp_path, BoundingBox(lat_min=20.0, lat_max=40.0, lon_min=120.0, lon_max=150.0), buffer_px=0
        )
        test_dataset = load_data_into_dataloader(data_config, split="test", seed=0, augment=True)
        assert test_dataset.augment is True


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestBuildLoss:
    _LATITUDES = np.linspace(25.0, 56.75, 8)

    def test_fss_returns_callable(self):
        loss_fn = _build_loss(
            loss_name="fractions_skill_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
        )
        assert callable(loss_fn)

    def test_neighborhood_brier_score_returns_callable(self):
        loss_fn = _build_loss(
            loss_name="neighborhood_brier_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
        )
        assert callable(loss_fn)

    def test_unrecognized_loss_name_raises(self):
        with pytest.raises(ValueError, match="Unrecognized loss_name"):
            _build_loss(
                loss_name="bogus",  # type: ignore[arg-type]
                loss_class_weights=None,
                latitudes=self._LATITUDES,
                fss_mask_size=(3, 3),
                nbs_tolerance_km=25.0,
                nbs_periodic_lon=False,
                nbs_lat_dependent_pool=False,
            )

    def test_fss_and_nbs_produce_different_losses_on_same_inputs(self):
        rng = np.random.default_rng(0)
        n_classes = 3
        y_true = tf.one_hot(rng.integers(0, n_classes, size=(2, 8, 8)), n_classes)
        y_pred = tf.nn.softmax(rng.standard_normal((2, 8, 8, n_classes)).astype(np.float32), axis=-1)

        fss_loss = _build_loss(
            loss_name="fractions_skill_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
        )
        nbs_loss = _build_loss(
            loss_name="neighborhood_brier_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
        )
        fss_value = float(tf.reduce_mean(fss_loss(y_true, y_pred)))
        nbs_value = float(tf.reduce_mean(nbs_loss(y_true, y_pred)))
        assert np.isfinite(fss_value)
        assert np.isfinite(nbs_value)

    def test_nbs_include_pixel_defaults_to_off(self):
        """nbs_include_pixel/nbs_pixel_weight are optional — omitting them must not change the loss."""
        rng = np.random.default_rng(1)
        n_classes = 3
        y_true = tf.one_hot(rng.integers(0, n_classes, size=(2, 8, 8)), n_classes)
        y_pred = tf.nn.softmax(rng.standard_normal((2, 8, 8, n_classes)).astype(np.float32), axis=-1)

        default_loss = _build_loss(
            loss_name="neighborhood_brier_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
        )
        explicit_off_loss = _build_loss(
            loss_name="neighborhood_brier_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
            nbs_include_pixel=False,
        )
        default_value = float(tf.reduce_mean(default_loss(y_true, y_pred)))
        explicit_off_value = float(tf.reduce_mean(explicit_off_loss(y_true, y_pred)))
        assert default_value == pytest.approx(explicit_off_value)

    def test_nbs_include_pixel_true_changes_loss(self):
        """Turning on nbs_include_pixel must add the un-pooled pixelwise term."""
        rng = np.random.default_rng(2)
        n_classes = 3
        y_true = tf.one_hot(rng.integers(0, n_classes, size=(2, 8, 8)), n_classes)
        y_pred = tf.nn.softmax(rng.standard_normal((2, 8, 8, n_classes)).astype(np.float32), axis=-1)

        pooled_only_loss = _build_loss(
            loss_name="neighborhood_brier_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
            nbs_include_pixel=False,
        )
        with_pixel_loss = _build_loss(
            loss_name="neighborhood_brier_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
            nbs_include_pixel=True,
            nbs_pixel_weight=0.5,
        )
        pooled_only_value = float(tf.reduce_mean(pooled_only_loss(y_true, y_pred)))
        with_pixel_value = float(tf.reduce_mean(with_pixel_loss(y_true, y_pred)))
        assert np.isfinite(with_pixel_value)
        assert with_pixel_value != pytest.approx(pooled_only_value)

    def test_neighborhood_brier_threads_pred_buffer_px(self):
        loss_fn = _build_loss(
            loss_name="neighborhood_brier_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
            nbs_pred_buffer_px=2,
        )
        y_true = np.zeros((1, 8, 8, 6), dtype=np.float32)
        y_true[..., 0] = 1.0
        y_pred = np.zeros((1, 12, 12, 6), dtype=np.float32)  # buffered by 2 on each side
        y_pred[..., 0] = 1.0
        result = loss_fn(y_true, y_pred).numpy()
        assert np.all(np.isfinite(result))


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestPredBufferPxFromDataConfig:
    def test_no_patch_config_returns_zero(self, data_config):
        assert _pred_buffer_px_from_data_config(data_config) == 0

    def test_patch_config_returns_its_buffer_px(self, data_config):
        import dataclasses as dc

        cfg = dc.replace(data_config, patch_config=PatchConfig(n_patches=9, patch_lon_width_px=128, buffer_px=16))
        assert _pred_buffer_px_from_data_config(cfg) == 16


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestTargetLatitudes:
    def test_returns_target_da_latitudes_not_input_ds(self, data_config):
        n_time, n_lat_core, n_lon = 2, 3, 4
        input_ds = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    np.zeros((n_time, n_lat_core + 2, n_lon), dtype=np.float32),
                    dims=["time", "latitude", "longitude"],
                    coords={"time": np.arange(n_time), "latitude": np.arange(n_lat_core + 2) + 100.0},
                )
            }
        )
        target_da = xr.DataArray(
            np.zeros((n_time, n_lat_core, n_lon), dtype=np.int32),
            dims=["time", "latitude", "longitude"],
            coords={"time": np.arange(n_time), "latitude": np.arange(n_lat_core) + 5.0},
        )
        ds = FrontsPyDataset(input_ds, target_da, data_config, batch_size=1)
        np.testing.assert_array_equal(_target_latitudes(ds), np.arange(n_lat_core) + 5.0)


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestShouldBuildTestVisualization:
    def test_true_when_wandb_and_cadence_set(self):
        assert _should_build_test_visualization("fronts", 1) is True

    def test_false_without_wandb_project(self):
        assert _should_build_test_visualization(None, 1) is False

    def test_false_without_cadence(self):
        assert _should_build_test_visualization("fronts", None) is False


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestBuildTestVisualizationCallback:
    """_build_test_visualization_callback must ignore data_config.patch_config.

    The callback's active-day map and per-office-region performance diagrams assume
    one whole-domain input/target pair per sample. Patch-mode training only changes how
    *training* samples are drawn; the model's input shape stays fully dynamic
    (Input(shape=(None, None, ...))), so whole-domain inference works regardless of
    patch_config, and the visualization dataset should always be loaded that way.
    """

    _TIMES = pd.date_range("2020-01-01", periods=3, freq="6h")
    _LAT_CORE = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    _LON_CORE = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0])
    _LAT_BUFFERED = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    _LON_BUFFERED = np.array([90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0])

    def _write_inputs(self, tmp_path) -> IcechunkStorageConfig:
        storage_config = IcechunkStorageConfig(store_path=str(tmp_path / "inputs"), branch_name="main")
        ds = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    np.zeros((len(self._TIMES), len(self._LAT_BUFFERED), len(self._LON_BUFFERED)), dtype=np.float32),
                    dims=["time", "latitude", "longitude"],
                    coords={"time": self._TIMES, "latitude": self._LAT_BUFFERED, "longitude": self._LON_BUFFERED},
                )
            }
        )
        write_or_append_icechunk_store(storage_config, ds)
        return storage_config

    def _write_targets(self, tmp_path) -> IcechunkStorageConfig:
        storage_config = IcechunkStorageConfig(store_path=str(tmp_path / "targets"), branch_name="main")
        data = np.zeros((len(self._TIMES), len(self._LAT_CORE), len(self._LON_CORE)), dtype=np.int32)
        data[0, 0, 0] = _ALL_CODES[0]
        ds = xr.Dataset(
            {
                "identifier": xr.DataArray(
                    data,
                    dims=["time", "latitude", "longitude"],
                    coords={"time": self._TIMES, "latitude": self._LAT_CORE, "longitude": self._LON_CORE},
                )
            }
        )
        write_or_append_icechunk_store(storage_config, ds)
        return storage_config

    def _data_config(
        self,
        inputs_store: IcechunkStorageConfig,
        targets_store: IcechunkStorageConfig,
        patch_config: PatchConfig | None,
    ) -> DatasetConfig:
        from fronts.utils import BoundingBox

        return DatasetConfig(
            inputs_icechunk_config=inputs_store,
            targets_icechunk_config=targets_store,
            variables=["temperature"],
            test_years=[2020],
            val_years=[],
            coordinates=BoundingBox(lat_min=10.0, lat_max=60.0, lon_min=100.0, lon_max=170.0),
            patch_config=patch_config,
        )

    def _callbacks_config(self) -> CallbacksConfig:
        return CallbacksConfig(test_viz_every_n_epochs=1, test_viz_sample_size=2)

    def test_whole_domain_shape_with_patch_config_set(self, tmp_path):
        data_config = self._data_config(
            self._write_inputs(tmp_path),
            self._write_targets(tmp_path),
            patch_config=PatchConfig(n_patches=2, patch_lon_width_px=4, buffer_px=1),
        )

        callback = _build_test_visualization_callback(data_config, self._callbacks_config(), seed=0)

        assert callback.active_day_x.shape == (len(self._LAT_CORE), len(self._LON_CORE), 1)
        assert callback.active_day_y.shape[:2] == (len(self._LAT_CORE), len(self._LON_CORE))
        assert callback.subsample_x.shape[1:3] == (len(self._LAT_CORE), len(self._LON_CORE))
        assert callback.subsample_y.shape[1:3] == (len(self._LAT_CORE), len(self._LON_CORE))
        np.testing.assert_array_equal(callback.lats, self._LAT_CORE)
        np.testing.assert_array_equal(callback.lons, self._LON_CORE)

    def test_output_identical_with_and_without_patch_config(self, tmp_path):
        inputs_store = self._write_inputs(tmp_path)
        targets_store = self._write_targets(tmp_path)
        callbacks_config = self._callbacks_config()

        no_patch_config = self._data_config(inputs_store, targets_store, patch_config=None)
        callback_no_patch = _build_test_visualization_callback(no_patch_config, callbacks_config, seed=0)

        with_patch_config = self._data_config(
            inputs_store, targets_store, patch_config=PatchConfig(n_patches=2, patch_lon_width_px=4, buffer_px=1)
        )
        callback_with_patch = _build_test_visualization_callback(with_patch_config, callbacks_config, seed=0)

        np.testing.assert_allclose(callback_with_patch.active_day_x, callback_no_patch.active_day_x)
        np.testing.assert_allclose(callback_with_patch.active_day_y, callback_no_patch.active_day_y)
        np.testing.assert_allclose(callback_with_patch.subsample_x, callback_no_patch.subsample_x)
        np.testing.assert_allclose(callback_with_patch.subsample_y, callback_no_patch.subsample_y)


class TestTrainConfigLossClassWeights:
    @pytest.fixture
    def train_config_cls(self):
        return pytest.importorskip("fronts.train").TrainConfig

    def test_null_yaml_value_parses_to_none(self, train_config_cls):
        from fronts import utils

        yaml_data = {"train_config": {"loss_class_weights": None, "epochs": 1}}
        cfg = utils.parse_config_section(yaml_data, train_config_cls, "train_config")
        assert cfg.loss_class_weights is None

    def test_explicit_weights_parse_to_list(self, train_config_cls):
        from fronts import utils

        weights = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        yaml_data = {"train_config": {"loss_class_weights": weights, "epochs": 1}}
        cfg = utils.parse_config_section(yaml_data, train_config_cls, "train_config")
        assert cfg.loss_class_weights == weights

    def test_nbs_include_pixel_defaults(self, train_config_cls):
        from fronts import utils

        yaml_data = {"train_config": {"loss_class_weights": None, "epochs": 1}}
        cfg = utils.parse_config_section(yaml_data, train_config_cls, "train_config")
        assert cfg.nbs_include_pixel is False
        assert cfg.nbs_pixel_weight == 0.1

    def test_sooner_ablations_config_parses_nbs_pixel_fields(self, train_config_cls):
        from fronts import utils

        yaml_data = utils.load_yaml("configs/sooner_ablations.yaml")
        cfg = utils.parse_config_section(yaml_data, train_config_cls, "train_config")
        assert cfg.nbs_include_pixel is True
        assert cfg.nbs_pixel_weight == 0.5

    def test_schooner_configs_parse(self, train_config_cls):
        from fronts import utils

        for path in [
            "configs/schooner_train.yaml",
            "configs/schooner_pipeline.yaml",
            "configs/schooner_train_3d.yaml",
        ]:
            yaml_data = utils.load_yaml(path)
            cfg = utils.parse_config_section(yaml_data, train_config_cls, "train_config")
            assert cfg.loss_class_weights is None
            assert cfg.loss_name == "neighborhood_brier_score"
            assert cfg.nbs_tolerance_km == 25.0

    def test_3d_config_parses(self):
        from fronts import utils
        from fronts.callbacks import CallbacksConfig
        from fronts.data.datasets import DatasetConfig
        from fronts.model import ModelConfig
        from fronts.train import TrainConfig

        yaml_data = utils.load_yaml("configs/schooner_train_3d.yaml")
        data_cfg = utils.parse_config_section(yaml_data, DatasetConfig, "data_config")
        model_cfg = utils.parse_config_section(yaml_data, ModelConfig, "model_config")
        train_cfg = utils.parse_config_section(yaml_data, TrainConfig, "train_config")
        callbacks_cfg = utils.parse_config_section(yaml_data, CallbacksConfig, "callbacks_config")

        assert data_cfg.volume_inputs is True
        assert len(data_cfg.variables) == 10
        assert model_cfg.squeeze_axes == 3
        assert list(model_cfg.pool_size) == [2, 2, 1]
        assert list(model_cfg.upsample_size) == [2, 2, 1]
        assert model_cfg.kernel_size == 5
        assert train_cfg.learning_rate == 1e-4
        assert train_cfg.gradient_clip_norm == 1.0
        assert callbacks_cfg.min_delta == 0.0
        assert callbacks_cfg.early_stopping_patience == 12

    def test_patch_buffer_ablation_config_parses(self, train_config_cls):
        from fronts import utils

        yaml_data = utils.load_yaml("configs/patch_buffer_ablation.yaml")
        data_cfg = utils.parse_config_section(yaml_data, DatasetConfig, "data_config", type_hooks=utils.YAML_TYPE_HOOKS)
        model_cfg = utils.parse_config_section(yaml_data, ModelConfig, "model_config")
        train_cfg = utils.parse_config_section(yaml_data, train_config_cls, "train_config")
        callbacks_cfg = utils.parse_config_section(yaml_data, CallbacksConfig, "callbacks_config")

        assert data_cfg.patch_config == PatchConfig(
            n_patches=9, patch_lon_width_px=128, buffer_px=16, flip_probability=0.25
        )
        assert data_cfg.coordinates == utils.BoundingBox(lat_min=25.0, lat_max=56.75, lon_min=228.0, lon_max=299.75)
        assert data_cfg.volume_inputs is True
        assert list(model_cfg.pool_size) == [2, 2, 1]
        assert callbacks_cfg.test_viz_every_n_epochs == 1
        assert train_cfg.loss_name == "neighborhood_brier_score"


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestFrontsPyDatasetVolume:
    """volume_inputs=True must yield (batch, lat, lon, level, variable) batches for a 3D model."""

    _N_TIME = 4
    _N_LAT = 6
    _N_LON = 8
    _LEVELS = (1000, 950)

    def _make_volume_inputs(self):
        rng = np.random.default_rng(11)
        times = np.arange(self._N_TIME)
        input_ds = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    rng.standard_normal((self._N_TIME, len(self._LEVELS), self._N_LAT, self._N_LON)).astype(np.float32),
                    dims=["time", "level", "latitude", "longitude"],
                    coords={"time": times, "level": list(self._LEVELS)},
                ),
                "mean_sea_level_pressure": xr.DataArray(
                    rng.standard_normal((self._N_TIME, self._N_LAT, self._N_LON)).astype(np.float32),
                    dims=["time", "latitude", "longitude"],
                    coords={"time": times},
                ),
            }
        )
        target_da = xr.DataArray(
            rng.integers(0, 2, size=(self._N_TIME, self._N_LAT, self._N_LON)).astype(np.int32),
            dims=["time", "latitude", "longitude"],
            coords={"time": times},
        )
        dummy_store = IcechunkStorageConfig(store_path="unused", branch_name="main")
        config = DatasetConfig(
            inputs_icechunk_config=dummy_store,
            targets_icechunk_config=dummy_store,
            variables=["temperature", "mean_sea_level_pressure"],
            test_years=[],
            val_years=[],
            volume_inputs=True,
        )
        return input_ds, target_da, config

    def test_batch_is_5d(self):
        input_ds, target_da, config = self._make_volume_inputs()
        ds = FrontsPyDataset(input_ds, target_da, config, batch_size=2)
        x_batch, y_batch = ds[0]
        assert x_batch.shape == (2, self._N_LAT, self._N_LON, len(self._LEVELS), 2)
        assert y_batch.shape == (2, self._N_LAT, self._N_LON, N_CLASSES)

    def test_single_level_variable_broadcast_in_batch(self):
        input_ds, target_da, config = self._make_volume_inputs()
        ds = FrontsPyDataset(input_ds, target_da, config, batch_size=2)
        x_batch, _ = ds[0]
        np.testing.assert_array_equal(x_batch[..., 0, 1], x_batch[..., 1, 1])


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestPatchConfigValidation:
    def test_n_patches_must_be_positive(self):
        with pytest.raises(ValueError, match="n_patches"):
            PatchConfig(n_patches=0, patch_lon_width_px=4)

    def test_patch_lon_width_px_must_be_positive(self):
        with pytest.raises(ValueError, match="patch_lon_width_px"):
            PatchConfig(n_patches=1, patch_lon_width_px=0)

    def test_buffer_px_must_be_non_negative(self):
        with pytest.raises(ValueError, match="buffer_px"):
            PatchConfig(n_patches=1, patch_lon_width_px=4, buffer_px=-1)

    def test_flip_probability_must_be_in_unit_interval(self):
        with pytest.raises(ValueError, match="flip_probability"):
            PatchConfig(n_patches=1, patch_lon_width_px=4, flip_probability=1.5)

    def test_defaults(self):
        cfg = PatchConfig(n_patches=9, patch_lon_width_px=128)
        assert cfg.buffer_px == 0
        assert cfg.flip_probability == 0.0


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestComputePatchLonStarts:
    def test_nine_patches_evenly_spaced_across_288(self):
        starts = compute_patch_lon_starts(n_lon_core=288, patch_width=128, n_patches=9)
        np.testing.assert_array_equal(starts, [0, 20, 40, 60, 80, 100, 120, 140, 160])

    def test_single_patch_starts_at_zero(self):
        starts = compute_patch_lon_starts(n_lon_core=20, patch_width=8, n_patches=1)
        np.testing.assert_array_equal(starts, [0])

    def test_patch_wider_than_core_raises(self):
        with pytest.raises(ValueError, match="exceeds"):
            compute_patch_lon_starts(n_lon_core=10, patch_width=12, n_patches=2)

    def test_last_patch_ends_exactly_at_core_width(self):
        starts = compute_patch_lon_starts(n_lon_core=12, patch_width=4, n_patches=3)
        assert starts[-1] + 4 == 12


def _build_small_unet(
    levels: int = 3,
    deep_supervision: bool = False,
    normalization_stat_a: np.ndarray | None = None,
    normalization_stat_b: np.ndarray | None = None,
) -> "tf.keras.Model":
    filter_num = [8, 16, 32, 64][:levels]
    return UNet3Plus(
        input_shape=(None, None, 4),
        num_classes=6,
        pool_size=(2, 2),
        upsample_size=(2, 2),
        levels=levels,
        filter_num=filter_num,
        deep_supervision=deep_supervision,
        output_activation="softmax",
        normalization_method="minmax",
        normalization_stat_a=normalization_stat_a,
        normalization_stat_b=normalization_stat_b,
    ).build()


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestPatchBufferEndToEnd:
    """Buffered patches must compose through the real model into the buffer-aware loss."""

    def test_buffered_patch_input_scores_against_unbuffered_core_target(self):
        core = 16
        buffer_px = 4
        buffered = core + 2 * buffer_px  # 24; still divisible by the 2-stage (levels=3) stride of 4
        model = _build_small_unet(levels=3, deep_supervision=False)  # input_shape=(None, None, 4)

        rng = np.random.default_rng(3)
        x = rng.standard_normal((2, buffered, buffered, 4)).astype(np.float32)
        y_true = tf.one_hot(rng.integers(0, 6, size=(2, core, core)), 6).numpy().astype(np.float32)

        y_pred = model(x, training=False)
        if isinstance(y_pred, list | tuple):
            y_pred = y_pred[0]
        assert y_pred.shape == (2, buffered, buffered, 6)

        loss_fn = losses.neighborhood_brier_score(
            latitudes=np.linspace(25.0, 30.0, core), tolerance_km=25.0, pred_buffer_px=buffer_px
        )
        result = loss_fn(y_true, y_pred).numpy()
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_compiled_model_trains_on_buffered_patch_batch(self):
        """_compile must thread pred_buffer_px into the HSS metric, not just the loss.

        Reproduces the crash from a real patch-buffer training run: model.fit failing
        inside compute_metrics because the buffered (wider) y_pred and unbuffered y_true
        reached heidke_skill_score with mismatched shapes.
        """
        core = 16
        buffer_px = 4
        buffered = core + 2 * buffer_px
        model = _build_small_unet(levels=3, deep_supervision=False)

        rng = np.random.default_rng(3)
        x = rng.standard_normal((2, buffered, buffered, 4)).astype(np.float32)
        y_true = tf.one_hot(rng.integers(0, 6, size=(2, core, core)), 6).numpy().astype(np.float32)

        train_cfg = TrainConfig(
            loss_class_weights=None,
            loss_name="neighborhood_brier_score",
            nbs_tolerance_km=25.0,
        )
        _compile(
            model=model,
            learning_rate=1e-4,
            metric_class_weights=None,
            train_cfg=train_cfg,
            latitudes=np.linspace(25.0, 30.0, core),
            pred_buffer_px=buffer_px,
        )

        result = model.train_on_batch(x, y_true, return_dict=True)
        assert np.isfinite(result["hss"])


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestLoadPretrainedWeights:
    def test_encoder_decoder_weights_transferred(self, tmp_path):
        min_val = np.zeros(4, dtype=np.float32)
        max_val = np.ones(4, dtype=np.float32)
        pretrained = _build_small_unet(normalization_stat_a=min_val, normalization_stat_b=max_val)
        checkpoint_path = str(tmp_path / "pretrained.keras")
        pretrained.save(checkpoint_path)

        fresh = _build_small_unet(normalization_stat_a=min_val, normalization_stat_b=max_val)
        pretrained_kernel = pretrained.get_layer("En1_Conv2D_1").get_weights()[0]
        fresh_kernel_before = fresh.get_layer("En1_Conv2D_1").get_weights()[0]
        assert not np.allclose(pretrained_kernel, fresh_kernel_before)

        _load_pretrained_weights(fresh, checkpoint_path, min_val, max_val)

        fresh_kernel_after = fresh.get_layer("En1_Conv2D_1").get_weights()[0]
        np.testing.assert_allclose(fresh_kernel_after, pretrained_kernel)

    def test_normalization_reset_to_new_stats_not_checkpoint_stats(self, tmp_path):
        checkpoint_min = np.array([0.0, -10.0, 100.0, 0.0], dtype=np.float32)
        checkpoint_max = np.array([1.0, 10.0, 200.0, 1.0], dtype=np.float32)
        pretrained = _build_small_unet(normalization_stat_a=checkpoint_min, normalization_stat_b=checkpoint_max)
        checkpoint_path = str(tmp_path / "pretrained.keras")
        pretrained.save(checkpoint_path)

        full_domain_min = np.array([-50.0, -80.0, 0.0, 0.0], dtype=np.float32)
        full_domain_max = np.array([50.0, 80.0, 1000.0, 1.0], dtype=np.float32)
        fresh = _build_small_unet(normalization_stat_a=full_domain_min, normalization_stat_b=full_domain_max)

        _load_pretrained_weights(fresh, checkpoint_path, full_domain_min, full_domain_max)

        norm_layer = fresh.get_layer("input_normalization")
        expected_scale = 1.0 / (full_domain_max - full_domain_min)
        expected_offset = -full_domain_min * expected_scale
        np.testing.assert_allclose(norm_layer.scale, expected_scale, atol=1e-5)
        np.testing.assert_allclose(norm_layer.offset, expected_offset, atol=1e-5)

    def test_mismatched_supervision_head_shape_skipped_not_fatal(self, tmp_path):
        min_val = np.zeros(4, dtype=np.float32)
        max_val = np.ones(4, dtype=np.float32)
        pretrained = _build_small_unet(
            levels=3, deep_supervision=True, normalization_stat_a=min_val, normalization_stat_b=max_val
        )
        checkpoint_path = str(tmp_path / "pretrained.keras")
        pretrained.save(checkpoint_path)

        fresh = _build_small_unet(
            levels=4, deep_supervision=True, normalization_stat_a=min_val, normalization_stat_b=max_val
        )
        pretrained_kernel = pretrained.get_layer("En1_Conv2D_1").get_weights()[0]

        _load_pretrained_weights(fresh, checkpoint_path, min_val, max_val)

        fresh_kernel_after = fresh.get_layer("En1_Conv2D_1").get_weights()[0]
        np.testing.assert_allclose(fresh_kernel_after, pretrained_kernel)


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestFreezeLayers:
    def test_prefix_match_freezes_expected_layers(self):
        unet = _build_small_unet(levels=3, deep_supervision=True)
        _freeze_layers(unet, ["En"])
        for layer in unet.layers:
            if layer.name.startswith("En"):
                assert layer.trainable is False
            elif layer.name.startswith("De") or layer.name.startswith("sup"):
                assert layer.trainable is True

    def test_no_prefixes_frozen_when_prefix_absent(self):
        unet = _build_small_unet(levels=3, deep_supervision=True)
        _freeze_layers(unet, ["NonexistentPrefix"])
        assert all(layer.trainable for layer in unet.layers)


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestModelConfigFreezeValidation:
    def test_freeze_prefixes_without_pretrained_path_raises(self):
        with pytest.raises(ValueError):
            ModelConfig(freeze_layer_prefixes=["En"], pretrained_weights_path=None)


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestCompileEma:
    _LATITUDES = np.linspace(25.0, 56.75, 8)

    def test_use_ema_defaults_to_false(self):
        unet = _build_small_unet()
        train_cfg = TrainConfig(loss_class_weights=None)
        _compile(unet, learning_rate=1e-4, metric_class_weights=None, train_cfg=train_cfg, latitudes=self._LATITUDES)
        assert unet.optimizer.use_ema is False

    def test_use_ema_true_sets_optimizer_flag_and_momentum(self):
        unet = _build_small_unet()
        train_cfg = TrainConfig(loss_class_weights=None, use_ema=True, ema_momentum=0.95)
        _compile(unet, learning_rate=1e-4, metric_class_weights=None, train_cfg=train_cfg, latitudes=self._LATITUDES)
        assert unet.optimizer.use_ema is True
        assert unet.optimizer.ema_momentum == pytest.approx(0.95)

    def test_use_ema_false_leaves_default_ema_momentum_inert(self):
        """ema_momentum must not affect a compiled optimizer when use_ema is False."""
        unet = _build_small_unet()
        train_cfg = TrainConfig(loss_class_weights=None, use_ema=False, ema_momentum=0.5)
        _compile(unet, learning_rate=1e-4, metric_class_weights=None, train_cfg=train_cfg, latitudes=self._LATITUDES)
        assert unet.optimizer.use_ema is False
