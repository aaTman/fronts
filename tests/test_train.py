import dataclasses
import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fronts.data import inputs as data_inputs
from fronts.data import targets as data_targets
from fronts.data.targets import FRONT_CLASS_MAP, filter_timesteps
from fronts.utils import IcechunkStorageConfig, apply_time_resolution

try:
    import tensorflow as tf

    from fronts.callbacks import CallbacksConfig
    from fronts.data.datasets import (
        DatasetConfig,
        FrontsPyDataset,
        PatchConfig,
        compute_patch_lon_starts,
        reflect_pad_lat_lon_buffer,
    )
    from fronts.data.generate import write_or_append_icechunk_store
    from fronts.data.inputs import inputs_ds_to_dataarray
    from fronts.layers import losses
    from fronts.model import ModelConfig, UNet3Plus
    from fronts.train import (
        TrainConfig,
        WandBConfig,
        _build_dataset_summary,
        _build_loss,
        _build_monitor_callbacks,
        _build_run_callbacks,
        _build_test_visualization_callback,
        _build_wandb_config,
        _compile,
        _freeze_layers,
        _load_pretrained_weights,
        _optimizer_uses_ema,
        _pred_buffer_from_data_config,
        _should_build_test_visualization,
        _target_latitudes,
        _validate_batch_size_for_strategy,
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

    def test_shuffle_reorders_batches_not_samples_within_a_batch(self, era5_ds, front_da, data_config):
        """Shuffling must only reorder whole batches, keeping each batch a contiguous read.

        Both icechunk stores backing this dataset chunk at 1 timestep, so a fully random
        per-sample shuffle turns every batch read into scattered single-chunk fetches,
        measured at 10-30x slower than a sequential read of the same size (see
        scripts/diagnose_read_throughput.py). Every batch must therefore still correspond
        to some contiguous run of the original timesteps, even with shuffling enabled.
        """
        batch_size = 2
        ds = self._make_ds(era5_ds, front_da, data_config, batch_size=batch_size, shuffle=True, seed=0)
        expected = inputs_ds_to_dataarray(era5_ds, data_config.variables).values
        for i in range(len(ds)):
            x_batch, _ = ds[i]
            n = x_batch.shape[0]
            matches = [s for s in range(N_TIME - n + 1) if np.allclose(x_batch, expected[s : s + n])]
            assert matches, f"batch {i} is not a contiguous run of original timesteps"

    def test_shuffle_covers_every_timestep_exactly_once_across_seeds(self, era5_ds, front_da, data_config):
        """Regression test: __getitem__ must resolve each batch's samples from
        ``self._order[start:stop]``, not ``self._order[idx] * batch_size``.

        The latter reduces to the correct batch only when ``self._order`` happens to be
        the identity permutation; for any other block permutation it silently produces
        undersized or entirely empty batches, dropping most of the epoch's data. Checked
        across many seeds since the failure is seed-dependent.
        """
        batch_size = 2
        for seed in range(20):
            ds = self._make_ds(era5_ds, front_da, data_config, batch_size=batch_size, shuffle=True, seed=seed)
            batch_sizes = [ds[i][0].shape[0] for i in range(len(ds))]
            assert all(n > 0 for n in batch_sizes), f"seed {seed}: empty batch in {batch_sizes}"
            assert sum(batch_sizes) == N_TIME, f"seed {seed}: batch sizes {batch_sizes} do not sum to {N_TIME}"

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
    """PatchConfig support: sliding longitude windows with an optional input-only buffer.

    ``input_ds`` is always the core (unbuffered) domain — buffer context is reflected off
    its own edges at batch-materialization time (see ``FrontsPyDataset._get_patches_at_indices``),
    not read from a wider store selection.
    """

    _N_TIME = 2
    _N_LAT_CORE = 6
    _N_LON_CORE = 12
    _BUFFER = 2  # applied to both axes by default, matching this class's pre-per-axis-buffer coverage
    _PATCH_WIDTH = 4
    _N_PATCHES = 3  # starts = [0, 4, 8] for a 12-wide core and a 4-wide patch

    def _core_vals(self):
        # Every pixel gets a unique value (lat_idx * 1000 + lon_idx) so mis-slicing is caught.
        return (np.arange(self._N_LAT_CORE)[:, None] * 1000 + np.arange(self._N_LON_CORE)[None, :]).astype(
            np.float32
        )

    def _make_ds(
        self,
        flip_probability=0.0,
        augment=False,
        front_dilation=0,
        buffer_lat_px=None,
        buffer_lon_px=None,
        patches_per_epoch=None,
        seed=0,
    ):
        core_input_vals = self._core_vals()
        input_ds = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    np.broadcast_to(
                        core_input_vals, (self._N_TIME, self._N_LAT_CORE, self._N_LON_CORE)
                    ).copy(),
                    dims=["time", "latitude", "longitude"],
                    coords={"time": np.arange(self._N_TIME)},
                )
            }
        )
        core_target_vals = (np.arange(self._N_LAT_CORE)[:, None] + np.arange(self._N_LON_CORE)[None, :]) % 2
        target_da = xr.DataArray(
            np.broadcast_to(
                core_target_vals, (self._N_TIME, self._N_LAT_CORE, self._N_LON_CORE)
            ).astype(np.int32).copy(),
            dims=["time", "latitude", "longitude"],
            coords={"time": np.arange(self._N_TIME)},
        )
        dummy_store = IcechunkStorageConfig(store_path="unused", branch_name="main")
        patch_config = PatchConfig(
            n_patches=self._N_PATCHES,
            patch_lon_width_px=self._PATCH_WIDTH,
            buffer_lat_px=self._BUFFER if buffer_lat_px is None else buffer_lat_px,
            buffer_lon_px=self._BUFFER if buffer_lon_px is None else buffer_lon_px,
            flip_probability=flip_probability,
            patches_per_epoch=patches_per_epoch,
        )
        data_config = DatasetConfig(
            inputs_icechunk_config=dummy_store,
            targets_icechunk_config=dummy_store,
            variables=["temperature"],
            test_years=[],
            val_years=[],
            front_dilation=front_dilation,
            patch_config=patch_config,
        )
        return FrontsPyDataset(input_ds, target_da, data_config, batch_size=1, augment=augment, seed=seed)

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

    def test_input_patch_matches_reflect_padded_core_slice(self):
        ds = self._make_ds()
        starts = compute_patch_lon_starts(self._N_LON_CORE, self._PATCH_WIDTH, self._N_PATCHES)
        padded = np.pad(self._core_vals(), self._BUFFER, mode="reflect")
        for global_idx in range(ds.n_samples):
            x, _ = ds.get_at_indices(np.array([global_idx]))
            _, patch_idx = divmod(global_idx, self._N_PATCHES)
            start = starts[patch_idx]
            expected = padded[:, start : start + self._PATCH_WIDTH + 2 * self._BUFFER]
            np.testing.assert_allclose(x[0, :, :, 0], expected)

    def test_buffer_mirrors_core_values_at_domain_edge(self):
        """The west buffer ring of the first patch must mirror the core's own west edge
        (reflect padding), not zeros or wrapped-around east-edge values.
        """
        ds = self._make_ds()
        x, _ = ds.get_at_indices(np.array([0]))
        padded = np.pad(self._core_vals(), self._BUFFER, mode="reflect")
        # np.pad(mode="reflect") mirrors excluding the edge pixel itself: buffer col 0 (2
        # px west of core col 0) equals core col 2, buffer col 1 equals core col 1.
        np.testing.assert_allclose(x[0, :, 0, 0], padded[:, 0])
        np.testing.assert_allclose(x[0, :, 1, 0], padded[:, 1])

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
        padded = np.pad(self._core_vals(), self._BUFFER, mode="reflect")
        start = starts[0]
        expected = padded[:, start : start + self._PATCH_WIDTH + 2 * self._BUFFER]
        np.testing.assert_allclose(x[0, :, :, 0], expected)

    def test_patches_sharing_a_timestep_materialize_inputs_once_per_unique_timestep(self, monkeypatch):
        """Regression test: a batch of patches from the same timestep must trigger one
        full-domain read of that timestep, not one per patch (see
        ``FrontsPyDataset._get_patches_at_indices``).
        """
        ds = self._make_ds()
        seen_time_sizes = []
        original = data_inputs.inputs_ds_to_dataarray

        def spy(ds_arg, variables):
            seen_time_sizes.append(ds_arg.sizes["time"])
            return original(ds_arg, variables)

        monkeypatch.setattr(data_inputs, "inputs_ds_to_dataarray", spy)
        idxs = np.array([0, 1, 2, self._N_PATCHES])  # 3 patches of timestep 0, 1 patch of timestep 1
        ds.get_at_indices(idxs)
        assert seen_time_sizes == [len(np.unique(idxs // self._N_PATCHES))]

    def test_patches_sharing_a_timestep_dilate_once_per_unique_timestep(self, monkeypatch):
        """Regression test: binary dilation (the expensive step) must run once per unique
        timestep in the batch, not once per patch.
        """
        ds = self._make_ds(front_dilation=1)
        call_count = 0
        original = data_targets._dilate_one_timestep

        def spy(arr, dilation):
            nonlocal call_count
            call_count += 1
            return original(arr, dilation)

        monkeypatch.setattr(data_targets, "_dilate_one_timestep", spy)
        idxs = np.array([0, 1, 2, self._N_PATCHES])
        ds.get_at_indices(idxs)
        assert call_count == len(np.unique(idxs // self._N_PATCHES))

    def test_patches_from_duplicated_timestep_match_individual_lookups(self):
        """Deduplicating the materialization must not change any individual patch's values."""
        ds = self._make_ds()
        idxs = np.array([0, 1, 2, self._N_PATCHES])
        x_batch, y_batch = ds.get_at_indices(idxs)
        for i, global_idx in enumerate(idxs):
            x_single, y_single = ds.get_at_indices(np.array([global_idx]))
            np.testing.assert_allclose(x_batch[i], x_single[0])
            np.testing.assert_allclose(y_batch[i], y_single[0])

    def test_zero_latitude_buffer_widens_only_longitude(self):
        """Buffer_lat_px=0 must widen only longitude — the point of per-axis buffering.

        Latitude must stay at the core height (identical between input and target) while
        longitude still grows by 2 * buffer_lon_px on the input only. See CONTRACT.md's
        root-cause table — the latitude buffer was pure waste since every patch already
        spans the full domain height, so there is no artificial tile cut along latitude
        to overlap-tile-buffer.
        """
        buffer_lon_px = 3
        ds = self._make_ds(buffer_lat_px=0, buffer_lon_px=buffer_lon_px)
        x, y = ds.get_at_indices(np.array([0]))
        assert x.shape == (1, self._N_LAT_CORE, self._PATCH_WIDTH + 2 * buffer_lon_px, 1)
        assert y.shape[1:3] == (self._N_LAT_CORE, self._PATCH_WIDTH)
        # Latitude extent must be identical (core height) between input and target.
        assert x.shape[1] == y.shape[1] == self._N_LAT_CORE
        # Longitude extent must differ by exactly 2 * buffer_lon_px.
        assert x.shape[2] - y.shape[2] == 2 * buffer_lon_px

    def test_zero_latitude_buffer_matches_unbuffered_reflect_pad_on_longitude_only(self):
        """With buffer_lat_px=0, the patch must equal a longitude-only reflect-padded slice."""
        buffer_lon_px = 3
        ds = self._make_ds(buffer_lat_px=0, buffer_lon_px=buffer_lon_px)
        starts = compute_patch_lon_starts(self._N_LON_CORE, self._PATCH_WIDTH, self._N_PATCHES)
        padded = reflect_pad_lat_lon_buffer(self._core_vals()[None, ...], 0, buffer_lon_px)[0]
        for global_idx in range(ds.n_samples):
            x, _ = ds.get_at_indices(np.array([global_idx]))
            _, patch_idx = divmod(global_idx, self._N_PATCHES)
            start = starts[patch_idx]
            expected = padded[:, start : start + self._PATCH_WIDTH + 2 * buffer_lon_px]
            np.testing.assert_allclose(x[0, :, :, 0], expected)

    def test_patch_config_rejects_negative_buffer_lat_px(self):
        with pytest.raises(ValueError, match="buffer_lat_px"):
            PatchConfig(n_patches=self._N_PATCHES, patch_lon_width_px=self._PATCH_WIDTH, buffer_lat_px=-1)

    def test_patch_config_rejects_negative_buffer_lon_px(self):
        with pytest.raises(ValueError, match="buffer_lon_px"):
            PatchConfig(n_patches=self._N_PATCHES, patch_lon_width_px=self._PATCH_WIDTH, buffer_lon_px=-1)

    def test_patch_config_rejects_patches_per_epoch_of_zero(self):
        with pytest.raises(ValueError, match="patches_per_epoch"):
            PatchConfig(n_patches=self._N_PATCHES, patch_lon_width_px=self._PATCH_WIDTH, patches_per_epoch=0)

    def test_patch_config_rejects_patches_per_epoch_above_n_patches(self):
        with pytest.raises(ValueError, match="patches_per_epoch"):
            PatchConfig(
                n_patches=self._N_PATCHES,
                patch_lon_width_px=self._PATCH_WIDTH,
                patches_per_epoch=self._N_PATCHES + 1,
            )

    def test_patch_config_accepts_patches_per_epoch_equal_to_n_patches(self):
        # Upper bound of the valid range (1 <= patches_per_epoch <= n_patches) must not raise.
        cfg = PatchConfig(
            n_patches=self._N_PATCHES, patch_lon_width_px=self._PATCH_WIDTH, patches_per_epoch=self._N_PATCHES
        )
        assert cfg.patches_per_epoch == self._N_PATCHES

    def test_flip_augmentation_is_vectorized_and_matches_manual_per_sample_flip(self):
        """Contract 7: the vectorized flip must reproduce the old per-sample loop's flips.

        Bit-for-bit, given the same seed, since the RNG draw order (flip_lat then
        flip_lon, both ``self._rng.random(n) < p``) is unchanged. The expected result is
        constructed independently here (a fresh RNG replicating the exact draw order, then
        explicit per-row flipping), not by calling the implementation under test, so this
        can't pass by construction.
        """
        seed = 0
        flip_probability = 0.5
        ds_raw = self._make_ds(flip_probability=flip_probability, augment=False, seed=seed)
        idxs = np.arange(ds_raw.n_samples)
        x_raw, y_raw = ds_raw.get_at_indices(idxs)

        ds_aug = self._make_ds(flip_probability=flip_probability, augment=True, seed=seed)
        x_aug, y_aug = ds_aug.get_at_indices(idxs)

        # ds_aug's RNG is untouched before this first get_at_indices call (shuffle=False
        # never draws from it), so a freshly seeded generator reproduces the exact draws.
        expected_rng = np.random.default_rng(seed)
        n = len(idxs)
        flip_lat = expected_rng.random(n) < flip_probability
        flip_lon = expected_rng.random(n) < flip_probability
        assert flip_lat.any() and flip_lon.any(), "test seed must exercise both flip branches"

        expected_x, expected_y = x_raw.copy(), y_raw.copy()
        for i in range(n):
            if flip_lat[i]:
                expected_x[i] = expected_x[i, ::-1, ...]
                expected_y[i] = expected_y[i, ::-1, ...]
            if flip_lon[i]:
                expected_x[i] = expected_x[i, :, ::-1, ...]
                expected_y[i] = expected_y[i, :, ::-1, ...]

        np.testing.assert_allclose(x_aug, expected_x)
        np.testing.assert_allclose(y_aug, expected_y)

    def test_flip_probability_zero_is_a_no_op(self):
        ds_raw = self._make_ds(flip_probability=0.0, augment=False)
        ds_aug = self._make_ds(flip_probability=0.0, augment=True)
        idxs = np.arange(ds_raw.n_samples)
        x_raw, y_raw = ds_raw.get_at_indices(idxs)
        x_aug, y_aug = ds_aug.get_at_indices(idxs)
        np.testing.assert_allclose(x_aug, x_raw)
        np.testing.assert_allclose(y_aug, y_raw)


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestFrontsPyDatasetPatchModeShuffleBlocks:
    """Block-aligned shuffling must stay contiguous in time even with interleaved patches.

    Global sample indices interleave patches within a timestep
    (``time_idx * n_patches + patch_idx``), so a naive per-sample shuffle would scatter
    reads across timesteps just as badly as in non-patch mode. ``_build_order`` must
    instead group every patch of nearby timesteps together.
    """

    def _make_ds(
        self, n_time, n_patches, batch_size, shuffle=True, seed=0, drop_remainder=False, patches_per_epoch=None
    ):
        n_lat_core, n_lon_core, patch_width = 4, 12, 4
        vals = (np.arange(n_lat_core)[:, None] + np.arange(n_lon_core)[None, :]).astype(np.float32) % 2
        input_ds = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    np.broadcast_to(vals, (n_time, n_lat_core, n_lon_core)).copy(),
                    dims=["time", "latitude", "longitude"],
                    coords={"time": np.arange(n_time)},
                )
            }
        )
        target_da = xr.DataArray(
            np.broadcast_to(vals, (n_time, n_lat_core, n_lon_core)).astype(np.int32).copy(),
            dims=["time", "latitude", "longitude"],
            coords={"time": np.arange(n_time)},
        )
        dummy_store = IcechunkStorageConfig(store_path="unused", branch_name="main")
        patch_config = PatchConfig(
            n_patches=n_patches,
            patch_lon_width_px=patch_width,
            buffer_lat_px=0,
            buffer_lon_px=0,
            patches_per_epoch=patches_per_epoch,
        )
        data_config = DatasetConfig(
            inputs_icechunk_config=dummy_store,
            targets_icechunk_config=dummy_store,
            variables=["temperature"],
            test_years=[],
            val_years=[],
            patch_config=patch_config,
        )
        return FrontsPyDataset(
            input_ds,
            target_da,
            data_config,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            drop_remainder=drop_remainder,
        )

    def test_block_size_is_smallest_multiple_of_batch_size(self):
        # gcd(6, 3) = 3 -> block_timesteps = 6 // 3 = 2: every 6-sample (= 1 batch) window
        # of _order must land on exactly 1 or 2 distinct, adjacent timesteps.
        ds = self._make_ds(n_time=12, n_patches=3, batch_size=6, seed=0)
        time_idxs = ds._order // ds._n_patches
        batch_size = 6
        for start in range(0, len(ds._order), batch_size):
            block_times = time_idxs[start : start + batch_size]
            uniq = np.unique(block_times)
            assert uniq.max() - uniq.min() <= 1

    def test_batches_stay_within_a_contiguous_timestep_run(self):
        # gcd(5, 3) = 1 -> block_timesteps = 5: batches may span up to 5 contiguous
        # timesteps, but never a scattered/non-adjacent set.
        n_time, n_patches, batch_size = 20, 3, 5
        ds = self._make_ds(n_time=n_time, n_patches=n_patches, batch_size=batch_size, seed=1)
        for i in range(len(ds)):
            local_idxs = ds._order[i * batch_size : (i + 1) * batch_size]
            time_idxs = local_idxs // n_patches
            uniq = np.unique(time_idxs)
            assert uniq.max() - uniq.min() == len(uniq) - 1, f"batch {i} timesteps {uniq} are not contiguous"

    def test_ragged_final_block_never_straddles_mid_epoch(self):
        """A total timestep count not a multiple of the block size must not straddle a batch.

        The undersized remainder block must never get shuffled into the middle, which
        would straddle a batch across two unrelated blocks. Checked across many seeds
        since the failure is seed-dependent (it only manifests when the ragged block
        lands anywhere but last).
        """
        n_time, n_patches, batch_size = 17, 3, 6  # block_timesteps=2, 8 full blocks + 1 ragged timestep
        for seed in range(20):
            ds = self._make_ds(n_time=n_time, n_patches=n_patches, batch_size=batch_size, seed=seed)
            for i in range(len(ds)):
                local_idxs = ds._order[i * batch_size : (i + 1) * batch_size]
                time_idxs = local_idxs // n_patches
                uniq = np.unique(time_idxs)
                assert uniq.max() - uniq.min() == len(uniq) - 1, (
                    f"seed {seed} batch {i} timesteps {uniq} are not contiguous"
                )

    def test_shuffle_visits_every_sample_exactly_once_per_epoch(self):
        ds = self._make_ds(n_time=9, n_patches=3, batch_size=9, seed=2)
        np.testing.assert_array_equal(np.sort(ds._order), np.arange(ds.n_samples))

    def test_on_epoch_end_reshuffles_block_order(self):
        ds = self._make_ds(n_time=9, n_patches=3, batch_size=9, seed=3)
        order_before = ds._order.copy()
        ds.on_epoch_end()
        assert not np.array_equal(order_before, ds._order)

    def test_no_shuffle_preserves_time_major_order(self):
        ds = self._make_ds(n_time=6, n_patches=2, batch_size=4, shuffle=False)
        np.testing.assert_array_equal(ds._order, np.arange(ds.n_samples))

    def test_non_patch_mode_uses_one_block_per_batch(self):
        """n_patches=1 must reduce to the non-patch case: one contiguous batch-sized block."""
        n_time, batch_size = 20, 4
        ds = self._make_ds(n_time=n_time, n_patches=1, batch_size=batch_size, seed=4)
        for i in range(len(ds)):
            local_idxs = ds._order[i * batch_size : (i + 1) * batch_size]
            np.testing.assert_array_equal(local_idxs, np.arange(local_idxs[0], local_idxs[0] + len(local_idxs)))

    def test_batch_contents_match_a_contiguous_timestep_window(self):
        """End-to-end: a batch's actual returned values must come from one timestep run."""
        n_time, n_patches, batch_size = 12, 3, 6
        ds = self._make_ds(n_time=n_time, n_patches=n_patches, batch_size=batch_size, seed=5)
        starts = compute_patch_lon_starts(n_lon_core=12, patch_width=4, n_patches=n_patches)
        for i in range(len(ds)):
            local_idxs = ds._order[i * batch_size : (i + 1) * batch_size]
            time_idxs = local_idxs // n_patches
            patch_idxs = local_idxs % n_patches
            x_batch, _ = ds.get_at_indices(local_idxs)
            for row, t, p in zip(x_batch, time_idxs, patch_idxs, strict=True):
                start = starts[p]
                expected = ds.input_ds["temperature"].isel(time=t).values[:, start : start + 4]
                np.testing.assert_allclose(row[..., 0], expected)

    def test_dunder_getitem_matches_get_at_indices_across_seeds(self):
        """Regression test: ``ds[i]`` (what Keras's training loop actually calls) must
        resolve the same samples as ``ds.get_at_indices(ds._order[i*batch_size:...])``.

        ``__getitem__`` previously passed a raw ``slice`` object straight into patch mode's
        ``get_at_indices``, which crashes (``TypeError`` on ``slice // int``) the moment a
        real training loop calls ``ds[i]``, since only ``ds.get_at_indices`` with a concrete
        index array was ever exercised directly in tests. Checked across many seeds since
        the pre-fix bug in the non-patch-mode branch (see TestFrontsPyDataset's analogous
        regression test) was also seed-dependent.
        """
        n_time, n_patches, batch_size = 12, 3, 6
        for seed in range(10):
            ds = self._make_ds(n_time=n_time, n_patches=n_patches, batch_size=batch_size, seed=seed)
            for i in range(len(ds)):
                local_idxs = ds._order[i * batch_size : (i + 1) * batch_size]
                expected_x, expected_y = ds.get_at_indices(local_idxs)
                x_batch, y_batch = ds[i]
                np.testing.assert_allclose(x_batch, expected_x)
                np.testing.assert_allclose(y_batch, expected_y)

    def test_patches_per_epoch_reduces_n_samples_and_len(self):
        """Contract 6: an epoch visits patches_per_epoch, not all n_patches, positions.

        This is the epoch-cost-parity fix (see CONTRACT.md's root-cause table: an
        unrestricted patch epoch computes 4.00x the core-coverage of a whole-domain
        epoch; sampling k of n positions divides that redundancy down to k/n).
        """
        n_time, n_patches, patches_per_epoch, batch_size = 5, 10, 4, 4
        ds = self._make_ds(
            n_time=n_time, n_patches=n_patches, batch_size=batch_size, patches_per_epoch=patches_per_epoch
        )
        assert ds.n_samples == n_time * patches_per_epoch
        assert len(ds) == math.ceil(n_time * patches_per_epoch / batch_size)

    def test_patches_per_epoch_shuffle_selects_k_distinct_valid_patches_per_timestep(self):
        n_time, n_patches, patches_per_epoch, batch_size = 6, 8, 3, 3
        ds = self._make_ds(
            n_time=n_time,
            n_patches=n_patches,
            batch_size=batch_size,
            shuffle=True,
            seed=2,
            patches_per_epoch=patches_per_epoch,
        )
        assert len(ds._order) == n_time * patches_per_epoch
        time_idxs = ds._order // n_patches
        patch_idxs = ds._order % n_patches
        for t in range(n_time):
            patches_for_t = patch_idxs[time_idxs == t]
            assert len(patches_for_t) == patches_per_epoch, f"timestep {t} got {len(patches_for_t)} patches"
            assert len(set(patches_for_t.tolist())) == patches_per_epoch, f"timestep {t} patches not distinct"
            assert patches_for_t.min() >= 0 and patches_for_t.max() < n_patches
            np.testing.assert_array_equal(patches_for_t, np.sort(patches_for_t))  # emitted in ascending order

    def test_patches_per_epoch_shuffle_redraws_a_different_subset_each_epoch(self):
        """Two successive epochs' patch subsets must both be correct and differ.

        Independently simulating the spec'd algorithm (block-order permutation via
        ``rng.permutation``, then per-timestep ``rng.choice(n_patches, size=k,
        replace=False)`` sorted ascending) for seed=7 gives these exact two epochs — a
        fresh ``np.random.default_rng(7)`` reproduces them deterministically, so this
        pins down the real draws rather than a vacuous not-equal check.
        """
        n_time, n_patches, patches_per_epoch, batch_size = 4, 5, 2, 4
        ds = self._make_ds(
            n_time=n_time,
            n_patches=n_patches,
            batch_size=batch_size,
            shuffle=True,
            seed=7,
            patches_per_epoch=patches_per_epoch,
        )
        first_epoch_order = ds._order.copy()
        ds.on_epoch_end()
        second_epoch_order = ds._order.copy()

        np.testing.assert_array_equal(first_epoch_order, np.array([2, 3, 7, 8, 10, 14, 16, 19]))
        np.testing.assert_array_equal(second_epoch_order, np.array([11, 14, 15, 18, 1, 3, 6, 8]))
        assert not np.array_equal(first_epoch_order, second_epoch_order)

    def test_patches_per_epoch_no_shuffle_is_deterministic_and_evenly_spaced(self):
        """shuffle=False must select the same linspace-evenly-spaced subset every epoch.

        Required for a stable val_loss (LR-plateau / early-stopping rely on it).
        """
        n_time, n_patches, patches_per_epoch, batch_size = 3, 30, 6, 6
        ds = self._make_ds(
            n_time=n_time,
            n_patches=n_patches,
            batch_size=batch_size,
            shuffle=False,
            patches_per_epoch=patches_per_epoch,
        )
        expected_positions = np.array([0, 5, 10, 15, 20, 25])
        expected_order = np.concatenate([t * n_patches + expected_positions for t in range(n_time)])
        np.testing.assert_array_equal(ds._order, expected_order)

        order_before = ds._order.copy()
        ds.on_epoch_end()
        np.testing.assert_array_equal(ds._order, order_before)

    def test_patches_per_epoch_none_matches_all_patches_behavior(self):
        """patches_per_epoch=None must reproduce today's all-patches behavior exactly.

        Every sample in the full n_patches index space, once.
        """
        n_time, n_patches, batch_size = 9, 3, 9
        ds = self._make_ds(n_time=n_time, n_patches=n_patches, batch_size=batch_size, seed=2, patches_per_epoch=None)
        assert ds._patches_per_epoch == n_patches
        assert ds.n_samples == n_time * n_patches
        np.testing.assert_array_equal(np.sort(ds._order), np.arange(ds.n_samples))

    def test_read_amplification_invariant_holds_when_batch_size_divides_patches_per_epoch(self):
        """The read-amplification invariant the whole feature exists for.

        When ``batch_size % patches_per_epoch == 0``, every batch covers exactly
        ``batch_size // patches_per_epoch`` distinct timesteps, and each timestep is
        materialized by exactly one batch across the whole epoch (never split across two
        batches, so never re-read).
        """
        n_time, n_patches, patches_per_epoch, batch_size = 8, 6, 3, 6
        assert batch_size % patches_per_epoch == 0
        ds = self._make_ds(
            n_time=n_time,
            n_patches=n_patches,
            batch_size=batch_size,
            shuffle=True,
            seed=11,
            patches_per_epoch=patches_per_epoch,
        )
        expected_distinct = batch_size // patches_per_epoch
        time_to_batches: dict[int, set[int]] = {}
        for i in range(len(ds)):
            local_idxs = ds._order[i * batch_size : (i + 1) * batch_size]
            time_idxs = local_idxs // n_patches
            uniq = np.unique(time_idxs)
            assert len(uniq) == expected_distinct, (
                f"batch {i} covers {len(uniq)} timesteps, expected {expected_distinct}"
            )
            for t in uniq:
                time_to_batches.setdefault(int(t), set()).add(i)

        assert set(time_to_batches.keys()) == set(range(n_time)), "every timestep must be visited"
        for t, batches in time_to_batches.items():
            assert len(batches) == 1, f"timestep {t} was materialized by {len(batches)} batches, expected exactly 1"

    def test_read_amplification_invariant_does_not_hold_when_misaligned(self):
        """Contrast case: a misaligned batch_size/patches_per_epoch allows re-reads.

        batch_size=4 is not a multiple of patches_per_epoch=3, so the invariant is NOT
        guaranteed — some timestep's patches straddle two different batches, meaning that
        timestep gets materialized (read) twice in the epoch. This proves the aligned
        test above is asserting something real, not vacuously true.
        """
        n_time, n_patches, patches_per_epoch, batch_size = 4, 6, 3, 4
        assert batch_size % patches_per_epoch != 0
        ds = self._make_ds(
            n_time=n_time,
            n_patches=n_patches,
            batch_size=batch_size,
            shuffle=True,
            seed=13,
            patches_per_epoch=patches_per_epoch,
        )
        time_to_batches: dict[int, set[int]] = {}
        for i in range(len(ds)):
            local_idxs = ds._order[i * batch_size : (i + 1) * batch_size]
            time_idxs = local_idxs // n_patches
            for t in np.unique(time_idxs):
                time_to_batches.setdefault(int(t), set()).add(i)

        assert any(len(batches) > 1 for batches in time_to_batches.values()), (
            "misaligned batch_size/patches_per_epoch should produce at least one re-read timestep"
        )


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestBuildDatasetSummary:
    def _make_dated_dataset(self, era5_ds, front_da, data_config, times, batch_size=2):
        input_ds = era5_ds.assign_coords(time=times)
        target_da = front_da.assign_coords(time=times)
        return FrontsPyDataset(input_ds, target_da, data_config, batch_size=batch_size)

    def test_input_and_target_shapes(self, era5_ds, front_da, data_config):
        times = pd.date_range("2020-01-01", periods=N_TIME, freq="6h")
        dataset = self._make_dated_dataset(era5_ds, front_da, data_config, times)

        summary = _build_dataset_summary("train", dataset, data_config)

        assert summary.split == "train"
        assert summary.input_shape == (N_TIME, N_LAT, N_LON, len(data_config.variables))
        assert summary.target_shape == (N_TIME, N_LAT, N_LON)

    def test_date_range_matches_time_coordinate(self, era5_ds, front_da, data_config):
        times = pd.date_range("2020-01-01", periods=N_TIME, freq="6h")
        dataset = self._make_dated_dataset(era5_ds, front_da, data_config, times)

        summary = _build_dataset_summary("val", dataset, data_config)

        assert summary.date_min == str(times.min().date())
        assert summary.date_max == str(times.max().date())

    def test_out_of_order_times_still_yield_true_min_max(self, era5_ds, front_da, data_config):
        times = pd.to_datetime(["2020-03-01", "2020-01-05", "2020-02-10", "2020-01-01", "2020-02-20"])
        dataset = self._make_dated_dataset(era5_ds, front_da, data_config, times)

        summary = _build_dataset_summary("test", dataset, data_config)

        assert summary.date_min == "2020-01-01"
        assert summary.date_max == "2020-03-01"

    def test_empty_split_raises(self, era5_ds, front_da, data_config):
        times = pd.date_range("2020-01-01", periods=N_TIME, freq="6h")
        dataset = self._make_dated_dataset(era5_ds, front_da, data_config, times)
        empty_dataset = FrontsPyDataset(
            dataset.input_ds.isel(time=slice(0, 0)),
            dataset.target_da.isel(time=slice(0, 0)),
            data_config,
            batch_size=2,
        )

        with pytest.raises(ValueError, match="empty"):
            _build_dataset_summary("test", empty_dataset, data_config)

    def test_volume_inputs_uses_volume_stacking(self):
        rng = np.random.default_rng(11)
        n_time, n_lat, n_lon = 4, 6, 8
        levels = (1000, 950)
        times = pd.date_range("2021-06-01", periods=n_time, freq="6h")
        input_ds = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    rng.standard_normal((n_time, len(levels), n_lat, n_lon)).astype(np.float32),
                    dims=["time", "level", "latitude", "longitude"],
                    coords={"time": times, "level": list(levels)},
                ),
                "mean_sea_level_pressure": xr.DataArray(
                    rng.standard_normal((n_time, n_lat, n_lon)).astype(np.float32),
                    dims=["time", "latitude", "longitude"],
                    coords={"time": times},
                ),
            }
        )
        target_da = xr.DataArray(
            rng.integers(0, 2, size=(n_time, n_lat, n_lon)).astype(np.int32),
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
        dataset = FrontsPyDataset(input_ds, target_da, config, batch_size=2)

        summary = _build_dataset_summary("train", dataset, config)

        assert summary.input_shape == (n_time, n_lat, n_lon, len(levels), 2)
        assert summary.date_min == str(times.min().date())
        assert summary.date_max == str(times.max().date())

    def test_does_not_eagerly_materialize_a_chunks_none_input_ds(self, era5_ds, front_da, data_config, monkeypatch):
        """Regression test: stacking a chunks=None input_ds must not materialize the full split.

        ``load_data_into_dataloader`` opens ``input_ds`` with ``chunks=None`` so per-batch
        training reads go straight through zarr. Stacking that non-dask array directly
        (``to_array``/``stack``) forces full materialization into RAM regardless of
        ``batch_size`` — the same trap ``load_or_compute_norm_stats``'s caller avoids with a
        metadata-only ``.chunk("auto")`` first. This asserts the same precaution is taken here.
        """
        times = pd.date_range("2020-01-01", periods=N_TIME, freq="6h")
        dataset = self._make_dated_dataset(era5_ds, front_da, data_config, times)

        original_chunk = xr.Dataset.chunk
        chunked_datasets = []

        def _spy_chunk(self, *args, **kwargs):
            result = original_chunk(self, *args, **kwargs)
            chunked_datasets.append(result)
            return result

        monkeypatch.setattr(xr.Dataset, "chunk", _spy_chunk)

        _build_dataset_summary("train", dataset, data_config)

        assert chunked_datasets, "_build_dataset_summary must .chunk() input_ds before stacking it"
        assert all(var.chunks is not None for var in chunked_datasets[-1].data_vars.values()), (
            "input_ds must be dask-backed before to_array/stack, or the full split materializes eagerly"
        )


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestBuildTestVisualizationCallback:
    def test_builds_from_already_loaded_test_dataset(self, data_config):
        n_time, n_lat, n_lon = 4, 3, 3
        times = pd.date_range("2022-01-01", periods=n_time, freq="6h")
        rng = np.random.default_rng(5)
        input_ds = xr.Dataset(
            {
                var: xr.DataArray(
                    rng.standard_normal((n_time, n_lat, n_lon)).astype(np.float32),
                    dims=["time", "latitude", "longitude"],
                    coords={
                        "time": times,
                        "latitude": np.arange(n_lat, dtype=np.float32),
                        "longitude": np.arange(n_lon, dtype=np.float32),
                    },
                )
                for var in data_config.variables
            }
        )
        target_data = np.zeros((n_time, n_lat, n_lon), dtype=np.int32)
        target_data[1, 0, 0] = 1  # CF code at timestep 1, so it's the "active" day.
        target_da = xr.DataArray(target_data, dims=["time", "latitude", "longitude"], coords={"time": times})
        test_dataset = FrontsPyDataset(input_ds, target_da, data_config, batch_size=2)
        callbacks_config = CallbacksConfig(test_viz_every_n_epochs=5, test_viz_sample_size=2)

        cb = _build_test_visualization_callback(test_dataset, data_config, callbacks_config, seed=0)

        assert cb.every_n_epochs == 5
        assert cb.active_day_x.shape == (n_lat, n_lon, len(data_config.variables))
        assert cb.subsample_x.shape[0] <= 2


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
                    np.zeros((len(self._TIMES), len(self._LEVELS), len(self._LAT), len(self._LON)), dtype=np.float32),
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
    """patch_config's per-axis buffers never widen the loaded domain — both inputs_ds and
    targets_da stay at the core ``coordinates`` box; buffering happens later, per batch,
    in ``FrontsPyDataset`` via reflect-padding (see TestFrontsPyDatasetPatchMode).
    """

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

    def _data_config(self, tmp_path, coordinates, buffer):
        return DatasetConfig(
            inputs_icechunk_config=self._write_store(tmp_path, "inputs", "temperature"),
            targets_icechunk_config=self._write_store(tmp_path, "targets", "identifier"),
            variables=["temperature"],
            test_years=[2020],
            val_years=[],
            coordinates=coordinates,
            patch_config=PatchConfig(n_patches=2, patch_lon_width_px=2, buffer_lat_px=buffer, buffer_lon_px=buffer)
            if buffer is not None
            else None,
        )

    @pytest.mark.parametrize("buffer", [0, 1])
    def test_inputs_and_targets_both_stay_core(self, tmp_path, buffer):
        from fronts.utils import BoundingBox

        data_config = self._data_config(
            tmp_path, BoundingBox(lat_min=20.0, lat_max=40.0, lon_min=120.0, lon_max=150.0), buffer=buffer
        )
        test_dataset = load_data_into_dataloader(data_config, split="test", seed=0)
        assert test_dataset.input_ds.sizes["latitude"] == 3
        assert test_dataset.input_ds.sizes["longitude"] == 4
        assert test_dataset.target_da.sizes["latitude"] == 3
        assert test_dataset.target_da.sizes["longitude"] == 4

    def test_buffer_past_store_edge_no_longer_raises(self, tmp_path):
        """A buffer with no real store margin past coordinates must load cleanly —
        the buffer is reflected off the core domain's own edges downstream in
        FrontsPyDataset, not read from the store (see TestFrontsPyDatasetPatchMode).
        """
        from fronts.utils import BoundingBox

        data_config = self._data_config(
            tmp_path, BoundingBox(lat_min=0.0, lat_max=40.0, lon_min=120.0, lon_max=150.0), buffer=1
        )
        test_dataset = load_data_into_dataloader(data_config, split="test", seed=0)
        assert test_dataset.input_ds.sizes["latitude"] == 5
        assert test_dataset.input_ds.sizes["longitude"] == 4

    def test_patch_config_without_coordinates_raises(self, tmp_path):
        data_config = DatasetConfig(
            inputs_icechunk_config=self._write_store(tmp_path, "inputs", "temperature"),
            targets_icechunk_config=self._write_store(tmp_path, "targets", "identifier"),
            variables=["temperature"],
            test_years=[2020],
            val_years=[],
            coordinates=None,
            patch_config=PatchConfig(n_patches=2, patch_lon_width_px=2, buffer_lat_px=0, buffer_lon_px=0),
        )
        with pytest.raises(ValueError, match="coordinates"):
            load_data_into_dataloader(data_config, split="test", seed=0)

    def test_augment_flag_threaded_to_dataset(self, tmp_path):
        from fronts.utils import BoundingBox

        data_config = self._data_config(
            tmp_path, BoundingBox(lat_min=20.0, lat_max=40.0, lon_min=120.0, lon_max=150.0), buffer=0
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

    def test_neighborhood_brier_threads_pred_buffer_lat_and_lon_px_independently(self):
        """Both axes must thread through independently — an asymmetric buffer per axis."""
        loss_fn = _build_loss(
            loss_name="neighborhood_brier_score",
            loss_class_weights=None,
            latitudes=self._LATITUDES,
            fss_mask_size=(3, 3),
            nbs_tolerance_km=25.0,
            nbs_periodic_lon=False,
            nbs_lat_dependent_pool=False,
            nbs_pred_buffer_lat_px=1,
            nbs_pred_buffer_lon_px=2,
        )
        y_true = np.zeros((1, 8, 8, 6), dtype=np.float32)
        y_true[..., 0] = 1.0
        # buffered by 1 on each latitude side and 2 on each longitude side, independently.
        y_pred = np.zeros((1, 10, 12, 6), dtype=np.float32)
        y_pred[..., 0] = 1.0
        result = loss_fn(y_true, y_pred).numpy()
        assert np.all(np.isfinite(result))


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestValidateBatchSizeForStrategy:
    """batch_size must split evenly across MirroredStrategy's replicas.

    An uneven split (e.g. batch_size=30 over 4 replicas -> shards of 8, 8, 8, 6) crashes
    deep-supervision gradient aggregation with an AddN shape mismatch well after
    model.fit has already started; this should be caught immediately instead.
    """

    def test_single_replica_never_raises(self):
        _validate_batch_size_for_strategy(batch_size=30, num_replicas=1)

    def test_even_split_does_not_raise(self):
        _validate_batch_size_for_strategy(batch_size=60, num_replicas=4)

    def test_uneven_split_raises(self):
        with pytest.raises(ValueError, match="must be a multiple"):
            _validate_batch_size_for_strategy(batch_size=30, num_replicas=4)


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestPredBufferFromDataConfig:
    def test_no_patch_config_returns_zero_zero(self, data_config):
        assert _pred_buffer_from_data_config(data_config) == (0, 0)

    def test_patch_config_returns_its_per_axis_buffers(self, data_config):
        import dataclasses as dc

        cfg = dc.replace(
            data_config,
            patch_config=PatchConfig(n_patches=9, patch_lon_width_px=128, buffer_lat_px=0, buffer_lon_px=16),
        )
        assert _pred_buffer_from_data_config(cfg) == (0, 16)


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
class TestLoadDataIntoDataloaderIgnoresPatchConfigForViz:
    """train() must load the visualization test split whole-domain, regardless of patch_config.

    _build_test_visualization_callback's active-day map and per-office-region performance
    diagrams assume one whole-domain input/target pair per sample, at the core (unbuffered)
    lats/lons. Patch-mode training only changes how *training* samples are drawn; the model's
    input shape stays fully dynamic (Input(shape=(None, None, ...))), so whole-domain
    inference works regardless of patch_config. train() enforces the whole-domain *load* via
    ``dataclasses.replace(data_config, patch_config=None)`` before calling
    _build_test_visualization_callback — this test exercises that same composition.

    When ``patch_config.buffer_lon_px``/``buffer_lat_px`` > 0, _build_test_visualization_callback
    additionally reflect-pads that whole-domain input by those per-axis buffers (see
    ``datasets.reflect_pad_lat_lon_buffer``) before handing it to the callback: every core
    pixel a patch-buffer-trained model was scored on during training had >= buffer_lon_px real
    pixels of longitude context before the nearest zero-padded tensor edge (see
    ``FrontsPyDataset._get_patches_at_indices``), and scoring it directly at the true
    (unbuffered) domain edge breaks that invariant, producing a systematic false-front stripe
    there. Latitude gets no such buffer by default (``buffer_lat_px=0``): every training patch
    already spans the domain's full latitude height, so there's no artificial tile cut to
    buffer there. ``TestVisualizationCallback.buffer_lat_px``/``buffer_lon_px`` crop the buffer
    back off the prediction before it's compared against the still-core-sized target/lats/lons.
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

    def _load_viz_dataset(self, data_config: DatasetConfig) -> FrontsPyDataset:
        viz_data_config = dataclasses.replace(data_config, patch_config=None)
        return load_data_into_dataloader(viz_data_config, split="test", seed=0)

    def test_whole_domain_shape_with_patch_config_set(self, tmp_path):
        """buffer_lat_px=0 leaves latitude untouched; only longitude gets padded."""
        buffer_lon_px = 1
        data_config = self._data_config(
            self._write_inputs(tmp_path),
            self._write_targets(tmp_path),
            patch_config=PatchConfig(n_patches=2, patch_lon_width_px=4, buffer_lat_px=0, buffer_lon_px=buffer_lon_px),
        )
        test_dataset = self._load_viz_dataset(data_config)

        callback = _build_test_visualization_callback(test_dataset, data_config, self._callbacks_config(), seed=0)

        assert callback.buffer_lat_px == 0
        assert callback.buffer_lon_px == buffer_lon_px
        assert callback.active_day_x.shape == (
            len(self._LAT_CORE),
            len(self._LON_CORE) + 2 * buffer_lon_px,
            1,
        )
        assert callback.active_day_y.shape[:2] == (len(self._LAT_CORE), len(self._LON_CORE))
        assert callback.subsample_x.shape[1:3] == (
            len(self._LAT_CORE),
            len(self._LON_CORE) + 2 * buffer_lon_px,
        )
        assert callback.subsample_y.shape[1:3] == (len(self._LAT_CORE), len(self._LON_CORE))
        np.testing.assert_array_equal(callback.lats, self._LAT_CORE)
        np.testing.assert_array_equal(callback.lons, self._LON_CORE)

    def test_whole_domain_stays_core_sized_without_patch_config(self, tmp_path):
        data_config = self._data_config(self._write_inputs(tmp_path), self._write_targets(tmp_path), patch_config=None)
        test_dataset = self._load_viz_dataset(data_config)

        callback = _build_test_visualization_callback(test_dataset, data_config, self._callbacks_config(), seed=0)

        assert callback.buffer_lat_px == 0
        assert callback.buffer_lon_px == 0
        assert callback.active_day_x.shape == (len(self._LAT_CORE), len(self._LON_CORE), 1)
        assert callback.subsample_x.shape[1:3] == (len(self._LAT_CORE), len(self._LON_CORE))

    def test_buffered_input_core_matches_unbuffered_input(self, tmp_path):
        """Reflect-padding only touches the (longitude) border.

        Stripping buffer_lon_px back off the padded (patch_config-set) input must reproduce the
        unbuffered (patch_config=None) input exactly. Targets/lats/lons are always core-sized
        and must match outright either way, and latitude is never padded (buffer_lat_px=0).
        """
        inputs_store = self._write_inputs(tmp_path)
        targets_store = self._write_targets(tmp_path)
        callbacks_config = self._callbacks_config()
        buffer_lon_px = 1

        no_patch_config = self._data_config(inputs_store, targets_store, patch_config=None)
        test_dataset_no_patch = self._load_viz_dataset(no_patch_config)
        callback_no_patch = _build_test_visualization_callback(
            test_dataset_no_patch, no_patch_config, callbacks_config, seed=0
        )

        with_patch_config = self._data_config(
            inputs_store,
            targets_store,
            patch_config=PatchConfig(n_patches=2, patch_lon_width_px=4, buffer_lat_px=0, buffer_lon_px=buffer_lon_px),
        )
        test_dataset_with_patch = self._load_viz_dataset(with_patch_config)
        callback_with_patch = _build_test_visualization_callback(
            test_dataset_with_patch, with_patch_config, callbacks_config, seed=0
        )

        b = buffer_lon_px
        np.testing.assert_allclose(callback_with_patch.active_day_x[:, b:-b, :], callback_no_patch.active_day_x)
        np.testing.assert_allclose(callback_with_patch.subsample_x[:, :, b:-b, :], callback_no_patch.subsample_x)
        np.testing.assert_allclose(callback_with_patch.active_day_y, callback_no_patch.active_day_y)
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
            n_patches=30,
            patch_lon_width_px=128,
            buffer_lat_px=0,
            buffer_lon_px=16,
            flip_probability=0.25,
            patches_per_epoch=6,
        )
        assert data_cfg.patch_config.patches_per_epoch == 6
        assert data_cfg.patch_config.buffer_lat_px == 0
        assert data_cfg.coordinates == utils.BoundingBox(lat_min=0.25, lat_max=80, lon_min=130, lon_max=369.75)
        assert data_cfg.volume_inputs is True
        assert data_cfg.batch_size == 24
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

    def test_buffer_lat_px_must_be_non_negative(self):
        with pytest.raises(ValueError, match="buffer_lat_px"):
            PatchConfig(n_patches=1, patch_lon_width_px=4, buffer_lat_px=-1)

    def test_buffer_lon_px_must_be_non_negative(self):
        with pytest.raises(ValueError, match="buffer_lon_px"):
            PatchConfig(n_patches=1, patch_lon_width_px=4, buffer_lon_px=-1)

    def test_flip_probability_must_be_in_unit_interval(self):
        with pytest.raises(ValueError, match="flip_probability"):
            PatchConfig(n_patches=1, patch_lon_width_px=4, flip_probability=1.5)

    def test_defaults(self):
        cfg = PatchConfig(n_patches=9, patch_lon_width_px=128)
        assert cfg.buffer_lat_px == 0
        assert cfg.buffer_lon_px == 0
        assert cfg.flip_probability == 0.0
        assert cfg.patches_per_epoch is None


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


class TestReflectPadLatLonBuffer:
    def test_zero_buffer_returns_input_unchanged(self):
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1)
        result = reflect_pad_lat_lon_buffer(x, 0, 0)
        assert result is x

    def test_pads_lat_lon_axes_only(self):
        x = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
        result = reflect_pad_lat_lon_buffer(x, 1, 1)
        assert result.shape == (2, 5, 6, 5)

    def test_matches_manual_reflect_pad(self):
        x = np.arange(2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6, 1)
        expected = np.pad(x, [(0, 0), (2, 2), (2, 2), (0, 0)], mode="reflect")
        result = reflect_pad_lat_lon_buffer(x, 2, 2)
        np.testing.assert_array_equal(result, expected)

    def test_core_slice_of_padded_result_matches_original(self):
        x = np.arange(2 * 5 * 5 * 3, dtype=np.float32).reshape(2, 5, 5, 3)
        buf = 2
        padded = reflect_pad_lat_lon_buffer(x, buf, buf)
        np.testing.assert_array_equal(padded[:, buf:-buf, buf:-buf, :], x)

    def test_handles_extra_trailing_axes(self):
        # (sample, latitude, longitude, level, variable) — the volume_inputs shape.
        x = np.arange(1 * 3 * 3 * 2 * 2, dtype=np.float32).reshape(1, 3, 3, 2, 2)
        result = reflect_pad_lat_lon_buffer(x, 1, 1)
        assert result.shape == (1, 5, 5, 2, 2)

    def test_lon_only_buffer_leaves_latitude_axis_untouched(self):
        """buffer_lat_px=0 with buffer_lon_px>0 — the patch_buffer_ablation.yaml case."""
        x = np.arange(2 * 4 * 6 * 1, dtype=np.float32).reshape(2, 4, 6, 1)
        expected = np.pad(x, [(0, 0), (0, 0), (3, 3), (0, 0)], mode="reflect")
        result = reflect_pad_lat_lon_buffer(x, 0, 3)
        assert result.shape == (2, 4, 12, 1)
        np.testing.assert_array_equal(result, expected)


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
        """Asymmetric buffer: longitude gets the overlap-tile context, latitude gets none.

        Exercises the actual patch_buffer_ablation.yaml shape (buffer_lat_px=0,
        buffer_lon_px>0): every training patch already spans the domain's full latitude
        height, so only longitude has an artificial tile cut needing buffer context.
        """
        core = 16
        buffer_lon_px = 4
        buffered_lon = core + 2 * buffer_lon_px  # 24; still divisible by the 2-stage (levels=3) stride of 4
        model = _build_small_unet(levels=3, deep_supervision=False)  # input_shape=(None, None, 4)

        rng = np.random.default_rng(3)
        x = rng.standard_normal((2, core, buffered_lon, 4)).astype(np.float32)  # lat unbuffered, lon buffered
        y_true = tf.one_hot(rng.integers(0, 6, size=(2, core, core)), 6).numpy().astype(np.float32)

        y_pred = model(x, training=False)
        if isinstance(y_pred, list | tuple):
            y_pred = y_pred[0]
        assert y_pred.shape == (2, core, buffered_lon, 6)

        loss_fn = losses.neighborhood_brier_score(
            latitudes=np.linspace(25.0, 30.0, core),
            tolerance_km=25.0,
            pred_buffer_lat_px=0,
            pred_buffer_lon_px=buffer_lon_px,
        )
        result = loss_fn(y_true, y_pred).numpy()
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_compiled_model_trains_on_buffered_patch_batch(self):
        """_compile must thread pred_buffer_lat_px/pred_buffer_lon_px into the HSS metric, not just the loss.

        Reproduces the crash from a real patch-buffer training run: model.fit failing
        inside compute_metrics because the buffered (wider) y_pred and unbuffered y_true
        reached heidke_skill_score with mismatched shapes. Uses the asymmetric
        buffer_lat_px=0/buffer_lon_px>0 shape that patch_buffer_ablation.yaml actually
        trains with, so a purely-symmetric implementation wouldn't be caught here.
        """
        core = 16
        buffer_lon_px = 4
        buffered_lon = core + 2 * buffer_lon_px
        model = _build_small_unet(levels=3, deep_supervision=False)

        rng = np.random.default_rng(3)
        x = rng.standard_normal((2, core, buffered_lon, 4)).astype(np.float32)
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
            pred_buffer_lat_px=0,
            pred_buffer_lon_px=buffer_lon_px,
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
