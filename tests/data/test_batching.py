import numpy as np
import pytest
import xarray as xr

from fronts.data.batching import ChunkPrefetcher, make_batch_dataset
from fronts.data.inputs import LazyTimeSource

try:
    import tensorflow as tf

    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

N_LAT = 32
N_LON = 64
N_CHANNELS = 30
N_CLASSES = 6


def _input_da(n_time: int = 16, n_channels: int = N_CHANNELS, seed: int = 0) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_time, N_LAT, N_LON, n_channels)).astype(np.float32)
    return xr.DataArray(
        data,
        dims=["time", "latitude", "longitude", "channel"],
        coords={"time": np.arange(n_time)},
    )


def _target_da(n_time: int = 16, seed: int = 1) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    data = rng.random((n_time, N_LAT, N_LON, N_CLASSES)).astype(np.float32)
    return xr.DataArray(
        data,
        dims=["time", "latitude", "longitude", "class"],
        coords={"time": np.arange(n_time)},
    )


def _prefetcher(
    inputs,
    target,
    n_out: int = 1,
    *,
    batch_size: int = 1,
    chunk_size: int | None = None,
    shuffle: bool = False,
    prefetch_chunks: int = 2,
    load_num_workers: int = 2,
    load_subblock: int = 32,
    seed: int | None = None,
) -> ChunkPrefetcher:
    sources = inputs if isinstance(inputs, list) else [inputs]
    sources = [s if isinstance(s, LazyTimeSource) else LazyTimeSource(s, np.arange(s.sizes["time"])) for s in sources]
    tgt = target if isinstance(target, LazyTimeSource) else LazyTimeSource(target, np.arange(target.sizes["time"]))
    total = len(tgt.positions)
    return ChunkPrefetcher(
        sources,
        tgt,
        n_out,
        batch_size=batch_size,
        chunk_size=chunk_size if chunk_size is not None else total,
        shuffle=shuffle,
        prefetch_chunks=prefetch_chunks,
        load_num_workers=load_num_workers,
        load_subblock=load_subblock,
        seed=seed,
    )


def _sample_order(samples, ref_da: xr.DataArray) -> list[int]:
    """Recover the native time index of each sample by matching its input array to ``ref_da``."""
    ref = ref_da.values
    order = []
    for x, _ in samples:
        match = next(t for t in range(ref.shape[0]) if np.array_equal(x, ref[t]))
        order.append(match)
    return order


class TestChunkPrefetcher:
    def test_positions_select_and_order_samples(self):
        era5 = _input_da(n_time=8)
        positions = np.array([4, 2])
        prefetcher = _prefetcher(LazyTimeSource(era5, positions), LazyTimeSource(_target_da(8), positions))
        samples = list(prefetcher.iter_samples())
        assert len(samples) == 2
        np.testing.assert_array_equal(samples[0][0], era5.isel(time=4).values)
        np.testing.assert_array_equal(samples[1][0], era5.isel(time=2).values)

    def test_subblock_gather_preserves_order_and_values(self):
        era5 = _input_da(n_time=8)
        positions = np.array([4, 0, 3, 1, 2])
        prefetcher = _prefetcher(
            LazyTimeSource(era5, positions), LazyTimeSource(_target_da(8), positions), load_subblock=2
        )
        samples = list(prefetcher.iter_samples())
        for sample, native in zip(samples, positions, strict=True):
            np.testing.assert_array_equal(sample[0], era5.isel(time=int(native)).values)

    def test_target_values_match_and_replicate(self):
        target = _target_da(8)
        positions = np.array([5, 1, 3])
        prefetcher = _prefetcher(LazyTimeSource(_input_da(8), positions), LazyTimeSource(target, positions), n_out=3)
        samples = list(prefetcher.iter_samples())
        for sample, native in zip(samples, positions, strict=True):
            _x, ys = sample
            assert len(ys) == 3
            for y in ys:
                np.testing.assert_array_equal(y, target.isel(time=int(native)).values)

    def test_loader_failure_propagates_instead_of_hanging(self):
        era5 = _input_da(8)
        bad = np.array([era5.sizes["time"] + 100])
        prefetcher = _prefetcher(LazyTimeSource(era5, bad), LazyTimeSource(_target_da(8), np.array([0])))
        with pytest.raises(IndexError):
            list(prefetcher.iter_samples())

    def test_multi_source_concat_on_channel(self):
        era5 = _input_da(8)
        positions = np.arange(8)
        prefetcher = _prefetcher(
            [LazyTimeSource(era5, positions), LazyTimeSource(era5, positions)],
            LazyTimeSource(_target_da(8), positions),
        )
        x, _ = next(prefetcher.iter_samples())
        assert x.shape == (N_LAT, N_LON, 2 * N_CHANNELS)

    def test_covers_all_timesteps(self):
        prefetcher = _prefetcher(_input_da(8), _target_da(8))
        assert sum(1 for _ in prefetcher.iter_samples()) == 8

    def test_chunking_covers_all_timesteps(self):
        prefetcher = _prefetcher(_input_da(16), _target_da(16), batch_size=2, chunk_size=3)
        assert sum(1 for _ in prefetcher.iter_samples()) == 16

    def test_shape_metadata(self):
        prefetcher = _prefetcher([_input_da(8), _input_da(8, n_channels=5)], _target_da(8))
        assert prefetcher.n_lat == N_LAT
        assert prefetcher.n_lon == N_LON
        assert prefetcher.n_channels == N_CHANNELS + 5
        assert prefetcher.n_classes == N_CLASSES

    def test_steps_per_epoch(self):
        prefetcher = _prefetcher(_input_da(10), _target_da(10), batch_size=4)
        assert prefetcher.steps_per_epoch == 3  # ceil(10 / 4)

    def test_no_shuffle_is_sequential(self):
        era5 = _input_da(16)
        prefetcher = _prefetcher(era5, _target_da(16), shuffle=False)
        assert _sample_order(prefetcher.iter_samples(), era5) == list(range(16))

    def test_shuffle_same_seed_same_order(self):
        era5 = _input_da(16)
        order_a = _sample_order(_prefetcher(era5, _target_da(16), shuffle=True, seed=7).iter_samples(), era5)
        order_b = _sample_order(_prefetcher(era5, _target_da(16), shuffle=True, seed=7).iter_samples(), era5)
        assert order_a == order_b

    def test_shuffle_different_seed_differs(self):
        era5 = _input_da(16)
        order_a = _sample_order(_prefetcher(era5, _target_da(16), shuffle=True, seed=1).iter_samples(), era5)
        order_b = _sample_order(_prefetcher(era5, _target_da(16), shuffle=True, seed=2).iter_samples(), era5)
        assert order_a != order_b

    def test_shuffle_is_a_permutation(self):
        era5 = _input_da(16)
        order = _sample_order(_prefetcher(era5, _target_da(16), shuffle=True, seed=3).iter_samples(), era5)
        assert sorted(order) == list(range(16))

    def test_consecutive_passes_reshuffle(self):
        era5 = _input_da(16)
        prefetcher = _prefetcher(era5, _target_da(16), shuffle=True, seed=0)
        first = _sample_order(prefetcher.iter_samples(), era5)
        second = _sample_order(prefetcher.iter_samples(), era5)
        assert first != second
        assert sorted(second) == list(range(16))

    def test_invalid_prefetch_chunks_raises(self):
        with pytest.raises(ValueError, match="prefetch_chunks"):
            _prefetcher(_input_da(8), _target_da(8), prefetch_chunks=0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="time lengths differ"):
            ChunkPrefetcher(
                [LazyTimeSource(_input_da(8), np.arange(8))],
                LazyTimeSource(_target_da(8), np.arange(5)),
                1,
                batch_size=1,
                chunk_size=8,
                shuffle=False,
                prefetch_chunks=2,
                load_num_workers=2,
                load_subblock=32,
            )


@pytest.mark.skipif(not _TF_AVAILABLE, reason="tensorflow not installed")
class TestMakeBatchDataset:
    def test_input_batch_shape(self, era5_da, front_da):
        ds, _ = make_batch_dataset(era5_da, front_da, 1, batch_size=2)
        x_batch, _ = next(iter(ds))
        assert x_batch.shape == (2, N_LAT, N_LON, N_CHANNELS)

    def test_target_batch_shape(self, era5_da, front_da):
        ds, _ = make_batch_dataset(era5_da, front_da, 1, batch_size=2)
        _, y_batch = next(iter(ds))
        assert y_batch[0].shape == (2, N_LAT, N_LON, N_CLASSES)

    def test_n_supervision_outputs(self, era5_da, front_da):
        for n_out in [1, 3, 5]:
            ds, _ = make_batch_dataset(era5_da, front_da, n_out, batch_size=2)
            _, y_batch = next(iter(ds))
            assert len(y_batch) == n_out

    def test_dtypes_are_float32(self, era5_da, front_da):
        ds, _ = make_batch_dataset(era5_da, front_da, 1, batch_size=2)
        x_batch, y_batch = next(iter(ds))
        assert x_batch.dtype == tf.float32
        assert y_batch[0].dtype == tf.float32

    def test_steps_per_epoch_returned(self, era5_da, front_da):
        _, steps = make_batch_dataset(era5_da, front_da, 1, batch_size=2)
        assert steps == 3  # ceil(5 / 2)

    def test_failure_surfaces_as_op_error(self, era5_da, front_da):
        bad = np.array([era5_da.sizes["time"] + 100])
        ds, _ = make_batch_dataset(
            LazyTimeSource(era5_da, bad), LazyTimeSource(front_da, np.array([0])), 1, batch_size=1
        )
        with pytest.raises(tf.errors.OpError):
            next(iter(ds))
