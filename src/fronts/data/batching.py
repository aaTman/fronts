"""Batched ``tf.data`` pipeline for streaming ERA5/fronts samples from lazy stores.

The data-loading machinery (scattered chunk gathering, background prefetch,
exception propagation) lives in the TensorFlow-free :class:`ChunkPrefetcher` so it
can be unit-tested without importing TensorFlow. :func:`make_batch_dataset` is a
thin wrapper that wires the prefetcher into a ``tf.data.Dataset``; TensorFlow is
imported lazily inside it.
"""

import concurrent.futures
import logging
import math
from collections import deque
from typing import Any

import dask
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from fronts.data.inputs import LazyTimeSource
from fronts.utils import process_rss_gb

logger = logging.getLogger(__name__)


def _as_source(data: "xr.DataArray | LazyTimeSource") -> LazyTimeSource:
    """Wrap a plain DataArray as an identity ``LazyTimeSource``; pass sources through unchanged."""
    if isinstance(data, LazyTimeSource):
        return data
    return LazyTimeSource(data, np.arange(data.sizes["time"]))


def _gather_inputs(input_sources: list[LazyTimeSource], local_idxs: np.ndarray, subblock: int) -> xr.DataArray:
    """Gather the given logical samples from each source and concatenate them along ``channel``."""
    pieces = [source.gather(local_idxs, subblock) for source in input_sources]
    return pieces[0] if len(pieces) == 1 else xr.concat(pieces, dim="channel")


def _preload(
    input_sources: list[LazyTimeSource], target_source: LazyTimeSource, load_subblock: int
) -> tuple[list[LazyTimeSource], LazyTimeSource]:
    """Materialize the entire dataset into RAM once via a single parallel dask compute.

    Eliminates all I/O during iteration (recommended for validation when the set fits
    in memory). Returns identity sources backed by the in-memory arrays.
    """
    total = len(target_source.positions)
    full = np.arange(total)
    logger.info("Pre-loading %d timesteps into RAM...", total)
    with dask.config.set(scheduler="threads", num_workers=16), ProgressBar():
        inputs_full = _gather_inputs(input_sources, full, load_subblock)
        targets_full = target_source.gather(full, load_subblock)
    logger.info("Pre-load complete (inputs %.1f GB, process RSS %.1f GB).", inputs_full.nbytes / 1e9, process_rss_gb())
    return [LazyTimeSource(inputs_full, full)], LazyTimeSource(targets_full, full)


class ChunkPrefetcher:
    """Yields model-ready ``(input, targets)`` samples, loading scattered timesteps in background chunks.

    A single parallel dask compute materializes each chunk of ``chunk_size`` samples
    rather than loading one sample at a time. A bounded pool keeps up to
    ``prefetch_chunks`` chunk loads in flight so the GPU overlaps training with I/O.
    Chunks are consumed strictly in order, and a failed background load surfaces as an
    exception from :meth:`iter_samples` rather than deadlocking the consumer.

    The class is TensorFlow-free; :func:`make_batch_dataset` adapts it into a
    ``tf.data.Dataset``.
    """

    def __init__(
        self,
        input_sources: list[LazyTimeSource],
        target_source: LazyTimeSource,
        n_supervision_outputs: int,
        *,
        batch_size: int,
        chunk_size: int,
        shuffle: bool,
        prefetch_chunks: int,
        load_num_workers: int,
        load_subblock: int,
        seed: int | None = None,
    ) -> None:
        """Initialize the prefetcher.

        Args:
            input_sources: One source per input, concatenated along ``channel``. Each carries
                shape (time, latitude, longitude, channel).
            target_source: Front target source of shape (time, latitude, longitude, class).
            n_supervision_outputs: Number of deep-supervision outputs; the target is replicated
                this many times per sample.
            batch_size: Number of timesteps per batch (used only to size ``steps_per_epoch``).
            chunk_size: Number of samples loaded per background chunk.
            shuffle: If True, iterate timesteps in a fresh random order on each pass.
            prefetch_chunks: Number of chunks kept in flight ahead of the consumer (>= 1).
            load_num_workers: Dask threads per background chunk load.
            load_subblock: Maximum timesteps materialized per dask ``compute`` when gathering.
            seed: Seed for the shuffle RNG. None draws from fresh entropy.
        """
        if prefetch_chunks < 1:
            raise ValueError(f"prefetch_chunks must be >= 1, got {prefetch_chunks}")
        self.input_sources = input_sources
        self.target_source = target_source
        self.n_supervision_outputs = n_supervision_outputs
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.prefetch_chunks = prefetch_chunks
        self.load_num_workers = load_num_workers
        self.load_subblock = load_subblock

        self.total = len(target_source.positions)
        mismatched = [len(s.positions) for s in input_sources if len(s.positions) != self.total]
        if mismatched:
            raise ValueError(f"Input and target time lengths differ: {mismatched} vs {self.total}")
        self.n_lat = input_sources[0].array.sizes["latitude"]
        self.n_lon = input_sources[0].array.sizes["longitude"]
        self.n_channels = sum(s.array.sizes["channel"] for s in input_sources)
        self.n_classes = target_source.array.sizes["class"]

        self._rng = np.random.default_rng(seed)
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None

    @property
    def steps_per_epoch(self) -> int:
        """Number of batches in one full pass over the dataset."""
        return math.ceil(self.total / self.batch_size)

    def _load_chunk(self, local_idxs: np.ndarray) -> tuple[xr.DataArray, xr.DataArray]:
        with dask.config.set(scheduler="threads", num_workers=self.load_num_workers):
            chunk_x = _gather_inputs(self.input_sources, local_idxs, self.load_subblock)
            chunk_y = self.target_source.gather(local_idxs, self.load_subblock)
        return chunk_x, chunk_y

    def _iter_chunk(self, chunk_x: xr.DataArray, chunk_y: xr.DataArray):
        for pos in range(chunk_x.sizes["time"]):
            x = np.ascontiguousarray(chunk_x.isel(time=pos).values)
            y = np.ascontiguousarray(chunk_y.isel(time=pos).values)
            yield x, tuple(y for _ in range(self.n_supervision_outputs))

    def iter_samples(self):
        """Yield ``(x, (y, ...))`` samples for one full pass, prefetching chunks in the background.

        Yields:
            Tuples of a float32 input array (latitude, longitude, channel) and a tuple of
            ``n_supervision_outputs`` identical target arrays (latitude, longitude, class).
        """
        if self._pool is None:
            self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.prefetch_chunks)
        order = self._rng.permutation(self.total) if self.shuffle else np.arange(self.total)
        chunks = [order[start : start + self.chunk_size] for start in range(0, self.total, self.chunk_size)]

        inflight: deque[concurrent.futures.Future] = deque()
        next_chunk = 0
        for _ in range(min(self.prefetch_chunks, len(chunks))):
            inflight.append(self._pool.submit(self._load_chunk, chunks[next_chunk]))
            next_chunk += 1
        while inflight:
            chunk_x, chunk_y = inflight.popleft().result()
            if next_chunk < len(chunks):
                inflight.append(self._pool.submit(self._load_chunk, chunks[next_chunk]))
                next_chunk += 1
            yield from self._iter_chunk(chunk_x, chunk_y)
            del chunk_x, chunk_y


def make_batch_dataset(
    input_data: "xr.DataArray | LazyTimeSource | list[LazyTimeSource]",
    target_data: "xr.DataArray | LazyTimeSource",
    n_supervision_outputs: int,
    batch_size: int = 4,
    shuffle: bool = False,
    preload: bool = False,
    epoch_steps: int | None = None,
    load_chunk_steps: int | None = None,
    prefetch_chunks: int = 2,
    load_num_workers: int = 4,
    load_subblock: int = 32,
    seed: int | None = None,
) -> tuple[Any, int]:
    """Create a batched ``tf.data.Dataset`` from ERA5 and fronts sources.

    Wraps a :class:`ChunkPrefetcher` in ``tf.data.Dataset.from_generator``. The load
    chunk size is ``load_chunk_steps * batch_size`` samples; set it smaller than
    ``epoch_steps`` to cap peak RAM per chunk while still overlapping I/O with training.
    Falls back to ``epoch_steps`` when ``load_chunk_steps`` is unset, and to the full
    dataset when neither is given.

    Thread safety: zarr's global ``ThreadPoolExecutor`` must be bounded before any zarr
    I/O (``zarr.config.update({"threading.max_workers": N})`` in the training process).

    Args:
        input_data: Input as a DataArray, a ``LazyTimeSource``, or a list of sources
            concatenated along ``channel``. Each carries shape (time, lat, lon, channel).
        target_data: Front target as a DataArray or ``LazyTimeSource`` of shape
            (time, lat, lon, class).
        n_supervision_outputs: Number of deep-supervision outputs; the target tuple is
            replicated this many times.
        batch_size: Number of timesteps per batch.
        shuffle: If True, iterate timesteps in a fresh random order each pass.
        preload: If True, materialize the whole dataset into RAM once at creation time.
        epoch_steps: Batches per epoch; used to size the load chunk when ``load_chunk_steps``
            is unset.
        load_chunk_steps: Steps' worth of samples to load per background prefetch.
        prefetch_chunks: Number of chunks kept loaded ahead of the generator (>= 1).
        load_num_workers: Dask threads per background chunk load. Peak host RAM scales with
            ``prefetch_chunks * load_num_workers``; keep small to bound memory.
        load_subblock: Maximum timesteps materialized per dask ``compute`` when gathering.
        seed: Seed for the shuffle RNG.

    Returns:
        Tuple of (tf.data.Dataset, steps_per_epoch).
    """
    import tensorflow as tf

    input_sources = [_as_source(s) for s in (input_data if isinstance(input_data, list) else [input_data])]
    target_source = _as_source(target_data)
    if preload:
        input_sources, target_source = _preload(input_sources, target_source, load_subblock)

    total = len(target_source.positions)
    effective_chunk_steps = load_chunk_steps if load_chunk_steps is not None else epoch_steps
    chunk_size = (effective_chunk_steps * batch_size) if effective_chunk_steps is not None else total

    prefetcher = ChunkPrefetcher(
        input_sources,
        target_source,
        n_supervision_outputs,
        batch_size=batch_size,
        chunk_size=chunk_size,
        shuffle=shuffle,
        prefetch_chunks=prefetch_chunks,
        load_num_workers=load_num_workers,
        load_subblock=load_subblock,
        seed=seed,
    )

    target_spec = tf.TensorSpec(shape=(prefetcher.n_lat, prefetcher.n_lon, prefetcher.n_classes), dtype=tf.float32)
    output_signature = (
        tf.TensorSpec(shape=(prefetcher.n_lat, prefetcher.n_lon, prefetcher.n_channels), dtype=tf.float32),
        tuple(target_spec for _ in range(n_supervision_outputs)),
    )
    ds = (
        tf.data.Dataset.from_generator(prefetcher.iter_samples, output_signature=output_signature)
        .batch(batch_size)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds, prefetcher.steps_per_epoch
