"""Benchmark the real ``ChunkPrefetcher`` pipeline against the training icechunk stores.

Exercises the same code path ``train.py`` uses (``load_training_data`` ->
``ChunkPrefetcher.iter_samples``) without TensorFlow or a GPU, so different
``load_num_workers`` / ``load_subblock`` / ``prefetch_chunks`` settings can be
compared on the real store for both throughput and peak memory before they're
written into a SLURM config. Must run on a host with access to the configured
icechunk store paths (e.g. schooner), not a local dev machine.

Usage:

    pixi run python scripts/benchmark_batching.py --config configs/schooner_train.yaml
    pixi run python scripts/benchmark_batching.py --load-num-workers 4 8 16 --n-chunks 3
"""

import argparse
import itertools
import logging
import time

from fronts import train, utils
from fronts.data.batching import ChunkPrefetcher
from fronts.data.loading import load_training_data
from fronts.utils import process_rss_gb

logger = logging.getLogger(__name__)


def _run_one(
    prefetcher: ChunkPrefetcher,
    n_chunks: int,
) -> dict[str, float]:
    """Drain up to ``n_chunks`` worth of samples and report throughput and peak RSS.

    Args:
        prefetcher: Configured prefetcher to drain.
        n_chunks: Number of chunks' worth of samples to consume before stopping early.

    Returns:
        Dict with samples_per_sec, mb_per_sec, peak_rss_gb, and elapsed_s.
    """
    n_samples_target = prefetcher.chunk_size * n_chunks
    rss_before = process_rss_gb()
    peak_rss = rss_before
    n_samples = 0
    n_bytes = 0
    t0 = time.monotonic()
    for x, _y in prefetcher.iter_samples():
        n_samples += 1
        n_bytes += x.nbytes
        if n_samples % max(prefetcher.batch_size, 1) == 0:
            peak_rss = max(peak_rss, process_rss_gb())
        if n_samples >= n_samples_target:
            break
    elapsed = time.monotonic() - t0
    peak_rss = max(peak_rss, process_rss_gb())
    return {
        "samples_per_sec": n_samples / max(elapsed, 1e-9),
        "mb_per_sec": n_bytes / 1e6 / max(elapsed, 1e-9),
        "peak_rss_gb": peak_rss,
        "rss_delta_gb": peak_rss - rss_before,
        "elapsed_s": elapsed,
        "n_samples": n_samples,
    }


def main() -> None:
    """Sweep load_num_workers / load_subblock / prefetch_chunks and report throughput + peak RSS."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/schooner_train.yaml", help="Path to YAML training config")
    parser.add_argument("--n-chunks", type=int, default=2, help="Chunks to drain per setting combination")
    parser.add_argument("--load-num-workers", type=int, nargs="+", default=None, help="Override(s) to sweep")
    parser.add_argument("--load-subblock", type=int, nargs="+", default=None, help="Override(s) to sweep")
    parser.add_argument("--prefetch-chunks", type=int, nargs="+", default=None, help="Override(s) to sweep")
    args = parser.parse_args()

    cfg = utils.open_config_yaml_as_dataclass(args.config, train.TrainConfig)
    data_cfg = cfg.data_config

    logger.info("Loading training data from %s ...", args.config)
    training_data = load_training_data(data_cfg, seed=cfg.seed)
    n_total = len(training_data.times)
    train_indices = list(range(n_total))
    train_input_sources = [s.select(train_indices) for s in training_data.input_sources]
    train_target_source = training_data.target_source.select(train_indices)
    n_channels = sum(s.array.sizes["channel"] for s in train_input_sources)
    logger.info("Loaded %d timesteps, %d channels.", n_total, n_channels)

    workers_grid = args.load_num_workers or [data_cfg.load_num_workers]
    subblock_grid = args.load_subblock or [data_cfg.load_subblock]
    prefetch_grid = args.prefetch_chunks or [data_cfg.prefetch_chunks]

    print(f"\n{'workers':>8} {'subblock':>9} {'prefetch':>9} {'samples/s':>10} {'MB/s':>8} {'peak RSS GB':>12}")
    for load_num_workers, load_subblock, prefetch_chunks in itertools.product(
        workers_grid, subblock_grid, prefetch_grid
    ):
        chunk_size = (data_cfg.load_chunk_steps or data_cfg.steps_per_epoch or 1) * data_cfg.batch_size
        prefetcher = ChunkPrefetcher(
            train_input_sources,
            train_target_source,
            n_supervision_outputs=1,
            batch_size=data_cfg.batch_size,
            chunk_size=chunk_size,
            shuffle=True,
            prefetch_chunks=prefetch_chunks,
            load_num_workers=load_num_workers,
            load_subblock=load_subblock,
            seed=cfg.seed,
        )
        result = _run_one(prefetcher, args.n_chunks)
        print(
            f"{load_num_workers:>8} {load_subblock:>9} {prefetch_chunks:>9} "
            f"{result['samples_per_sec']:>10.2f} {result['mb_per_sec']:>8.0f} {result['peak_rss_gb']:>12.2f}"
        )


if __name__ == "__main__":
    main()
