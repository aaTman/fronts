"""CLI entrypoint invoked by the systemd timer on the Proxmox VM. Determines
the most recent completed IFS synoptic cycle from the current UTC time,
loads both Keras models, and runs `run_cycle` for both.

Not covered by unit tests (loads real Keras models + hits real network) --
`run_cycle.run_cycle`/`run_one_model` carry the tested logic; this module is
just wiring. Smoke-test on the Proxmox VM before enabling the timer.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

from frontfinder.config.manifests import BEST_LOSS_MANIFEST, MODEL_1702_MANIFEST
from frontfinder.ingest.ecmwf_ifs import EcmwfOpenDataSource, IFSCycle
from frontfinder.inference.engine import KerasPredictor
from frontfinder.scheduler.run_cycle import ModelRunConfig, run_cycle

SYNOPTIC_HOURS = (0, 6, 12, 18)


def most_recent_completed_cycle(now: datetime, publish_lag_hours: int = 7) -> IFSCycle:
    """The most recent synoptic cycle whose IFS open-data files should
    already be published, given ECMWF's typical ~6-8h publish lag after
    the nominal cycle time. `publish_lag_hours` is a first-pass estimate --
    tune it against actual observed availability on the Proxmox VM; if runs
    start failing with "file not found" on the ECMWF side, increase it
    rather than assume the pipeline is broken.
    """
    candidate = now - timedelta(hours=publish_lag_hours)
    cycle_hour = max(h for h in SYNOPTIC_HOURS if h <= candidate.hour)
    cycle_dt = candidate.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    return IFSCycle(date=cycle_dt.strftime("%Y-%m-%d"), run_hour=cycle_hour)


def build_run_configs(model_dir: str) -> list[ModelRunConfig]:
    import os

    return [
        ModelRunConfig(
            manifest=BEST_LOSS_MANIFEST,
            predictor=KerasPredictor(os.path.join(model_dir, BEST_LOSS_MANIFEST.weights_filename)),
        ),
        ModelRunConfig(
            manifest=MODEL_1702_MANIFEST,
            predictor=KerasPredictor(os.path.join(model_dir, MODEL_1702_MANIFEST.weights_filename)),
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one frontfinder IFS cycle for both models.")
    parser.add_argument("--model-dir", required=True, help="directory containing the .keras/.h5 weight files")
    parser.add_argument("--output-root", required=True, help="directory to write zarr pyramids + latest.json into")
    parser.add_argument("--cache-dir", default="/tmp/frontfinder_ifs_cache")
    parser.add_argument("--publish-lag-hours", type=int, default=7)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cycle = most_recent_completed_cycle(datetime.now(timezone.utc), args.publish_lag_hours)
    source = EcmwfOpenDataSource(cache_dir=args.cache_dir)
    run_configs = build_run_configs(args.model_dir)

    results = run_cycle(run_configs, source, cycle, args.output_root)
    logging.getLogger(__name__).info("cycle %s complete: %s", cycle, list(results))
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
