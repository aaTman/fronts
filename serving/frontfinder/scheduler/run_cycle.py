"""Orchestrates one IFS cycle's worth of frontfinder work: for each of the two
models, fetch/assemble input -> tiled inference -> zarr pyramid write.

Intended to run as a scheduled batch job (systemd timer / cron) on the
Proxmox VM, triggered shortly after each IFS 00/06/12/18Z cycle's 0.25deg
open-data files become available. Not run on a per-request basis -- see the
deployment doc for the timer schedule and offset.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from frontfinder.config.manifests import MANIFESTS, ModelManifest
from frontfinder.inference.engine import Predictor, run_tiled_inference
from frontfinder.ingest.ecmwf_ifs import IFSCycle, IFSFieldSource, assemble_model_input
from frontfinder.zarrio.pyramid import FrontFields, build_front_pyramid, write_front_pyramid

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelRunConfig:
    manifest: ModelManifest
    predictor: Predictor
    patch_size: int = 256
    overlap: int = 32
    batch_size: int = 8
    n_pyramid_levels: int = 6


def run_one_model(
    run_config: ModelRunConfig,
    source: IFSFieldSource,
    cycle: IFSCycle,
    output_root: str,
) -> str:
    """Runs one model end-to-end for one cycle. Returns the zarr store path
    written."""
    manifest = run_config.manifest
    logger.info("assembling input for model=%s cycle=%s", manifest.name, cycle)
    input_grid = assemble_model_input(manifest, source, cycle)

    logger.info("running tiled inference for model=%s", manifest.name)
    served_probs = run_tiled_inference(
        run_config.predictor,
        input_grid,
        manifest,
        patch_size=run_config.patch_size,
        overlap=run_config.overlap,
        batch_size=run_config.batch_size,
    )

    probabilities = {
        cls: served_probs[..., i] for i, cls in enumerate(manifest.served_classes)
    }
    valid_time = f"{cycle.date}T{cycle.run_hour:02d}:00:00"
    fields = FrontFields(
        probabilities=probabilities,
        lat=source.lat,
        lon=source.lon,
        valid_time=valid_time,
        cycle_time=valid_time,
    )

    logger.info("building + writing zarr pyramid for model=%s", manifest.name)
    pyramid = build_front_pyramid(fields, manifest, n_levels=run_config.n_pyramid_levels)

    store_name = f"{cycle.date}T{cycle.run_hour:02d}Z.zarr"
    store_path = os.path.join(output_root, manifest.name, store_name)
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    write_front_pyramid(pyramid, store_path)
    logger.info("wrote %s", store_path)

    _write_latest_pointer(output_root, manifest.name, store_name, valid_time)
    return store_path


def _write_latest_pointer(output_root: str, model_name: str, store_name: str, cycle_time: str) -> None:
    """Writes `<output_root>/<model>/latest.json`, read by the webapp to find
    the most recent successfully-published run without listing the store
    directory. Only updated after a model's zarr write succeeds, so a failed
    run never points the frontend at a partial/missing store."""
    pointer_path = os.path.join(output_root, model_name, "latest.json")
    with open(pointer_path, "w") as f:
        json.dump({"store": store_name, "cycle_time": cycle_time}, f)


def run_cycle(
    run_configs: list[ModelRunConfig],
    source: IFSFieldSource,
    cycle: IFSCycle,
    output_root: str,
) -> dict[str, str]:
    """Runs every configured model for one IFS cycle. A failure in one model
    does not stop the others -- partial results (e.g. only model_1702 updates)
    are better than none, and are logged for the "latest" pointer to skip."""
    results: dict[str, str] = {}
    for run_config in run_configs:
        try:
            results[run_config.manifest.name] = run_one_model(run_config, source, cycle, output_root)
        except Exception:
            logger.exception("model %s failed for cycle %s", run_config.manifest.name, cycle)
    return results


def known_model_names() -> list[str]:
    return sorted(MANIFESTS)
