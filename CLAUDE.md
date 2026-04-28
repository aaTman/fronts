# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FrontFinder AI — trains UNet-style deep learning models to predict atmospheric frontal boundaries (cold fronts, warm fronts, etc.) from ERA5 reanalysis data. Targets are front labels from NOAA XML or TWC GML files.

## Environment & Package Manager

This project uses **pixi** (not pip or conda directly). There are four environments:

| Environment | Purpose | Platform |
|-------------|---------|----------|
| `default` / `dev` | Full training with GPU TF | linux-64 |
| `test` | Lightweight tests, no TF/GPU | any |
| `mac` | Local dry-run, CPU TF 2.16 | osx-arm64 |

Activate environments with `pixi shell -e <env>` or prefix commands with `pixi run -e <env>`.

## Common Commands

```bash
# Run all tests (lightweight, no GPU needed)
pixi run -e test test
# Equivalent to:
PYTHONPATH=src python -m pytest tests/ -v

# Run a single test
PYTHONPATH=src python -m pytest tests/test_model.py::test_name -v

# Lint / format
ruff check src/
ruff format src/

# Train a model
PYTHONPATH=src python src/fronts/train.py -tc configs/1702.yaml

# Dry run (validates config + data pipeline + model build without training or WandB)
PYTHONPATH=src python src/fronts/train.py -tc configs/1702.yaml --dry_run

# Generate dry-run fixture data (run once before dry runs)
python scripts/make_dryrun_data.py
```

Log level is controlled by `FRONTS_LOG_LEVEL` (default: `INFO`). WandB API key is read from `WANDB_KEY`.

## Architecture

### Config-driven dataclass pattern

Everything flows through typed dataclasses with `.build()` methods, loaded from YAML via `dacite`. The entry point is `TrainConfig` in `train.py`, loaded with `open_config_yaml_as_dataclass()`. See `configs/1702.yaml` for the canonical reference config.

```
YAML → dacite → TrainConfig.build() → Trainer.train()
                     ├── ModelConfig.build()  → compiled tf.keras.Model
                     ├── DataConfig.build()   → ModelTrainingData (train/val/test tf.data.Datasets)
                     └── CallbacksConfig.build() → [Keras callbacks]
```

### Data pipeline (`src/fronts/data/`)

- **`era5.py`** — `ERA5Config` loads ERA5 from an ARCO zarr store on GCS (`gs://gcp-public-data-arco-era5/...`) or a local zarr cache. Handles variable subsetting, level stacking (surface + pressure levels), and derived variable computation (dewpoint, theta_e, etc.).
- **`targets.py`** — `TargetDataConfig` loads front label data from an icechunk store or NetCDF files. Applies front dilation (pixel expansion).
- **`config.py`** — `DataConfig` assembles ERA5 + targets into `tf.data.Dataset` splits by year. `PredictConfig` is the inference-time equivalent. `AugmentationConfig` handles runtime lat/lon flips with wind sign correction.
- **`batch.py`** — `create_dataloader()` wraps xbatcher `BatchGenerator` → `tf.data.Dataset`.

Input DataArray shape: `(time, latitude, longitude, level, variable)`. 3D inputs (multiple pressure levels) are squeezed to 2D targets inside the model's final layer.

### Model (`src/fronts/model.py`, `src/fronts/layers/`)

- `ModelConfig` / `Model` in `model.py` — builds and compiles any UNet variant via `UNetRegistry`.
- `src/fronts/layers/unets.py` — `UNetRegistry` dispatches to UNet, UNet+, UNet++, UNet3+, Attention UNet dataclass implementations.
- `src/fronts/layers/modules.py` — shared convolution/pooling/upsampling building blocks used by all UNet variants.
- `src/fronts/layers/activations.py` — 30 custom activation functions as Keras `Layer` subclasses.
- `src/fronts/layers/losses.py` — custom losses (e.g. Fractions Skill Score).
- `src/fronts/layers/metrics.py` — custom metrics (e.g. Critical Success Index).
- `src/fronts/utils/keras_builders.py` — config dataclasses for Keras objects (`OptimizerConfig`, `LossConfig`, `ActivationConfig`, etc.) that `.build()` into real Keras objects.

### Training (`src/fronts/train.py`)

`Trainer.train()` calls `model.fit()`. Supports `MirroredStrategy` for multi-GPU via `distribution: "mirrored"` in config. Deep supervision (multiple outputs) replicates targets to match output count. WandB integration wraps the fit loop in `wandb.init()`.

### Evaluation (`src/fronts/evaluation/`)

- `predict_tf.py` — generates prediction NetCDF files from a trained model + TF dataset.
- `generate_performance_stats.py` — computes CSI, POD, FAR, HSS over neighborhoods (50–250 km).
- `calibrate_model.py` — Platt scaling / isotonic calibration.
- `generate_permutations.py` — permutation importance.

## Tests

Tests mock TensorFlow, wandb, xbatcher, and geospatial libs (`conftest.py`) so the test suite runs without GPU or heavy data deps. The `test` pixi environment installs only pytest + dacite + numpy + pyyaml. Tests exercise the config ingestion pipeline (YAML → dataclass → `.build()` call paths).
