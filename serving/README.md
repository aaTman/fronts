# frontfinder serving (fronts.espr.ai)

Batch inference + zarr serving pipeline: runs two Keras front-detection
models (`_best_loss.keras`, `model_1702.h5`) against ECMWF IFS open-data
global 0.25deg fields, writes cold/warm/occluded/stationary front
probabilities to a GeoZarr pyramid (via `topozarr`), and serves them to a
maplibre-gl + `@carbonplan/zarr-layer` viewer at fronts.espr.ai, hosted on
mandelhub via Proxmox.

## Architecture decision log (read this first)

Before any code was written, mandelhub's spec (i7-4770, 4c/8t, 16GB->32GB
RAM, no GPU, 8gbps internet) was checked against this workload. Verdict:
buildable, contingent on three things Taylor confirmed:

1. **Scheduled batch, not on-demand.** Inference runs on the IFS cycle
   cadence (00/06/12/18Z), not per user request -- CPU-only inference has
   hours of slack, not milliseconds.
2. **Most/all of mandelhub's CPU+RAM dedicated to this VM/CT.**
3. **Plenty of disk (500GB+)** for GRIB inputs + pyramid storage.

Two things surfaced during config review that change the shape of this
system and are worth restating here:

- Both training configs (`sooner_ablations.yaml`, `generate_conus.yaml`)
  use a **CONUS bounding box** (`[25.0, 56.75, 228.0, 299.75]`), not
  global. Taylor explicitly chose **true global inference** anyway --
  meaning both models are being run well outside the domain they were
  validated on. This is a modeling risk, not an engineering one: expect to
  sanity-check output quality outside CONUS before trusting it, especially
  near the poles and over open ocean far from any training analog.
- `_best_loss.keras`'s pressure levels weren't in the config
  (`n_channels: 30` implied 6 levels but didn't list them) -- Taylor
  confirmed `[1000, 925, 850, 700, 500, 300]`.

## Why no GPU is fine here, and where the CPU constraint actually shows up

`frontfinder/inference/tiling.py` + `engine.py` never run one whole-globe
forward pass. The grid is split into overlapping, 16-divisible patches
(the models' architecture constraint), inference runs patch-by-patch, and
results are blended back together -- bounding peak RAM regardless of how
large the global grid is, at the cost of some wall-clock time that's fine
given the multi-hour window between IFS cycles.

## Package layout

```
frontfinder/
  config/manifests.py     # per-model variable/pressure-level manifests (TESTED)
  ingest/derive.py         # theta-e (Bolton 1980) + isobaric PV formulas (TESTED)
  ingest/ecmwf_ifs.py       # IFS field source + model-input assembly (TESTED via fake source;
                             # EcmwfOpenDataSource itself is untested, network-only)
  inference/tiling.py       # patch generation + blended stitching (TESTED)
  inference/engine.py       # tiled inference runner (TESTED via fake predictor;
                             # KerasPredictor itself is untested, needs real weights)
  zarrio/pyramid.py         # topozarr-based GeoZarr pyramid builder (TESTED, real topozarr)
  scheduler/run_cycle.py    # per-cycle, per-model orchestration + latest.json (TESTED end-to-end
                             # with fakes)
  scheduler/cli.py          # systemd timer entrypoint (untested wiring; smoke-test on mandelhub)
webapp/
  index.html                # maplibre-gl + @carbonplan/zarr-layer viewer, model-swap toggle
deploy/
  Caddyfile, systemd/*      # Proxmox VM deployment config
```

## Setup

Package management is [`uv`](https://docs.astral.sh/uv/), driven entirely by
`pyproject.toml` + `uv.lock` (no `requirements.txt`). From this directory:

```
uv run --group dev pytest -q   # installs into .venv on first run, then tests
uv run python -m frontfinder.scheduler.cli --model-dir ... --output-root ...
```

`uv lock` regenerates `uv.lock` after editing dependencies in
`pyproject.toml`; commit the updated lockfile alongside. `requires-python =
">=3.11"` is set by `topozarr==0.0.4`'s own requirement, not an arbitrary
choice -- `uv lock` caught this when it was first set to `>=3.10`.

67 tests, all green, `uv run --group dev pytest -q` from this directory. TDD was used
throughout: tiling, the derived-variable formulas, manifest validation, and
the full assembly/inference/pyramid/scheduler chain are all exercised
against fakes before touching real models or real network calls -- the only
things NOT covered by the test suite are the two integration points this
sandbox genuinely cannot exercise: `EcmwfOpenDataSource` (needs live
network + `cfgrib`/`eccodes`) and `KerasPredictor` (needs the real
`.keras`/`.h5` weight files and TensorFlow). Both are thin, isolated
adapters specifically so the untested surface area is as small as possible.

## What's NOT done yet / needs your input before this goes live

1. **Smoke-test `EcmwfOpenDataSource` for real.** It's written against my
   best understanding of the `ecmwf-opendata` client API and IFS open-data
   GRIB shortnames -- I have not run it against a live ECMWF request. Some
   parameters (e.g. whether `q` is published at all 6 of best_loss's
   pressure levels in the 0.25deg open-data feed) need verification against
   https://www.ecmwf.int/en/forecasts/datasets/open-data.
2. **Verify the `@carbonplan/zarr-layer` frontend against a real pyramid.**
   See `webapp/README.md` -- the constructor options came from its README,
   not from running it in a browser.
3. **theta-e and PV are approximations**, not whatever exact ERA5-derived
   fields/pipeline the model actually saw in training (see
   `ingest/derive.py` docstring). If validation accuracy looks off, this is
   the first place to check.
4. **32GB RAM upgrade** on mandelhub -- not a blocker, but recommended
   before this runs unattended in production.
5. Model weight files (`_best_loss.keras`, `model_1702.h5`) need to actually
   land on the Proxmox VM at the path `scheduler/cli.py --model-dir` points
   to -- not included in this repo.
