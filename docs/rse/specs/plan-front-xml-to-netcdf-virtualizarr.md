# Implementation Plan: Config-driven front XML→netCDF conversion with virtual icechunk registration

---
**Date:** 2026-09-03
**Author:** AI Assistant
**Status:** Complete (see [Implementation Summary](implement-front-xml-to-netcdf-virtualizarr.md) for the executed
result, including one dependency-resolution deviation from this plan)
**Related Documents:**
- [Research: Front XML to netCDF conversion with a config-driven, virtualizarr-backed icechunk store](research-front-xml-to-netcdf-virtualizarr.md)

---

## Overview

`master:convert_front_xml_to_netcdf.py` converts MPC/OPC surface-analysis front XML files
into per-timestep netCDF files, using CLI flags and hardcoded domain/grid tables. This plan
replaces it with a `feat/2.0.0`-style, YAML-config-driven module,
`src/fronts/data/generate_fronts.py`, that mirrors `src/fronts/data/generate.py`'s
inspect→diff→write shape: it lists which front XML files are new (by valid time), converts
only those to netCDF on the grid the ERA5 icechunk store already uses, and registers the new
netCDF files as **virtual chunks** (via VirtualiZarr) in the fronts icechunk store — a single
commit, no data copy — instead of writing real array data.

**Goal:** Running `python -m fronts.data.generate_fronts --config configs/generate_fronts.yaml`
against the 2025 XML directory converts every not-yet-registered XML file to netCDF and leaves
the fronts icechunk store's `time` coordinate covering every one of those valid times, with the
underlying `identifier` array readable through `utils.open_readonly_icechunk_store` exactly as
it is today for the pre-existing store content.

**Motivation:** The 2025 raw front XML files use a different filename convention than the
legacy script assumes (`YYYYMMDD_HHMM_cycle_MPC_final-anal_OPC_SFC_ANAL.xml`, valid to the
minute, not the top of the hour), the legacy script has no config file and no notion of "what's
already converted," and there is currently no code anywhere in `feat/2.0.0` for *writing*
virtual chunk references — only for reading them (`virtual_chunk_local_path` on
`utils.IcechunkStorageConfig`). The fronts store that `configs/schooner_train.yaml` and others
already point at needs a way to grow incrementally as new XML analyses arrive.

## Current State Analysis

**Existing Implementation:**
- `convert_front_xml_to_netcdf.py` (master, whole file) — CLI-only, single-date-at-a-time,
  processes only `files[-1:]`, depends on `utils/data_utils.py` (not on `feat/2.0.0`) for
  `haversine`/`reverse_haversine`/`geometric`/`redistribute_vertices`, and writes netCDF with
  `(longitude, latitude)` dim order and its own bespoke, non-ERA5-aligned `domain_coords`.
- `src/fronts/data/generate.py:428-473` (`main`) — the config-loading → inspect-store →
  diff → write pattern to mirror exactly.
- `src/fronts/data/generate.py:258-289` (`inspect_store`) and `:297-347`
  (`determine_write_strategy`) — the inspect/diff shape to adapt for a single-variable,
  time-only store.
- `src/fronts/utils.py:28-53` (`IcechunkStorageConfig`) — already has
  `virtual_chunk_local_path`, reused unchanged as the fronts store's config.
- `src/fronts/utils.py:432-465` (`get_icechunk_snapshot_id`) and `:468-513`
  (`open_readonly_icechunk_store`) — both duplicate an identical
  `VirtualChunkContainer`/`containers_credentials` setup block; no writable counterpart exists.
- `src/fronts/utils.py:138-151` (`attach_periodic_lon_index`), `:154-176`
  (`select_spatial_domain`), `:196-217` (`unwrap_longitude`) — the periodic-longitude
  machinery the ERA5 store's `"full"` domain already relies on; the new grid must reuse this
  directly rather than reimplementing domain math, so front and ERA5 coordinates match exactly.
- `configs/schooner_train.yaml:16-19` — the real fronts store this plan targets:
  `store_path: /ourdisk/hpc/ai2es/tman/restructured_front_data/icechunk`,
  `virtual_chunk_local_path: /ourdisk/hpc/ai2es/tman/restructured_front_data/netcdf/`,
  no `group_name` (root group).
- `pyproject.toml` — `virtualizarr` is not a dependency anywhere; `shapely`/`netcdf4` are
  `train`-feature-only; `data`-feature already has `icechunk`, `obstore` (pypi), `s3fs` (pypi).
- `tests/data/conftest.py`, `tests/data/test_generate.py` — real `tmp_path`-backed
  fixtures, no mocks; a static `tests/data/test_generate.yaml` fixture is used only to test
  `open_config_yaml_as_dataclass` parsing, not `main()` end-to-end.

**Current Behavior:** No config-driven, `feat/2.0.0`-native way exists to convert new front
XML files or to grow the fronts icechunk store; the only working conversion path is the
unconfigured `master` script producing files that must be added to the store by hand.

**Current Limitations:**
- Legacy script can't parse minute-precision valid times (2025 filenames need this).
- Legacy script's grid is not the ERA5-aligned periodic grid the rest of `feat/2.0.0` assumes.
- No "what's missing" check — every run is a manual, all-or-nothing conversion.
- No code path writes virtual chunk references into an icechunk store.

## Desired End State

**New Behavior:** `python -m fronts.data.generate_fronts --config
configs/generate_fronts.yaml` discovers front XML files in a configured directory and date
range, compares their valid times against what the target icechunk store already has, converts
only the missing ones to netCDF on the ERA5-aligned grid, and appends them as virtual chunk
references to the store in one commit. Running it twice in a row with no new XML files is a
no-op on the second run.

**Success Looks Like:**
- `src/fronts/data/generate_fronts.py` exists, is importable, and every public function has a
  Google-style docstring and a corresponding test.
- `configs/generate_fronts.yaml` is a working example config using the real paths given by the
  user (`xml_indir: /ourdisk/hpc/ai2es/fronts/raw_front_data/xml_tars/2025/xml`, matching the
  real `restructured_front_data` store paths already used by `configs/schooner_train.yaml`).
- A local round trip (synthetic XML → netCDF → virtual icechunk write → `xr.open_zarr` read
  back) reproduces the expected `identifier` values and time coordinate, both for the
  store-creation case and the append-to-existing-store case.
- `pixi run -e data generate-fronts` (and `pixi run -e data test`) succeed with the new
  dependencies resolved.

## What We're NOT Doing

- [ ] Porting `master`'s model-native domains (`hrrr`, `nam_12km`, `rap`, `namnest_conus`) —
      those require external GRIB template files not present anywhere on `feat/2.0.0` and
      aren't used by any current config; only the fixed-grid ERA5-aligned domain is supported.
- [ ] Building or migrating the *existing* `/ourdisk/hpc/ai2es/tman/restructured_front_data`
      store's historical contents — this plan only adds a way to grow it going forward.
- [ ] Deleting or deprecating `convert_front_xml_to_netcdf.py` on `master` — it is left as-is;
      this plan only adds the new script on a branch off `feat/2.0.0`.
- [ ] SLURM/dask-jobqueue parallelism for the conversion loop (unlike `generate.py`'s
      `--slurm` option) — front XML files are small (single-file XML parses), and each run's
      missing-file count is expected to be modest (new analyses arriving incrementally), so a
      plain sequential loop is sufficient.
- [ ] Handling XML `pgenType` values outside the 16 already listed in `master`'s
      `pgenType_identifiers` — an unrecognized type raises `ValueError` rather than silently
      being dropped or guessed at (see Edge Cases).

**Rationale:** Keeps this change scoped to what the user asked for — a working config-driven
converter plus store update for the real 2025 XML directory — without speculative
generalization to grids/products nothing currently uses.

## Implementation Approach

**Technical Strategy:** New module `src/fronts/data/generate_fronts.py`, structured exactly
like `src/fronts/data/generate.py` (module logger setup, dataclass config(s), pure
inspect/convert/write functions, thin `main()`). A new `configs/generate_fronts.yaml` mirrors
`configs/generate_icechunk.yaml`'s two-section shape. One small, justified refactor in
`src/fronts/utils.py` extracts the virtual-chunk-container setup (currently duplicated in two
read-only functions) into a shared helper, and adds one new writable counterpart function that
`generate_fronts.py` uses — introduced because this plan adds the third call site, not as
unrelated cleanup.

**Key Architectural Decisions:**

1. **Decision:** Reuse `utils.select_spatial_domain` (and `utils.unwrap_longitude`) to build
   the front raster grid, instead of porting `master`'s standalone `domain_coords` dict.
   - **Rationale:** The fronts store must be pixel-aligned with the ERA5 icechunk store for
     training (`src/fronts/train.py` merges inputs and targets by coordinate); reusing the
     exact function the ERA5 write path itself doesn't call but the *read* path
     (`select_spatial_domain`, used throughout `evaluate.py`/`plot.py`) relies on guarantees
     identical coordinate values, including the `"full"` domain's dateline-wrapping
     `[130...359.75, 0...9.75]` layout, with no hand-rolled equivalent to keep in sync.
   - **Trade-offs:** Couples `generate_fronts.py` to `utils.BoundingBox` instead of a
     `domain: "conus" | "full" | "global"` enum; this is *more* consistent with
     `ERA5DataLoaderConfig.coordinates`, which already takes a `BoundingBox`.
   - **Alternatives considered:** Porting `master`'s `domain_coords` dict verbatim — rejected
     because its longitude values (`-179.75..9.75` then `130..180`) don't match the ERA5
     store's native `0..359.75` convention at all, which would silently misalign fronts and
     ERA5 inputs during training.

2. **Decision:** Bucket front-line points onto the grid using a **single modular longitude
   shift**, `((lon - lon_min) % 360) + lon_min`, against `utils.unwrap_longitude`'s
   monotonic-ascending form of the grid (index-aligned with the raw, possibly wrap-crossing,
   output coordinate) — instead of `master`'s separate "does this line cross the dateline"
   special-casing combined with per-domain branching.
   - **Rationale:** `np.digitize` requires monotonic bins; the `"full"` domain's raw longitude
     coordinate (`[130...359.75, 0...9.75]`) is not monotonic, so digitizing against it
     directly (as a naive port of `master`'s code would) silently corrupts bucket indices at
     the wrap point. The modular shift is one formula that is correct for wrapping and
     non-wrapping domains alike, and is index-compatible with the raw output coordinate by
     construction (`unwrap_longitude` only changes values, never order/length).
   - **Trade-offs:** One extra `utils.unwrap_longitude` call per conversion; negligible cost
     relative to XML parsing.
   - **Alternatives considered:** Keeping `master`'s domain-specific branching — rejected as
     both more code and silently wrong for the wrap-crossing `"full"` domain.

3. **Decision:** Parse XML with `defusedxml.ElementTree` instead of the stdlib
   `xml.etree.ElementTree` `master`'s script uses.
   - **Rationale:** Stdlib XML parsers are vulnerable to XXE and billion-laughs
     entity-expansion attacks by default; `defusedxml` is a drop-in replacement with the same
     API that disables these by default, at negligible cost, for a script that will run
     unattended against files dropped into an HPC directory by an external ingest process.
   - **Trade-offs:** One additional lightweight dependency (`defusedxml`).
   - **Alternatives considered:** Keeping `xml.etree.ElementTree` (matching `master`) —
     rejected; the fix is free and the files aren't from a fully trusted, access-controlled
     source in the way in-repo config files are.

4. **Decision:** Convert all missing XML files to netCDF first, then issue **one**
   `open_virtual_mfdataset` + **one** `to_icechunk` + **one** `session.commit()` for the whole
   batch, rather than looping per-file commits.
   - **Rationale:** Matches this repo's established convention (single-commit icechunk
     writes; see `write_new_variables_to_icechunk_store` for the analogous non-batched write
     in `generate.py`) and avoids partial-store states between missing files that all belong
     to one logical "catch the store up" run.
   - **Trade-offs:** A run converting many files does all XML parsing before anything is
     committed — acceptable since front XML files are tiny and no dask/streaming step exists.
   - **Alternatives considered:** Per-file commits (`generate.py`'s `write_batch_size` for
     ERA5) — rejected; that batching exists there to bound *memory* for large eager ERA5
     writes, which doesn't apply to tiny virtual-chunk references.

**Patterns to Follow:**
- `src/fronts/data/generate.py:428-473` (`main`) — config load → inspect → diff → write.
- `src/fronts/data/generate.py:1-26` — module logger setup (`logger`, `StreamHandler`,
  `Formatter`), copied verbatim.
- `src/fronts/utils.py:396-416` (`open_config_yaml_as_dataclass`) and `:330-333`
  (`YAML_TYPE_HOOKS`) — YAML→dataclass loading, including the `BoundingBox`/`datetime` hooks.
- `tests/data/conftest.py` / `tests/data/test_generate.py` — real `tmp_path` fixtures, no
  mocks; static YAML fixture used only for config-parsing tests.

## Implementation Phases

### Phase 1: Grid alignment and front-line rasterization math

**Objective:** Pure functions (no XML, no I/O) that build the ERA5-aligned grid and rasterize
one front line's points onto it — the riskiest numerical logic, tested against known values
before anything else depends on it.

**Tasks:**

- [ ] **Write the failing test** for the ERA5-aligned grid helper.
  - File: `tests/data/test_generate_fronts.py` (new)

  ```python
  import numpy as np
  import pytest

  from fronts.data import generate_fronts
  from fronts.utils import BoundingBox


  def test_grid_coordinates_matches_era5_full_domain_shape():
      bb = BoundingBox(lat_min=0.25, lat_max=80.0, lon_min=130.0, lon_max=369.75)
      latitude, longitude, longitude_unwrapped = generate_fronts.grid_coordinates(bb)
      assert latitude.shape == (320,)
      assert longitude.shape == (960,)
      assert longitude[0] == pytest.approx(130.0)
      assert longitude[-1] == pytest.approx(9.75)  # wraps past 360, raw form
      assert np.all(np.diff(longitude_unwrapped) > 0)  # unwrapped form is monotonic
      assert longitude_unwrapped[-1] == pytest.approx(369.75)
  ```

- [ ] **Run it, watch it fail:**
  `pixi run -e data python -m pytest tests/data/test_generate_fronts.py::test_grid_coordinates_matches_era5_full_domain_shape -v`
  → expect FAIL (`ModuleNotFoundError`/`AttributeError`: `generate_fronts` doesn't exist yet)

- [ ] **Implement the minimal code** — create the module and grid helper.
  - File: `src/fronts/data/generate_fronts.py` (new)

  ```python
  """Convert MPC/OPC surface-analysis front XML files to netCDF and register the new
  files as virtual chunks in an icechunk store, without copying array data.
  """

  import numpy as np
  import xarray as xr

  from fronts import utils

  _NATIVE_GRID_RESOLUTION_DEG = 0.25


  def _native_grid_template() -> xr.Dataset:
      """Coordinate-only Dataset spanning the full 0.25-degree global ERA5-aligned grid."""
      longitude = np.round(np.arange(0.0, 360.0, _NATIVE_GRID_RESOLUTION_DEG), 2).astype("float32")
      latitude = np.round(
          np.arange(90.0, -90.0 - _NATIVE_GRID_RESOLUTION_DEG, -_NATIVE_GRID_RESOLUTION_DEG), 2
      ).astype("float32")
      return xr.Dataset(coords={"latitude": latitude, "longitude": longitude})


  def grid_coordinates(coordinates: utils.BoundingBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
      """Return the front-raster grid, aligned to the ERA5 icechunk store's coordinates.

      Args:
          coordinates: Spatial bounding box, same convention as
              ``ERA5DataLoaderConfig.coordinates``.

      Returns:
          ``(latitude, longitude, longitude_unwrapped)``: the output ``latitude``/``longitude``
          coordinate arrays (produced by ``utils.select_spatial_domain`` on the native grid, so
          they match the ERA5 store exactly, including any dateline-wrapping layout), plus
          ``longitude_unwrapped`` — the same coordinate made monotonically increasing via
          ``utils.unwrap_longitude`` (index-aligned with ``longitude``), for use as
          ``np.digitize`` bins.
      """
      cropped = utils.select_spatial_domain(_native_grid_template(), coordinates)
      unwrapped = utils.unwrap_longitude(cropped)
      return cropped["latitude"].values, cropped["longitude"].values, unwrapped["longitude"].values
  ```

- [ ] **Run it, watch it pass:**
  `pixi run -e data python -m pytest tests/data/test_generate_fronts.py::test_grid_coordinates_matches_era5_full_domain_shape -v`
  → expect PASS
- [ ] **Commit:** `git commit -m "feat: add ERA5-aligned front raster grid helper"`

- [ ] **Write the failing test** for the haversine round trip (known values, ported from
      `master`'s docstring examples as pinned reference values).
  - File: `tests/data/test_generate_fronts.py`

  ```python
  def test_haversine_known_value():
      x, y = generate_fronts._haversine(np.array([-95.0]), np.array([35.0]))
      assert x[0] == pytest.approx(-10077.330945462296)
      assert y[0] == pytest.approx(3892.875)


  def test_haversine_reverse_haversine_round_trip():
      lon, lat = np.array([-95.0, 10.0, 170.0]), np.array([35.0, 40.0, -20.0])
      x, y = generate_fronts._haversine(lon, lat)
      lon_out, lat_out = generate_fronts._reverse_haversine(x, y)
      np.testing.assert_allclose(lon_out, lon)
      np.testing.assert_allclose(lat_out, lat)
  ```

- [ ] **Run it, watch it fail:**
  `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -k haversine -v`
  → expect FAIL (`AttributeError`: no `_haversine`/`_reverse_haversine`)

- [ ] **Implement the minimal code:**
  - File: `src/fronts/data/generate_fronts.py`

  ```python
  _EARTH_CIRCUMFERENCE_KM = 40041  # average circumference of Earth in kilometers


  def _haversine(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
      """Transform lon/lat points (degrees) to an x/y Cartesian plane (kilometers)."""
      x = lon * _EARTH_CIRCUMFERENCE_KM * np.cos(lat * np.pi / 360) / 360
      y = lat * _EARTH_CIRCUMFERENCE_KM / 360
      return x, y


  def _reverse_haversine(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
      """Inverse of ``_haversine``: transform x/y kilometers back to lon/lat degrees."""
      lon = x * 360 / np.cos(y * np.pi / _EARTH_CIRCUMFERENCE_KM) / _EARTH_CIRCUMFERENCE_KM
      lat = y * 360 / _EARTH_CIRCUMFERENCE_KM
      return lon, lat
  ```

- [ ] **Run it, watch it pass:**
  `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -k haversine -v`
  → expect PASS
- [ ] **Commit:** `git commit -m "feat: port haversine/reverse_haversine coordinate transforms"`

- [ ] **Write the failing test** for vertex redistribution (a straight 2-point line of known
      length should interpolate to the expected vertex count).
  - File: `tests/data/test_generate_fronts.py`

  ```python
  from shapely.geometry import LineString


  def test_redistribute_vertices_even_spacing():
      line = LineString([(0.0, 0.0), (10.0, 0.0)])  # 10 km long
      out = generate_fronts._redistribute_vertices(line, distance=2.5)
      xs = [pt[0] for pt in out.coords]
      assert len(xs) == 5  # 10 / 2.5 + 1 vertices
      np.testing.assert_allclose(xs, [0.0, 2.5, 5.0, 7.5, 10.0])


  def test_redistribute_vertices_shorter_than_distance_returns_endpoints():
      line = LineString([(0.0, 0.0), (1.0, 0.0)])  # shorter than distance
      out = generate_fronts._redistribute_vertices(line, distance=5.0)
      assert len(out.coords) == 2
  ```

- [ ] **Run it, watch it fail:**
  `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -k redistribute -v`
  → expect FAIL (`AttributeError`: no `_redistribute_vertices`)

- [ ] **Implement the minimal code:**
  - File: `src/fronts/data/generate_fronts.py`

  ```python
  from shapely.geometry import LineString


  def _redistribute_vertices(linestring: LineString, distance: float) -> LineString:
      """Interpolate points along ``linestring`` at even ``distance``-km spacing.

      Args:
          linestring: Front line in x/y kilometers (see ``_haversine``).
          distance: Target spacing between interpolated vertices, in kilometers.

      Returns:
          A new LineString with vertices evenly spaced at (approximately) ``distance``.
      """
      num_vertices = max(round(linestring.length / distance), 1)
      return LineString(
          [linestring.interpolate(fraction, normalized=True) for fraction in np.linspace(0, 1, num_vertices + 1)]
      )
  ```

- [ ] **Run it, watch it pass:**
  `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -k redistribute -v`
  → expect PASS
- [ ] **Commit:** `git commit -m "feat: port front-line vertex redistribution"`

**Dependencies:** `shapely` importable (already a `train`-feature dep; added to `data`/`test`
features in Phase 5).

**Verification:**
- [ ] `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -v` → all Phase 1
      tests PASS.

### Phase 2: XML filename parsing and single-file conversion

**Objective:** Parse the 2025 XML filename convention and rasterize one XML file's `Line`
elements into a single-timestep `identifier` Dataset, using the Phase 1 grid.

**Tasks:**

- [ ] **Write the failing test** for filename parsing (the three example filenames from the
      user's directory).
  - File: `tests/data/test_generate_fronts.py`

  ```python
  import pandas as pd


  @pytest.mark.parametrize(
      ("filename", "expected"),
      [
          ("20250511_0345_00_MPC_final-anal_OPC_SFC_ANAL.xml", pd.Timestamp("2025-05-11T03:45")),
          ("20250815_1545_12_MPC_final-anal_OPC_SFC_ANAL.xml", pd.Timestamp("2025-08-15T15:45")),
          ("20251021_0945_06_MPC_final-anal_OPC_SFC_ANAL.xml", pd.Timestamp("2025-10-21T09:45")),
      ],
  )
  def test_parse_xml_valid_time(filename, expected):
      assert generate_fronts.parse_xml_valid_time(filename) == expected


  def test_parse_xml_valid_time_returns_none_for_unrecognized_filename():
      assert generate_fronts.parse_xml_valid_time("readme.txt") is None
  ```

- [ ] **Run it, watch it fail:**
  `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -k parse_xml_valid_time -v`
  → expect FAIL (`AttributeError`: no `parse_xml_valid_time`)

- [ ] **Implement the minimal code:**
  - File: `src/fronts/data/generate_fronts.py`

  ```python
  import datetime
  import re

  import pandas as pd

  _XML_FILENAME_PATTERN = re.compile(
      r"^(?P<date>\d{8})_(?P<time>\d{4})_(?P<cycle_hour>\d{2})_MPC_final-anal_OPC_SFC_ANAL\.xml$"
  )


  def parse_xml_valid_time(filename: str) -> pd.Timestamp | None:
      """Parse the analysis valid time from an MPC/OPC front XML filename.

      Args:
          filename: Basename of the XML file, e.g.
              ``"20250511_0345_00_MPC_final-anal_OPC_SFC_ANAL.xml"``.

      Returns:
          The valid time as a ``pandas.Timestamp``, or None if ``filename`` does not match the
          expected naming pattern.
      """
      match = _XML_FILENAME_PATTERN.match(filename)
      if match is None:
          return None
      return pd.Timestamp(datetime.datetime.strptime(match["date"] + match["time"], "%Y%m%d%H%M"))
  ```

- [ ] **Run it, watch it pass:**
  `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -k parse_xml_valid_time -v`
  → expect PASS
- [ ] **Commit:** `git commit -m "feat: parse front XML valid time from filename"`

- [ ] **Write the failing test** for XML→Dataset conversion, using a minimal synthetic XML
      fixture with one cold-front `Line` whose points fall on known grid cells.
  - File: `tests/data/conftest.py` (add fixture)

  ```python
  @pytest.fixture
  def cold_front_xml(tmp_path) -> pathlib.Path:
      xml = """<?xml version="1.0" encoding="utf-8"?>
      <Product>
        <Line pgenType="COLD_FRONT">
          <Point Lon="-100.0" Lat="40.0"/>
          <Point Lon="-99.0" Lat="40.0"/>
        </Line>
      </Product>
      """
      path = tmp_path / "20250511_0345_00_MPC_final-anal_OPC_SFC_ANAL.xml"
      path.write_text(xml)
      return path
  ```

  - File: `tests/data/test_generate_fronts.py`

  ```python
  def test_convert_xml_to_dataset_places_cold_front_code(cold_front_xml):
      bb = BoundingBox(lat_min=0.25, lat_max=80.0, lon_min=130.0, lon_max=369.75)
      valid_time = pd.Timestamp("2025-05-11T03:45")
      ds = generate_fronts.convert_xml_to_dataset(str(cold_front_xml), valid_time, bb, distance_km=25.0)
      assert ds["identifier"].dims == ("time", "latitude", "longitude")
      assert ds["time"].values[0] == valid_time.to_datetime64()
      assert (ds["identifier"].values == generate_fronts.PGEN_TYPE_IDENTIFIERS["COLD_FRONT"]).any()
      assert ds["identifier"].values.max() == generate_fronts.PGEN_TYPE_IDENTIFIERS["COLD_FRONT"]


  def test_convert_xml_to_dataset_raises_on_unknown_front_type(tmp_path):
      xml = """<?xml version="1.0" encoding="utf-8"?>
      <Product>
        <Line pgenType="NOT_A_REAL_TYPE">
          <Point Lon="-100.0" Lat="40.0"/>
          <Point Lon="-99.0" Lat="40.0"/>
        </Line>
      </Product>
      """
      path = tmp_path / "bad.xml"
      path.write_text(xml)
      bb = BoundingBox(lat_min=0.25, lat_max=80.0, lon_min=130.0, lon_max=369.75)
      with pytest.raises(ValueError, match="NOT_A_REAL_TYPE"):
          generate_fronts.convert_xml_to_dataset(str(path), pd.Timestamp("2025-01-01"), bb, distance_km=25.0)
  ```

- [ ] **Run it, watch it fail:**
  `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -k convert_xml_to_dataset -v`
  → expect FAIL (`AttributeError`: no `convert_xml_to_dataset`/`PGEN_TYPE_IDENTIFIERS`)

- [ ] **Implement the minimal code:**
  - File: `src/fronts/data/generate_fronts.py`

  ```python
  import defusedxml.ElementTree as ET

  PGEN_TYPE_IDENTIFIERS = {
      "COLD_FRONT": 1,
      "WARM_FRONT": 2,
      "STATIONARY_FRONT": 3,
      "OCCLUDED_FRONT": 4,
      "COLD_FRONT_FORM": 5,
      "WARM_FRONT_FORM": 6,
      "STATIONARY_FRONT_FORM": 7,
      "OCCLUDED_FRONT_FORM": 8,
      "COLD_FRONT_DISS": 9,
      "WARM_FRONT_DISS": 10,
      "STATIONARY_FRONT_DISS": 11,
      "OCCLUDED_FRONT_DISS": 12,
      "INSTABILITY": 13,
      "TROF": 14,
      "TROPICAL_TROF": 15,
      "DRY_LINE": 16,
  }


  def convert_xml_to_dataset(
      xml_path: str, valid_time: pd.Timestamp, coordinates: utils.BoundingBox, distance_km: float
  ) -> xr.Dataset:
      """Rasterize one front-analysis XML file onto the ERA5-aligned grid.

      Args:
          xml_path: Path to a single MPC/OPC front XML file.
          valid_time: Analysis valid time to assign to the output's ``time`` coordinate.
          coordinates: Spatial bounding box the output grid is cropped to.
          distance_km: Spacing, in kilometers, used to interpolate front-line vertices before
              bucketing them onto the grid.

      Returns:
          Single-timestep Dataset with a float32 ``identifier`` variable on dims
          ``(time, latitude, longitude)``, one code per pixel from ``PGEN_TYPE_IDENTIFIERS``
          (0 where no front is present).

      Raises:
          ValueError: If a ``Line`` element's ``pgenType`` attribute is not a recognized front
              type.
      """
      latitude, longitude, longitude_unwrapped = grid_coordinates(coordinates)
      identifier = np.zeros((len(latitude), len(longitude)), dtype=np.float32)

      root = ET.parse(xml_path, parser=ET.XMLParser(encoding="utf-8")).getroot()
      for line in root.iter("Line"):
          front_type = line.get("pgenType")
          if front_type not in PGEN_TYPE_IDENTIFIERS:
              raise ValueError(f"Unrecognized front type {front_type!r} in {xml_path}")

          points = [(float(point.get("Lon")), float(point.get("Lat"))) for point in line.iter("Point")]
          lons, lats = np.array([p[0] for p in points]), np.array([p[1] for p in points])

          crosses_dateline = np.max(np.abs(np.diff(lons))) > 180
          if crosses_dateline:
              lons = np.where(lons < 0, lons + 360, lons)

          x_km, y_km = _haversine(lons, lats)
          vertices = _redistribute_vertices(LineString(list(zip(x_km, y_km))), distance_km)
          x_new, y_new = np.array(vertices.xy)
          lon_new, lat_new = _reverse_haversine(x_new, y_new)

          lon_shifted = np.mod(lon_new - coordinates.lon_min, 360) + coordinates.lon_min
          lat_idx = np.digitize(lat_new, latitude)
          lon_idx = np.digitize(lon_shifted, longitude_unwrapped)
          grid_idx = np.unique(np.stack([lat_idx, lon_idx], axis=-1), axis=0)
          in_bounds = (grid_idx[:, 0] < len(latitude)) & (grid_idx[:, 1] < len(longitude))
          grid_idx = grid_idx[in_bounds]
          identifier[grid_idx[:, 0], grid_idx[:, 1]] = PGEN_TYPE_IDENTIFIERS[front_type]

      return xr.Dataset(
          {"identifier": (("time", "latitude", "longitude"), identifier[np.newaxis])},
          coords={"time": [valid_time], "latitude": latitude, "longitude": longitude},
      )
  ```

- [ ] **Run it, watch it pass:**
  `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -k convert_xml_to_dataset -v`
  → expect PASS
- [ ] **Commit:** `git commit -m "feat: rasterize front XML files onto the ERA5-aligned grid"`

**Dependencies:** Phase 1 (`grid_coordinates`, `_haversine`, `_reverse_haversine`,
`_redistribute_vertices`).

**Verification:**
- [ ] `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -v` → all Phase 1–2
      tests PASS.

### Phase 3: Config dataclass, XML discovery, and store-times inspection

**Objective:** The YAML-loadable config, a directory scan that finds new XML files by date
range, and a read-only "what times does the store already have" check — the diff logic that
decides what Phase 2's conversion needs to run on.

**Tasks:**

- [ ] **Write the failing test** for `FrontConversionConfig` YAML loading.
  - File: `tests/data/test_generate_fronts.yaml` (new)

  ```yaml
  front_conversion_config:
    xml_indir: "/tmp/test_xml"
    netcdf_outdir: "/tmp/test_netcdf"
    date_start: 2025-05-01T00:00:00
    date_end: 2025-05-31T23:59:00
    coordinates: [0.25, 80, 130, 369.75]
    distance: 1.0

  icechunk_storage_config:
    store_path: "/tmp/test_fronts_store"
    branch_name: "main"
    commit_message: "test commit"
    virtual_chunk_local_path: "/tmp/test_netcdf"
  ```

  - File: `tests/data/test_generate_fronts.py`

  ```python
  FRONTS_YAML_PATH = pathlib.Path(__file__).parent / "test_generate_fronts.yaml"


  def test_front_conversion_config_from_yaml():
      with open(FRONTS_YAML_PATH) as f:
          front_config = utils.parse_config_section(
              yaml.safe_load(f), generate_fronts.FrontConversionConfig, "front_conversion_config", utils.YAML_TYPE_HOOKS
          )
      assert front_config.xml_indir == "/tmp/test_xml"
      assert front_config.coordinates == BoundingBox(lat_min=0.25, lat_max=80.0, lon_min=130.0, lon_max=369.75)
      assert front_config.distance == 1.0
  ```

- [ ] **Run it, watch it fail:**
  `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -k config_from_yaml -v`
  → expect FAIL (`AttributeError`: no `FrontConversionConfig`)

- [ ] **Implement the minimal code:**
  - File: `src/fronts/data/generate_fronts.py`

  ```python
  import dataclasses


  @dataclasses.dataclass
  class FrontConversionConfig:
      """Configuration for converting MPC/OPC surface-analysis front XML files to netCDF.

      Attributes:
          xml_indir: Directory containing raw front XML files, named
              ``<YYYYMMDD>_<HHMM>_<cycle_hour>_MPC_final-anal_OPC_SFC_ANAL.xml``.
          netcdf_outdir: Directory to write converted front netCDF files into. Must equal the
              paired ``IcechunkStorageConfig.virtual_chunk_local_path`` so the icechunk store's
              virtual chunk references resolve to these files.
          date_start: Inclusive start of the XML valid-time range to convert.
          date_end: Inclusive end of the XML valid-time range to convert.
          coordinates: Spatial bounding box for the output grid, in the same convention as
              ``ERA5DataLoaderConfig.coordinates`` (so front and ERA5 data share coordinates).
          distance: Interpolation distance, in kilometers, used to redistribute front-line
              vertices before bucketing them onto the grid.
      """

      xml_indir: str
      netcdf_outdir: str
      date_start: datetime.datetime
      date_end: datetime.datetime
      coordinates: utils.BoundingBox
      distance: float
  ```

- [ ] **Run it, watch it pass:**
  `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -k config_from_yaml -v`
  → expect PASS
- [ ] **Commit:** `git commit -m "feat: add FrontConversionConfig YAML schema"`

- [ ] **Write the failing test** for XML discovery by date range.
  - File: `tests/data/test_generate_fronts.py`

  ```python
  @pytest.fixture
  def xml_dir(tmp_path) -> pathlib.Path:
      d = tmp_path / "xml"
      d.mkdir()
      for name in [
          "20250511_0345_00_MPC_final-anal_OPC_SFC_ANAL.xml",
          "20250815_1545_12_MPC_final-anal_OPC_SFC_ANAL.xml",
          "not_a_front_file.txt",
      ]:
          (d / name).write_text("<Product/>")
      return d


  def test_discover_xml_files_filters_by_date_range_and_pattern(xml_dir):
      found = generate_fronts.discover_xml_files(
          str(xml_dir), datetime.datetime(2025, 5, 1), datetime.datetime(2025, 5, 31)
      )
      assert list(found) == [pd.Timestamp("2025-05-11T03:45")]
      assert found[pd.Timestamp("2025-05-11T03:45")].endswith(
          "20250511_0345_00_MPC_final-anal_OPC_SFC_ANAL.xml"
      )
  ```

- [ ] **Run it, watch it fail:**
  `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -k discover_xml_files -v`
  → expect FAIL (`AttributeError`: no `discover_xml_files`)

- [ ] **Implement the minimal code:**
  - File: `src/fronts/data/generate_fronts.py`

  ```python
  import os


  def discover_xml_files(
      xml_indir: str, date_start: datetime.datetime, date_end: datetime.datetime
  ) -> dict[pd.Timestamp, str]:
      """List front XML files in ``xml_indir`` whose valid time falls within [date_start, date_end].

      Args:
          xml_indir: Directory containing raw front XML files.
          date_start: Inclusive start of the valid-time range.
          date_end: Inclusive end of the valid-time range.

      Returns:
          Mapping from valid time to the full path of its XML file, one entry per matching file
          found directly inside ``xml_indir``.
      """
      start, end = pd.Timestamp(date_start), pd.Timestamp(date_end)
      files: dict[pd.Timestamp, str] = {}
      for name in sorted(os.listdir(xml_indir)):
          valid_time = parse_xml_valid_time(name)
          if valid_time is not None and start <= valid_time <= end:
              files[valid_time] = os.path.join(xml_indir, name)
      return files
  ```

- [ ] **Run it, watch it pass:**
  `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -k discover_xml_files -v`
  → expect PASS
- [ ] **Commit:** `git commit -m "feat: discover front XML files by valid-time range"`

- [ ] **Write the failing test** for store-times inspection, on a real (empty and populated)
      local icechunk store.
  - File: `tests/data/test_generate_fronts.py`

  ```python
  @pytest.fixture
  def fronts_storage_config(tmp_path) -> utils.IcechunkStorageConfig:
      return utils.IcechunkStorageConfig(
          store_path=str(tmp_path / "fronts_store"),
          branch_name="main",
          commit_message="test commit",
          virtual_chunk_local_path=str(tmp_path / "netcdf") + "/",
      )


  def test_inspect_fronts_store_times_returns_none_for_nonexistent_store(fronts_storage_config):
      assert generate_fronts.inspect_fronts_store_times(fronts_storage_config) is None


  def test_inspect_fronts_store_times_returns_written_times(fronts_storage_config, tmp_path, cold_front_xml):
      netcdf_dir = pathlib.Path(fronts_storage_config.virtual_chunk_local_path)
      netcdf_dir.mkdir()
      bb = BoundingBox(lat_min=0.25, lat_max=80.0, lon_min=130.0, lon_max=369.75)
      valid_time = pd.Timestamp("2025-05-11T03:45")
      ds = generate_fronts.convert_xml_to_dataset(str(cold_front_xml), valid_time, bb, distance_km=25.0)
      netcdf_path = netcdf_dir / "FrontObjects_202505110345_full.nc"
      ds.to_netcdf(netcdf_path, engine="netcdf4", mode="w")

      generate_fronts.write_netcdfs_to_icechunk_store(fronts_storage_config, [str(netcdf_path)], append=False)

      times = generate_fronts.inspect_fronts_store_times(fronts_storage_config)
      assert times is not None
      assert list(times) == [valid_time.to_datetime64()]
  ```

  (This test drives Phase 4's `write_netcdfs_to_icechunk_store` too — written now so Phase 3's
  read side and Phase 4's write side are verified together via one real round trip.)

- [ ] **Run it, watch it fail:**
  `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -k inspect_fronts_store_times -v`
  → expect FAIL (`AttributeError`: no `inspect_fronts_store_times`/`write_netcdfs_to_icechunk_store`)

**Dependencies:** Phase 2 (`convert_xml_to_dataset`); Phase 4 must land before the second test
above can pass (noted here, resolved by sequencing Phase 4 immediately after).

**Verification:**
- [ ] `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -k "config_from_yaml or discover_xml_files" -v`
      → PASS (the `inspect_fronts_store_times` tests are completed in Phase 4).

### Phase 4: Writable virtual-chunk icechunk store

**Objective:** Add the missing writable-virtual-chunk-store capability: a small shared helper
in `src/fronts/utils.py` (removing the duplicated container-setup block across three call
sites) and the VirtualiZarr-based commit function in `generate_fronts.py`.

**Tasks:**

- [ ] **Write the failing test** for the new `utils.open_writable_icechunk_repo` helper.
  - File: `tests/test_utils.py`

  ```python
  def test_open_writable_icechunk_repo_creates_new_store(tmp_path):
      store_path = str(tmp_path / "store")
      repo = utils.open_writable_icechunk_repo(store_path)
      session = repo.writable_session("main")
      ds = xr.Dataset({"x": (("i",), np.array([1, 2, 3]))})
      icechunk.xarray.to_icechunk(ds, session, safe_chunks=False)
      session.commit("init")

      read_back = utils.open_readonly_icechunk_store(store_path, "main", chunks=None)
      np.testing.assert_array_equal(read_back["x"].values, [1, 2, 3])


  def test_open_writable_icechunk_repo_reopens_existing_store(tmp_path):
      store_path = str(tmp_path / "store")
      utils.open_writable_icechunk_repo(store_path)
      repo = utils.open_writable_icechunk_repo(store_path)  # second call must not error
      assert repo is not None
  ```

  (Add `import icechunk.xarray` and `import numpy as np` to `tests/test_utils.py`'s imports if
  not already present.)

- [ ] **Run it, watch it fail:**
  `pixi run -e data python -m pytest tests/test_utils.py -k open_writable_icechunk_repo -v`
  → expect FAIL (`AttributeError`: no `open_writable_icechunk_repo`)

- [ ] **Implement the minimal code** — extract the shared helper and add the writable
      function.
  - File: `src/fronts/utils.py:432-513` (replace the duplicated blocks)

  ```python
  def _configure_virtual_chunk_access(
      repo_config: ic.RepositoryConfig, virtual_chunk_local_path: str | None
  ) -> Any:
      """Register a VirtualChunkContainer for local netcdf-backed virtual chunks, if configured.

      Mutates ``repo_config`` in place when ``virtual_chunk_local_path`` is given.

      Args:
          repo_config: Repository config to attach the virtual chunk container to.
          virtual_chunk_local_path: Local directory containing netcdf files referenced by
              virtual chunks, or None for stores with no virtual chunks.

      Returns:
          The value to pass as ``authorize_virtual_chunk_access``, or None.
      """
      if virtual_chunk_local_path is None:
          return None
      url_prefix = f"file://{virtual_chunk_local_path}"
      repo_config.set_virtual_chunk_container(
          ic.VirtualChunkContainer(url_prefix=url_prefix, store=ic.local_filesystem_store(virtual_chunk_local_path))
      )
      return ic.containers_credentials({url_prefix: None})


  def get_icechunk_snapshot_id(
      store_path: str,
      branch: str,
      virtual_chunk_local_path: str | None = None,
  ) -> str:
      """Return the snapshot ID at the tip of a branch in an icechunk store.

      Args:
          store_path: Path to the icechunk store directory.
          branch: Branch name to read from.
          virtual_chunk_local_path: Local directory containing netcdf files referenced by
              virtual chunks. Leave None for stores with no virtual chunks.

      Returns:
          The snapshot ID string for the branch tip.
      """
      storage = ic.local_filesystem_storage(store_path)
      repo_config = ic.RepositoryConfig.default()
      authorize_virtual_chunk_access = _configure_virtual_chunk_access(repo_config, virtual_chunk_local_path)
      repo = ic.Repository.open(
          storage,
          config=repo_config,
          authorize_virtual_chunk_access=authorize_virtual_chunk_access,
      )
      session = repo.readonly_session(branch)
      return session.snapshot_id


  def open_readonly_icechunk_store(
      store_path: str,
      branch: str,
      group: str | None = None,
      zarr_format: int = 3,
      virtual_chunk_local_path: str | None = None,
      chunks: Any = "auto",
  ) -> xr.Dataset:
      """Open a local icechunk store in read-only mode and return it as an xarray datatype.

      Args:
          store_path: Path to the icechunk store directory.
          branch: Branch name to read from.
          group: Optional group name within the zarr store to open.
          zarr_format: Zarr format version to use when opening the store (default is 3).
          virtual_chunk_local_path: Local directory containing the netcdf files referenced by
              virtual chunks (e.g. ``/ourdisk/hpc/data/netcdf/``). When provided, registers a
              VirtualChunkContainer and authorizes access so those chunks can be fetched. Leave
              None for stores with no virtual chunks.
          chunks: Forwarded to ``xr.open_zarr``. ``"auto"`` (default) returns a dask-backed
              Dataset suitable for chunked reductions (e.g. normalization stats). ``None``
              returns a Dataset backed directly by the zarr store with no dask graph, for
              callers that only ever read small, explicit slices themselves.

      Returns:
          An xarray Dataset or DataArray containing the data from the icechunk store.
      """
      storage = ic.local_filesystem_storage(store_path)
      repo_config = ic.RepositoryConfig.default()
      authorize_virtual_chunk_access = _configure_virtual_chunk_access(repo_config, virtual_chunk_local_path)
      repo = ic.Repository.open(
          storage,
          config=repo_config,
          authorize_virtual_chunk_access=authorize_virtual_chunk_access,
      )
      session = repo.readonly_session(branch)
      return xr.open_zarr(session.store, group=group, zarr_format=zarr_format, consolidated=False, chunks=chunks)


  def open_writable_icechunk_repo(store_path: str, virtual_chunk_local_path: str | None = None) -> ic.Repository:
      """Open or create a local icechunk repository, registering virtual chunk access if configured.

      Args:
          store_path: Path to the icechunk store directory.
          virtual_chunk_local_path: Local directory containing netcdf files referenced by
              virtual chunks. Leave None for stores with no virtual chunks.

      Returns:
          An open (or newly created) icechunk Repository, ready for ``repo.writable_session(...)``.
      """
      storage = ic.local_filesystem_storage(store_path)
      repo_config = ic.RepositoryConfig.default()
      authorize_virtual_chunk_access = _configure_virtual_chunk_access(repo_config, virtual_chunk_local_path)
      return ic.Repository.open_or_create(
          storage, config=repo_config, authorize_virtual_chunk_access=authorize_virtual_chunk_access
      )
  ```

- [ ] **Run it, watch it pass:**
  `pixi run -e data python -m pytest tests/test_utils.py -k open_writable_icechunk_repo -v` →
  PASS, and `pixi run -e data python -m pytest tests/test_utils.py -v` → all pre-existing
  `utils` tests still PASS (regression check on the refactor).
- [ ] **Commit:** `git commit -m "refactor: extract virtual chunk container setup; add open_writable_icechunk_repo"`

- [ ] **Implement `write_netcdfs_to_icechunk_store`** (completes the Phase 3 round-trip test).
  - File: `src/fronts/data/generate_fronts.py`

  ```python
  from obspec_utils.registry import ObjectStoreRegistry
  from obstore.store import LocalStore
  from virtualizarr import open_virtual_mfdataset
  from virtualizarr.parsers import HDFParser


  def write_netcdfs_to_icechunk_store(
      icechunk_config: utils.IcechunkStorageConfig, netcdf_paths: list[str], append: bool
  ) -> None:
      """Register netCDF files as virtual chunks in the icechunk store, in a single commit.

      Args:
          icechunk_config: Configuration for the target icechunk store. Its
              ``virtual_chunk_local_path`` must be set and must be the directory
              ``netcdf_paths`` live in.
          netcdf_paths: Paths of the new front netCDF files to register, one time step each;
              concatenated along ``time`` in sorted order.
          append: True to append to an existing ``time`` dimension; False to create it (the
              store or group has no data yet).

      Raises:
          ValueError: If ``netcdf_paths`` is empty or ``icechunk_config.virtual_chunk_local_path``
              is not set.
      """
      if not netcdf_paths:
          raise ValueError("netcdf_paths must not be empty")
      if icechunk_config.virtual_chunk_local_path is None:
          raise ValueError("icechunk_config.virtual_chunk_local_path must be set")

      url_prefix = f"file://{icechunk_config.virtual_chunk_local_path}"
      registry = ObjectStoreRegistry({url_prefix: LocalStore()})
      urls = [f"file://{path}" for path in sorted(netcdf_paths)]
      virtual_ds = open_virtual_mfdataset(
          urls, registry=registry, parser=HDFParser(), concat_dim="time", coords="minimal", combine="nested"
      )

      repo = utils.open_writable_icechunk_repo(icechunk_config.store_path, icechunk_config.virtual_chunk_local_path)
      session = repo.writable_session(icechunk_config.branch_name)
      virtual_ds.virtualize.to_icechunk(
          session.store, group=icechunk_config.group_name, append_dim="time" if append else None
      )
      session.commit(icechunk_config.commit_message)
      logger.info(f"Committed {len(netcdf_paths)} new front netCDF file(s) to {icechunk_config.store_path}")
  ```

  - File: `src/fronts/data/generate_fronts.py` (add `inspect_fronts_store_times`, completing
    Phase 3)

  ```python
  import icechunk as ic
  import zarr.errors


  def inspect_fronts_store_times(icechunk_config: utils.IcechunkStorageConfig) -> pd.DatetimeIndex | None:
      """Return the time steps currently present in the fronts icechunk store.

      Args:
          icechunk_config: Configuration for the target icechunk store.

      Returns:
          DatetimeIndex of stored times, or None if the store (or its group) doesn't exist yet.
      """
      storage = ic.local_filesystem_storage(icechunk_config.store_path)
      if not ic.Repository.exists(storage):
          return None
      try:
          ds = utils.open_readonly_icechunk_store(
              icechunk_config.store_path,
              icechunk_config.branch_name,
              group=icechunk_config.group_name,
              zarr_format=icechunk_config.zarr_format,
              virtual_chunk_local_path=icechunk_config.virtual_chunk_local_path,
              chunks=None,
          )
      except (FileNotFoundError, KeyError, zarr.errors.GroupNotFoundError):
          return None
      if "time" not in ds.coords:
          return None
      return pd.DatetimeIndex(ds["time"].values)
  ```

- [ ] **Run it, watch it pass:**
  `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -v` → all Phase 1–4
  tests PASS, including `test_inspect_fronts_store_times_returns_written_times`.

- [ ] **Write the failing test** for the append case (second batch of XML files, existing
      store already has one time step).
  - File: `tests/data/test_generate_fronts.py`

  ```python
  def test_write_netcdfs_to_icechunk_store_append_increases_time_steps(fronts_storage_config, cold_front_xml):
      netcdf_dir = pathlib.Path(fronts_storage_config.virtual_chunk_local_path)
      netcdf_dir.mkdir()
      bb = BoundingBox(lat_min=0.25, lat_max=80.0, lon_min=130.0, lon_max=369.75)

      first_time = pd.Timestamp("2025-05-11T03:45")
      first_ds = generate_fronts.convert_xml_to_dataset(str(cold_front_xml), first_time, bb, distance_km=25.0)
      first_path = netcdf_dir / "FrontObjects_202505110345_full.nc"
      first_ds.to_netcdf(first_path, engine="netcdf4", mode="w")
      generate_fronts.write_netcdfs_to_icechunk_store(fronts_storage_config, [str(first_path)], append=False)

      second_time = pd.Timestamp("2025-05-11T09:45")
      second_ds = generate_fronts.convert_xml_to_dataset(str(cold_front_xml), second_time, bb, distance_km=25.0)
      second_path = netcdf_dir / "FrontObjects_202505110945_full.nc"
      second_ds.to_netcdf(second_path, engine="netcdf4", mode="w")
      generate_fronts.write_netcdfs_to_icechunk_store(fronts_storage_config, [str(second_path)], append=True)

      times = generate_fronts.inspect_fronts_store_times(fronts_storage_config)
      assert list(times) == [first_time.to_datetime64(), second_time.to_datetime64()]
  ```

- [ ] **Run it, watch it fail then pass:**
  `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -k append_increases_time_steps -v`
  → expect immediate PASS (implementation already complete from the prior tasks; this test
  documents and locks in the append behavior) — if it fails, fix `write_netcdfs_to_icechunk_store`
  before proceeding.
- [ ] **Commit:** `git commit -m "feat: append new front netCDFs as virtual chunks in a single commit"`

**Dependencies:** Phase 3 (`FrontConversionConfig`, `discover_xml_files`); `virtualizarr`,
`obstore`, `h5py`, `netcdf4` importable (added to `data`/`test` pixi features in Phase 5 — if
Phase 4 is executed before Phase 5's dependency changes land, install them ad hoc first:
`pixi add -f data virtualizarr h5py`).

**Verification:**
- [ ] `pixi run -e data python -m pytest tests/data/test_generate_fronts.py tests/test_utils.py -v`
      → all PASS.

### Phase 5: CLI entry point, dependencies, and example config

**Objective:** Wire everything into a runnable `main()`, add the new dependencies to
`pyproject.toml`, and ship the example config for the real 2025 XML directory.

**Tasks:**

- [ ] **Write the failing test** for the netcdf_outdir/virtual_chunk_local_path consistency
      check `main()` performs before doing any work.
  - File: `tests/data/test_generate_fronts.py`

  ```python
  def test_main_raises_when_netcdf_outdir_does_not_match_virtual_chunk_local_path(tmp_path, monkeypatch):
      config_path = tmp_path / "config.yaml"
      config_path.write_text(f"""
  front_conversion_config:
    xml_indir: "{tmp_path / "xml"}"
    netcdf_outdir: "{tmp_path / "netcdf_a"}"
    date_start: 2025-01-01T00:00:00
    date_end: 2025-12-31T23:59:00
    coordinates: [0.25, 80, 130, 369.75]
    distance: 1.0

  icechunk_storage_config:
    store_path: "{tmp_path / "store"}"
    branch_name: "main"
    virtual_chunk_local_path: "{tmp_path / "netcdf_b"}"
  """)
      monkeypatch.setattr("sys.argv", ["generate_fronts.py", "--config", str(config_path)])
      with pytest.raises(ValueError, match="virtual_chunk_local_path"):
          generate_fronts.main()
  ```

- [ ] **Run it, watch it fail:**
  `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -k main_raises -v`
  → expect FAIL (`AttributeError`: no `main`)

- [ ] **Implement the minimal code:**
  - File: `src/fronts/data/generate_fronts.py`

  ```python
  import argparse
  import logging
  import sys

  logger = logging.getLogger(__name__)
  logger.setLevel(logging.INFO)
  handler = logging.StreamHandler(sys.stdout)
  handler.setLevel(logging.DEBUG)
  formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
  handler.setFormatter(formatter)
  logger.addHandler(handler)


  def main() -> None:
      """Entry point: convert new front XML files to netCDF and register them in the icechunk store."""
      parser = argparse.ArgumentParser(description="Convert front XML files to netCDF and update the icechunk store")
      parser.add_argument("--config", type=str, required=True, help="Path to YAML front conversion config")
      args = parser.parse_args()

      front_config = utils.open_config_yaml_as_dataclass(
          args.config, FrontConversionConfig, config_key="front_conversion_config", type_hooks=utils.YAML_TYPE_HOOKS
      )
      icechunk_config = utils.open_config_yaml_as_dataclass(
          args.config, utils.IcechunkStorageConfig, config_key="icechunk_storage_config"
      )
      logger.info(f"Front conversion config loaded: {front_config}")
      logger.info(f"Icechunk storage config loaded: {icechunk_config}")

      if icechunk_config.virtual_chunk_local_path is None or os.path.normpath(
          front_config.netcdf_outdir
      ) != os.path.normpath(icechunk_config.virtual_chunk_local_path):
          raise ValueError(
              "icechunk_storage_config.virtual_chunk_local_path must be set and match "
              f"front_conversion_config.netcdf_outdir (got "
              f"{icechunk_config.virtual_chunk_local_path!r} vs {front_config.netcdf_outdir!r})"
          )

      available = discover_xml_files(front_config.xml_indir, front_config.date_start, front_config.date_end)
      existing_times = inspect_fronts_store_times(icechunk_config)
      existing_times_set = set(existing_times) if existing_times is not None else set()
      missing_times = sorted(t for t in available if t.to_datetime64() not in existing_times_set)

      if not missing_times:
          logger.info("All requested front XML files are already represented in the icechunk store.")
          return

      logger.info(f"Converting {len(missing_times)} new front XML file(s) to netCDF...")
      os.makedirs(front_config.netcdf_outdir, exist_ok=True)
      netcdf_paths = []
      for valid_time in missing_times:
          ds = convert_xml_to_dataset(
              available[valid_time], valid_time, front_config.coordinates, front_config.distance
          )
          netcdf_path = os.path.join(
              front_config.netcdf_outdir, f"FrontObjects_{valid_time.strftime('%Y%m%d%H%M')}_full.nc"
          )
          ds.to_netcdf(netcdf_path, engine="netcdf4", mode="w")
          netcdf_paths.append(netcdf_path)

      write_netcdfs_to_icechunk_store(icechunk_config, netcdf_paths, append=existing_times is not None)
      logger.info("Front XML to netCDF conversion and icechunk store update complete.")


  if __name__ == "__main__":
      main()
  ```

- [ ] **Run it, watch it pass:**
  `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -k main_raises -v` →
  PASS
- [ ] **Commit:** `git commit -m "feat: wire generate_fronts CLI entry point"`

- [ ] **Add dependencies** to `pyproject.toml`.
  - File: `pyproject.toml`, `[tool.pixi.feature.data.dependencies]` (add three lines):

  ```toml
  shapely = ">=2.0.1,<3"
  netcdf4 = ">=1.6.2,<2"
  h5py = ">=3.11.0"
  defusedxml = ">=0.7.1,<1"
  ```

  - File: `pyproject.toml`, `[tool.pixi.feature.data.pypi-dependencies]` (add one line):

  ```toml
  virtualizarr = ">=2.7,<3"
  ```

  - File: `pyproject.toml`, `[tool.pixi.feature.data.tasks]` (add one line):

  ```toml
  generate-fronts = "python -m fronts.data.generate_fronts --config configs/generate_fronts.yaml"
  ```

  - File: `pyproject.toml`, `[tool.pixi.feature.test.dependencies]` (add three lines; `h5py`
    already present):

  ```toml
  shapely = ">=2.0.1,<3"
  netcdf4 = ">=1.6.2,<2"
  defusedxml = ">=0.7.1,<1"
  ```

  - File: `pyproject.toml` (add new table after `[tool.pixi.feature.test.dependencies]`):

  ```toml
  [tool.pixi.feature.test.pypi-dependencies]
  virtualizarr = ">=2.7,<3"
  ```

- [ ] **Run it:** `pixi install -e data` → resolves without conflicts (exit 0).
- [ ] **Commit:** `git commit -m "build: add shapely, netcdf4, h5py, virtualizarr to data/test features"`

- [ ] **Add the example config** for the real 2025 XML directory.
  - File: `configs/generate_fronts.yaml` (new)

  ```yaml
  front_conversion_config:
    xml_indir: "/ourdisk/hpc/ai2es/fronts/raw_front_data/xml_tars/2025/xml"
    netcdf_outdir: "/ourdisk/hpc/ai2es/tman/restructured_front_data/netcdf/"
    date_start: 2025-01-01T00:00:00
    date_end: 2025-12-31T23:59:00
    coordinates: [0.25, 80, 130, 369.75]  # [lat_min, lat_max, lon_min, lon_max]
    distance: 1.0  # km, front-line vertex interpolation spacing

  icechunk_storage_config:
    store_path: "/ourdisk/hpc/ai2es/tman/restructured_front_data/icechunk"
    branch_name: "main"
    commit_message: "Add 2025 front netCDFs from MPC/OPC surface analyses"
    virtual_chunk_local_path: "/ourdisk/hpc/ai2es/tman/restructured_front_data/netcdf/"
  ```

- [ ] **Run it:**
  `pixi run -e data python -c "from fronts import utils; from fronts.data import generate_fronts; print(utils.open_config_yaml_as_dataclass('configs/generate_fronts.yaml', generate_fronts.FrontConversionConfig, config_key='front_conversion_config', type_hooks=utils.YAML_TYPE_HOOKS))"`
  → prints the loaded `FrontConversionConfig` with no error.
- [ ] **Commit:** `git commit -m "docs: add generate_fronts example config for the 2025 XML directory"`

**Dependencies:** Phases 1–4 complete.

**Verification:**
- [ ] `pixi run -e data test` → full suite passes, including all new `generate_fronts`/`utils`
      tests.
- [ ] `pixi run -e data generate-fronts --help`-equivalent smoke check:
      `pixi run -e data python -m fronts.data.generate_fronts --help` exits 0 and prints usage.

## Success Criteria

### Automated Verification

- [x] `pixi run -e data test` passes (full suite, including new tests) — 446 passed, 73
      skipped, 1 pre-existing unrelated failure (`test_3d_config_parses`, missing `wandb` in
      the `data` env, confirmed present on `feat/2.0.0` before this branch via `git stash`).
- [x] `pixi run -e data python -m pytest tests/data/test_generate_fronts.py -v` passes with no
      skips (20/20).
- [x] `pixi run -e data python -m pytest tests/test_utils.py -v` passes (refactor regression
      check; 38/38, including the 2 new `open_writable_icechunk_repo` tests).
- [x] `pixi run -e data ruff check src/fronts/data/generate_fronts.py src/fronts/utils.py`
      passes with no errors.
- [x] `pixi install -e data` exits 0 (dependency resolution succeeds) — required pinning
      `zarr>=3.1,<3.2` and `virtualizarr>=2.4,<2.7` instead of the plan's `virtualizarr>=2.7,<3`;
      see the Implementation Summary for why.
- [x] File `src/fronts/data/generate_fronts.py` exists and is importable:
      `pixi run -e data python -c "from fronts.data import generate_fronts"`.
- [x] File `configs/generate_fronts.yaml` exists and parses via
      `utils.open_config_yaml_as_dataclass` (see Phase 5 verification command).

### Manual Verification

- [ ] On the HPC system (where `/ourdisk` is mounted), run
      `pixi run -e data python -m fronts.data.generate_fronts --config configs/generate_fronts.yaml`
      against a handful of real 2025 XML files and confirm: (a) new netCDF files appear under
      `netcdf_outdir` matching the XML valid times, (b) `utils.open_readonly_icechunk_store`
      against the fronts store shows the new times, and (c) rerunning the same command logs
      "All requested front XML files are already represented" and makes no further writes.
- [ ] Open one converted netCDF and visually sanity-check the `identifier` raster against the
      source XML's front lines (e.g. plot with `cartopy`, compare against the MPC surface
      analysis chart for that valid time).
- [ ] Confirm `pgenType` values found in a real 2025 XML file are all covered by
      `PGEN_TYPE_IDENTIFIERS`; if not, extend the map (see Edge Cases) before running at scale.

## Testing Strategy

**Unit Test Coverage (summary, written in-phase):**
- [ ] Grid alignment (`grid_coordinates`) against the ERA5 `"full"` domain's known
      wrap-crossing shape.
- [ ] Coordinate transforms (`_haversine`, `_reverse_haversine`) against pinned reference
      values and round-trip invariants.
- [ ] Vertex redistribution (`_redistribute_vertices`) against known vertex counts/spacing.
- [ ] Filename parsing (`parse_xml_valid_time`) against the three real example filenames plus
      a non-matching filename.
- [ ] XML rasterization (`convert_xml_to_dataset`) against a synthetic single-front XML fixture
      and an unrecognized-front-type error case.
- [ ] Config YAML parsing (`FrontConversionConfig`).
- [ ] XML discovery by date range (`discover_xml_files`).
- [ ] Store inspection (`inspect_fronts_store_times`) on nonexistent and populated stores.
- [ ] Writable virtual-chunk store creation/reopening (`utils.open_writable_icechunk_repo`).
- [ ] Virtual-chunk write/append, single commit (`write_netcdfs_to_icechunk_store`), including
      the multi-batch append case.
- [ ] `main()`'s config-consistency guard.
- No mocks anywhere — every icechunk/netCDF interaction in these tests uses a real local store
  or file under `tmp_path`, matching `tests/data/conftest.py`'s existing convention.

**Integration Tests:**
- [ ] `test_inspect_fronts_store_times_returns_written_times` and
      `test_write_netcdfs_to_icechunk_store_append_increases_time_steps` (Phases 3–4) are the
      integration coverage: real XML → real netCDF → real icechunk write → real read-back,
      covering both store-creation and append paths end-to-end.

**Manual Testing:** See Manual Verification above — requires the HPC-mounted `/ourdisk` path,
not available in this sandbox.

**Test Data Requirements:**
- Synthetic XML fixtures (`cold_front_xml` and inline XML strings in Phase 2/5 tests) — no
  real XML files are available in this sandbox (`/ourdisk` is not mounted here), so filename
  parsing is verified against the three real example filenames given by the user, and rasterization
  correctness is verified against synthetic fixtures built to match the same `<Line
  pgenType="..."><Point Lon="..." Lat="..."/></Line>` schema `master`'s script already parses.

## Edge Cases and Error Handling

**Edge Cases:**
1. **Case:** An XML file's `Line` has a `pgenType` not in `PGEN_TYPE_IDENTIFIERS`.
   - **Expected Behavior (revised during manual verification on the HPC system — see
     [Implementation Summary](implement-front-xml-to-netcdf-virtualizarr.md)):** the real
     "final-anal" product's XML encodes the *entire* surface analysis — fronts, pressure
     centers, contours, text labels — and non-front features (`LINE_SOLID`, `DOUBLE_LINE`,
     `ZZZ_LINE`, ...) share the same `<Line pgenType="...">` element as real fronts. This plan
     originally assumed (based on `master`'s pre-filtered source files) that every `Line` was a
     front and specified raising `ValueError` on an unrecognized type; that assumption was
     wrong for the real 2025 files and raised on the very first real conversion attempt.
     `convert_xml_to_dataset` now silently skips any `Line` whose `pgenType` is not in
     `PGEN_TYPE_IDENTIFIERS`, rather than raising.
   - **Implementation:** `src/fronts/data/generate_fronts.py` (`convert_xml_to_dataset`,
     Phase 2).
2. **Case:** A front line's points straddle the antimeridian (e.g. crossing from 179°E to
   179°W).
   - **Expected Behavior:** The line is still interpolated as one continuous geometry (not
     torn into two disconnected segments) and its rasterized points land in the correct grid
     cells even where the output grid itself is discontinuous (the `"full"` domain wraps at
     `130°/9.75°`).
   - **Implementation:** Per-line dateline continuity fix (`crosses_dateline` branch) plus the
     modular longitude shift against `longitude_unwrapped` (Architectural Decision 2), both in
     `convert_xml_to_dataset`.
3. **Case:** A front line's redistributed points fall outside the configured `coordinates`
   bounding box (e.g. a front partly over open ocean south of `lat_min`).
   - **Expected Behavior:** Out-of-bounds points are dropped rather than raising or wrapping
     into the wrong row/column; the front is still represented by whichever of its points fall
     inside the domain.
   - **Implementation:** The `in_bounds` mask in `convert_xml_to_dataset` (Phase 2), matching
     `master`'s equivalent bounds-check.
4. **Case:** `discover_xml_files` finds zero files in the configured date range (e.g. a config
   pointed at an empty or wrong directory).
   - **Expected Behavior:** `main()` logs "All requested front XML files are already
     represented..." only when files were found but already stored; when zero files are found
     at all, `missing_times` is also empty, so the same no-op path is taken and logged —
     acceptable since both cases correctly result in "nothing to do," and `discover_xml_files`
     itself doesn't distinguish an empty directory from a wrong one (a config error here shows
     up as an unexpectedly-quiet run, not a crash).
5. **Case:** `netcdf_outdir` doesn't match `icechunk_storage_config.virtual_chunk_local_path`.
   - **Expected Behavior:** `main()` raises `ValueError` immediately, before any XML parsing or
     file I/O, naming both mismatched values.
   - **Implementation:** `main()`'s consistency check (Phase 5).

**Error Scenarios:**
1. **Error:** `xml.etree.ElementTree.ParseError` from a malformed/truncated XML file.
   - **Handling:** Not caught — propagates as-is, since a malformed source file is a data
     problem the operator must investigate, not a condition `generate_fronts.py` can safely
     paper over (matches `master`'s behavior, which also does not catch this).

## Documentation Updates

- [ ] Module docstring on `src/fronts/data/generate_fronts.py` (Phase 1) documents the
      script's purpose and the ERA5-grid-alignment design decision.
- [ ] `configs/generate_fronts.yaml` (Phase 5) is itself the primary usage documentation,
      following the existing convention (`configs/generate_icechunk.yaml` has no separate
      prose doc either).
- No README/top-level doc changes — out of scope per "What We're NOT Doing."

## Open Questions

None.

---

## References

**Research Documents:**
- [Research: Front XML to netCDF conversion with a config-driven, virtualizarr-backed icechunk store](research-front-xml-to-netcdf-virtualizarr.md)

**Files Analyzed:**
- `convert_front_xml_to_netcdf.py` (master)
- `src/fronts/data/generate.py`
- `src/fronts/utils.py`
- `src/fronts/data/config.py`
- `configs/generate_icechunk.yaml`, `configs/schooner_train.yaml`
- `tests/data/conftest.py`, `tests/data/test_generate.py`, `tests/data/test_generate.yaml`
- `pyproject.toml`

**External Documentation:**
- VirtualiZarr `2.7.3` API, verified via a local scratch-venv install and a real
  convert→write→read round trip (no stable public docs URL captured; see research doc).

---

## Review History

### Version 1.0 — 2026-09-03
- Initial plan created (Direct mode, per explicit user directive with research already done).

### Version 1.1 — 2026-09-03
- Executed. One deviation from the plan as written: `virtualizarr>=2.7,<3` (validated during
  research) turned out to be unsatisfiable against this repo's `numpy<2.1` pin (kept for
  TensorFlow compatibility). Resolved by pinning `virtualizarr>=2.4,<2.7` plus an explicit
  `zarr>=3.1,<3.2` (virtualizarr<2.7 reaches into a zarr-internal module path removed in zarr
  3.2), re-validated end-to-end against the actual resolved versions
  (`virtualizarr==2.4.0`, `zarr==3.1.6`, `icechunk==2.0.6`, `numpy==2.0.2`). A second
  real bug surfaced only under this deviation: appending across separate
  `write_netcdfs_to_icechunk_store` calls corrupted previously-written times, because the
  writer defaults to a batch-relative time encoding reference; fixed by pinning an explicit,
  batch-independent `time` encoding (`"minutes since 1970-01-01"`) before every write. See
  [Implementation Summary](implement-front-xml-to-netcdf-virtualizarr.md) for full detail.
