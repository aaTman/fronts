# Research: Front XML→NetCDF conversion with a config-driven, virtualizarr-backed icechunk store

**Date:** 2026-09-03
**Scope:** internal codebase
**Related Documents:** none

## Question / Scope

The user wants a new, config-yaml-driven script (branched off `feat/2.0.0`) that:
1. Converts MPC/OPC surface-analysis front XML files (e.g.
   `20250511_0345_00_MPC_final-anal_OPC_SFC_ANAL.xml`, at
   `/ourdisk/hpc/ai2es/fronts/raw_front_data/xml_tars/2025/xml`) to netCDF, replacing the
   legacy `master:convert_front_xml_to_netcdf.py` with something that reads its parameters
   from YAML and is easier to read/audit, matching `feat/2.0.0` conventions.
2. Checks whether the resulting netCDFs are already represented in the existing
   *virtual* icechunk store (virtual chunk references into netCDF files on disk, no data
   copy), and appends virtual references for any that are missing.

In scope: how master converts XML to netCDF, how `feat/2.0.0` structures config-driven
scripts and icechunk stores, whether virtual-chunk/virtualizarr machinery already exists,
and what the real filename convention/front types are for the 2025 raw XML files. Codebase
state: `feat/2.0.0` at `a85269c`, working branch at `61c0067`.

## Codebase Findings

### Legacy conversion script (`master`)

`master:convert_front_xml_to_netcdf.py` is a single flat script (argparse, no config file):
- Parses one `--date` (year, month, day) at a time, globs matching XML files, and only ever
  processes the **last** match (`files[-1:]`).
- Filename→time parsing assumes the *hour* is embedded at whole-hour precision
  (`date[-2:]`) — inadequate for the 2025 files, which encode minutes too (see below).
- Builds a plate-carrée or model-native (HRRR/NAM/RAP Lambert-conformal) target grid from
  hardcoded `domain_coords` dicts, buckets each front polyline's interpolated points onto
  that grid via `np.digitize`, and writes one `identifier` NetCDF per date/domain
  (`FrontObjects_<date>_<domain>.nc`).
- Depends on `utils/data_utils.py` (not present on `feat/2.0.0`) for the coordinate math:
  `haversine`/`reverse_haversine` (lon/lat ↔ km on a spherical approximation),
  `geometric`/`redistribute_vertices` (shapely `LineString` interpolation at a fixed
  spacing), and `lambert_conformal_to_cartesian` (only needed for the model-native domains,
  not `"full"`).
- `pgenType_identifiers` maps 16 raw front-type codes (COLD_FRONT, WARM_FRONT, …, DRY_LINE)
  to integer labels 1–16.

### `feat/2.0.0` config-yaml + icechunk conventions

- Config-driven scripts follow one shape: a `@dataclasses.dataclass` config per YAML
  top-level section (e.g. `ERA5DataLoaderConfig`, `utils.IcechunkStorageConfig`,
  `config.SlurmConfig`), loaded via `utils.open_config_yaml_as_dataclass(path, cls,
  config_key=...)` (`src/fronts/utils.py:396`), which supports `${var}` interpolation from
  top-level YAML scalars (`src/fronts/utils.py:336`) and `type_hooks` for non-primitive
  fields like `BoundingBox`/`datetime` (`src/fronts/utils.py:330`). Every dataclass field
  and the class itself carries a Google-style docstring.
- `src/fronts/data/generate.py` is the direct analogue for what's being asked: it inspects
  an icechunk store (`inspect_store`, line 258), diffs requested vs. present
  variables/times (`determine_write_strategy`, line 297), and only writes what's missing
  (`WriteStrategy.execute`, line 108). `main()` (line 428) wires config loading → inspect →
  diff → write, with a `--config` CLI arg and structured logging via a module-level
  `logger`.
- `configs/generate_icechunk.yaml` mirrors that dataclass structure 1:1 (top-level keys
  `era5_config:`, `icechunk_storage_config:`, `slurm_config:`), and its pixi task is
  `[tool.pixi.feature.data.tasks] generate = "python -m fronts.data.generate --config
  configs/generate_icechunk.yaml"` (`pyproject.toml`).
- `utils.IcechunkStorageConfig` (`src/fronts/utils.py:28`) already has a
  `virtual_chunk_local_path` field, documented as "If the store contains virtual chunks
  referencing local netcdf files, set this to the directory those files live in
  ... Leave None for stores with no virtual chunks." This field is threaded through
  `open_readonly_icechunk_store` (line 468) and `get_icechunk_snapshot_id` (line 432),
  which both register an `ic.VirtualChunkContainer` for `file://{virtual_chunk_local_path}`
  and call `ic.containers_credentials({url_prefix: None})` before opening the repo — but
  only for **reading**. No writable-virtual-chunk helper exists yet anywhere in the repo.
- The real fronts store is already configured and read from in multiple places
  (`src/fronts/train.py:143`, `src/fronts/evaluate.py:430`,
  `src/fronts/model_1702/run_eval.py:166`, `src/fronts/plot/plot.py:595`,
  `configs/schooner_train.yaml:17`):
  ```yaml
  targets_icechunk_config:
    store_path: "/ourdisk/hpc/ai2es/tman/restructured_front_data/icechunk"
    branch_name: "main"
    virtual_chunk_local_path: "/ourdisk/hpc/ai2es/tman/restructured_front_data/netcdf/"
  ```
  No `group_name` is set (root group). There is no committed script that built this store —
  it predates `feat/2.0.0`'s tracked history, or was built manually/out-of-repo.
- `virtualizarr` is **not** a dependency anywhere (`pyproject.toml`, `pixi.lock`) and is not
  importable in the project's pixi env. `icechunk` is (`>=2`, resolved to `2.0.6` in
  `default`). `shapely` and `netcdf4` are `train`-feature deps only, not `data`-feature —
  the new script will need both plus `virtualizarr` (and its `h5py` HDF parser dependency)
  added to the `data` feature.
- Test conventions (`tests/data/test_generate.py`, `tests/data/conftest.py`): real
  `tmp_path`-backed fixtures, no mocking — an `era5_zarr`/`storage_config` fixture pair
  builds real local zarr/icechunk stores and a `populated_store` fixture round-trips
  through the actual `write_or_append_icechunk_store`. A parallel YAML fixture
  (`tests/data/test_generate.yaml`) exercises `main()`'s config-loading path end-to-end.

### Real 2025 XML filenames and front types

The sample filenames given —
`20250511_0345_00_MPC_final-anal_OPC_SFC_ANAL.xml`,
`20250815_1545_12_MPC_final-anal_OPC_SFC_ANAL.xml`,
`20251021_0945_06_MPC_final-anal_OPC_SFC_ANAL.xml` — differ structurally from anything
`master`'s script or `pgenType_identifiers` filename-globbing assumes (`pres*_...f000.xml`,
`IBM*_...`). Their fields are `<YYYYMMDD>_<HHMM>_<cycle-hour>_MPC_final-anal_OPC_SFC_ANAL.xml`:
date and **valid time to the minute** are in fields 1–2 (e.g. `03:45`, `15:45`, `09:45` UTC —
not top-of-hour), and field 3 (`00`/`12`/`06`) is the analysis cycle hour, not a forecast
hour. `/ourdisk` is not mounted in this sandbox, so the actual XML contents (available
`pgenType` codes) could not be inspected directly; the existing 16-entry
`pgenType_identifiers` map is assumed to still cover MPC/OPC surface analysis front types
but should be validated against a real file when run on the HPC system.

### Validated VirtualiZarr + icechunk write path

`virtualizarr` (PyPI, current `2.7.3`) was installed in a scratch venv alongside `icechunk`,
`h5py`, and `netcdf4` to validate the exact API needed (not previously used anywhere in this
repo), since `pip`/no network access inside the actual pixi env made this otherwise
unverifiable:
- `virtualizarr.open_virtual_mfdataset(urls, registry, parser=HDFParser(), concat_dim="time",
  coords="minimal", combine="nested")` opens multiple netCDF files as one lazy
  `ManifestArray`-backed `xr.Dataset`, given an `obspec_utils.registry.ObjectStoreRegistry`
  mapping a `file://.../` URL prefix to an `obstore.store.LocalStore()`.
- `vds.virtualize.to_icechunk(session.store, group=..., append_dim="time")` (the
  `VirtualiZarrDatasetAccessor`) writes those virtual references into an icechunk session;
  `mode=None` defaults to create-if-absent, and `append_dim` appends along an existing
  dimension — this is the write-side counterpart `generate.py`'s
  `write_or_append_icechunk_store` doesn't have, since that function only ever loads real
  data.
- The icechunk-side `url_prefix` **must** end in `/` (confirmed via a runtime `ValueError`)
  — matching the existing `virtual_chunk_local_path` convention's trailing slash in
  `configs/schooner_train.yaml` etc.
- End-to-end round trip (write two synthetic front-shaped netCDFs as virtual chunks, commit,
  reopen with `xr.open_zarr` through a readonly session) reproduced values correctly.
- `ic.containers_credentials({url_prefix: None})` — the exact pattern already used in
  `src/fronts/utils.py` — still works on `icechunk==2.2.0` but now emits a
  `DeprecationWarning` recommending `ic.credentials.LocalFileSystemAccess`; not present with
  the repo's pinned `icechunk==2.0.6`. Not itself in scope to fix (pre-existing helper), but
  worth flagging.

## Synthesis

There is no existing writable-virtual-chunk helper in `feat/2.0.0`; only the read side
(`open_readonly_icechunk_store`) and the field documenting the convention
(`virtual_chunk_local_path`) exist. The new script is genuinely new functionality, not a
refactor of something already there, and needs to introduce `virtualizarr` as a dependency
for the first time. The natural shape — closely mirroring `src/fronts/data/generate.py` —
is: an XML-conversion config dataclass (xml dir, netcdf out dir, date range/domain,
interpolation distance) + the existing `utils.IcechunkStorageConfig` (already has
`virtual_chunk_local_path`) for the target store, an `inspect_store`-style function that
lists which dates the icechunk store currently has, a diff against XML files found on disk
for the requested range, XML→netCDF conversion only for the missing dates (porting the
`haversine`/`geometric`/`redistribute_vertices` math since it doesn't exist on
`feat/2.0.0`), and a virtualizarr-based append for the resulting new netCDF files. Open
question for planning: whether interpolation/grid math should live in a new
`src/fronts/data/*.py` module (parallel to `generate.py`) or as a `scripts/*.py` (like
`rechunk_era5_store.py`), and exact config schema/field names.

## References / Sources

- Code: `convert_front_xml_to_netcdf.py` (master, full file)
- Code: `src/fronts/data/generate.py:1-428` (feat/2.0.0)
- Code: `src/fronts/utils.py:28-54,330-350,396-416,432-465,468-513` (feat/2.0.0)
- Code: `configs/generate_icechunk.yaml`, `configs/schooner_train.yaml:16-19` (feat/2.0.0)
- Code: `tests/data/test_generate.py`, `tests/data/conftest.py` (feat/2.0.0)
- Code: `pyproject.toml` (feat/2.0.0, dependency/feature-group layout)
- External: VirtualiZarr `2.7.3` installed API, inspected via
  `python -c "import inspect; ..."` against `virtualizarr.open_virtual_mfdataset`,
  `virtualizarr.VirtualiZarrDatasetAccessor.to_icechunk`,
  `virtualizarr.parsers.HDFParser`, `obspec_utils.registry.ObjectStoreRegistry` — no public
  docs URL captured, verified by running a real conversion+write+read round trip locally.
