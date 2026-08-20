"""ECMWF IFS open-data source abstraction + model-input assembly.

Design goal: the assembly logic (mapping a ModelManifest's variables/levels
to fetch calls, deriving theta-e and PV where needed, stacking channels in
manifest order) is pure and unit-testable against a fake data source. The
real network-backed source (`EcmwfOpenDataSource`) is a thin adapter around
the `ecmwf-opendata` client and is not exercised by unit tests -- it needs
an integration/smoke test run on the Proxmox VM where network access and
the real package are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np

from frontfinder.config.manifests import ModelManifest, VariableSpec
from frontfinder.ingest.derive import equivalent_potential_temperature, potential_vorticity_isobaric, potential_temperature

# Native IFS open-data base variables directly available at pressure levels
# or single-level, keyed by the ERA5-style name used in the model configs.
DIRECT_PRESSURE_LEVEL_VARIABLES = {
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
}
DIRECT_SINGLE_LEVEL_VARIABLES = {
    "surface_pressure",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
}
# Variables the model configs ask for that IFS open-data doesn't publish
# directly -- computed in `derive.py` from the direct variables above.
DERIVED_VARIABLES = {"equivalent_potential_temperature", "potential_vorticity"}

# ERA5-style variable name -> IFS open-data GRIB shortname, for the real
# network client (see EcmwfOpenDataSource).
ERA5_NAME_TO_IFS_SHORTNAME = {
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
    "surface_pressure": "sp",
    "2m_temperature": "2t",
    "2m_dewpoint_temperature": "2d",
    "10m_u_component_of_wind": "10u",
    "10m_v_component_of_wind": "10v",
}


@dataclass(frozen=True)
class IFSCycle:
    """One IFS operational forecast cycle, e.g. 2026-08-19 12Z."""

    date: str  # "YYYY-MM-DD"
    run_hour: int  # one of 0, 6, 12, 18
    step: int = 0  # forecast lead time in hours; 0 == analysis/T+0

    def __post_init__(self) -> None:
        if self.run_hour not in (0, 6, 12, 18):
            raise ValueError(f"run_hour must be one of 0/6/12/18, got {self.run_hour}")
        # raises ValueError if malformed
        datetime.strptime(self.date, "%Y-%m-%d")


class IFSFieldSource(Protocol):
    """Anything that can hand back IFS fields on the global 0.25deg grid."""

    @property
    def lat(self) -> np.ndarray: ...

    @property
    def lon(self) -> np.ndarray: ...

    def fetch_pressure_level(self, variable: str, level_hpa: int, cycle: IFSCycle) -> np.ndarray: ...

    def fetch_single_level(self, variable: str, cycle: IFSCycle) -> np.ndarray: ...


class FakeIFSFieldSource:
    """Deterministic synthetic field source for tests -- no network."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray, seed: int = 0):
        self._lat = lat
        self._lon = lon
        self._rng = np.random.default_rng(seed)
        self._cache: dict[tuple, np.ndarray] = {}

    @property
    def lat(self) -> np.ndarray:
        return self._lat

    @property
    def lon(self) -> np.ndarray:
        return self._lon

    def _grid(self, base: float, spread: float) -> np.ndarray:
        return base + spread * self._rng.standard_normal((len(self._lat), len(self._lon)))

    def fetch_pressure_level(self, variable: str, level_hpa: int, cycle: IFSCycle) -> np.ndarray:
        key = ("pl", variable, level_hpa, cycle.date, cycle.run_hour)
        if key not in self._cache:
            if variable == "temperature":
                base = 288.0 - 0.05 * (1000 - level_hpa)  # cooler aloft
                self._cache[key] = self._grid(base, 5.0)
            elif variable == "geopotential":
                self._cache[key] = self._grid(9.8 * (44330.0 * (1 - (level_hpa / 1013.25) ** 0.1903)), 50.0)
            elif variable == "specific_humidity":
                self._cache[key] = np.clip(self._grid(0.005, 0.002), 1e-6, 0.03)
            else:
                self._cache[key] = self._grid(0.0, 5.0)
        return self._cache[key]

    def fetch_single_level(self, variable: str, cycle: IFSCycle) -> np.ndarray:
        key = ("sl", variable, cycle.date, cycle.run_hour)
        if key not in self._cache:
            if variable == "surface_pressure":
                self._cache[key] = self._grid(101325.0, 500.0)
            elif "temperature" in variable or "dewpoint" in variable:
                self._cache[key] = self._grid(288.0, 5.0)
            else:
                self._cache[key] = self._grid(0.0, 5.0)
        return self._cache[key]


class EcmwfOpenDataSource:
    """Real network-backed IFSFieldSource, using the `ecmwf-opendata` client
    to pull the global 0.25deg operational IFS grid.

    NOT covered by unit tests -- needs `ecmwf-opendata`, `cfgrib`/`eccodes`,
    and live network access, none of which are available in this sandbox.
    Treat this class as a first-pass implementation to smoke-test on the
    Proxmox VM before it runs unattended: verify each shortname in
    `ERA5_NAME_TO_IFS_SHORTNAME` actually resolves against a real IFS
    open-data request (some fields, e.g. `q` on pressure levels, are only
    published on a subset of levels for the 0.25deg open-data feed -- check
    against https://www.ecmwf.int/en/forecasts/datasets/open-data before
    relying on 300hPa specific humidity being present).
    """

    def __init__(self, cache_dir: str = "/tmp/frontfinder_ifs_cache"):
        import os

        os.makedirs(cache_dir, exist_ok=True)
        self._cache_dir = cache_dir
        self._client = None  # lazy: constructed on first fetch, see _get_client
        self._lat = np.linspace(90.0, -90.0, 721)  # IFS open-data 0.25deg grid, N->S
        self._lon = np.linspace(0.0, 359.75, 1440)
        self._field_cache: dict[tuple, np.ndarray] = {}

    def _get_client(self):
        if self._client is None:
            from ecmwf.opendata import Client  # local import: not a unit-test dependency

            self._client = Client(source="ecmwf")
        return self._client

    @property
    def lat(self) -> np.ndarray:
        return self._lat

    @property
    def lon(self) -> np.ndarray:
        return self._lon

    def _fetch_grib(self, param: str, cycle: IFSCycle, levelist: list[int] | None) -> "xr.Dataset":
        import os

        import xarray as xr

        key = (param, tuple(levelist or ()), cycle.date, cycle.run_hour, cycle.step)
        target = os.path.join(
            self._cache_dir,
            f"{cycle.date}_{cycle.run_hour:02d}z_{param}_{'-'.join(map(str, levelist or []))}.grib2",
        )
        if not os.path.exists(target):
            request = dict(
                date=cycle.date.replace("-", ""),
                time=cycle.run_hour,
                step=cycle.step,
                stream="oper",
                type="fc" if cycle.step > 0 else "an",
                param=param,
                target=target,
            )
            if levelist:
                request["levelist"] = levelist
            self._get_client().retrieve(**request)
        return xr.open_dataset(target, engine="cfgrib")

    def fetch_pressure_level(self, variable: str, level_hpa: int, cycle: IFSCycle) -> np.ndarray:
        key = ("pl", variable, level_hpa, cycle.date, cycle.run_hour, cycle.step)
        if key not in self._field_cache:
            shortname = ERA5_NAME_TO_IFS_SHORTNAME[variable]
            ds = self._fetch_grib(shortname, cycle, levelist=[level_hpa])
            data_var = next(iter(ds.data_vars))
            self._field_cache[key] = ds[data_var].sel(isobaricInhPa=level_hpa).values
        return self._field_cache[key]

    def fetch_single_level(self, variable: str, cycle: IFSCycle) -> np.ndarray:
        key = ("sl", variable, cycle.date, cycle.run_hour, cycle.step)
        if key not in self._field_cache:
            shortname = ERA5_NAME_TO_IFS_SHORTNAME[variable]
            ds = self._fetch_grib(shortname, cycle, levelist=None)
            data_var = next(iter(ds.data_vars))
            self._field_cache[key] = ds[data_var].values
        return self._field_cache[key]


def _resolve_direct_variable(
    var: VariableSpec, source: IFSFieldSource, cycle: IFSCycle
) -> list[np.ndarray]:
    if var.levels is None:
        return [source.fetch_single_level(var.name, cycle)]
    return [source.fetch_pressure_level(var.name, lvl, cycle) for lvl in var.levels]


def _resolve_derived_variable(
    var: VariableSpec, source: IFSFieldSource, cycle: IFSCycle
) -> list[np.ndarray]:
    if var.name == "equivalent_potential_temperature":
        if var.levels is None:
            raise ValueError("equivalent_potential_temperature requires pressure levels")
        fields = []
        for lvl in var.levels:
            t = source.fetch_pressure_level("temperature", lvl, cycle)
            q = source.fetch_pressure_level("specific_humidity", lvl, cycle)
            fields.append(equivalent_potential_temperature(t, q, pressure_hpa=lvl))
        return fields

    if var.name == "potential_vorticity":
        if var.levels is None:
            raise ValueError("potential_vorticity requires pressure levels")
        theta_by_level = {
            lvl: potential_temperature(source.fetch_pressure_level("temperature", lvl, cycle), lvl)
            for lvl in var.levels
        }
        u_by_level = {lvl: source.fetch_pressure_level("u_component_of_wind", lvl, cycle) for lvl in var.levels}
        v_by_level = {lvl: source.fetch_pressure_level("v_component_of_wind", lvl, cycle) for lvl in var.levels}
        return [
            potential_vorticity_isobaric(
                u_by_level, v_by_level, theta_by_level, source.lat, source.lon, level_hpa=lvl
            )
            for lvl in var.levels
        ]

    raise ValueError(f"no derivation defined for {var.name!r}")


def assemble_model_input(
    manifest: ModelManifest, source: IFSFieldSource, cycle: IFSCycle
) -> np.ndarray:
    """Fetch + derive + stack every channel `manifest` needs, in channel order.

    Returns an array of shape (len(source.lat), len(source.lon), manifest.n_channels).
    """
    channels: list[np.ndarray] = []
    for var in manifest.variables:
        if var.name in DIRECT_PRESSURE_LEVEL_VARIABLES or var.name in DIRECT_SINGLE_LEVEL_VARIABLES:
            channels.extend(_resolve_direct_variable(var, source, cycle))
        elif var.name in DERIVED_VARIABLES:
            channels.extend(_resolve_derived_variable(var, source, cycle))
        else:
            raise ValueError(
                f"variable {var.name!r} is neither a known direct IFS field nor a "
                "known derived field -- add a mapping in ecmwf_ifs.py before running"
            )

    stacked = np.stack(channels, axis=-1)
    expected_shape = (len(source.lat), len(source.lon), manifest.n_channels)
    if stacked.shape != expected_shape:
        raise AssertionError(f"assembled input shape {stacked.shape} != expected {expected_shape}")
    return stacked
