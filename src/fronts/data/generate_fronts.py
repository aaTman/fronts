"""Convert MPC/OPC surface-analysis front XML files to netCDF and register the new files.

The new files are registered as virtual chunks in an icechunk store, without copying array
data. The output grid is built from the ERA5 icechunk store's own coordinate machinery
(``utils.select_spatial_domain``/``utils.unwrap_longitude``) rather than a standalone domain
table, so front rasters are pixel-aligned with the ERA5 inputs store used during training.
"""

import argparse
import dataclasses
import datetime
import logging
import os
import re
import sys

import icechunk as ic
import numpy as np
import pandas as pd
import xarray as xr
import zarr.errors
from defusedxml import ElementTree
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from shapely.geometry import LineString
from virtualizarr import open_virtual_mfdataset
from virtualizarr.parsers import HDFParser

from fronts import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


_NATIVE_GRID_RESOLUTION_DEG = 0.25
_EARTH_CIRCUMFERENCE_KM = 40041  # average circumference of Earth in kilometers

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

_XML_FILENAME_PATTERN = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{4})_(?P<cycle_hour>\d{2})_MPC_final-anal_OPC_SFC_ANAL\.xml$"
)


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


def _native_grid_template() -> xr.Dataset:
    """Coordinate-only Dataset spanning the full 0.25-degree global ERA5-aligned grid."""
    longitude = np.round(np.arange(0.0, 360.0, _NATIVE_GRID_RESOLUTION_DEG), 2).astype("float32")
    latitude = np.round(np.arange(90.0, -90.0 - _NATIVE_GRID_RESOLUTION_DEG, -_NATIVE_GRID_RESOLUTION_DEG), 2).astype(
        "float32"
    )
    return xr.Dataset(coords={"latitude": latitude, "longitude": longitude})


def grid_coordinates(coordinates: utils.BoundingBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the front-raster grid, aligned to the ERA5 icechunk store's coordinates.

    Args:
        coordinates: Spatial bounding box, same convention as ``ERA5DataLoaderConfig.coordinates``.

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

    root = ElementTree.parse(xml_path).getroot()
    for line in root.iter("Line"):
        front_type = line.get("pgenType")
        if front_type not in PGEN_TYPE_IDENTIFIERS:
            raise ValueError(f"Unrecognized front type {front_type!r} in {xml_path}")

        points = [(float(point.get("Lon")), float(point.get("Lat"))) for point in line.iter("Point")]
        lons = np.array([p[0] for p in points])
        lats = np.array([p[1] for p in points])

        crosses_dateline = np.max(np.abs(np.diff(lons))) > 180
        if crosses_dateline:
            lons = np.where(lons < 0, lons + 360, lons)

        x_km, y_km = _haversine(lons, lats)
        vertices = _redistribute_vertices(LineString(list(zip(x_km, y_km, strict=True))), distance_km)
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


def write_netcdfs_to_icechunk_store(
    icechunk_config: utils.IcechunkStorageConfig, netcdf_paths: list[str], append: bool
) -> None:
    """Register netCDF files as virtual chunks in the icechunk store, in a single commit.

    Args:
        icechunk_config: Configuration for the target icechunk store. Its
            ``virtual_chunk_local_path`` must be set and must be the directory ``netcdf_paths``
            live in.
        netcdf_paths: Paths of the new front netCDF files to register, one time step each;
            concatenated along ``time`` in sorted order.
        append: True to append to an existing ``time`` dimension; False to create it (the store
            or group has no data yet).

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
    # Fix the time encoding to a batch-independent reference epoch: the writer otherwise
    # defaults to "<unit> since <this batch's first time>", which is inconsistent across
    # separate append calls and corrupts previously-written values on readback.
    virtual_ds["time"].encoding = {
        "units": "minutes since 1970-01-01",
        "calendar": "proleptic_gregorian",
        "dtype": "int64",
    }

    repo = utils.open_writable_icechunk_repo(icechunk_config.store_path, icechunk_config.virtual_chunk_local_path)
    session = repo.writable_session(icechunk_config.branch_name)
    virtual_ds.vz.to_icechunk(session.store, group=icechunk_config.group_name, append_dim="time" if append else None)
    session.commit(icechunk_config.commit_message)
    logger.info(f"Committed {len(netcdf_paths)} new front netCDF file(s) to {icechunk_config.store_path}")


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
    missing_times = sorted(t for t in available if t not in existing_times_set)

    if not missing_times:
        logger.info("All requested front XML files are already represented in the icechunk store.")
        return

    logger.info(f"Converting {len(missing_times)} new front XML file(s) to netCDF...")
    os.makedirs(front_config.netcdf_outdir, exist_ok=True)
    netcdf_paths = []
    for valid_time in missing_times:
        ds = convert_xml_to_dataset(available[valid_time], valid_time, front_config.coordinates, front_config.distance)
        netcdf_path = os.path.join(
            front_config.netcdf_outdir, f"FrontObjects_{valid_time.strftime('%Y%m%d%H%M')}_full.nc"
        )
        ds.to_netcdf(netcdf_path, engine="netcdf4", mode="w")
        netcdf_paths.append(netcdf_path)

    write_netcdfs_to_icechunk_store(icechunk_config, netcdf_paths, append=existing_times is not None)
    logger.info("Front XML to netCDF conversion and icechunk store update complete.")


if __name__ == "__main__":
    main()
