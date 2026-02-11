import xarray as xr

import datetime
from collections import namedtuple
import dataclasses
from fronts.utils import calc
from typing import Callable

ARCO_ERA5_GCP_URI = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

BoundingBox = namedtuple("BoundingBox", ["lat_min", "lat_max", "lon_min", "lon_max"])


def convert_domain_extent_to_bounding_box(domain_extent: list[float]) -> BoundingBox:
    """Converts a domain extent from constants.py to a BoundingBox namedtuple.

    Args:
        domain_extent: A list of four floats representing the domain extent in the
            format [lat_min, lat_max, lon_min, lon_max].

    Returns a BoundingBox named tuple with the corresponding values.
    """
    if len(domain_extent) != 4:
        raise ValueError("Domain extent must be a list of four floats.")
    return BoundingBox(
        lon_min=domain_extent[0],
        lon_max=domain_extent[1],
        lat_min=domain_extent[2],
        lat_max=domain_extent[3],
    )


def load_arco_era5(
    store: str = ARCO_ERA5_GCP_URI,
    chunks: dict[str, int] = {"time": 48},
    consolidated: bool = True,
):
    """Opens the Google ARCO ERA5 analysis-ready dataset as an xarray Dataset.

    Args:
        store: The URI of the zarr store to open. Defaults to the Google ARCO ERA5
            analysis-ready dataset link.
        chunks: The chunk sizes to use when opening the dataset. Defaults to chunking
            the time dimension into 48-hour chunks.
        consolidated: Whether to use consolidated metadata when opening the dataset.
            Defaults to True.

    Returns an xarray Dataset containing the ERA5 analysis-ready data.
    """
    era5_ds = xr.open_zarr(
        store=store,
        chunks=chunks,
        consolidated=consolidated,
    )

    return era5_ds


def subset_arco_era5(
    ds: xr.Dataset,
    variables: list[str],
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    bounding_box: BoundingBox,
    levels: list[int],
):
    """Subsets the ARCO ERA5 dataset by variables, specific time range, and geographic bounding box.

    Args:
        ds: The input xarray Dataset containing the ARCO ERA5 data.
        variables: A list of variable names to subset from the dataset.
        start_date: The start date of the time range to subset (inclusive).
        end_date: The end date of the time range to subset (inclusive).
        bounding_box: A BoundingBox named tuple defining the geographic bounding box
            for subsetting. Defaults to a bounding box covering the contiguous United
            States.
        levels: A list of pressure levels to subset from the dataset.
    """
    variables_to_postprocess = [var for var in variables if var not in ds.data_vars]
    if variables_to_postprocess:
        variables.pop(variables_to_postprocess)

    if not any(
        [n for n in variables_to_postprocess if n in calc.callable_mapping.keys()]
    ):
        raise ValueError(
            f"Variables {variables_to_postprocess} not found in dataset and no "
            "post-processing functions available for them."
        )
    ds = ds[variables]
    ds = ds.sel(
        latitude=slice(bounding_box.lat_max, bounding_box.lat_min),
        longitude=slice(bounding_box.lon_min, bounding_box.lon_max),
    )
    ds = ds.sel(time=slice(start_date, end_date))
    ds = ds.sel(level=levels)
    return ds


@dataclasses.dataclass
class ERA5TrainingDataConfig:
    """A dataclass for generating data from the ARCO ERA5 dataset.

    This class provides methods for loading and subsetting the variables, spatial
        bounds, and time of the ARCO ERA5 dataset.

    Attributes:
        domain_extent: A list of four floats representing the geographic domain extent
            in the format [lon_min, lon_max, lat_min, lat_max].
        variables: A list of variable names to subset from the dataset.
        start_date: The start date of the time range to subset (inclusive).
        end_date: The end date of the time range to subset (inclusive).
        store: The URI of the zarr store to open.
        chunks: The chunk sizes to use when opening the dataset.
        consolidated: Whether to use consolidated metadata when opening the dataset.
    """

    domain_extent: list[float]
    variables: list[str]
    start_date: datetime.datetime
    end_date: datetime.datetime
    levels: list[int]
    store: str
    chunks: dict[str, int]
    consolidated: bool

    def build(self) -> xr.Dataset:
        """Builds the training dataset by loading and subsetting the ARCO ERA5 dataset.

        Returns an xarray Dataset containing the subset ARCO ERA5 data.
        """
        # Load the ARCO ERA5 dataset with default params
        ds = load_arco_era5(
            store=self.store, chunks=self.chunks, consolidated=self.consolidated
        )

        # Subset the dataset by variables, time range, and geographic bounding box
        ds = subset_arco_era5(
            ds,
            variables=self.variables,
            start_date=self.start_date,
            end_date=self.end_date,
            bounding_box=convert_domain_extent_to_bounding_box(self.domain_extent),
            levels=self.levels,
        )
        return ds


def _default_postprocess(ds: xr.Dataset):
    """Default postprocessor that passes through data unmodified."""
    return ds


def maybe_postprocess_era5(
    ds: xr.Dataset, postprocess_func: Callable = _default_postprocess, **kwargs
) -> xr.Dataset:
    """Applies any necessary post-processing steps to the ERA5 dataset.

    This function is a placeholder for any future post-processing steps that may be
    required for the ERA5 dataset. Currently, it returns the dataset unchanged.

    Args:
        ds: The input xarray Dataset containing the ERA5 data.
        postprocess_func: A callable function that takes an xarray Dataset as input.
            Defaults to a no-op function that returns the dataset unchanged.
        **kwargs: Additional keyword arguments to pass to the post-processing function.

    Returns the possibly post-processed Dataset.
    """
    ds = postprocess_func(ds, **kwargs)
    return ds


def dewpoint_postprocessor(ds: xr.Dataset):
    ds["dewpoint"] = calc.dewpoint_from_specific_humidity(
        ds.level, ds.specific_humidity
    )
    return ds


def potential_temperature_postprocessor(ds: xr.Dataset):
    ds["potential_temperature"] = calc.potential_temperature(ds.level, ds.temperature)
    return ds


def equivalent_potential_temperature_postprocessor(ds: xr.Dataset):
    ds["equivalent_potential_temperature"] = calc.equivalent_potential_temperature(
        ds.level, ds.temperature, ds.dewpoint
    )
    return ds


def virtual_potential_temperature_postprocessor(ds: xr.Dataset):
    ds["virtual_potential_temperature"] = calc.virtual_potential_temperature(
        ds.level, ds.temperature, ds.dewpoint
    )
    return ds


def wet_bulb_temperature_postprocessor(ds: xr.Dataset):
    ds["wet_bulb_temperature"] = calc.wet_bulb_temperature(ds.temperature, ds.dewpoint)
    return ds


def wet_bulb_potential_temperature_postprocessor(ds: xr.Dataset):
    ds["wet_bulb_potential_temperature"] = calc.wet_bulb_potential_temperature(
        ds.level, ds.temperature, ds.dewpoint
    )
    return ds


def relative_humidity_postprocessor(ds: xr.Dataset):
    ds["relative_humidity"] = calc.relative_humidity_from_dewpoint(
        ds.temperature, ds.dewpoint
    )
    return ds


callable_mapping = {
    "dewpoint": dewpoint_postprocessor,
    "potential_temperature": potential_temperature_postprocessor,
    "equivalent_potential_temperature": equivalent_potential_temperature_postprocessor,
    "virtual_potential_temperature": virtual_potential_temperature_postprocessor,
    "wet_bulb_temperature": wet_bulb_temperature_postprocessor,
    "wet_bulb_potential_temperature": wet_bulb_potential_temperature_postprocessor,
    "relative_humidity": relative_humidity_postprocessor,
}
