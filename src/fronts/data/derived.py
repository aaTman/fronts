import dataclasses
from collections.abc import Callable

import numpy as np
import xarray as xr

_R_D = 287.05  # dry air gas constant, J kg-1 K-1
_C_PD = 1004.0  # specific heat of dry air at constant pressure, J kg-1 K-1
_L_V = 2.501e6  # latent heat of vaporization at 0 °C, J kg-1
_EPSILON = 0.622  # ratio of molar masses of water vapour to dry air


@dataclasses.dataclass
class DerivedVariableSpec:
    """Specification for computing a variable not present in the ARCO ERA5 store.

    Attributes:
        required_inputs: ERA5 variable names that must be downloaded before compute.
        compute: Callable that takes the subsetted Dataset and returns a DataArray
            with dims (time, level, latitude, longitude).
    """

    required_inputs: list[str]
    compute: Callable[[xr.Dataset], xr.DataArray]


def _compute_wind_speed(ds: xr.Dataset) -> xr.DataArray:
    return (ds["u_component_of_wind"] ** 2 + ds["v_component_of_wind"] ** 2) ** 0.5


def _pressure_pa(ds: xr.Dataset) -> xr.DataArray:
    """Broadcast pressure levels (hPa → Pa) to (time, level, lat, lon)."""
    level_pa = ds["level"].values.astype(np.float64) * 100.0
    ref_var = next(iter(ds.data_vars))
    return xr.DataArray(
        np.broadcast_to(
            level_pa[np.newaxis, :, np.newaxis, np.newaxis],
            ds[ref_var].shape,
        ).copy(),
        dims=ds[ref_var].dims,
        coords=ds[ref_var].coords,
    )


def _compute_potential_temperature(ds: xr.Dataset) -> xr.DataArray:
    """Dry potential temperature via Poisson's equation: θ = T (p₀/p)^(R_d/c_pd)."""
    p = _pressure_pa(ds)
    return ds["temperature"] * (100000.0 / p) ** (_R_D / _C_PD)


def _saturation_vapour_pressure(t: xr.DataArray) -> xr.DataArray:
    """Saturation vapour pressure in hPa via Bolton (1980) eq. 10."""
    return 6.112 * xr.apply_ufunc(np.exp, 17.67 * (t - 273.15) / (t - 29.65))


def _vapour_pressure(ds: xr.Dataset) -> xr.DataArray:
    """Actual vapour pressure in hPa from specific humidity and pressure level."""
    q = ds["specific_humidity"]
    p_hpa = _pressure_pa(ds) / 100.0
    r = q / (1.0 - q)
    return r / (_EPSILON + r) * p_hpa


def _compute_equivalent_potential_temperature(ds: xr.Dataset) -> xr.DataArray:
    """Equivalent potential temperature via Bolton (1980) eq. 43.

    Reference: Bolton, D. (1980). Mon. Wea. Rev., 108, 1046-1053.
    """
    t = ds["temperature"]
    q = ds["specific_humidity"]
    p = _pressure_pa(ds)
    e = _vapour_pressure(ds)
    r = q / (1.0 - q)
    log_e = xr.apply_ufunc(np.log, e / 6.112)
    t_d = 243.5 * log_e / (17.67 - log_e) + 273.15
    t_l = 1.0 / (1.0 / (t_d - 56.0) + xr.apply_ufunc(np.log, t / t_d) / 800.0) + 56.0
    return t * (100000.0 / p) ** (_R_D / _C_PD) * xr.apply_ufunc(np.exp, (_L_V * r) / (_C_PD * t_l))


def _compute_virtual_temperature(ds: xr.Dataset) -> xr.DataArray:
    """Virtual temperature: T_v = T (1 + q/ε) / (1 + q)."""
    t = ds["temperature"]
    q = ds["specific_humidity"]
    return t * (1.0 + q / _EPSILON) / (1.0 + q)


def _compute_dewpoint_temperature(ds: xr.Dataset) -> xr.DataArray:
    """Dewpoint temperature via Bolton (1980) eq. 11."""
    e = _vapour_pressure(ds)
    log_e = xr.apply_ufunc(np.log, e / 6.112)
    return 243.5 * log_e / (17.67 - log_e) + 273.15


def _compute_relative_humidity(ds: xr.Dataset) -> xr.DataArray:
    """Relative humidity as a fraction (0-1) from specific humidity and pressure."""
    t = ds["temperature"]
    e = _vapour_pressure(ds)
    e_s = _saturation_vapour_pressure(t)
    return e / e_s


DERIVED_VARIABLE_REGISTRY: dict[str, DerivedVariableSpec] = {
    "wind_speed": DerivedVariableSpec(
        required_inputs=["u_component_of_wind", "v_component_of_wind"],
        compute=_compute_wind_speed,
    ),
    "potential_temperature": DerivedVariableSpec(
        required_inputs=["temperature"],
        compute=_compute_potential_temperature,
    ),
    "equivalent_potential_temperature": DerivedVariableSpec(
        required_inputs=["temperature", "specific_humidity"],
        compute=_compute_equivalent_potential_temperature,
    ),
    "virtual_temperature": DerivedVariableSpec(
        required_inputs=["temperature", "specific_humidity"],
        compute=_compute_virtual_temperature,
    ),
    "dewpoint_temperature": DerivedVariableSpec(
        required_inputs=["temperature", "specific_humidity"],
        compute=_compute_dewpoint_temperature,
    ),
    "relative_humidity": DerivedVariableSpec(
        required_inputs=["temperature", "specific_humidity"],
        compute=_compute_relative_humidity,
    ),
}


def classify_variables(
    requested: list[str],
    arco_available: set[str],
) -> tuple[list[str], list[str]]:
    """Split requested variable names into direct (in ARCO) and derived.

    Args:
        requested: Variable names from the user config.
        arco_available: Variable names present in the ARCO ERA5 Zarr store.

    Returns:
        Tuple of (direct_vars, derived_vars) where direct_vars are available
        in ARCO and derived_vars must be computed via the registry.

    Raises:
        ValueError: If any variable is neither in ``arco_available`` nor in
            ``DERIVED_VARIABLE_REGISTRY``.
    """
    direct_vars: list[str] = []
    derived_vars: list[str] = []
    unknown: list[str] = []

    for var in requested:
        if var in arco_available:
            direct_vars.append(var)
        elif var in DERIVED_VARIABLE_REGISTRY:
            derived_vars.append(var)
        else:
            unknown.append(var)

    if unknown:
        raise ValueError(
            f"Variables not available in ARCO ERA5 and have no derivation function: {unknown}. "
            f"Registered derivable variables: {sorted(DERIVED_VARIABLE_REGISTRY)}"
        )

    return direct_vars, derived_vars


def resolve_download_variables(
    requested: list[str],
    direct_vars: list[str],
    derived_vars: list[str],
) -> list[str]:
    """Return the full list of variables to download from ARCO.

    Includes all direct variables plus any required inputs for derived variables
    that are not already in the direct list.

    Args:
        requested: The original variable list from config (preserves order for directs).
        direct_vars: Variables available directly in ARCO.
        derived_vars: Variables that require derivation.

    Returns:
        Deduplicated list of variable names to fetch from ARCO.
    """
    to_download: list[str] = list(direct_vars)
    requested_set = set(requested)
    seen = set(direct_vars)

    for var in derived_vars:
        spec = DERIVED_VARIABLE_REGISTRY[var]
        for inp in spec.required_inputs:
            if inp not in seen:
                if inp not in requested_set:
                    to_download.append(inp)
                seen.add(inp)

    return to_download
