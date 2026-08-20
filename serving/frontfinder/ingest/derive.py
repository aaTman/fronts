"""Derived-variable formulas for fields the models need that IFS open-data
does not archive directly.

Both models were trained on ERA5. `equivalent_potential_temperature` and
`potential_vorticity` are not standard IFS open-data pressure-level params,
so they're computed here from base fields (temperature, specific humidity,
wind, geopotential height) that IFS *does* publish. These are standard
synoptic-meteorology formulas (Bolton 1980 for theta-e; isobaric PV from
relative vorticity + static stability), but they are an approximation of
whatever ERA5-native fields/derivation the fronts model was actually trained
on -- flagged here as a science-validation item, not just an engineering one.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_000.0
OMEGA = 7.292115e-5  # Earth's rotation rate, rad/s
G = 9.80665  # m/s^2
RD = 287.05  # J/(kg K), dry air gas constant
CP = 1004.6  # J/(kg K), specific heat of dry air at constant pressure
P0 = 1000.0  # hPa reference pressure


def potential_temperature(temperature_k: np.ndarray, pressure_hpa: float) -> np.ndarray:
    """theta = T * (P0/P)^(Rd/Cp)"""
    return temperature_k * (P0 / pressure_hpa) ** (RD / CP)


def mixing_ratio(specific_humidity: np.ndarray) -> np.ndarray:
    """r = q / (1 - q), specific_humidity in kg/kg."""
    return specific_humidity / (1.0 - specific_humidity)


def equivalent_potential_temperature(
    temperature_k: np.ndarray,
    specific_humidity: np.ndarray,
    pressure_hpa: float,
) -> np.ndarray:
    """Bolton (1980) approximation of equivalent potential temperature.

    theta_e = theta * exp((3.376/T_L - 0.00254) * r * 1000 * (1 + 0.81e-3 * r * 1000))

    Uses the simplified assumption T_L ~= T (skips the LCL temperature
    correction term) since frontfinder only needs theta_e as a CNN input
    feature, not a precise thermodynamic diagnostic.
    """
    theta = potential_temperature(temperature_k, pressure_hpa)
    r = mixing_ratio(specific_humidity)
    r_g_per_kg = r * 1000.0
    return theta * np.exp((3.376 / temperature_k - 0.00254) * r_g_per_kg * (1 + 0.81e-3 * r_g_per_kg))


def coriolis_parameter(lat_deg: np.ndarray) -> np.ndarray:
    """f = 2*Omega*sin(lat)"""
    return 2.0 * OMEGA * np.sin(np.deg2rad(lat_deg))


def relative_vorticity(
    u: np.ndarray,
    v: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
) -> np.ndarray:
    """zeta = dv/dx - du/dy on a regular lat/lon grid.

    `u`, `v` are 2D (lat, lon) arrays. `lat_deg` and `lon_deg` are 1D
    coordinate arrays matching those axes. Uses centered finite differences
    with spherical map factors; edges use one-sided differences.
    """
    if u.shape != v.shape:
        raise ValueError(f"u and v shape mismatch: {u.shape} vs {v.shape}")
    if u.shape != (len(lat_deg), len(lon_deg)):
        raise ValueError(
            f"u/v shape {u.shape} does not match (len(lat), len(lon)) = "
            f"({len(lat_deg)}, {len(lon_deg)})"
        )

    lat_rad = np.deg2rad(lat_deg)
    dlat = np.gradient(lat_rad)
    dlon = np.gradient(np.deg2rad(lon_deg))

    dy = EARTH_RADIUS_M * dlat  # meters per grid step, per row
    cos_lat = np.cos(lat_rad)
    cos_lat_safe = np.where(np.abs(cos_lat) < 1e-6, 1e-6, cos_lat)
    dx = EARTH_RADIUS_M * cos_lat_safe[:, None] * dlon[None, :]  # meters per grid step, per (row, col)

    dv_dx = np.gradient(v, axis=1) / dx
    du_dy = np.gradient(u, axis=0) / dy[:, None]
    return dv_dx - du_dy


def potential_vorticity_isobaric(
    u_by_level: dict[int, np.ndarray],
    v_by_level: dict[int, np.ndarray],
    theta_by_level: dict[int, np.ndarray],
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    level_hpa: int,
) -> np.ndarray:
    """Isobaric potential vorticity at `level_hpa` (in PVU, 1e-6 K m^2 kg^-1 s^-1):

        PV = -g * (zeta + f) * (d theta / d p)

    `d theta / d p` is estimated from the nearest available levels above and
    below `level_hpa` in `theta_by_level` (falls back to a one-sided
    difference at the top/bottom of the column).
    """
    levels = sorted(theta_by_level.keys())
    if level_hpa not in levels:
        raise ValueError(f"level {level_hpa} not present in theta_by_level: {levels}")

    idx = levels.index(level_hpa)
    if idx == 0:
        lower, upper = levels[0], levels[1]
    elif idx == len(levels) - 1:
        lower, upper = levels[-2], levels[-1]
    else:
        lower, upper = levels[idx - 1], levels[idx + 1]

    # pressure decreases with altitude; d theta/d p computed with p in Pa
    dtheta = theta_by_level[upper] - theta_by_level[lower]
    dp = (upper - lower) * 100.0  # hPa -> Pa
    dtheta_dp = dtheta / dp

    zeta = relative_vorticity(u_by_level[level_hpa], v_by_level[level_hpa], lat_deg, lon_deg)
    f = coriolis_parameter(lat_deg)[:, None]
    pv_si = -G * (zeta + f) * dtheta_dp
    return pv_si * 1e6  # SI -> PVU
