import numpy as np
import pytest

from frontfinder.ingest.derive import (
    coriolis_parameter,
    equivalent_potential_temperature,
    mixing_ratio,
    potential_temperature,
    potential_vorticity_isobaric,
    relative_vorticity,
)


def test_potential_temperature_equals_actual_temperature_at_reference_pressure():
    t = np.array([280.0, 300.0])
    theta = potential_temperature(t, pressure_hpa=1000.0)
    np.testing.assert_allclose(theta, t)


def test_potential_temperature_increases_at_lower_pressure():
    t = np.array([280.0])
    theta_1000 = potential_temperature(t, pressure_hpa=1000.0)
    theta_500 = potential_temperature(t, pressure_hpa=500.0)
    assert theta_500 > theta_1000


def test_mixing_ratio_zero_humidity():
    assert mixing_ratio(np.array([0.0]))[0] == 0.0


def test_equivalent_potential_temperature_equals_theta_when_dry():
    t = np.array([290.0])
    theta_e = equivalent_potential_temperature(t, specific_humidity=np.array([0.0]), pressure_hpa=850.0)
    theta = potential_temperature(t, pressure_hpa=850.0)
    np.testing.assert_allclose(theta_e, theta, rtol=1e-6)


def test_equivalent_potential_temperature_increases_with_humidity():
    t = np.array([290.0, 290.0, 290.0])
    q = np.array([0.0, 0.005, 0.015])
    theta_e = equivalent_potential_temperature(t, q, pressure_hpa=850.0)
    assert theta_e[0] < theta_e[1] < theta_e[2]


def test_coriolis_parameter_zero_at_equator():
    assert coriolis_parameter(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-12)


def test_coriolis_parameter_sign_flips_across_equator():
    f_nh = coriolis_parameter(np.array([45.0]))[0]
    f_sh = coriolis_parameter(np.array([-45.0]))[0]
    assert f_nh > 0
    assert f_sh < 0
    assert f_nh == pytest.approx(-f_sh)


def test_relative_vorticity_is_zero_for_uniform_flow():
    lat = np.linspace(30.0, 40.0, 11)
    lon = np.linspace(0.0, 10.0, 11)
    u = np.full((11, 11), 5.0)
    v = np.full((11, 11), 3.0)
    zeta = relative_vorticity(u, v, lat, lon)
    np.testing.assert_allclose(zeta, 0.0, atol=1e-12)


def test_relative_vorticity_matches_analytic_shear():
    lat = np.linspace(30.0, 40.0, 11)
    lon = np.linspace(0.0, 10.0, 11)
    u = np.zeros((11, 11))
    # v varies linearly with longitude index only
    v = np.tile(np.arange(11.0), (11, 1))
    zeta = relative_vorticity(u, v, lat, lon)

    dlon_rad = np.deg2rad(lon[1] - lon[0])
    earth_radius_m = 6_371_000.0
    dx = earth_radius_m * np.cos(np.deg2rad(lat))[:, None] * dlon_rad
    expected = np.broadcast_to(1.0 / dx, zeta.shape)  # dv/d(index) == 1 everywhere for this linear ramp
    np.testing.assert_allclose(zeta, expected, rtol=1e-6)


def test_relative_vorticity_rejects_shape_mismatch():
    lat = np.linspace(30.0, 40.0, 11)
    lon = np.linspace(0.0, 10.0, 11)
    u = np.zeros((11, 11))
    v = np.zeros((10, 11))
    with pytest.raises(ValueError):
        relative_vorticity(u, v, lat, lon)


def test_potential_vorticity_is_positive_in_stable_northern_hemisphere_atmosphere():
    lat = np.linspace(30.0, 40.0, 9)
    lon = np.linspace(0.0, 10.0, 9)
    u0 = np.zeros((9, 9))
    v0 = np.zeros((9, 9))
    # a stably stratified column: theta increases with height (lower pressure)
    theta_by_level = {
        1000: np.full((9, 9), 290.0),
        850: np.full((9, 9), 300.0),
        700: np.full((9, 9), 315.0),
    }
    pv = potential_vorticity_isobaric(
        u_by_level={1000: u0, 850: u0, 700: u0},
        v_by_level={1000: v0, 850: v0, 700: v0},
        theta_by_level=theta_by_level,
        lat_deg=lat,
        lon_deg=lon,
        level_hpa=850,
    )
    assert np.all(pv > 0)


def test_potential_vorticity_rejects_missing_level():
    lat = np.linspace(30.0, 40.0, 9)
    lon = np.linspace(0.0, 10.0, 9)
    z = np.zeros((9, 9))
    with pytest.raises(ValueError):
        potential_vorticity_isobaric(
            u_by_level={1000: z}, v_by_level={1000: z}, theta_by_level={1000: z},
            lat_deg=lat, lon_deg=lon, level_hpa=850,
        )
