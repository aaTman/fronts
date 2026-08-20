import numpy as np
import pytest

from frontfinder.config.manifests import BEST_LOSS_MANIFEST, MODEL_1702_MANIFEST
from frontfinder.ingest.ecmwf_ifs import (
    FakeIFSFieldSource,
    IFSCycle,
    assemble_model_input,
)


@pytest.fixture
def small_source():
    lat = np.linspace(20.0, 50.0, 17)
    lon = np.linspace(200.0, 300.0, 21)
    return FakeIFSFieldSource(lat, lon, seed=42)


def test_ifs_cycle_rejects_bad_run_hour():
    with pytest.raises(ValueError):
        IFSCycle(date="2026-08-19", run_hour=9)


def test_ifs_cycle_rejects_bad_date():
    with pytest.raises(ValueError):
        IFSCycle(date="not-a-date", run_hour=0)


def test_ifs_cycle_accepts_valid_synoptic_hours():
    for h in (0, 6, 12, 18):
        IFSCycle(date="2026-08-19", run_hour=h)


def test_assemble_best_loss_input_has_30_channels(small_source):
    cycle = IFSCycle(date="2026-08-19", run_hour=12)
    arr = assemble_model_input(BEST_LOSS_MANIFEST, small_source, cycle)
    assert arr.shape == (17, 21, 30)
    assert np.all(np.isfinite(arr))


def test_assemble_model_1702_input_has_25_channels(small_source):
    cycle = IFSCycle(date="2026-08-19", run_hour=12)
    arr = assemble_model_input(MODEL_1702_MANIFEST, small_source, cycle)
    assert arr.shape == (17, 21, 25)
    assert np.all(np.isfinite(arr))


def test_assemble_is_deterministic_for_same_cycle(small_source):
    cycle = IFSCycle(date="2026-08-19", run_hour=12)
    arr1 = assemble_model_input(BEST_LOSS_MANIFEST, small_source, cycle)
    arr2 = assemble_model_input(BEST_LOSS_MANIFEST, small_source, cycle)
    np.testing.assert_array_equal(arr1, arr2)


def test_assemble_differs_between_cycles(small_source):
    cycle_a = IFSCycle(date="2026-08-19", run_hour=0)
    cycle_b = IFSCycle(date="2026-08-19", run_hour=12)
    arr_a = assemble_model_input(BEST_LOSS_MANIFEST, small_source, cycle_a)
    arr_b = assemble_model_input(BEST_LOSS_MANIFEST, small_source, cycle_b)
    assert not np.array_equal(arr_a, arr_b)


def test_assemble_channel_order_matches_manifest_channel_names(small_source):
    # equivalent_potential_temperature block comes first for best_loss,
    # so channels 0-5 should all be theta_e derived (finite, plausible K range)
    cycle = IFSCycle(date="2026-08-19", run_hour=12)
    arr = assemble_model_input(BEST_LOSS_MANIFEST, small_source, cycle)
    theta_e_block = arr[..., 0:6]
    assert np.all(theta_e_block > 200) and np.all(theta_e_block < 500)
