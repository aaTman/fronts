"""Tests for the DataConfig pipeline: YAML -> dacite -> dataclasses.

Tests verify that all DataConfig classes (ERA5PredictorConfig, FrontsDataConfig,
BatchGeneratorConfig, DataConfig) parse correctly from dicts/YAML via dacite and
have the expected field types. .build() methods are tested with mocked I/O.

Does NOT require TensorFlow, GPU, or real data files — follows the same pattern
as conftest.py (mocks installed at module load time).
"""

import datetime
import os
from unittest.mock import MagicMock, patch

import dacite
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from fronts.data.batch import BatchGeneratorConfig
from fronts.data.config import (
    DataConfig,
    ERA5PredictorConfig,
    FrontsDataConfig,
    PredictConfig,
    TimeSelection,
    ModelData,
    SURFACE_VARIABLE_MAP,
    SURFACE_ONLY_VARIABLES,
)
from fronts.train import TrainConfig, open_config_yaml_as_dataclass

DACITE_CONFIG = dacite.Config(cast=[tuple], check_types=False)


# ---------------------------------------------------------------------------
# Shared xarray fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_era5_ds():
    """Minimal ERA5-like Dataset with unified level coordinate."""
    times = pd.date_range("2010-01-01", periods=4, freq="6h")
    lats = np.array([30.0, 35.0, 40.0])
    lons = np.array([-100.0, -95.0, -90.0])
    levels = ["surface", 1000, 850]

    return xr.Dataset(
        {
            "temperature": (
                ["time", "level", "latitude", "longitude"],
                np.random.rand(4, 3, 3, 3).astype("float32"),
            ),
        },
        coords={
            "time": times,
            "level": levels,
            "latitude": lats,
            "longitude": lons,
        },
    )


@pytest.fixture
def sample_fronts_ds():
    """Minimal fronts Dataset with identifier variable."""
    times = pd.date_range("2010-01-01", periods=4, freq="6h")
    lats = np.array([30.0, 35.0, 40.0])
    lons = np.array([-100.0, -95.0, -90.0])

    return xr.Dataset(
        {
            "identifier": (
                ["time", "latitude", "longitude"],
                np.zeros((4, 3, 3), dtype="int32"),
            ),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )


# ---------------------------------------------------------------------------
# ERA5PredictorConfig
# ---------------------------------------------------------------------------


class TestERA5PredictorConfig:
    def _minimal_dict(self):
        return {
            "domain_extent": [-140.0, -60.0, 20.0, 60.0],
            "variables": ["temperature", "mean_sea_level_pressure"],
            "levels": ["surface", 1000, 850],
            "years": [2010, 2011],
            "store": "gs://fake-store",
            "chunks": {"time": 48},
            "consolidated": True,
        }

    def test_dacite_parses(self):
        cfg = dacite.from_dict(ERA5PredictorConfig, self._minimal_dict(), DACITE_CONFIG)
        assert cfg.levels == ["surface", 1000, 850]
        assert cfg.years == [2010, 2011]
        assert cfg.variables == ["temperature", "mean_sea_level_pressure"]
        assert cfg.consolidated is True

    def test_domain_extent_length(self):
        cfg = dacite.from_dict(ERA5PredictorConfig, self._minimal_dict(), DACITE_CONFIG)
        assert len(cfg.domain_extent) == 4

    def test_levels_pressure_only(self):
        """levels may contain only integer hPa values (no surface)."""
        d = self._minimal_dict()
        d["levels"] = [1000, 900, 750]
        cfg = dacite.from_dict(ERA5PredictorConfig, d, DACITE_CONFIG)
        assert cfg.levels == [1000, 900, 750]
        assert "surface" not in cfg.levels

    def test_build_calls_open_zarr(self, sample_era5_ds):
        """ERA5PredictorConfig.build() opens the zarr store and subsets spatially/temporally."""
        times = pd.date_range("2010-01-01", periods=8, freq="6h")
        lats = np.linspace(20.0, 60.0, 5)
        lons = np.linspace(-140.0, -60.0, 5)
        levels = [1000, 850]

        raw_ds = xr.Dataset(
            {
                "temperature": (
                    ["time", "level", "latitude", "longitude"],
                    np.random.rand(8, 2, 5, 5).astype("float32"),
                ),
                "2m_temperature": (
                    ["time", "latitude", "longitude"],
                    np.random.rand(8, 5, 5).astype("float32"),
                ),
            },
            coords={
                "time": times,
                "level": levels,
                "latitude": lats,
                "longitude": lons,
            },
        )

        cfg = ERA5PredictorConfig(
            domain_extent=[-140.0, -60.0, 20.0, 60.0],
            variables=["temperature"],
            levels=["surface", 1000, 850],
            years=[2010],
            store="gs://fake-store",
            chunks={"time": 48},
            consolidated=True,
        )

        with patch("xarray.open_zarr", return_value=raw_ds):
            result = cfg.build()

        assert "temperature" in result
        # temperature should have level coord including "surface"
        assert "surface" in result["temperature"].coords["level"].values

    def test_build_surface_only_variable_added(self):
        """Surface-only variables appear in result with level=["surface"]."""
        times = pd.date_range("2010-01-01", periods=4, freq="6h")
        lats = np.linspace(20.0, 60.0, 3)
        lons = np.linspace(-140.0, -60.0, 3)

        raw_ds = xr.Dataset(
            {
                "temperature": (
                    ["time", "level", "latitude", "longitude"],
                    np.random.rand(4, 1, 3, 3).astype("float32"),
                ),
                "mean_sea_level_pressure": (
                    ["time", "latitude", "longitude"],
                    np.random.rand(4, 3, 3).astype("float32"),
                ),
            },
            coords={
                "time": times,
                "level": [1000],
                "latitude": lats,
                "longitude": lons,
            },
        )

        cfg = ERA5PredictorConfig(
            domain_extent=[-140.0, -60.0, 20.0, 60.0],
            variables=["temperature", "mean_sea_level_pressure"],
            levels=["surface", 1000],
            years=[2010],
            store="gs://fake-store",
            chunks={"time": 48},
            consolidated=True,
        )

        with patch("xarray.open_zarr", return_value=raw_ds):
            result = cfg.build()

        assert "mean_sea_level_pressure" in result
        assert "surface" in result["mean_sea_level_pressure"].coords["level"].values

    def test_build_pressure_only_levels(self):
        """When levels has no 'surface', only pressure-level data is returned."""
        times = pd.date_range("2010-01-01", periods=4, freq="6h")
        lats = np.linspace(20.0, 60.0, 3)
        lons = np.linspace(-140.0, -60.0, 3)

        raw_ds = xr.Dataset(
            {
                "temperature": (
                    ["time", "level", "latitude", "longitude"],
                    np.random.rand(4, 2, 3, 3).astype("float32"),
                ),
            },
            coords={
                "time": times,
                "level": [1000, 850],
                "latitude": lats,
                "longitude": lons,
            },
        )

        cfg = ERA5PredictorConfig(
            domain_extent=[-140.0, -60.0, 20.0, 60.0],
            variables=["temperature"],
            levels=[1000, 850],
            years=[2010],
            store="gs://fake-store",
            chunks={"time": 48},
            consolidated=True,
        )

        with patch("xarray.open_zarr", return_value=raw_ds):
            result = cfg.build()

        assert "temperature" in result
        assert "surface" not in result["temperature"].coords["level"].values


# ---------------------------------------------------------------------------
# FrontsDataConfig
# ---------------------------------------------------------------------------


class TestFrontsDataConfig:
    def test_dacite_parses_string_front_types(self):
        d = {"directory": "/tmp/fronts", "years": [2010], "front_types": "MERGED-ALL"}
        cfg = dacite.from_dict(FrontsDataConfig, d, DACITE_CONFIG)
        assert cfg.front_types == "MERGED-ALL"
        assert cfg.years == [2010]

    def test_dacite_parses_list_front_types(self):
        d = {"directory": "/tmp/fronts", "years": [2010], "front_types": ["CF", "WF"]}
        cfg = dacite.from_dict(FrontsDataConfig, d, DACITE_CONFIG)
        assert cfg.front_types == ["CF", "WF"]

    def test_dacite_parses_null_front_types(self):
        d = {"directory": "/tmp/fronts", "years": [2010], "front_types": None}
        cfg = dacite.from_dict(FrontsDataConfig, d, DACITE_CONFIG)
        assert cfg.front_types is None

    def test_build_with_mocked_mfdataset(self, sample_fronts_ds, tmp_path):
        """FrontsDataConfig.build() calls open_mfdataset and returns a Dataset."""
        cfg = FrontsDataConfig(
            directory=str(tmp_path),
            years=[2010],
            front_types=None,
        )
        with patch("xarray.open_mfdataset", return_value=sample_fronts_ds):
            ds = cfg.build()
        assert "identifier" in ds

    def test_build_calls_reformat_fronts_when_front_types_set(self, sample_fronts_ds, tmp_path):
        """FrontsDataConfig.build() calls reformat_fronts when front_types is given."""
        cfg = FrontsDataConfig(
            directory=str(tmp_path),
            years=[2010],
            front_types="MERGED-ALL",
        )
        with patch("xarray.open_mfdataset", return_value=sample_fronts_ds):
            with patch("fronts.utils.data_utils.reformat_fronts", return_value=sample_fronts_ds) as mock_rf:
                cfg.build()
        mock_rf.assert_called_once()

    def test_build_skips_reformat_fronts_when_none(self, sample_fronts_ds, tmp_path):
        """FrontsDataConfig.build() does not call reformat_fronts when front_types is None."""
        cfg = FrontsDataConfig(
            directory=str(tmp_path),
            years=[2010],
            front_types=None,
        )
        with patch("xarray.open_mfdataset", return_value=sample_fronts_ds):
            with patch("fronts.utils.data_utils.reformat_fronts") as mock_rf:
                cfg.build()
        mock_rf.assert_not_called()


# ---------------------------------------------------------------------------
# BatchGeneratorConfig
# ---------------------------------------------------------------------------


class TestBatchGeneratorConfig:
    def test_dacite_parses(self):
        d = {
            "input_sizes": {"time": 1, "latitude": 128, "longitude": 128},
            "target_sizes": {"time": 1, "latitude": 128, "longitude": 128},
            "prefetch_number": 3,
            "preload_batch": False,
        }
        cfg = dacite.from_dict(BatchGeneratorConfig, d, DACITE_CONFIG)
        assert cfg.input_sizes == {"time": 1, "latitude": 128, "longitude": 128}
        assert cfg.prefetch_number == 3

    def test_defaults(self):
        cfg = BatchGeneratorConfig()
        assert cfg.input_sizes is None
        assert cfg.target_sizes is None
        assert cfg.prefetch_number == 3
        assert cfg.preload_batch is False


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------


def _minimal_data_config_dict():
    return {
        "train_years": [2010],
        "val_years": [2011],
        "test_years": [],
        "shuffle": True,
        "normalization_method": "standard",
        "era5": {
            "domain_extent": [-140.0, -60.0, 20.0, 60.0],
            "variables": ["temperature", "mean_sea_level_pressure"],
            "levels": ["surface", 1000, 850],
            "years": [],
            "store": "gs://fake-store",
            "chunks": {"time": 48},
            "consolidated": True,
        },
        "fronts": {
            "directory": "/tmp/fronts",
            "years": [],
            "front_types": "MERGED-ALL",
        },
        "batch": {
            "input_sizes": {"time": 1, "latitude": 128, "longitude": 128},
            "target_sizes": {"time": 1, "latitude": 128, "longitude": 128},
            "prefetch_number": 3,
            "preload_batch": False,
        },
    }


class TestDataConfig:
    def test_dacite_parses_full_config(self):
        cfg = dacite.from_dict(DataConfig, _minimal_data_config_dict(), DACITE_CONFIG)
        assert cfg.train_years == [2010]
        assert cfg.val_years == [2011]
        assert cfg.test_years == []
        assert cfg.shuffle is True
        assert cfg.normalization_method == "standard"
        assert isinstance(cfg.era5, ERA5PredictorConfig)
        assert isinstance(cfg.fronts, FrontsDataConfig)
        assert isinstance(cfg.batch, BatchGeneratorConfig)

    def test_nested_era5_fields(self):
        cfg = dacite.from_dict(DataConfig, _minimal_data_config_dict(), DACITE_CONFIG)
        assert cfg.era5.levels == ["surface", 1000, 850]
        assert cfg.era5.variables == ["temperature", "mean_sea_level_pressure"]

    def test_nested_fronts_fields(self):
        cfg = dacite.from_dict(DataConfig, _minimal_data_config_dict(), DACITE_CONFIG)
        assert cfg.fronts.front_types == "MERGED-ALL"
        assert cfg.fronts.directory == "/tmp/fronts"

    def test_empty_test_years_produces_none_test_data(self, sample_era5_ds, sample_fronts_ds):
        """DataConfig.build() returns test_data=None when test_years is empty."""
        cfg = dacite.from_dict(DataConfig, _minimal_data_config_dict(), DACITE_CONFIG)

        mock_tf_ds = MagicMock()
        mock_tf_ds.shuffle = MagicMock(return_value=mock_tf_ds)

        with patch.object(ERA5PredictorConfig, "build", return_value=sample_era5_ds):
            with patch.object(FrontsDataConfig, "build", return_value=sample_fronts_ds):
                with patch("fronts.utils.data_utils.normalize_dataset", return_value=sample_era5_ds):
                    with patch("fronts.data.config.create_dataloader", return_value=mock_tf_ds):
                        result = cfg.build()

        assert isinstance(result, ModelData)
        assert result.test_data is None

    def test_shuffle_called_on_train_ds(self, sample_era5_ds, sample_fronts_ds):
        """DataConfig.build() calls .shuffle() on the training dataset when shuffle=True."""
        cfg = dacite.from_dict(DataConfig, _minimal_data_config_dict(), DACITE_CONFIG)

        mock_tf_ds = MagicMock()
        shuffled_ds = MagicMock()
        mock_tf_ds.shuffle = MagicMock(return_value=shuffled_ds)

        with patch.object(ERA5PredictorConfig, "build", return_value=sample_era5_ds):
            with patch.object(FrontsDataConfig, "build", return_value=sample_fronts_ds):
                with patch("fronts.utils.data_utils.normalize_dataset", return_value=sample_era5_ds):
                    with patch("fronts.data.config.create_dataloader", return_value=mock_tf_ds):
                        result = cfg.build()

        mock_tf_ds.shuffle.assert_called_once()
        assert result.train_data is shuffled_ds

    def test_no_shuffle_when_disabled(self, sample_era5_ds, sample_fronts_ds):
        """DataConfig.build() does not shuffle when shuffle=False."""
        d = _minimal_data_config_dict()
        d["shuffle"] = False
        cfg = dacite.from_dict(DataConfig, d, DACITE_CONFIG)

        mock_tf_ds = MagicMock()
        mock_tf_ds.shuffle = MagicMock()

        with patch.object(ERA5PredictorConfig, "build", return_value=sample_era5_ds):
            with patch.object(FrontsDataConfig, "build", return_value=sample_fronts_ds):
                with patch("fronts.utils.data_utils.normalize_dataset", return_value=sample_era5_ds):
                    with patch("fronts.data.config.create_dataloader", return_value=mock_tf_ds):
                        cfg.build()

        mock_tf_ds.shuffle.assert_not_called()

    def test_years_injected_via_replace(self, sample_era5_ds, sample_fronts_ds):
        """DataConfig.build() injects split-specific years into ERA5/Fronts configs."""
        cfg = dacite.from_dict(DataConfig, _minimal_data_config_dict(), DACITE_CONFIG)
        # era5.years in the dict is [] — DataConfig.build() injects train_years=[2010], val_years=[2011]

        all_era5_years = []
        all_fronts_years = []

        def mock_era5_build(self):
            all_era5_years.append(self.years)
            return sample_era5_ds

        def mock_fronts_build(self):
            all_fronts_years.append(self.years)
            return sample_fronts_ds

        mock_tf_ds = MagicMock()
        mock_tf_ds.shuffle = MagicMock(return_value=mock_tf_ds)

        with patch.object(ERA5PredictorConfig, "build", mock_era5_build):
            with patch.object(FrontsDataConfig, "build", mock_fronts_build):
                with patch("fronts.utils.data_utils.normalize_dataset", return_value=sample_era5_ds):
                    with patch("fronts.data.config.create_dataloader", return_value=mock_tf_ds):
                        cfg.build()

        # train_years=[2010] and val_years=[2011] both get injected; test_years=[] is skipped
        assert [2010] in all_era5_years
        assert [2011] in all_era5_years
        assert [2010] in all_fronts_years
        assert [2011] in all_fronts_years


# ---------------------------------------------------------------------------
# Integration: DataConfig in TrainConfig YAML round-trip
# ---------------------------------------------------------------------------


class TestDataConfigInTrainConfig:
    @pytest.fixture
    def sample_yaml_with_data(self, tmp_path, sample_config_dict):
        """Extend sample_config_dict with a minimal data block."""
        config = dict(sample_config_dict)
        config["data"] = _minimal_data_config_dict()
        path = tmp_path / "config_with_data.yaml"
        path.write_text(yaml.dump(config, default_flow_style=False))
        return str(path)

    def test_train_config_parses_data_block(self, sample_yaml_with_data):
        config = open_config_yaml_as_dataclass(
            path=sample_yaml_with_data, config_class=TrainConfig
        )
        assert config is not None
        assert isinstance(config.data, DataConfig)
        assert config.data.train_years == [2010]
        assert isinstance(config.data.era5, ERA5PredictorConfig)
        assert isinstance(config.data.fronts, FrontsDataConfig)

    def test_actual_1702_yaml_parses_data_block(self):
        path = os.path.join(os.path.dirname(__file__), "..", "configs", "1702.yaml")
        if not os.path.exists(path):
            pytest.skip("1702.yaml not found")
        config = open_config_yaml_as_dataclass(path=path, config_class=TrainConfig)
        assert isinstance(config.data, DataConfig)
        assert config.data.train_years == list(range(2010, 2020))
        assert config.data.era5.store == (
            "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
        )
        # New unified levels field
        assert "surface" in config.data.era5.levels
        assert 1000 in config.data.era5.levels
        assert "temperature" in config.data.era5.variables
        assert "mean_sea_level_pressure" in config.data.era5.variables


# ---------------------------------------------------------------------------
# SURFACE_VARIABLE_MAP and SURFACE_ONLY_VARIABLES constants
# ---------------------------------------------------------------------------


class TestSurfaceVariableConstants:
    def test_surface_variable_map_has_temperature(self):
        assert "temperature" in SURFACE_VARIABLE_MAP
        assert SURFACE_VARIABLE_MAP["temperature"] == "2m_temperature"

    def test_surface_variable_map_has_wind_components(self):
        assert "u_component_of_wind" in SURFACE_VARIABLE_MAP
        assert "v_component_of_wind" in SURFACE_VARIABLE_MAP
        assert SURFACE_VARIABLE_MAP["u_component_of_wind"] == "10m_u_component_of_wind"
        assert SURFACE_VARIABLE_MAP["v_component_of_wind"] == "10m_v_component_of_wind"

    def test_surface_only_variables_contains_mslp(self):
        assert "mean_sea_level_pressure" in SURFACE_ONLY_VARIABLES

    def test_surface_variable_map_and_surface_only_are_disjoint(self):
        """No variable should appear in both sets."""
        overlap = set(SURFACE_VARIABLE_MAP.keys()) & SURFACE_ONLY_VARIABLES
        assert len(overlap) == 0


# ---------------------------------------------------------------------------
# TimeSelection
# ---------------------------------------------------------------------------


class TestTimeSelection:
    # --- Construction / validation ---

    def test_most_recent_valid(self):
        ts = TimeSelection(most_recent=True)
        assert ts.most_recent is True

    def test_timestamps_valid(self):
        ts = TimeSelection(timestamps=[datetime.datetime(2024, 6, 1, 12)])
        assert ts.timestamps is not None

    def test_date_range_valid(self):
        ts = TimeSelection(
            date_range=[datetime.datetime(2024, 6, 1), datetime.datetime(2024, 6, 7)]
        )
        assert ts.date_range is not None

    def test_raises_if_none_set(self):
        with pytest.raises(ValueError, match="Exactly one"):
            TimeSelection()

    def test_raises_if_multiple_set(self):
        with pytest.raises(ValueError, match="Exactly one"):
            TimeSelection(
                most_recent=True,
                timestamps=[datetime.datetime(2024, 6, 1, 12)],
            )

    def test_raises_if_date_range_wrong_length(self):
        with pytest.raises(ValueError, match="exactly two"):
            TimeSelection(date_range=[datetime.datetime(2024, 6, 1)])

    # --- dacite parsing ---

    def test_dacite_parses_most_recent(self):
        d = {"most_recent": True}
        ts = dacite.from_dict(TimeSelection, d, DACITE_CONFIG)
        assert ts.most_recent is True
        assert ts.timestamps is None
        assert ts.date_range is None

    def test_dacite_parses_timestamps(self):
        """dacite accepts pre-converted datetime objects in timestamps list."""
        ts1 = datetime.datetime(2024, 6, 1, 12, 0, 0)
        ts2 = datetime.datetime(2024, 6, 2, 0, 0, 0)
        d = {"timestamps": [ts1, ts2]}
        ts = dacite.from_dict(TimeSelection, d, DACITE_CONFIG)
        assert ts.timestamps is not None
        assert len(ts.timestamps) == 2
        assert ts.timestamps[0] == ts1

    def test_dacite_parses_date_range(self):
        """dacite accepts pre-converted datetime objects in date_range list."""
        start = datetime.datetime(2024, 6, 1, 0, 0, 0)
        end = datetime.datetime(2024, 6, 7, 18, 0, 0)
        d = {"date_range": [start, end]}
        ts = dacite.from_dict(TimeSelection, d, DACITE_CONFIG)
        assert ts.date_range is not None
        assert len(ts.date_range) == 2
        assert ts.date_range[0] == start
        assert ts.date_range[1] == end

    # --- apply() ---

    @pytest.fixture
    def time_ds(self):
        """Tiny xarray Dataset with 5 timesteps."""
        times = pd.date_range("2024-06-01", periods=5, freq="6h")
        return xr.Dataset(
            {"x": (["time"], np.arange(5, dtype="float32"))},
            coords={"time": times},
        )

    def test_apply_most_recent_returns_last_timestep(self, time_ds):
        ts = TimeSelection(most_recent=True)
        result = ts.apply(time_ds)
        assert len(result.time) == 1
        assert result.time.values[0] == time_ds.time.values[-1]

    def test_apply_timestamps_selects_correct_times(self, time_ds):
        target = datetime.datetime(2024, 6, 1, 6, 0)
        ts = TimeSelection(timestamps=[target])
        result = ts.apply(time_ds)
        assert len(result.time) == 1

    def test_apply_date_range_selects_slice(self, time_ds):
        ts = TimeSelection(
            date_range=[
                datetime.datetime(2024, 6, 1, 0),
                datetime.datetime(2024, 6, 1, 12),
            ]
        )
        result = ts.apply(time_ds)
        # 00Z, 06Z, 12Z = 3 timesteps
        assert len(result.time) == 3


# ---------------------------------------------------------------------------
# PredictConfig
# ---------------------------------------------------------------------------


def _minimal_predict_config_dict():
    return {
        "time_selection": {"most_recent": True},
        "normalization_method": "standard",
        "era5": {
            "domain_extent": [-140.0, -60.0, 20.0, 60.0],
            "variables": ["temperature", "mean_sea_level_pressure"],
            "levels": ["surface", 1000, 850],
            "store": "gs://fake-store",
            "chunks": {"time": 48},
            "consolidated": True,
        },
    }


class TestPredictConfig:
    def test_dacite_parses_predict_config(self):
        cfg = dacite.from_dict(PredictConfig, _minimal_predict_config_dict(), DACITE_CONFIG)
        assert isinstance(cfg.era5, ERA5PredictorConfig)
        assert isinstance(cfg.time_selection, TimeSelection)
        assert cfg.normalization_method == "standard"

    def test_most_recent_mode_parsed(self):
        cfg = dacite.from_dict(PredictConfig, _minimal_predict_config_dict(), DACITE_CONFIG)
        assert cfg.time_selection.most_recent is True

    def test_era5_levels_parsed(self):
        cfg = dacite.from_dict(PredictConfig, _minimal_predict_config_dict(), DACITE_CONFIG)
        assert "surface" in cfg.era5.levels
        assert 1000 in cfg.era5.levels

    def test_era5_variables_parsed(self):
        cfg = dacite.from_dict(PredictConfig, _minimal_predict_config_dict(), DACITE_CONFIG)
        assert "temperature" in cfg.era5.variables
        assert "mean_sea_level_pressure" in cfg.era5.variables

    def test_timestamps_mode_parsed(self):
        """dacite accepts pre-converted datetime objects in PredictConfig.time_selection."""
        d = _minimal_predict_config_dict()
        ts = datetime.datetime(2024, 6, 1, 12, 0, 0)
        d["time_selection"] = {"timestamps": [ts]}
        cfg = dacite.from_dict(PredictConfig, d, DACITE_CONFIG)
        assert cfg.time_selection.timestamps is not None
        assert cfg.time_selection.timestamps[0] == ts

    def test_date_range_mode_parsed(self):
        """dacite accepts pre-converted datetime objects in date_range."""
        d = _minimal_predict_config_dict()
        start = datetime.datetime(2024, 6, 1, 0, 0, 0)
        end = datetime.datetime(2024, 6, 7, 18, 0, 0)
        d["time_selection"] = {"date_range": [start, end]}
        cfg = dacite.from_dict(PredictConfig, d, DACITE_CONFIG)
        assert cfg.time_selection.date_range is not None
        assert len(cfg.time_selection.date_range) == 2

    @pytest.fixture
    def raw_zarr_ds(self):
        """ERA5-like Dataset as it would come from the zarr store: pure integer levels,
        with a separate surface variable (no level dim) and pressure-level variable.
        Matches levels=["surface", 1000, 850] in the minimal predict config."""
        times = pd.date_range("2024-06-01", periods=4, freq="6h")
        lats = np.linspace(20.0, 60.0, 3)
        lons = np.linspace(-140.0, -60.0, 3)
        levels = [1000, 850]

        return xr.Dataset(
            {
                "temperature": (
                    ["time", "level", "latitude", "longitude"],
                    np.random.rand(4, 2, 3, 3).astype("float32"),
                ),
                "2m_temperature": (
                    ["time", "latitude", "longitude"],
                    np.random.rand(4, 3, 3).astype("float32"),
                ),
                "mean_sea_level_pressure": (
                    ["time", "latitude", "longitude"],
                    np.random.rand(4, 3, 3).astype("float32"),
                ),
            },
            coords={
                "time": times,
                "level": levels,
                "latitude": lats,
                "longitude": lons,
            },
        )

    def test_build_returns_dataset(self, raw_zarr_ds):
        """PredictConfig.build() returns a normalized xr.Dataset."""
        cfg = dacite.from_dict(PredictConfig, _minimal_predict_config_dict(), DACITE_CONFIG)

        with patch("xarray.open_zarr", return_value=raw_zarr_ds):
            with patch(
                "fronts.utils.data_utils.normalize_dataset", return_value=raw_zarr_ds
            ) as mock_norm:
                result = cfg.build()

        mock_norm.assert_called_once()
        assert isinstance(result, xr.Dataset)

    def test_build_calls_time_selection_apply(self, raw_zarr_ds):
        """PredictConfig.build() delegates time filtering to TimeSelection.apply()."""
        cfg = dacite.from_dict(PredictConfig, _minimal_predict_config_dict(), DACITE_CONFIG)

        with patch("xarray.open_zarr", return_value=raw_zarr_ds):
            with patch(
                "fronts.utils.data_utils.normalize_dataset", return_value=raw_zarr_ds
            ):
                with patch.object(
                    TimeSelection, "apply", wraps=cfg.time_selection.apply
                ) as mock_apply:
                    cfg.build()

        mock_apply.assert_called_once()

    def test_predict_config_yaml_roundtrip(self, tmp_path):
        """predict_1702.yaml loads into PredictConfig via open_config_yaml_as_dataclass."""
        predict_yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "predict_1702.yaml"
        )
        if not os.path.exists(predict_yaml_path):
            pytest.skip("predict_1702.yaml not found")

        cfg = open_config_yaml_as_dataclass(
            path=predict_yaml_path, config_class=PredictConfig
        )
        assert isinstance(cfg, PredictConfig)
        assert isinstance(cfg.time_selection, TimeSelection)
        assert cfg.time_selection.most_recent is True
        # New unified levels: should include "surface" and integer hPa values
        assert "surface" in cfg.era5.levels
        assert 1000 in cfg.era5.levels
        assert 950 in cfg.era5.levels
        assert 900 in cfg.era5.levels
        assert 850 in cfg.era5.levels
        # Variables field
        assert "temperature" in cfg.era5.variables
        assert "mean_sea_level_pressure" in cfg.era5.variables
