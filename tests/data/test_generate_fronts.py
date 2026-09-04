import datetime
import pathlib

import numpy as np
import pandas as pd
import pytest
import yaml
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from shapely.geometry import LineString

from fronts import utils
from fronts.data import generate_fronts
from fronts.utils import BoundingBox

FULL_DOMAIN_BB = BoundingBox(lat_min=0.25, lat_max=80.0, lon_min=130.0, lon_max=369.75)
FRONTS_YAML_PATH = pathlib.Path(__file__).parent / "test_generate_fronts.yaml"


def test_grid_coordinates_matches_era5_full_domain_shape():
    latitude, longitude, longitude_unwrapped = generate_fronts.grid_coordinates(FULL_DOMAIN_BB)
    assert latitude.shape == (320,)
    assert longitude.shape == (960,)
    assert longitude[0] == pytest.approx(130.0)
    assert longitude[-1] == pytest.approx(9.75)  # wraps past 360, raw form
    assert np.all(np.diff(longitude_unwrapped) > 0)  # unwrapped form is monotonic
    assert longitude_unwrapped[-1] == pytest.approx(369.75)


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


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        # The middle HHMM field is when the file was issued, not the valid time; valid time is
        # the date combined with the trailing synoptic cycle hour (00/06/12/18 UTC).
        ("20250511_0345_00_MPC_final-anal_OPC_SFC_ANAL.xml", pd.Timestamp("2025-05-11T00:00")),
        ("20250815_1545_12_MPC_final-anal_OPC_SFC_ANAL.xml", pd.Timestamp("2025-08-15T12:00")),
        ("20251021_0945_06_MPC_final-anal_OPC_SFC_ANAL.xml", pd.Timestamp("2025-10-21T06:00")),
    ],
)
def test_parse_xml_valid_time(filename, expected):
    assert generate_fronts.parse_xml_valid_time(filename) == expected


def test_parse_xml_valid_time_returns_none_for_unrecognized_filename():
    assert generate_fronts.parse_xml_valid_time("readme.txt") is None


def test_convert_xml_to_dataset_places_cold_front_code(cold_front_xml):
    valid_time = pd.Timestamp("2025-05-11T00:00")
    ds = generate_fronts.convert_xml_to_dataset(str(cold_front_xml), valid_time, FULL_DOMAIN_BB, distance_km=25.0)
    assert ds["identifier"].dims == ("time", "latitude", "longitude")
    assert ds["time"].values[0] == valid_time.to_datetime64()
    assert (ds["identifier"].values == generate_fronts.PGEN_TYPE_IDENTIFIERS["COLD_FRONT"]).any()
    assert ds["identifier"].values.max() == generate_fronts.PGEN_TYPE_IDENTIFIERS["COLD_FRONT"]


def test_convert_xml_to_dataset_skips_non_front_line_types(tmp_path):
    # The real "final-anal" product's XML mixes fronts with non-front map features
    # (pressure-center symbols, contours, generic lines) under the same <Line> tag.
    xml = """<?xml version="1.0" encoding="utf-8"?>
<Product>
  <Line pgenType="LINE_SOLID">
    <Point Lon="-100.0" Lat="40.0"/>
    <Point Lon="-99.0" Lat="40.0"/>
  </Line>
  <Line pgenType="COLD_FRONT">
    <Point Lon="-100.0" Lat="45.0"/>
    <Point Lon="-99.0" Lat="45.0"/>
  </Line>
</Product>
"""
    path = tmp_path / "mixed.xml"
    path.write_text(xml)
    ds = generate_fronts.convert_xml_to_dataset(str(path), pd.Timestamp("2025-01-01"), FULL_DOMAIN_BB, distance_km=25.0)
    present_codes = set(np.unique(ds["identifier"].values))
    assert present_codes == {0.0, float(generate_fronts.PGEN_TYPE_IDENTIFIERS["COLD_FRONT"])}


def _single_front_document(lon: float, lat: float, lon2: float, lat2: float) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
  <Products xmlns:ns2="http://www.example.org/productType">
    <Product name="Default" type="Default">
      <Layer name="Default">
        <DrawableElement>
          <Line pgenType="COLD_FRONT">
            <Point Lat="{lat}" Lon="{lon}"/>
            <Point Lat="{lat2}" Lon="{lon2}"/>
          </Line>
        </DrawableElement>
      </Layer>
    </Product>
  </Products>
"""


def test_iter_line_elements_handles_concatenated_documents(tmp_path):
    # Some real files concatenate multiple complete XML documents (each with its own <?xml
    # declaration) into a single file rather than merging them into one document.
    xml = _single_front_document(-100.0, 40.0, -99.0, 40.0) + _single_front_document(-100.0, 45.0, -99.0, 45.0)
    path = tmp_path / "concatenated.xml"
    path.write_text(xml)
    lines = generate_fronts._iter_line_elements(str(path))
    assert len(lines) == 2
    assert all(line.get("pgenType") == "COLD_FRONT" for line in lines)


def test_iter_line_elements_handles_single_document(tmp_path):
    path = tmp_path / "single.xml"
    path.write_text(_single_front_document(-100.0, 40.0, -99.0, 40.0))
    lines = generate_fronts._iter_line_elements(str(path))
    assert len(lines) == 1


def test_iter_line_elements_skips_incomplete_fragment(tmp_path):
    # Observed on a real file: a repeated header ending mid-element with no closing tags,
    # followed by one genuinely complete document.
    incomplete_fragment = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
  <Products xmlns:ns2="http://www.example.org/productType">
    <Product name="Default" type="Default">
      <Layer name="Default">
        <DrawableElement>
"""
    xml = incomplete_fragment + _single_front_document(-100.0, 40.0, -99.0, 40.0)
    path = tmp_path / "fragment_then_complete.xml"
    path.write_text(xml)
    lines = generate_fronts._iter_line_elements(str(path))
    assert len(lines) == 1
    assert lines[0].get("pgenType") == "COLD_FRONT"


def test_convert_xml_to_dataset_reads_fronts_from_all_concatenated_documents(tmp_path):
    xml = _single_front_document(-100.0, 40.0, -99.0, 40.0) + _single_front_document(-100.0, 60.0, -99.0, 60.0)
    path = tmp_path / "concatenated.xml"
    path.write_text(xml)
    ds = generate_fronts.convert_xml_to_dataset(str(path), pd.Timestamp("2025-01-01"), FULL_DOMAIN_BB, distance_km=25.0)
    front_code = generate_fronts.PGEN_TYPE_IDENTIFIERS["COLD_FRONT"]
    hit_latitudes = ds["latitude"].values[(ds["identifier"].values[0] == front_code).any(axis=1)]
    assert np.any(np.isclose(hit_latitudes, 40.0, atol=0.25))
    assert np.any(np.isclose(hit_latitudes, 60.0, atol=0.25))


def test_convert_xml_to_dataset_handles_dateline_crossing_front(tmp_path):
    xml = """<?xml version="1.0" encoding="utf-8"?>
<Product>
  <Line pgenType="COLD_FRONT">
    <Point Lon="179.0" Lat="50.0"/>
    <Point Lon="-179.0" Lat="50.0"/>
  </Line>
</Product>
"""
    path = tmp_path / "dateline.xml"
    path.write_text(xml)
    ds = generate_fronts.convert_xml_to_dataset(str(path), pd.Timestamp("2025-01-01"), FULL_DOMAIN_BB, distance_km=25.0)
    assert (ds["identifier"].values == generate_fronts.PGEN_TYPE_IDENTIFIERS["COLD_FRONT"]).any()


def test_front_conversion_config_from_yaml():
    with open(FRONTS_YAML_PATH) as f:
        front_config = utils.parse_config_section(
            yaml.safe_load(f), generate_fronts.FrontConversionConfig, "front_conversion_config", utils.YAML_TYPE_HOOKS
        )
    assert front_config.xml_indir == "/tmp/test_xml"
    assert front_config.coordinates == BoundingBox(lat_min=0.25, lat_max=80.0, lon_min=130.0, lon_max=369.75)
    assert front_config.distance == 1.0


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
    assert list(found) == [pd.Timestamp("2025-05-11T00:00")]
    assert found[pd.Timestamp("2025-05-11T00:00")].endswith("20250511_0345_00_MPC_final-anal_OPC_SFC_ANAL.xml")


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


def test_inspect_fronts_store_times_returns_written_times(fronts_storage_config, cold_front_xml):
    netcdf_dir = pathlib.Path(fronts_storage_config.virtual_chunk_local_path)
    netcdf_dir.mkdir()
    valid_time = pd.Timestamp("2025-05-11T00:00")
    ds = generate_fronts.convert_xml_to_dataset(str(cold_front_xml), valid_time, FULL_DOMAIN_BB, distance_km=25.0)
    netcdf_path = netcdf_dir / "FrontObjects_202505110000_full.nc"
    ds.to_netcdf(netcdf_path, engine="netcdf4", mode="w")

    generate_fronts.write_netcdfs_to_icechunk_store(fronts_storage_config, [str(netcdf_path)], append=False)

    times = generate_fronts.inspect_fronts_store_times(fronts_storage_config)
    assert times is not None
    assert list(times) == [valid_time]


def test_write_netcdfs_to_icechunk_store_append_increases_time_steps(fronts_storage_config, cold_front_xml):
    netcdf_dir = pathlib.Path(fronts_storage_config.virtual_chunk_local_path)
    netcdf_dir.mkdir()

    first_time = pd.Timestamp("2025-05-11T00:00")
    first_ds = generate_fronts.convert_xml_to_dataset(str(cold_front_xml), first_time, FULL_DOMAIN_BB, distance_km=25.0)
    first_path = netcdf_dir / "FrontObjects_202505110000_full.nc"
    first_ds.to_netcdf(first_path, engine="netcdf4", mode="w")
    generate_fronts.write_netcdfs_to_icechunk_store(fronts_storage_config, [str(first_path)], append=False)

    second_time = pd.Timestamp("2025-05-11T06:00")
    second_ds = generate_fronts.convert_xml_to_dataset(
        str(cold_front_xml), second_time, FULL_DOMAIN_BB, distance_km=25.0
    )
    second_path = netcdf_dir / "FrontObjects_202505110600_full.nc"
    second_ds.to_netcdf(second_path, engine="netcdf4", mode="w")
    generate_fronts.write_netcdfs_to_icechunk_store(fronts_storage_config, [str(second_path)], append=True)

    times = generate_fronts.inspect_fronts_store_times(fronts_storage_config)
    assert list(times) == [first_time, second_time]


def test_write_netcdfs_to_icechunk_store_append_reuses_existing_time_encoding(fronts_storage_config, cold_front_xml):
    # Regression test for a real incident: a store built by an independent process (here,
    # "hours since 2000-01-01") had its newly-appended times corrupted because
    # write_netcdfs_to_icechunk_store previously always forced its own fixed encoding
    # ("minutes since 1970-01-01") regardless of what the store already used, producing a
    # scale mismatch against the array's actual stored units/calendar attrs. Seeds the store
    # through the same virtual-chunk (netCDF-backed) write path write_netcdfs_to_icechunk_store
    # itself uses, so the seeded "identifier" chunk's codec matches what gets appended later
    # (a real, non-virtual write would use a different codec and fail with an unrelated
    # "cannot concatenate arrays which were stored using different codecs" error).
    netcdf_dir = pathlib.Path(fronts_storage_config.virtual_chunk_local_path)
    netcdf_dir.mkdir()

    pre_existing_time = pd.Timestamp("2019-06-01T00:00")
    seed_local_ds = generate_fronts.convert_xml_to_dataset(
        str(cold_front_xml), pre_existing_time, FULL_DOMAIN_BB, distance_km=25.0
    )
    seed_path = netcdf_dir / "FrontObjects_201906010000_full.nc"
    seed_local_ds.to_netcdf(seed_path, engine="netcdf4", mode="w")

    url_prefix = f"file://{fronts_storage_config.virtual_chunk_local_path}"
    registry = ObjectStoreRegistry({url_prefix: LocalStore()})
    seed_virtual_ds = generate_fronts.open_virtual_mfdataset(
        [f"file://{seed_path}"],
        registry=registry,
        parser=generate_fronts.HDFParser(),
        concat_dim="time",
        coords="minimal",
        combine="nested",
    )
    seed_virtual_ds["time"].encoding = {
        "units": "hours since 2000-01-01",
        "calendar": "proleptic_gregorian",
        "dtype": "int64",
    }
    repo = utils.open_writable_icechunk_repo(
        fronts_storage_config.store_path, fronts_storage_config.virtual_chunk_local_path
    )
    session = repo.writable_session(fronts_storage_config.branch_name)
    seed_virtual_ds.vz.to_icechunk(session.store, group=fronts_storage_config.group_name)
    session.commit("seed store with an independent time encoding")

    new_time = pd.Timestamp("2025-05-11T00:00")
    new_ds = generate_fronts.convert_xml_to_dataset(str(cold_front_xml), new_time, FULL_DOMAIN_BB, distance_km=25.0)
    new_path = netcdf_dir / "FrontObjects_202505110000_full.nc"
    new_ds.to_netcdf(new_path, engine="netcdf4", mode="w")
    generate_fronts.write_netcdfs_to_icechunk_store(fronts_storage_config, [str(new_path)], append=True)

    times = generate_fronts.inspect_fronts_store_times(fronts_storage_config)
    assert list(times) == [pre_existing_time, new_time]


def test_write_netcdfs_to_icechunk_store_raises_on_empty_paths(fronts_storage_config):
    with pytest.raises(ValueError, match="netcdf_paths"):
        generate_fronts.write_netcdfs_to_icechunk_store(fronts_storage_config, [], append=False)


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


def test_main_end_to_end_converts_and_registers_new_files(tmp_path, monkeypatch):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    netcdf_dir = tmp_path / "netcdf"
    (xml_dir / "20250511_0345_00_MPC_final-anal_OPC_SFC_ANAL.xml").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<Product>
  <Line pgenType="COLD_FRONT">
    <Point Lon="-100.0" Lat="40.0"/>
    <Point Lon="-99.0" Lat="40.0"/>
  </Line>
</Product>
"""
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
front_conversion_config:
  xml_indir: "{xml_dir}"
  netcdf_outdir: "{netcdf_dir}"
  date_start: 2025-01-01T00:00:00
  date_end: 2025-12-31T23:59:00
  coordinates: [0.25, 80, 130, 369.75]
  distance: 1.0

icechunk_storage_config:
  store_path: "{tmp_path / "store"}"
  branch_name: "main"
  virtual_chunk_local_path: "{netcdf_dir}/"
""")
    monkeypatch.setattr("sys.argv", ["generate_fronts.py", "--config", str(config_path)])
    generate_fronts.main()

    storage_config = utils.IcechunkStorageConfig(
        store_path=str(tmp_path / "store"), branch_name="main", virtual_chunk_local_path=f"{netcdf_dir}/"
    )
    times = generate_fronts.inspect_fronts_store_times(storage_config)
    assert list(times) == [pd.Timestamp("2025-05-11T00:00")]

    # Second run with no new files is a no-op (does not raise, store unchanged).
    generate_fronts.main()
    times_after_rerun = generate_fronts.inspect_fronts_store_times(storage_config)
    assert list(times_after_rerun) == [pd.Timestamp("2025-05-11T00:00")]
