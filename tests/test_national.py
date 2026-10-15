"""Checks on the per-segment table: grid transfer, HPMS milepost join, NLCD
shares, GBIF counts, and collision windows."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from capstone import national as n


def line_5070(x0, y0, x1, y1):
    return LineString([(x0, y0), (x1, y1)])


@pytest.fixture
def segments():
    """Three short segments in EPSG:5070 (returned in EPSG:4326 like the real
    segments). A: inside cell 0_0 then 1_0. B: entirely in cell 5_5. C: far
    from every panel cell."""
    geoms = [line_5070(100, 400, 1300, 400),       # 700 m in 0_0, 500 m in 1_0
             line_5070(4100, 4400, 4700, 4400),    # all in 5_5
             line_5070(90000, 90000, 91600, 90000)]
    g = gpd.GeoDataFrame({"segment_id": ["A", "B", "C"], "hwy": ["001"] * 3, "mkey": ["00"] * 3,
                          "mile": [0, 1, 2], "begin_mp": [0.0, 1.0, 2.0], "end_mp": [1.0, 2.0, 3.0],
                          "length_mi": [1.0, 1.0, 1.0], "is_short": [False] * 3},
                         geometry=geoms, crs=n.GRID_CRS)
    return g.to_crs("EPSG:4326")


@pytest.fixture
def panel():
    """Panel rows for cells 0_0 (two segments), 1_0 and 5_5, two months each."""
    rows = []
    cells = {"seg_0": "0_0", "seg_1": "0_0", "seg_2": "1_0", "seg_3": "5_5"}
    elev = {"seg_0": 100, "seg_1": 300, "seg_2": 1000, "seg_3": 50}
    for sid, cell in cells.items():
        for year, month, ppt, t in ((2015, 1, 100, 2), (2015, 7, 10, 20)):
            rows.append(dict(segment_id=sid, grid_id=cell, year=year, month=month, ppt_mean=ppt + elev[sid] / 100,
                             tmean_mean=t, elevation_mean_m=elev[sid], elevation_min_m=elev[sid] - 10,
                             elevation_max_m=elev[sid] + 10, slope_mean_deg=1.0, slope_max_deg=5.0,
                             pop_density_per_km2=2.0))
    return pd.DataFrame(rows)


def test_grid_id_matches_panel_scheme():
    assert list(n.grid_id([100, -100, 1600], [400, 400, -1])) == ["0_0", "-1_0", "2_-1"]
    centers = n.cell_centers(["0_0", "-1_2"])
    assert centers.tolist() == [[400.0, 400.0], [-400.0, 2000.0]]


def test_segment_cells_weights_sum_to_one(segments):
    cells = n.segment_cells(segments)
    sums = cells.groupby("segment_id")["weight"].sum()
    assert np.allclose(sums.values, 1.0)
    a = cells[cells["segment_id"] == "A"].set_index("grid_id")["weight"]
    assert set(a.index) == {"0_0", "1_0"} and a["0_0"] > a["1_0"] > 0.4


def test_transfer_static_and_borrowing(segments, panel):
    static, weather = n.panel_by_cell(panel)
    assert static.set_index("grid_id").loc["0_0", "elevation_mean_m"] == 200  # mean of the two segments
    assert static.set_index("grid_id").loc["0_0", "elevation_min_m"] == 90
    cells, cov = n.cover_segments(n.segment_cells(segments), segments, static["grid_id"])
    cov = cov.set_index("segment_id")
    assert cov.loc["A", "panel_coverage"] == pytest.approx(1.0)
    assert cov.loc["C", "panel_coverage"] == 0 and cov.loc["C", "panel_borrow_m"] > 80000
    terrain = n.transfer_static(cells, static).set_index("segment_id")
    assert 200 < terrain.loc["A", "elevation_mean_m"] < 1000       # weighted between cells 0_0 and 1_0
    assert terrain.loc["A", "elevation_max_m"] == 1010
    assert terrain.loc["B", "elevation_mean_m"] == 50
    assert terrain.loc["C", "elevation_mean_m"] == 50               # borrowed from the nearest cell (5_5)


def test_transfer_weather_summaries(segments, panel):
    static, weather = n.panel_by_cell(panel)
    cells, _ = n.cover_segments(n.segment_cells(segments), segments, static["grid_id"])
    climate = n.transfer_weather(cells, weather).set_index("segment_id")
    assert climate.loc["B", "ppt_month_min_mm"] == pytest.approx(10.5)
    assert climate.loc["B", "ppt_month_max_mm"] == pytest.approx(100.5)
    assert climate.loc["B", "tmean_month_max_c"] == 20 and climate.loc["B", "tmean_winter_c"] == 2
    assert climate.loc["B", "tmean_summer_c"] == 20
    assert np.isnan(climate.loc["B", "ppt_annual_mm"])              # no complete year in the fixture
    assert climate.loc["B", "weather_months"] == 2


def test_hpms_milepost_join(segments):
    hpms = pd.DataFrame({
        "route_id": ["00100I00", "00100I00", "00100D00", "00100I00", "001AAI00"],
        "begin_point": [0.0, 0.5, 0.0, 1.0, 0.0], "end_point": [0.5, 1.0, 1.0, 1.4, 5.0],
        "aadt": [1000, 3000, 9999, 0, 9999], "f_system": [1, 3, 1, 3, 7], "urban_id": [99999, 123, 1, 99999, 1],
        "through_lanes": [2, 2, 9, 2, 9], "speed_limit": [55, 0, 0, 65, 0], "access_control": [1, 1, 1, 2, 1],
        "aadt_single_unit": [100, 300, 0, 0, 0], "aadt_combination": [0, 0, 0, 0, 0],
    })
    f = n.hpms_features(segments, hpms).set_index("segment_id")
    assert f.loc["A", "hpms_aadt"] == pytest.approx(2000)           # length-weighted over the two halves
    assert f.loc["A", "hpms_truck_share"] == pytest.approx(0.1)
    assert f.loc["A", "hpms_speed_limit"] == 55                      # 0 means not collected
    assert f.loc["A", "hpms_rural_share"] == pytest.approx(0.5)
    assert f.loc["A", "hpms_f_system"] in (1, 3) and f.loc["A", "hpms_functional_class"] in ("interstate", "principal_arterial")
    assert f.loc["A", "hpms_coverage"] == pytest.approx(1.0)
    assert f.loc["B", "hpms_coverage"] == pytest.approx(0.4) and np.isnan(f.loc["B", "hpms_aadt"])
    assert f.loc["C", "hpms_coverage"] == 0


def test_nlcd_shares(tmp_path, segments):
    """A raster whose left half is forest (42) and right half developed (22),
    in EPSG:5070 covering segment A."""
    arr = np.full((100, 100), 22, dtype=np.uint8)
    arr[:, :50] = 42
    arr[0, 0] = n.NLCD_NODATA
    path = tmp_path / "nlcd.tif"
    with rasterio.open(path, "w", driver="GTiff", height=100, width=100, count=1, dtype="uint8",
                       crs=n.GRID_CRS, transform=from_origin(-500, 1900, 30, 30), nodata=n.NLCD_NODATA) as dst:
        dst.write(arr, 1)
    s = n.nlcd_shares(segments.iloc[:1], path, buffer_m=100).set_index("segment_id")
    assert s.loc["A", "nlcd_forest"] + s.loc["A", "nlcd_developed"] == pytest.approx(1.0)
    assert 0.5 < s.loc["A", "nlcd_forest"] < 0.9
    assert s.loc["A", "nlcd_dominant"] in ("forest", "developed") and s.loc["A", "nlcd_pixels"] > 0


def test_gbif_counts_radius_and_uncertainty(segments):
    near = segments.to_crs(n.GRID_CRS).geometry.iloc[0].interpolate(0.5, normalized=True)
    pts = gpd.GeoSeries([near, near, near], crs=n.GRID_CRS).to_crs("EPSG:4326")
    far = gpd.GeoSeries([line_5070(0, 0, 1, 1).interpolate(0)], crs=n.GRID_CRS).translate(0, 20000).to_crs("EPSG:4326")
    recs = pd.DataFrame({"decimalLatitude": list(pts.y) + list(far.y), "decimalLongitude": list(pts.x) + list(far.x),
                         "coordinateUncertaintyInMeters": [5, None, 50000, 5]})
    c = n.gbif_counts(segments, {"deer": recs}, radius_m=5000).set_index("segment_id")
    assert c.loc["A", "gbif_deer"] == 2        # the 50 km-uncertainty record is dropped, the far one is outside
    assert c.loc["C", "gbif_deer"] == 0


def test_collision_windows(segments):
    assigned = pd.DataFrame({
        "segment_id": ["A", "A", "A", "B", "B", "C"], "status": ["assigned"] * 5 + ["off_network_milepost"],
        "year": [2015, 2016, 2021, 2015, 2015, 2015], "ANIMAL": ["Deer", "Bear", "Elk", "Deer", "Deer", "Deer"],
    })
    w = n.collision_windows(assigned, segments, ("Deer", "Elk"), [(2015, 2020), (2021, 2025)]).set_index("segment_id")
    assert w.loc["A", "coll_all_2015_2020"] == 2 and w.loc["A", "coll_deer_elk_2015_2020"] == 1
    assert w.loc["A", "coll_all_2021_2025"] == 1 and w.loc["A", "coll_all_2021_2025_per_yr"] == pytest.approx(0.2)
    assert w.loc["B", "coll_all_2015_2020_per_yr"] == pytest.approx(2 / 6)
    assert w.loc["C", "coll_all_2015_2020"] == 0
    years = n.collision_years(assigned)
    assert years[(years["segment_id"] == "B") & (years["ANIMAL"] == "Deer")]["n"].item() == 2
