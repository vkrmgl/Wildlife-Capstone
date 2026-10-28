"""Covariates on synthetic rasters and polygons: terrain, climate,
population, land cover, GBIF counts, and the dictionary."""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

from capstone import climate, gbif, landcover, population, terrain
from capstone.states import STATES
from capstone.table import check_dictionary, dictionary

MI = 1609.344
SEP = 4 * MI  # distance between the two test segments


@pytest.fixture
def segments():
    """Two one-mile segments in EPSG:5070 near lon -123, lat 44.5, returned
    in EPSG:4326 like the real segments."""
    origin = gpd.GeoSeries(gpd.points_from_xy([-122.5], [44.5]), crs="EPSG:4326").to_crs("EPSG:5070")
    x0, y0 = float(origin.x.iloc[0]), float(origin.y.iloc[0])
    g = gpd.GeoDataFrame({"segment_id": ["A", "B"], "length_mi": [1.0, 1.0]},
                         geometry=[LineString([(x0, y0), (x0 + MI, y0)]), LineString([(x0, y0 + SEP), (x0 + MI, y0 + SEP)])],
                         crs="EPSG:5070")
    return g.to_crs("EPSG:4326")


def write_raster(path, arr, transform, crs, nodata=None, dtype="float32"):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=dtype, crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(arr.astype(dtype), 1)


def test_terrain_tiles_and_stats(tmp_path, segments):
    assert terrain.tile_name(44, -123) == "n45w123"
    assert terrain.tiles_for((-122.9, 44.2, -122.1, 44.6)) == ["n45w123"]
    assert set(terrain.tiles_for((-123.1, 44.9, -122.9, 45.1))) == {"n45w124", "n45w123", "n46w124", "n46w123"}
    # A 1-degree tile at 1 arc-second with elevation rising 1 m per column (eastward).
    res = 1 / 3600
    cols = np.arange(3600, dtype="float64")
    elev = np.tile(cols, (3600, 1))
    write_raster(tmp_path / "USGS_1_n45w123.tif", elev, from_origin(-123, 45, res, res), "EPSG:4269", nodata=-999999)
    out = terrain.terrain_features(segments, tile_dir=tmp_path).set_index("segment_id")
    assert out.loc["A", "terrain_pixels"] > 50 and out.loc["A", "elevation_max_m"] > out.loc["A", "elevation_min_m"]
    # 1 m rise per pixel of ~22 m (cos 44.5 deg * 30.9 m): slope about 2.6 degrees everywhere
    expected = np.degrees(np.arctan(1 / (res * terrain.M_PER_DEG * np.cos(np.radians(44.5)))))
    assert out.loc["A", "slope_mean_deg"] == pytest.approx(expected, rel=0.02)
    assert out.loc["A", "slope_max_deg"] == pytest.approx(expected, rel=0.02)


def test_climate_sampling_and_summaries(tmp_path, segments):
    # A CONUS-like 4 km grid where ppt = month and tmean = 10 + month, nodata elsewhere.
    res = 1 / 24
    transform = from_origin(-125, 50, res, res)
    h, w = int(10 / res), int(10 / res)
    for year in (2015, 2016):
        for month in range(1, 13):
            for var, val in (("ppt", float(month)), ("tmean", 10.0 + month)):
                arr = np.full((h, w), val)
                if var == "tmean" and year == 2016 and month == 7:
                    arr[:] = -9999  # a month with no data over the segments
                path = climate.grid_path(var, year, month, tmp_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                write_raster(path.with_suffix(".tif"), arr, transform, "EPSG:4269", nodata=-9999)
                path.with_suffix(".tif").rename(path)
    lon, lat, idx = climate.sample_points(segments)
    assert len(lon) == len(idx) and set(idx) == {0, 1} and (idx == 0).sum() >= 8
    monthly = climate.monthly_means(segments, tmp_path, years=(2015, 2016))
    assert len(monthly) == 2 * 24
    out = climate.climate_summaries(monthly).set_index("segment_id")
    assert out.loc["A", "ppt_month_mean_mm"] == pytest.approx(6.5)
    assert out.loc["A", "ppt_month_min_mm"] == 1 and out.loc["A", "ppt_month_max_mm"] == 12
    assert out.loc["A", "ppt_annual_mm"] == pytest.approx(78)
    assert out.loc["A", "ppt_winter_month_mm"] == pytest.approx(5) and out.loc["A", "ppt_summer_month_mm"] == pytest.approx(7)
    assert out.loc["A", "tmean_month_max_c"] == 22 and out.loc["A", "tmean_summer_c"] == pytest.approx(17)
    assert out.loc["A", "weather_months"] == 24


def test_population_length_weighted(segments):
    seg5070 = segments.to_crs("EPSG:5070")
    x0, y0 = seg5070.geometry.iloc[0].coords[0]
    tracts = gpd.GeoDataFrame({"pop_density_per_km2": [10.0, 50.0, 1000.0]},
                              geometry=[box(x0 - 10, y0 - 100, x0 + 0.25 * MI, y0 + 100),
                                        box(x0 + 0.25 * MI, y0 - 100, x0 + 2 * MI, y0 + 100),
                                        box(x0 - 10, y0 + SEP + 500, x0 + MI, y0 + SEP + 600)],
                              crs="EPSG:5070")
    out = population.population_features(segments, tracts).set_index("segment_id")
    assert out.loc["A", "pop_density_per_km2"] == pytest.approx(0.25 * 10 + 0.75 * 50, rel=1e-3)
    assert out.loc["B", "pop_density_per_km2"] == 1000.0   # no tract touches B: nearest one


def test_nlcd_shares(tmp_path, segments):
    seg5070 = segments.to_crs("EPSG:5070")
    minx, miny, maxx, maxy = seg5070.total_bounds
    transform = from_origin(minx - 1000, maxy + 1000, 30, 30)
    h, w = int((maxy - miny + 2000) / 30) + 1, int((maxx - minx + 2000) / 30) + 1
    arr = np.full((h, w), 42, dtype="uint8")                    # evergreen forest everywhere ...
    row_a = int((maxy + 1000 - seg5070.geometry.iloc[0].coords[0][1]) / 30)
    arr[row_a - 2:row_a + 3, :] = 22                              # ... except a developed strip under segment A
    write_raster(tmp_path / "nlcd.tif", arr, transform, "EPSG:5070", nodata=250, dtype="uint8")
    out = landcover.nlcd_shares(segments, tmp_path / "nlcd.tif").set_index("segment_id")
    assert out.loc["B", "nlcd_forest"] == 1.0 and out.loc["B", "nlcd_dominant"] == "forest"
    assert 0.2 < out.loc["A", "nlcd_developed"] < 0.5 and out.loc["A", "nlcd_forest"] + out.loc["A", "nlcd_developed"] == pytest.approx(1)


def test_gbif_counts_and_bbox(segments):
    seg5070 = segments.to_crs("EPSG:5070")
    x0, y0 = seg5070.geometry.iloc[0].coords[0]
    pts = gpd.GeoSeries(gpd.points_from_xy([x0, x0 + 500, x0, x0 + 2000, x0],
                                           [y0 + 100, y0 - 4000, y0 + SEP, y0 + 100, y0 + 50]),
                        crs="EPSG:5070").to_crs("EPSG:4326")
    rec = pd.DataFrame({"decimalLatitude": pts.y, "decimalLongitude": pts.x,
                        "coordinateUncertaintyInMeters": [None, 100, 5_000, None, 50_000]})
    out = gbif.gbif_counts(segments, {"deer": rec}).set_index("segment_id")
    assert out.loc["A", "gbif_deer"] == 3           # points 1, 2, 4; the last is too uncertain
    assert out.loc["B", "gbif_deer"] == 1           # only the point on B; the others are 4 mi away
    assert gbif.bbox_wkt((-124.5, 41.9, -116.4, 46.3)) == \
        "POLYGON((-124.5 41.9,-116.4 41.9,-116.4 46.3,-124.5 46.3,-124.5 41.9))"


def test_dictionary_covers_table_columns():
    dic = dictionary(STATES["OR"])
    assert dic["column"].is_unique
    table = pd.DataFrame(columns=dic["column"])
    check_dictionary(table, dic)
    with pytest.raises(AssertionError):
        check_dictionary(table.drop(columns=["hpms_aadt"]), dic)
    assert "IDFG" in dictionary(STATES["ID"])["source"].values[12]
