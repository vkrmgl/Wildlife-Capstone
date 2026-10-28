"""Species groups, record preparation, assignment by milepost and by
coordinates, conservation, and window counts."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from capstone import collisions as c
from capstone.states import STATES

OR, ID = STATES["OR"], STATES["ID"]
MI = 1609.344


def test_species_groups():
    labels = ["Deer", "Mule Deer (Odocoileus hemionus)", "Elk (Cervus canadensis)", "Moose (Alces alces)",
              "Deer Mouse (Peromyscus maniculatus)", "Killdeer (Charadrius vociferus)", "Common Nighthawk",
              "Red-tailed Hawk (Buteo jamaicensis)", "Great Horned Owl (Bubo virginianus)", "Domestic Pet",
              "Wild Game - Small", "Antelope", "Pronghorn (Antilocapra americana)", "Big Horn Sheep",
              "Domestic Sheep (Ovis aries)", "Grizzly Bear (Ursus arctos)", "Mountain Lion, Cougar, or Puma",
              "Mt Goat", "Bald Eagle", "Golden Eagle (Aquila chrysaetos)", None]
    got = list(c.species_group(labels))
    assert got == ["deer", "deer", "elk", "moose", "other", "other", "other", "hawk", "owl", "other", "other",
                   "pronghorn", "pronghorn", "bighorn_sheep", "other", "bear", "cougar", "mt_goat", "bald_eagle",
                   "golden_eagle", "unknown"]


@pytest.fixture
def segments():
    """Route 00100I00 miles 0-2 (x = 0..2 mi along y = 0) and route 00200I00
    mile 5 (x = 10..11 mi), in EPSG:5070."""
    rows = [("00100I00_0", "00100I00", "001", 0, 0.0, 1.0, 1.0, LineString([(0, 0), (1 * MI, 0)])),
            ("00100I00_1", "00100I00", "001", 1, 1.0, 2.0, 1.0, LineString([(1 * MI, 0), (2 * MI, 0)])),
            ("00200I00_5", "00200I00", "002", 5, 5.0, 6.0, 1.0, LineString([(10 * MI, 0), (11 * MI, 0)]))]
    g = gpd.GeoDataFrame(rows, columns=["segment_id", "route_id", "route", "mile", "begin_mp", "end_mp",
                                        "length_mi", "geometry"], crs="EPSG:5070")
    return g


def lonlat(x_m, y_m):
    p = gpd.GeoSeries(gpd.points_from_xy([x_m], [y_m]), crs="EPSG:5070").to_crs("EPSG:4326")
    return float(p.y.iloc[0]), float(p.x.iloc[0])


def test_prepare_odot_maps_divided_roadway():
    raw = pd.DataFrame({"OBJECTID": [1, 2, 3], "LRM_KEY": ["00100I00", "00100D00", "bad"], "MP": [0.5, 1.5, 1.0],
                        "YEAR": ["2016", "2017", "2018"], "ANIMAL": ["Deer", "Elk", "Bear"],
                        "DATE": pd.to_datetime(["2016-01-01", "2017-01-01", "2018-01-01"]),
                        "LAT": [45.0, 45.0, 45.0], "LONGTD": [-120.0, -120.0, -120.0]})
    rec = c.prepare_odot(raw)
    assert list(rec["route_id"]) == ["00100I00", "00100I00", pd.NA] or pd.isna(rec["route_id"].iloc[2])
    assert list(rec["route_id"][:2]) == ["00100I00", "00100I00"]
    assert list(rec["group"]) == ["deer", "elk", "bear"] and list(rec["year"]) == [2016, 2017, 2018]


def test_prepare_idfg_dedupes_and_maps_routes(segments):
    base = dict(species="Mule Deer (Odocoileus hemionus)", source="IDFG Roadkill & Salvage Reports",
                observed=pd.Timestamp("2019-05-02"), highway="US-95", milepost=12.3, lat=44.0, lon=-116.0,
                sex="F", note="")
    raw = pd.DataFrame([dict(base, OBJECTID=1, GlobalID="a", path="p1"),
                        dict(base, OBJECTID=2, GlobalID="b", path="p2"),                 # double submission
                        dict(base, OBJECTID=3, GlobalID="c", path="p3", sex="M"),        # another animal
                        dict(base, OBJECTID=4, GlobalID="d", path="p4", highway="Hwy 95")])  # free text
    seg = segments.copy()
    seg["route"] = ["US-95", "US-95", "ID-3"]
    rec = c.prepare_idfg(raw, seg)
    assert c.prepare_idfg.duplicates == 1 and list(rec["record_id"]) == [1, 3, 4]
    assert list(rec["route_id"][:2]) == ["00100I00", "00100I00"] and pd.isna(rec["route_id"].iloc[2])
    assert list(rec["year"]) == [2019, 2019, 2019]


def test_assignment_milepost_then_coordinates(segments):
    on_mile1 = lonlat(1.5 * MI, 20)         # 20 m off the road at x = 1.5 mi
    far = lonlat(1.5 * MI, 500)             # 500 m away: beyond SNAP_MAX_M
    rec = pd.DataFrame({
        "record_id": [1, 2, 3, 4, 5, 6],
        "source": "t", "species": "Deer", "group": "deer",
        "date": pd.Timestamp("2016-06-01"), "year": pd.array([2016, 2016, 2016, 2021, 2016, 2016], dtype="Int64"),
        "lat": [np.nan, on_mile1[0], far[0], on_mile1[0], np.nan, np.nan],
        "lon": [np.nan, on_mile1[1], far[1], on_mile1[1], np.nan, np.nan],
        "route_id": pd.array(["00100I00", None, None, "00100I00", "00200I00", "00100I00"], dtype="string"),
        "mp": [0.5, np.nan, np.nan, 9.5, 6.0, np.nan],
    })
    out = c.assign_collisions(rec, segments, OR).set_index("record_id")
    assert out.loc[1, "status"] == "assigned_milepost" and out.loc[1, "segment_id"] == "00100I00_0"
    assert out.loc[2, "status"] == "assigned_coordinates" and out.loc[2, "segment_id"] == "00100I00_1"
    assert out.loc[2, "snap_m"] == pytest.approx(20, abs=1)
    assert out.loc[3, "status"] == "off_network"
    assert out.loc[4, "status"] == "assigned_coordinates"      # milepost off the network, coordinates rescue it
    assert out.loc[5, "segment_id"] == "00200I00_5"            # exact end of the route -> last segment
    assert out.loc[6, "status"] == "no_location"

    report = c.conservation_report(out.reset_index())
    assert report["assigned"] == 4 and report["off_network"] == 1 and report["no_location"] == 1
    assert report["deer_elk_assigned"] == 4

    # Idaho trusts coordinates first: record 4 would snap by coordinates even with a milepost
    rec_id = rec.copy()
    rec_id.loc[0, ["lat", "lon"]] = on_mile1
    out_id = c.assign_collisions(rec_id, segments, ID).set_index("record_id")
    assert out_id.loc[1, "status"] == "assigned_coordinates" and out_id.loc[1, "segment_id"] == "00100I00_1"

    windows = c.collision_windows(out.reset_index(), segments).set_index("segment_id")
    assert windows.loc["00100I00_0", "coll_deer_elk_2015_2020"] == 1
    assert windows.loc["00100I00_1", "coll_all_2015_2020"] == 1 and windows.loc["00100I00_1", "coll_all_2021_2025"] == 1
    assert windows.loc["00100I00_1", "coll_deer_2021_2025_per_yr"] == pytest.approx(0.2)
    assert windows.loc["00200I00_5", "coll_moose_2015_2020"] == 0
    years = c.collision_years(out.reset_index())
    assert years["n"].sum() == 4 and set(years["group"]) == {"deer"}
