"""Segments from HPMS sections: mainline rules, orientation, cutting at
mileposts, and the milepost join of traffic attributes."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from capstone import network as n
from capstone.states import STATES

OR, ID = STATES["OR"], STATES["ID"]


def section(route_id, b, e, x0, x1, facility_type=2, reverse=False, **attrs):
    """A straight east-west section along y=0 in EPSG:5070 metres."""
    coords = [(x0, 0.0), (x1, 0.0)]
    row = dict(route_id=route_id, begin_point=b, end_point=e, facility_type=facility_type, ownership=1,
               f_system=3, urban_id=99999, through_lanes=2, speed_limit=55, access_control=3,
               aadt=1000, aadt_single_unit=50, aadt_combination=50, sectionlength=e - b)
    row.update(attrs)
    row["geometry"] = LineString(coords[::-1] if reverse else coords)
    return row


MI = 1609.344


@pytest.fixture
def hpms():
    rows = [
        section("00100I00", 0.0, 0.6, 0, 0.6 * MI),
        section("00100I00", 0.6, 2.3, 0.6 * MI, 2.3 * MI, reverse=True, aadt=3000),   # digitized backwards
        section("00100D00", 1.0, 2.0, 1.0 * MI, 2.0 * MI, facility_type=1),         # second roadway: not mainline
        section("001NKI00", 0.0, 0.3, 0, 0.3 * MI, facility_type=4),                 # ramp
        section("00100IZ1", 1.5, 1.8, 2.3 * MI, 2.6 * MI),                            # overlapping mileage key
        section("00200I00", 5.0, 6.0, 10 * MI, 11 * MI, urban_id=12345, speed_limit=0),
    ]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:5070")


def test_mainline_rules(hpms):
    kept = n.mainline_sections(hpms, OR)
    assert sorted(kept["route_id"].unique()) == ["00100I00", "00100IZ1", "00200I00"]
    idaho = pd.DataFrame({"route_id": ["01540AUS095", "01540DUS095", "01540AUS095"], "facility_type": [2, 6, 4]})
    assert list(ID.mainline(idaho)) == [True, False, False]
    assert list(ID.route_label(pd.Series(["01540AUS095", "00001AIN084", "01234ASH021"]))) == ["US-95", "I-84", "ID-21"]
    assert list(OR.route_label(pd.Series(["00100I00", "00100IZ1"]))) == ["001", "001Z1"]


def test_orientation_follows_mileposts(hpms):
    secs = n.orient_sections(n.mainline_sections(hpms, OR))
    first, second = secs[secs["route_id"] == "00100I00"].geometry.tolist()
    assert first.coords[-1] == second.coords[0]                   # flipped section now continues from the first
    assert second.coords[0][0] < second.coords[-1][0]            # runs west to east like its mileposts
    assert np.allclose(n.geometry_check(secs), 1.0)


def test_segments_cut_at_mileposts(hpms):
    seg = n.build_segments(n.orient_sections(n.mainline_sections(hpms, OR)), OR)
    s = seg.set_index("segment_id")
    assert list(s.index) == ["00100I00_0", "00100I00_1", "00100I00_2", "00100IZ1_1", "00200I00_5"]
    assert s.loc["00100I00_0", "length_mi"] == 1.0 and s.loc["00100I00_0", "n_pieces"] == 2
    assert s.loc["00100I00_2", "length_mi"] == pytest.approx(0.3) and not s.loc["00100I00_2", "is_short"]
    assert s.loc["00100IZ1_1", "route"] == "001Z1" and s.loc["00200I00_5", "route"] == "002"
    # geometry of mile 1 runs from x = 1 mi to x = 2 mi
    g = s.loc["00100I00_1", "geometry"]
    assert g.bounds[0] == pytest.approx(1 * MI, abs=1) and g.bounds[2] == pytest.approx(2 * MI, abs=1)
    assert seg.crs == hpms.crs


def test_hpms_features_length_weighted(hpms):
    secs = n.orient_sections(n.mainline_sections(hpms, OR))
    seg = n.build_segments(secs, OR)
    f = n.hpms_features(seg, secs).set_index("segment_id")
    # mile 0: 0.6 mi at 1000 and 0.4 mi at 3000
    assert f.loc["00100I00_0", "hpms_aadt"] == pytest.approx(0.6 * 1000 + 0.4 * 3000)
    assert f.loc["00100I00_0", "hpms_truck_share"] == pytest.approx(100 / (0.6 * 1000 + 0.4 * 3000))
    assert f.loc["00100I00_0", "hpms_coverage"] == pytest.approx(1.0)
    assert f.loc["00100I00_0", "hpms_rural_share"] == 1.0 and f.loc["00200I00_5", "hpms_rural_share"] == 0.0
    assert np.isnan(f.loc["00200I00_5", "hpms_speed_limit"])      # 0 = not collected
    assert f.loc["00200I00_5", "hpms_functional_class"] == "principal_arterial"
