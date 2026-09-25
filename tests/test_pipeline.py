"""Checks on the steps that would invalidate the results if they broke."""

import numpy as np
import pandas as pd
import pytest

from capstone import data, features, models
from capstone.evaluate import capture_at_k


def piece(hwy, beg, end, sfx="00", rdwy="1", typ="reg", mlge="0", ovlp="0"):
    return dict(HWYNUMB=hwy, ST_HWY_SFX=sfx, RDWY_ID=rdwy, RDWY_TYP=typ,
                MLGE_TYP=mlge, OVLAP_CD=ovlp, BEGMP=beg, ENDMP=end)


@pytest.fixture
def highways():
    return pd.DataFrame([
        piece("001", 0.0, 0.6),
        piece("001", 0.6, 2.3),
        piece("001", 1.0, 2.0, rdwy="2"),            # second roadway of a divided section
        piece("001", 0.4, 0.7, sfx="AA", typ="con"),  # ramp
        piece("001", 1.5, 1.8, mlge="Z", ovlp="1"),   # Z-mileage (mileage equation)
        piece("002", 5.0, 6.0),
        piece("002", 7.0, 8.0),                       # no collisions ever
    ])


@pytest.fixture
def seg(highways):
    return data.build_segments(data.mainline_pieces(highways))


def record(lrm, mp, year, animal="Deer"):
    return dict(LRM_KEY=lrm, MP=mp, YEAR=str(year), ANIMAL=animal)


@pytest.fixture
def raw_collisions():
    return pd.DataFrame([
        record("00100I00", 0.5, 2012),
        record("00100D00", 1.5, 2016, "Elk"),   # non-add roadway -> same mainline mile
        record("00100I00", 2.3, 2013),          # terminal endpoint of mile 2
        record("00200I00", 6.0, 2014),          # exact end of network at mile 6 -> mile 5
        record("00100IZ1", 1.6, 2015),          # Z-mileage record
        record("001AAI00", 0.5, 2012),          # ramp
        record("00100I00", 9.5, 2012),          # no segment at that mile
        record("00100I00", np.nan, 2012),       # no milepost
        record("00100I00", 0.2, 2012, "Domestic Pet"),
        record("00100I00", 1.7, 2021),          # target-period record
    ])


@pytest.fixture
def assigned(raw_collisions, seg):
    return data.assign_collisions(data.prepare_collisions(raw_collisions), seg)


def test_segments_ids_lengths(seg):
    lengths = seg.set_index("segment_id")["length_mi"].to_dict()
    assert lengths == pytest.approx({"001_0": 1.0, "001_1": 1.0, "001_2": 0.3,
                                     "001Z1_1": 0.3, "002_5": 1.0, "002_7": 1.0})
    assert (seg["length_mi"] > 0).all()
    assert seg["segment_id"].is_unique


def test_collision_conservation(assigned):
    report = data.conservation_report(assigned)
    assert report["raw_records"] == 10
    assert report["dropped_species"] == 1
    assert report["deer_elk_records"] == 9
    assert report["non_mainline_roadway"] == 1
    assert report["off_network_milepost"] == 1
    assert report["invalid_location"] == 1
    assert report["assigned"] == 6
    assert report["assigned"] + report["unassigned"] == report["deer_elk_records"]
    got = assigned.loc[assigned["status"] == "assigned"].set_index("mp")["segment_id"].to_dict()
    assert got == {0.5: "001_0", 1.5: "001_1", 2.3: "001_2", 6.0: "002_5", 1.6: "001Z1_1", 1.7: "001_1"}


def test_fold_keeps_zero_history_segments(assigned, seg):
    fold = features.build_fold(assigned, seg, 2011, 2014, 2015, 2017)
    assert len(fold) == len(seg)
    assert set(fold["segment_id"]) == set(seg["segment_id"])
    assert fold.set_index("segment_id").loc["002_7", "history_count"] == 0


def test_history_excludes_target_years(assigned, seg):
    fold = features.build_fold(assigned, seg, 2011, 2014, 2015, 2017).set_index("segment_id")
    # 001_1 has an elk in 2016 (target) and a deer in 2021 (outside both windows).
    assert fold.loc["001_1", "history_count"] == 0
    assert fold.loc["001_1", "deer_elk_count"] == 1
    with pytest.raises(ValueError):
        features.build_fold(assigned, seg, 2011, 2015, 2015, 2017)


def test_no_post_cutoff_data_in_history(assigned, seg):
    before = features.build_fold(assigned, seg, 2011, 2014, 2015, 2017)
    extra = assigned.iloc[[0]].copy()
    extra["year"] = pd.array([2015], dtype="Int64")  # one year after the cutoff
    after = features.build_fold(pd.concat([assigned, extra]), seg, 2011, 2014, 2015, 2017)
    pd.testing.assert_series_equal(before["history_count"], after["history_count"])
    pd.testing.assert_series_equal(before["history_rate"], after["history_rate"])


def traffic_row(hwy, beg, end, rdwy="1", mlge="0", ovlp="0", **values):
    return dict(HWYNUMB=hwy, ST_HWY_SFX="00", RDWY_ID=rdwy, MLGE_TYP=mlge, OVLP_CD=ovlp,
                BEGMP=beg, ENDMP=end, **values)


def test_traffic_join_one_row_per_segment_and_length_weighted(seg, highways):
    aadt = pd.DataFrame([
        traffic_row("001", 0.0, 1.25, AADT=1000, TRK_AADT=100, TRK_PCT=10),
        traffic_row("001", 1.25, 2.3, AADT=3000, TRK_AADT=300, TRK_PCT=10),
        traffic_row("001", 1.0, 2.0, AADT=99999, TRK_AADT=0, TRK_PCT=0, rdwy="2"),  # ignored
        traffic_row("002", 5.0, 5.5, AADT=500, TRK_AADT=50, TRK_PCT=10),
    ])
    speed = pd.DataFrame([traffic_row("001", 0.0, 2.3, SPEED=55), traffic_row("002", 5.0, 8.0, SPEED=45)])
    lanes = pd.DataFrame([traffic_row("001", 0.0, 2.3, NO_LANES=2), traffic_row("001", 1.0, 2.0, rdwy="2", NO_LANES=2),
                          traffic_row("002", 5.0, 8.0, NO_LANES=2)])
    tf = features.traffic_features(seg, aadt, speed, lanes, highways)

    assert len(tf) == len(seg) and tf["segment_id"].is_unique
    t = tf.set_index("segment_id")
    assert t.loc["001_1", "aadt"] == pytest.approx(0.25 * 1000 + 0.75 * 3000)
    assert t.loc["001_1", "lanes"] == pytest.approx(4.0)
    assert t.loc["001_1", "divided_share"] == pytest.approx(1.0)
    assert t.loc["002_5", "aadt_imputed"] == 0 and t.loc["002_5", "aadt_coverage"] == pytest.approx(0.5)
    # No overlapping interval: carried from the nearest interval and flagged.
    assert t.loc["002_7", "aadt_imputed"] == 1 and t.loc["002_7", "aadt"] == 500
    # Z-mileage segment with no Z-keyed count: regular-mileage fallback, flagged.
    assert t.loc["001Z1_1", "aadt_imputed"] == 1 and t.loc["001Z1_1", "aadt"] == 3000
    assert tf["aadt"].notna().all()


def test_offset_uses_segment_length():
    fold = pd.DataFrame({"length_mi": [1.0, 0.3], "target_years": [5, 5]})
    np.testing.assert_allclose(models.offset(fold), np.log([5.0, 1.5]))


def test_capture_at_k_splits_tied_blocks():
    # 4 one-mile segments; top 50% = 2 miles. Two segments tie for the second slot.
    share = capture_at_k(score=[3, 1, 1, 0], collisions=[5, 4, 0, 1], miles=[1, 1, 1, 1], k=0.5)
    assert share == pytest.approx((5 + 0.5 * 4) / 10)


def test_pooled_target_table_one_row_per_segment(assigned, seg):
    t = features.target_table(assigned, seg, 2015, 2020)
    assert len(t) == len(seg) and t["segment_id"].is_unique
    assert t.set_index("segment_id").loc["001_1", "deer_elk_count"] == 1  # 2016 elk; 2021 deer excluded
    assert (t["target_years"] == 6).all()


def test_count_model_rejects_repeated_segments():
    rows = pd.DataFrame({"segment_id": ["a", "b", "a"], "length_mi": [1.0, 1.0, 1.0], "target_years": [3, 3, 3],
                         "deer_elk_count": [1, 2, 3], "log_aadt": [7.0, 8.0, 7.0], "truck_pct": [10.0, 12.0, 10.0],
                         "posted_speed": [55.0, 55.0, 55.0], "lanes": [2.0, 2.0, 2.0], "divided_share": [0.0, 0.0, 0.0]})
    with pytest.raises(ValueError):
        models.fit_b1(rows, groups=["x", "y", "x"])
