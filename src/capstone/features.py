"""Segment-level collision windows and traffic/road features.

Traffic layers are current snapshots (AADT 2024; speed and lanes 2025) and are
used as fixed road attributes for every year.
"""

import numpy as np
import pandas as pd

from capstone.data import MAINLINE, REGULAR_MILEAGE, mileage_key


def window_counts(assigned, segments, start, end):
    """Deer/elk records per segment for years start..end (inclusive); every
    segment is returned, with 0 where there are no records."""
    ok = (assigned["status"] == "assigned") & assigned["year"].between(start, end)
    counts = assigned.loc[ok].groupby("segment_id").size()
    return segments["segment_id"].map(counts).fillna(0).astype(int).to_numpy()


def target_table(assigned, segments, start, end):
    """One row per segment with the count pooled over start..end. Model fits
    use this so each physical segment is one observation."""
    out = segments[["segment_id", "length_mi"]].copy()
    out["deer_elk_count"] = window_counts(assigned, segments, start, end)
    out["target_years"] = end - start + 1
    out["deer_elk_rate"] = out["deer_elk_count"] / out["length_mi"] / out["target_years"]
    return out


def build_fold(assigned, segments, history_start, history_end, target_start, target_end):
    """History counts from history_start..history_end and the target from
    target_start..target_end. The windows may not overlap."""
    if history_end >= target_start:
        raise ValueError(f"History {history_start}-{history_end} overlaps target {target_start}-{target_end}")
    fold = target_table(assigned, segments, target_start, target_end)
    fold["history_count"] = window_counts(assigned, segments, history_start, history_end)
    fold["history_years"] = history_end - history_start + 1
    fold["history_rate"] = fold["history_count"] / fold["length_mi"] / fold["history_years"]
    return fold


def interval_layer(layer, value_cols, roadway="1"):
    """Mainline milepost intervals of one roadway from a network or traffic layer."""
    m = (layer["ST_HWY_SFX"] == MAINLINE) & (layer["RDWY_ID"] == roadway)
    if "RDWY_TYP" in layer:
        m &= layer["RDWY_TYP"] == "reg"
    ovlp = layer["OVLP_CD"] if "OVLP_CD" in layer else layer["OVLAP_CD"]  # spelled differently per layer
    out = pd.DataFrame({
        "hwy": layer.loc[m, "HWYNUMB"].values,
        "mkey": mileage_key(layer.loc[m, "MLGE_TYP"], ovlp.loc[m]).values,
        "begin": layer.loc[m, "BEGMP"].astype(float).values,
        "end": layer.loc[m, "ENDMP"].astype(float).values,
    })
    for c in value_cols:
        out[c] = pd.to_numeric(layer.loc[m, c], errors="coerce").values
    return out[out["end"] - out["begin"] > 1e-9]


def interval_join(segments, intervals, value_cols):
    """Length-weighted mean of each value over the intervals overlapping each
    segment (intervals can cross segment boundaries), plus covered_mi. One row
    per segment; segments with no overlap get NaN and covered_mi 0."""
    j = segments[["segment_id", "hwy", "mkey", "begin_mp", "end_mp"]].merge(intervals, on=["hwy", "mkey"])
    j["overlap"] = np.minimum(j["end_mp"], j["end"]) - np.maximum(j["begin_mp"], j["begin"])
    j = j[j["overlap"] > 1e-9]
    out = pd.DataFrame({"segment_id": segments["segment_id"].values})
    out["covered_mi"] = out["segment_id"].map(j.groupby("segment_id")["overlap"].sum()).fillna(0.0).values
    for c in value_cols:
        ok = j[j[c].notna()]
        wmean = (ok[c] * ok["overlap"]).groupby(ok["segment_id"]).sum() / ok.groupby("segment_id")["overlap"].sum()
        out[c] = out["segment_id"].map(wmean).values
    return out


def nearest_interval(segments, intervals, col, same_mkey=True):
    """Value of the nearest interval (by milepost gap) on the same highway,
    and the same mileage key unless same_mkey is False (then regular mileage)."""
    ints = intervals[["hwy", "mkey", "begin", "end", col]].dropna()
    keys = ["hwy", "mkey"]
    if not same_mkey:
        ints = ints[ints["mkey"] == REGULAR_MILEAGE].drop(columns="mkey")
        keys = ["hwy"]
    j = segments[["segment_id", "hwy", "mkey", "begin_mp", "end_mp"]].merge(ints, on=keys)
    j["gap"] = np.maximum(0, np.maximum(j["begin"] - j["end_mp"], j["begin_mp"] - j["end"]))
    best = j.sort_values(["segment_id", "gap"]).drop_duplicates("segment_id").set_index("segment_id")
    return segments["segment_id"].map(best[col])


def traffic_features(segments, aadt, speed, lanes, highways):
    """One row per segment: aadt, log_aadt, truck_aadt, truck_pct, aadt_imputed,
    posted_speed, lanes, divided_share, and coverage shares for reporting."""
    out = segments[["segment_id"]].copy()
    length = segments["length_mi"].values

    a_int = interval_layer(aadt, ["AADT", "TRK_AADT", "TRK_PCT"])
    a = interval_join(segments, a_int, ["AADT", "TRK_AADT", "TRK_PCT"])
    out["aadt_coverage"] = np.minimum(1.0, a["covered_mi"].values / length)
    out["aadt"], out["truck_aadt"], out["truck_pct"] = a["AADT"].values, a["TRK_AADT"].values, a["TRK_PCT"].values
    # Segments with no AADT interval take the nearest one on the same highway.
    # Z-mileage segments have no Z-keyed counts, so they fall back to regular
    # mileage. Filled values are flagged.
    out["aadt_imputed"] = out["aadt"].isna().astype(int)
    for same_mkey in (True, False):
        todo = out["aadt"].isna()
        for src, dst in (("AADT", "aadt"), ("TRK_AADT", "truck_aadt"), ("TRK_PCT", "truck_pct")):
            out.loc[todo, dst] = nearest_interval(segments, a_int, src, same_mkey)[todo].values
    out["log_aadt"] = np.log(out["aadt"])

    s = interval_join(segments, interval_layer(speed, ["SPEED"]), ["SPEED"])
    out["speed_coverage"] = np.minimum(1.0, s["covered_mi"].values / length)
    out["posted_speed"] = s["SPEED"].values

    # Divided highways: add the second roadway's lanes over the share it covers.
    l1 = interval_join(segments, interval_layer(lanes, ["NO_LANES"], roadway="1"), ["NO_LANES"])
    l2 = interval_join(segments, interval_layer(lanes, ["NO_LANES"], roadway="2"), ["NO_LANES"])
    out["lanes_coverage"] = np.minimum(1.0, l1["covered_mi"].values / length)
    share2 = np.minimum(1.0, l2["covered_mi"].values / length)
    out["lanes"] = l1["NO_LANES"].values + np.nan_to_num(l2["NO_LANES"].values) * share2

    d = interval_join(segments, interval_layer(highways.assign(flag=1.0), ["flag"], roadway="2"), ["flag"])
    out["divided_share"] = np.minimum(1.0, d["covered_mi"].values / length)
    return out
