"""Road network and traffic from FHWA HPMS (Highway Performance Monitoring
System), which every state submits with section geometry, mileposts
(begin_point/end_point, in miles along route_id), traffic, and road
attributes.

State-owned mainline sections are cut at integer mileposts into one-mile
segments. A segment is the network inside [k, k+1) on one route_id; its
length is the covered length, so terminal miles and miles broken by a
mileage equation are short.
"""

import math

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import line_merge
from shapely.geometry import MultiLineString

from capstone.data import RAW, SHORT_SEGMENT_MI, download_layer, piece_part, record_download
from capstone.features import interval_join

HPMS_YEAR = 2024
HPMS_URL = "https://geo.dot.gov/server/rest/services/Hosted/HPMS_FULL_{state}_{year}/FeatureServer/0"
HPMS_FIELDS = ["objectid", "route_id", "begin_point", "end_point", "f_system", "urban_id", "ownership",
               "facility_type", "through_lanes", "speed_limit", "access_control", "aadt",
               "aadt_single_unit", "aadt_combination", "sectionlength"]
HPMS_STATE_OWNED = "ownership=1"
HPMS_RURAL_URBAN_ID = 99999
HPMS_F_SYSTEM = {1: "interstate", 2: "freeway_expressway", 3: "principal_arterial", 4: "minor_arterial",
                 5: "major_collector", 6: "minor_collector", 7: "local"}
HPMS_NUMERIC = ["aadt", "f_system", "urban_id", "facility_type", "through_lanes", "speed_limit",
                "access_control", "aadt_single_unit", "aadt_combination"]


def hpms_path(state):
    return RAW / f"hpms_sections_{state.code.lower()}_{HPMS_YEAR}.parquet"


def download_hpms(state):
    """State-owned HPMS sections with geometry, to data/raw/. Skipped if present."""
    path = hpms_path(state)
    if path.exists():
        print(f"{path.name} exists, skipping")
        return path
    url = HPMS_URL.format(state=state.code, year=HPMS_YEAR)
    print(f"HPMS {state.code} {HPMS_YEAR}: {url}")
    gdf, total = download_layer(url, geometry=True, where=HPMS_STATE_OWNED, id_field="objectid",
                                out_fields=",".join(HPMS_FIELDS))
    gdf.to_parquet(path, index=False)
    record_download(path.stem, path, len(gdf), total, source=f"{url} where {HPMS_STATE_OWNED}")
    print(f"  {len(gdf):,} sections")
    return path


# --- Sections -> segments -----------------------------------------------------

def mainline_sections(hpms, state):
    """Inventory-direction mainline sections with a positive milepost length
    and a geometry, sorted along each route."""
    h = hpms.copy()
    for c in ("begin_point", "end_point", *HPMS_NUMERIC):
        h[c] = pd.to_numeric(h[c], errors="coerce")
    keep = state.mainline(h) & (h["end_point"] - h["begin_point"] > 1e-9).to_numpy()
    keep &= h.geometry.notna().to_numpy() & ~h.geometry.is_empty.to_numpy()
    out = h.loc[keep].sort_values(["route_id", "begin_point"], kind="stable").reset_index(drop=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=hpms.crs)


def _endpoints(geom):
    parts = getattr(geom, "geoms", [geom])
    return np.asarray(parts[0].coords[0][:2]), np.asarray(parts[-1].coords[-1][:2])


def _dist(a, b):
    return float(np.hypot(*(a - b)))


def orient_sections(sections):
    """Make each section's line run from begin_point to end_point. HPMS does
    not say which way a line was digitized, so a section is flipped when its
    end, not its start, touches the previous section on the route; the first
    section of a route is oriented so that its end is the endpoint closer to
    the next section."""
    geoms = []
    for _, grp in sections.groupby("route_id", sort=False):
        lines = [line_merge(g) if isinstance(g, MultiLineString) else g for g in grp.geometry]
        prev_end = None
        for i, line in enumerate(lines):
            s, e = _endpoints(line)
            if prev_end is not None:
                flip = _dist(e, prev_end) < _dist(s, prev_end)
            elif i + 1 < len(lines):
                ns, ne = _endpoints(lines[i + 1])
                flip = min(_dist(s, ns), _dist(s, ne)) < min(_dist(e, ns), _dist(e, ne))
            else:
                flip = False
            if flip:
                line = shapely.reverse(line)
            prev_end = _endpoints(line)[1]
            geoms.append(line)
    out = sections.copy()
    out["geometry"] = gpd.GeoSeries(geoms, index=out.index, crs=sections.crs)
    return out


def build_segments(sections, state):
    """One-mile segments from oriented mainline sections. Returns a
    GeoDataFrame in the sections' CRS with segment_id (route_id + mile),
    route (readable label), mileposts, covered length, and geometry."""
    parts = []
    for row in sections.itertuples(index=False):
        b, e = float(row.begin_point), float(row.end_point)
        for k in range(math.floor(b), math.ceil(e)):
            lo, hi = max(b, k), min(e, k + 1)
            if hi - lo <= 1e-9:
                continue
            geom = piece_part(row.geometry, (lo - b) / (e - b), (hi - b) / (e - b))
            parts.append((row.route_id, k, lo, hi, hi - lo, geom))
    p = pd.DataFrame(parts, columns=["route_id", "mile", "lo", "hi", "len", "geometry"])
    g = p.groupby(["route_id", "mile"], sort=True)
    seg = g.agg(begin_mp=("lo", "min"), end_mp=("hi", "max"), length_mi=("len", "sum"),
                n_pieces=("len", "size")).reset_index()
    seg["length_mi"] = seg["length_mi"].round(4)
    seg.insert(0, "segment_id", seg["route_id"] + "_" + seg["mile"].astype(str))
    seg.insert(2, "route", state.route_label(seg["route_id"]))
    seg["is_short"] = seg["length_mi"] < SHORT_SEGMENT_MI
    seg["geometry"] = g["geometry"].agg(lambda s: line_merge(MultiLineString(
        [ln for x in s if x is not None for ln in getattr(x, "geoms", [x])]))).values
    return gpd.GeoDataFrame(seg, geometry="geometry", crs=sections.crs)


def geometry_check(sections):
    """Ratio of each section's drawn length to its milepost length (miles).
    Values far from 1 mean the line and the mileposts disagree."""
    drawn = sections.to_crs("EPSG:5070").geometry.length / 1609.344
    return (drawn / (sections["end_point"] - sections["begin_point"])).to_numpy()


# --- Traffic and road attributes ----------------------------------------------

def hpms_intervals(sections):
    """Milepost intervals with the attributes used as features. HPMS codes
    "not collected" as 0 for these items; that is treated as missing."""
    out = pd.DataFrame({"route_id": sections["route_id"].values,
                        "begin": sections["begin_point"].astype(float).values,
                        "end": sections["end_point"].astype(float).values})
    for c in ("aadt", "f_system", "urban_id", "through_lanes", "speed_limit", "access_control",
              "aadt_single_unit", "aadt_combination"):
        out[c] = pd.to_numeric(sections[c], errors="coerce").values
    for c in ("aadt", "speed_limit", "through_lanes", "access_control", "f_system"):
        out.loc[out[c] <= 0, c] = np.nan
    out["rural"] = (out["urban_id"] == HPMS_RURAL_URBAN_ID).astype(float)
    out["truck_aadt"] = out["aadt_single_unit"].fillna(0) + out["aadt_combination"].fillna(0)
    return out[out["end"] - out["begin"] > 1e-9]


def interval_mode(segments, intervals, col):
    """Value of col covering the most of each segment (length-weighted mode)."""
    j = segments[["segment_id", "route_id", "begin_mp", "end_mp"]].merge(intervals, on="route_id")
    j["overlap"] = np.minimum(j["end_mp"], j["end"]) - np.maximum(j["begin_mp"], j["begin"])
    j = j[(j["overlap"] > 1e-9) & j[col].notna()]
    best = j.groupby(["segment_id", col])["overlap"].sum().reset_index() \
            .sort_values(["segment_id", "overlap"], ascending=[True, False]).drop_duplicates("segment_id")
    return segments["segment_id"].map(best.set_index("segment_id")[col])


def hpms_features(segments, sections):
    """One row per segment: hpms_coverage, hpms_aadt, hpms_log_aadt,
    hpms_truck_share, hpms_lanes, hpms_speed_limit, hpms_rural_share,
    hpms_f_system, hpms_functional_class, hpms_access_control. Values are
    length-weighted over the sections overlapping the segment."""
    ints = hpms_intervals(sections)
    j = interval_join(segments, ints, ["aadt", "truck_aadt", "through_lanes", "speed_limit", "rural"],
                      keys=("route_id",))
    out = segments[["segment_id"]].copy()
    out["hpms_coverage"] = np.minimum(1.0, j["covered_mi"].values / segments["length_mi"].values)
    out["hpms_aadt"] = j["aadt"].values
    out["hpms_log_aadt"] = np.log(out["hpms_aadt"])
    out["hpms_truck_share"] = j["truck_aadt"].values / j["aadt"].values
    out["hpms_lanes"] = j["through_lanes"].values
    out["hpms_speed_limit"] = j["speed_limit"].values
    out["hpms_rural_share"] = j["rural"].values
    out["hpms_f_system"] = interval_mode(segments, ints, "f_system").values
    out["hpms_functional_class"] = out["hpms_f_system"].map(HPMS_F_SYSTEM)
    out["hpms_access_control"] = interval_mode(segments, ints, "access_control").values
    return out
