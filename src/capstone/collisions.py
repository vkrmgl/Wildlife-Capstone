"""Collision records: one loader per state agency, a common record format,
and assignment of records to one-mile segments.

Common record columns: record_id, source, date, year, species (the agency's
label), group (a species group shared across states), lat, lon, route_id
(the HPMS route_id the record's milepost refers to, if any), mp.

A record is assigned by milepost (route_id + milepost, exact) or by
coordinates (nearest segment within SNAP_MAX_M). Each state says which to
trust first: ODOT locates records by milepost and derives coordinates from
it; IDFG records are pinned on a map and their highway/milepost fields are
free text that newer reports no longer fill in.
"""

import math
import re

import geopandas as gpd
import numpy as np
import pandas as pd

from capstone.data import ENDPOINT_TOL_MI, LAYERS, RAW, download_layer, record_download

SNAP_MAX_M = 100
TRAIN_YEARS = (2015, 2020)
TEST_YEARS = (2021, 2025)

# Species groups, in the order the rules are tried. A label falls in the
# first group whose pattern it matches; "other" otherwise. The groups match
# the GBIF taxa in capstone.gbif so collision and occurrence columns line up.
GROUP_RULES = [
    ("other", r"mouse|domestic|farm|pet\b|bridge"),
    ("deer", r"\bdeer\b|odocoileus"),
    ("elk", r"\belk\b|cervus canadensis"),
    ("moose", r"\bmoose\b|alces"),
    ("bear", r"\bbears?\b|ursus"),
    ("cougar", r"cougar|mountain lion|\bpuma\b"),
    ("pronghorn", r"pronghorn|antelope|antilocapra"),
    ("mt_goat", r"mountain goat|\bmt\.? goat|oreamnos"),
    ("bighorn_sheep", r"big ?horn|ovis canadensis"),
    ("owl", r"\bowls?\b|strigiformes|strigidae"),
    ("hawk", r"\bhawks?\b|accipiter|buteo"),
    ("bald_eagle", r"bald eagle|haliaeetus"),
    ("golden_eagle", r"golden eagle|aquila chrysaetos"),
]
GROUPS = [g for g, _ in GROUP_RULES if g != "other"] + ["other"]
TARGET_GROUPS = {"all": None, "deer_elk": ("deer", "elk"), "deer": ("deer",), "elk": ("elk",), "moose": ("moose",)}


def species_group(labels):
    """Map agency species labels to GROUPS; missing labels become 'unknown'."""
    s = pd.Series(labels, dtype="string").str.lower().str.strip()
    out = pd.Series("other", index=s.index, dtype="string")
    done = pd.Series(False, index=s.index)
    for group, pattern in GROUP_RULES:
        hit = s.str.contains(pattern, regex=True).fillna(False) & ~done
        out[hit] = group
        done |= hit
    out[s.isna()] = "unknown"
    return out.to_numpy()


# --- Sources ------------------------------------------------------------------

IDFG_URL = "https://gisportal-idfg.idaho.gov/hosting/rest/services/Roadkill/Roadkill_Observations/FeatureServer/1"


def download_odot(path):
    """ODOT animal incident records (carcass pickups with LRM_KEY and milepost)."""
    url = LAYERS["odot_wildlife_collisions"]
    df, total = download_layer(url)
    df["DATE"] = pd.to_datetime(df["DATE"], unit="ms")
    df.to_parquet(path, index=False)
    record_download("odot_wildlife_collisions", path, len(df), total)
    return df


def prepare_odot(raw, segments=None):
    key = raw["LRM_KEY"].astype("string")
    # The second roadway of a divided highway ("D") shares the mainline's
    # mileposts, so its records go to the inventory-direction route.
    route_id = (key.str[:5] + "I" + key.str[6:8]).where(key.str.len() == 8)
    return pd.DataFrame({
        "record_id": raw["OBJECTID"].astype(int).values,
        "source": "ODOT",
        "date": pd.to_datetime(raw["DATE"]).values,
        "year": pd.to_numeric(raw["YEAR"], errors="coerce").astype("Int64").values,
        "species": raw["ANIMAL"].astype("string").values,
        "group": species_group(raw["ANIMAL"]),
        "lat": pd.to_numeric(raw["LAT"], errors="coerce").values,
        "lon": pd.to_numeric(raw["LONGTD"], errors="coerce").values,
        "route_id": route_id.values,
        "mp": pd.to_numeric(raw["MP"], errors="coerce").values,
    })


def download_idfg(path):
    """IDFG roadkill observations (public salvage reports, IDFG and ITD
    apps). Coordinates come from the point geometry; the latitude and
    longitude attributes are swapped on many ITD records."""
    gdf, total = download_layer(IDFG_URL, geometry=True, id_field="OBJECTID")
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    df["lon"], df["lat"] = gdf.geometry.x.values, gdf.geometry.y.values
    for c in ("observed", "reported"):
        df[c] = pd.to_datetime(pd.to_numeric(df[c], errors="coerce"), unit="ms")
    df.to_parquet(path, index=False)
    record_download("idfg_roadkill_observations", path, len(df), total, source=IDFG_URL)
    return df


IDFG_ID_COLS = ["OBJECTID", "GlobalID", "path", "lat", "lon"]
IDFG_ROUTE = re.compile(r"^(I|US|ID)-(\d+)[A-Z]?$")


def primary_routes(segments):
    """route label -> the route_id carrying the most miles under that label,
    so a milepost on 'US-95' refers to the main US-95 route, not a spur."""
    miles = segments.groupby(["route", "route_id"])["length_mi"].sum().reset_index()
    best = miles.sort_values(["route", "length_mi"], ascending=[True, False]).drop_duplicates("route")
    return dict(zip(best["route"], best["route_id"]))


def prepare_idfg(raw, segments=None):
    """Submissions identical in every field except the system ids are
    double submissions and are dropped (prepare_idfg.duplicates says how
    many). Records with a standard highway label ('US-95', 'I-84', 'ID-21')
    and a milepost get the route_id of that route's primary HPMS route."""
    df = raw[~raw.duplicated([c for c in raw.columns if c not in IDFG_ID_COLS])].copy()
    prepare_idfg.duplicates = len(raw) - len(df)
    observed = pd.to_datetime(df["observed"], errors="coerce")
    label = df["highway"].astype("string").str.strip().str.upper()
    label = label.where(label.str.match(IDFG_ROUTE.pattern).fillna(False))
    routes = primary_routes(segments) if segments is not None else {}
    return pd.DataFrame({
        "record_id": df["OBJECTID"].astype(int).values,
        "source": df["source"].astype("string").values,
        "date": observed.values,
        "year": observed.dt.year.astype("Int64").values,
        "species": df["species"].astype("string").values,
        "group": species_group(df["species"]),
        "lat": pd.to_numeric(df["lat"], errors="coerce").values,
        "lon": pd.to_numeric(df["lon"], errors="coerce").values,
        "route_id": label.map(routes).astype("string").values,
        "mp": pd.to_numeric(df["milepost"], errors="coerce").values,
    })


SOURCES = {
    "odot": ("odot_wildlife_collisions.parquet", download_odot, prepare_odot),
    "idfg": ("idfg_roadkill_observations.parquet", download_idfg, prepare_idfg),
}


def collisions_path(state):
    return RAW / SOURCES[state.collisions][0]


def download_collisions(state):
    path = collisions_path(state)
    if path.exists():
        print(f"{path.name} exists, skipping")
        return path
    print(f"collisions {state.code}: {state.collisions}")
    df = SOURCES[state.collisions][1](path)
    print(f"  {len(df):,} records")
    return path


def prepare_collisions(state, segments):
    raw = pd.read_parquet(collisions_path(state))
    return SOURCES[state.collisions][2](raw, segments)


# --- Assignment ---------------------------------------------------------------

def assign_by_milepost(records, segments):
    """segment_id for each record from route_id and floor(mp); a record at the
    exact end of a route (within ENDPOINT_TOL_MI past the last segment's
    end) belongs to that last segment. NA where nothing matches."""
    by_key = segments.set_index(["route_id", "mile"])
    ids = []
    for route_id, mp in zip(records["route_id"], records["mp"]):
        if pd.isna(route_id) or not np.isfinite(mp):
            ids.append(pd.NA)
            continue
        k = math.floor(mp)
        sid = by_key["segment_id"].get((route_id, k))
        prev = (route_id, k - 1)
        if sid is None and mp - k <= ENDPOINT_TOL_MI and prev in by_key.index \
                and by_key.loc[prev, "end_mp"] >= k - ENDPOINT_TOL_MI:
            sid = by_key.loc[prev, "segment_id"]
        ids.append(sid if sid is not None else pd.NA)
    return pd.Series(ids, index=records.index, dtype="string")


def assign_by_coordinates(records, segments, max_m=SNAP_MAX_M):
    """(segment_id, distance in m) of the nearest segment within max_m of
    each record's point; NA beyond that or without coordinates."""
    ok = records["lat"].notna() & records["lon"].notna()
    pts = gpd.GeoDataFrame({"i": records.index[ok]},
                           geometry=gpd.points_from_xy(records.loc[ok, "lon"], records.loc[ok, "lat"]),
                           crs="EPSG:4326").to_crs("EPSG:5070")
    seg = segments[["segment_id", "geometry"]].to_crs("EPSG:5070")
    hit = gpd.sjoin_nearest(pts, seg, how="inner", max_distance=max_m, distance_col="snap_m")
    hit = hit.sort_values("snap_m").drop_duplicates("i").set_index("i")
    sid = pd.Series(pd.NA, index=records.index, dtype="string")
    dist = pd.Series(np.nan, index=records.index)
    sid[hit.index] = hit["segment_id"].astype("string")
    dist[hit.index] = hit["snap_m"]
    return sid, dist


def assign_collisions(records, segments, state, snap_max_m=SNAP_MAX_M):
    """Add segment_id, status, and snap_m to the records. Statuses:
    assigned_milepost, assigned_coordinates, no_location (neither a usable
    milepost nor coordinates), off_network (located, but on no segment)."""
    df = records.copy()
    df["segment_id"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df["snap_m"] = np.nan
    has_mp = df["route_id"].notna() & np.isfinite(df["mp"].astype(float))
    has_xy = df["lat"].notna() & df["lon"].notna()
    df["status"] = np.where(has_mp | has_xy, "off_network", "no_location")
    for method in state.assign_order:
        todo = df["segment_id"].isna()
        if method == "milepost":
            todo &= has_mp
            sid = assign_by_milepost(df[todo], segments)
            df.loc[todo, "segment_id"] = sid
            df.loc[todo & df["segment_id"].notna(), "status"] = "assigned_milepost"
        elif method == "coordinates":
            todo &= has_xy
            sid, dist = assign_by_coordinates(df[todo], segments, snap_max_m)
            df.loc[todo, "segment_id"] = sid
            df.loc[todo, "snap_m"] = dist
            df.loc[todo & df["segment_id"].notna(), "status"] = "assigned_coordinates"
        else:
            raise ValueError(f"unknown assignment method {method!r}")
    return df


def conservation_report(assigned, target=("deer", "elk")):
    """Record counts by outcome. Raises if they do not add up."""
    s = assigned["status"].value_counts()
    is_target = assigned["group"].isin(target)
    ok = assigned["status"].str.startswith("assigned")
    located = len(assigned) - int(s.get("no_location", 0))
    rows = {
        "raw_records": len(assigned),
        "no_location": int(s.get("no_location", 0)),
        "off_network": int(s.get("off_network", 0)),
        "assigned_milepost": int(s.get("assigned_milepost", 0)),
        "assigned_coordinates": int(s.get("assigned_coordinates", 0)),
        "assigned": int(ok.sum()),
        "assignment_rate_of_located": round(float(ok.sum() / located), 4) if located else 0.0,
        "deer_elk_records": int(is_target.sum()),
        "deer_elk_assigned": int((is_target & ok).sum()),
    }
    if rows["no_location"] + rows["off_network"] + rows["assigned"] != len(assigned):
        raise AssertionError(f"Collision conservation failed: {rows}")
    return rows


# --- Counts per segment -------------------------------------------------------

def collision_years(assigned):
    """Assigned records per segment, year, and group (long)."""
    ok = assigned[assigned["status"].str.startswith("assigned") & assigned["year"].notna()]
    return ok.groupby(["segment_id", "year", "group"]).size().rename("n").reset_index()


def collision_windows(assigned, segments, windows=(TRAIN_YEARS, TEST_YEARS), groups=TARGET_GROUPS):
    """Counts and per-year rates per segment for each (start, end) window and
    each target group: coll_{group}_{start}_{end} and ..._per_yr."""
    out = segments[["segment_id"]].copy()
    ok = assigned[assigned["status"].str.startswith("assigned")]
    for start, end in windows:
        w = ok[ok["year"].between(start, end)]
        years = end - start + 1
        for label, members in groups.items():
            sub = w if members is None else w[w["group"].isin(members)]
            n = out["segment_id"].map(sub.groupby("segment_id").size()).fillna(0).astype(int)
            out[f"coll_{label}_{start}_{end}"] = n.values
            out[f"coll_{label}_{start}_{end}_per_yr"] = (n / years).values
    return out
