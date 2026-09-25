"""Download ODOT layers, build 1-mile state highway segments, and assign
deer/elk collision records to them.

ODOT's linear reference key (LRM_KEY) has 8 characters: highway (3), suffix (2,
"00" = mainline, others = ramps/connections/frontage roads), roadway (1, "I" =
add roadway, "D" = second roadway of a divided highway), and mileage key (2,
"00" = regular mileage, "Z1".."Z7" = overlapping mileage from a mileage
equation). Collision records and the network layers share these keys.
"""

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely import line_merge
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import substring

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

ODOT = "https://gis.odot.state.or.us/arcgis1006/rest/services"
LAYERS = {
    "odot_wildlife_collisions": f"{ODOT}/data_layers/wildlife_collisions/MapServer/0",
    "odot_state_highways": f"{ODOT}/data_layers/state_highways/MapServer/0",
    "odot_aadt": f"{ODOT}/transgis/catalog/MapServer/155",
    "odot_posted_speed": f"{ODOT}/transgis/catalog/MapServer/158",
    "odot_lanes": f"{ODOT}/transgis/catalog/MapServer/126",
}

TARGET_SPECIES = ("Deer", "Elk")
MAINLINE = "00"
REGULAR_MILEAGE = "00"
SHORT_SEGMENT_MI = 0.25
# A record at the exact end of a highway belongs to the last segment.
ENDPOINT_TOL_MI = 0.005


# --- Download ---------------------------------------------------------------

def get_json(url, params, attempts=6):
    """GET an ArcGIS endpoint with exponential backoff. ArcGIS reports some
    failures as HTTP 200 with an "error" body, so that is retried too."""
    for i in range(attempts):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(data["error"])
            return data
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            if i == attempts - 1:
                raise
            print(f"  retrying after error: {exc}")
            time.sleep(min(60, 2 ** (i + 1)))


def esri_to_shapely(geom):
    if not geom:
        return None
    if "x" in geom:
        return Point(geom["x"], geom["y"])
    paths = [[pt[:2] for pt in path] for path in geom["paths"] if len(path) >= 2]
    return LineString(paths[0]) if len(paths) == 1 else MultiLineString(paths)


def download_layer(url, geometry=False):
    """Page through a layer ordered by OBJECTID. Returns (frame, server count);
    the frame is a GeoDataFrame in EPSG:4326 when geometry=True."""
    page_size = min(int(get_json(url, {"f": "json"}).get("maxRecordCount") or 1000), 2000)
    total = int(get_json(f"{url}/query", {"where": "1=1", "returnCountOnly": "true", "f": "json"})["count"])
    rows, geoms = [], []
    for _ in range(math.ceil(total / page_size) + 5):
        params = {"where": "1=1", "outFields": "*", "orderByFields": "OBJECTID", "f": "json",
                  "resultOffset": len(rows), "resultRecordCount": page_size,
                  "returnGeometry": str(geometry).lower()}
        if geometry:
            params["outSR"] = 4326
        features = get_json(f"{url}/query", params).get("features", [])
        if not features:
            break
        rows += [f["attributes"] for f in features]
        geoms += [esri_to_shapely(f.get("geometry")) for f in features]
        print(f"\r  {len(rows):,} / {total:,}", end="", flush=True)
        time.sleep(0.25)
    print()
    df = pd.DataFrame(rows)
    if df["OBJECTID"].duplicated().any():
        raise RuntimeError(f"{url}: duplicate OBJECTIDs from paging")
    if geometry:
        df = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")
    return df, total


def record_download(name, path, rows, server_count):
    """Log source, row counts, date, and checksum in data/raw/manifest.json."""
    manifest_path = RAW / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[name] = {
        "source": LAYERS[name],
        "rows": rows,
        "server_count": server_count,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


# --- Collisions ---------------------------------------------------------------

def mileage_key(mlge_typ, ovlp_cd):
    """The 2-character mileage key used in LRM_KEY, from the layer fields."""
    mlge = pd.Series(mlge_typ, dtype="string").fillna("0")
    ovlp = pd.Series(ovlp_cd, dtype="string").fillna("0")
    return (mlge + ovlp).where(mlge != "0", REGULAR_MILEAGE)


def prepare_collisions(raw):
    """Decode LRM_KEY and flag target species. No rows are dropped here."""
    df = raw.copy()
    key = df["LRM_KEY"].astype("string")
    df["hwy"], df["sfx"], df["direction"], df["mkey"] = key.str[:3], key.str[3:5], key.str[5], key.str[6:8]
    df["year"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    df["mp"] = pd.to_numeric(df["MP"], errors="coerce")
    df["is_target_species"] = df["ANIMAL"].isin(TARGET_SPECIES)
    df["valid_location"] = df["mp"].notna() & np.isfinite(df["mp"]) & (key.str.len() == 8).fillna(False)
    return df


# --- Segments ---------------------------------------------------------------

def mainline_pieces(highways, roadway="1"):
    h = highways
    mask = (h["ST_HWY_SFX"] == MAINLINE) & (h["RDWY_TYP"] == "reg") & (h["RDWY_ID"] == roadway)
    out = h.loc[mask].copy()
    out["hwy"] = out["HWYNUMB"]
    out["mkey"] = mileage_key(out["MLGE_TYP"], out["OVLAP_CD"]).values
    return out


def piece_part(geom, frac0, frac1):
    """Part of a piece between two length fractions, assuming mileposts are
    proportional to distance along the piece (for mapping only)."""
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, MultiLineString):
        geom = line_merge(geom)
        if isinstance(geom, MultiLineString):
            return geom
    return substring(geom, frac0, frac1, normalized=True)


def build_segments(pieces):
    """Split mainline pieces at integer mileposts. A segment is the network
    inside [k, k+1) on one highway and mileage key; its length is the covered
    length, so terminal miles and miles broken by a mileage equation are short.
    Z-mileage keys are segmented separately because they repeat milepost
    numbers that exist on the regular mileage."""
    parts = []
    has_geom = "geometry" in pieces
    for row in pieces.itertuples(index=False):
        b, e = float(row.BEGMP), float(row.ENDMP)
        for k in range(math.floor(b), math.ceil(e)):
            lo, hi = max(b, k), min(e, k + 1)
            if hi - lo <= 1e-9:
                continue
            geom = piece_part(row.geometry, (lo - b) / (e - b), (hi - b) / (e - b)) if has_geom else None
            parts.append((row.hwy, row.mkey, k, lo, hi, hi - lo, geom))

    p = pd.DataFrame(parts, columns=["hwy", "mkey", "mile", "lo", "hi", "len", "geometry"])
    g = p.groupby(["hwy", "mkey", "mile"], sort=True)
    seg = g.agg(begin_mp=("lo", "min"), end_mp=("hi", "max"), length_mi=("len", "sum"),
                n_pieces=("len", "size")).reset_index()
    seg["length_mi"] = seg["length_mi"].round(4)
    seg.insert(0, "segment_id", [f"{h}_{k}" if m == REGULAR_MILEAGE else f"{h}{m}_{k}"
                                 for h, m, k in zip(seg.hwy, seg.mkey, seg.mile)])
    seg["is_short"] = seg["length_mi"] < SHORT_SEGMENT_MI
    if has_geom:
        seg["geometry"] = g["geometry"].agg(lambda s: line_merge(MultiLineString(
            [ln for x in s if x is not None for ln in getattr(x, "geoms", [x])]))).values
    else:
        seg["geometry"] = None
    return gpd.GeoDataFrame(seg, geometry="geometry", crs="EPSG:4326")


def assign_collisions(coll, segments):
    """Give every prepared record a status and, if assigned, a segment_id.

    Statuses: assigned, dropped_species, invalid_location, non_mainline_roadway
    (ramp/connection/frontage road), off_network_milepost. Records on the "D"
    roadway share mileposts with the mainline and go to the same mile."""
    df = coll.copy()
    df["segment_id"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df["status"] = "assigned"
    df.loc[~df["is_target_species"], "status"] = "dropped_species"
    df.loc[df["is_target_species"] & ~df["valid_location"], "status"] = "invalid_location"
    df.loc[(df["status"] == "assigned") & (df["sfx"] != MAINLINE), "status"] = "non_mainline_roadway"

    todo = df["status"] == "assigned"
    by_key = segments.set_index(["hwy", "mkey", "mile"])
    ids = []
    for hwy, mkey, mp in zip(df.loc[todo, "hwy"], df.loc[todo, "mkey"], df.loc[todo, "mp"]):
        k = math.floor(mp)
        sid = by_key["segment_id"].get((hwy, mkey, k))
        prev = (hwy, mkey, k - 1)
        if sid is None and mp - k <= ENDPOINT_TOL_MI and prev in by_key.index \
                and by_key.loc[prev, "end_mp"] >= k - ENDPOINT_TOL_MI:
            sid = by_key.loc[prev, "segment_id"]
        ids.append(sid)
    df.loc[todo, "segment_id"] = pd.array(ids, dtype="string")
    df.loc[todo & df["segment_id"].isna(), "status"] = "off_network_milepost"
    return df


def conservation_report(assigned):
    """Record counts by outcome. Raises if they do not add up."""
    s = assigned["status"].value_counts()
    target = int(assigned["is_target_species"].sum())
    n_assigned = int(s.get("assigned", 0))
    rows = {
        "raw_records": len(assigned),
        "dropped_species": int(s.get("dropped_species", 0)),
        "deer_elk_records": target,
        "invalid_location": int(s.get("invalid_location", 0)),
        "non_mainline_roadway": int(s.get("non_mainline_roadway", 0)),
        "off_network_milepost": int(s.get("off_network_milepost", 0)),
        "assigned": n_assigned,
        "unassigned": target - n_assigned,
        "assignment_rate": round(n_assigned / target, 4),
    }
    if rows["dropped_species"] + target != len(assigned):
        raise AssertionError(f"Collision conservation failed: {rows}")
    return rows
