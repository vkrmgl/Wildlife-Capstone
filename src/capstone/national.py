"""One row per one-mile segment, with covariates that exist for every state.

Sources
- Terrain, population, and climate: carried over from the segment x month
  panel (data/processed/master_panel.parquet) through its 800 m grid cells.
  The panel has no geometry, but grid_id pins each of its segments to a cell
  in EPSG:5070, so a one-mile segment takes the values of the cells it
  crosses, weighted by how much of it lies in each.
- Traffic and road attributes: FHWA HPMS (Highway Performance Monitoring
  System), which every state submits. Oregon's route_id is ODOT's LRM_KEY and
  begin/end points are mileposts, so sections join to segments by milepost.
- Land cover: USGS Annual NLCD, shares of each class in a buffer around the
  segment.
- Species occurrences: GBIF records within a radius of the segment, one
  column per animal type in the ODOT collision records. eBird is excluded
  because its checklists are 97% of bird records and mark birding sites, not
  roads.
"""

import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import shapely
from rasterio.windows import from_bounds
from rasterstats import zonal_stats
from scipy.spatial import cKDTree

from capstone.features import interval_join

GRID_CRS = "EPSG:5070"  # NAD83 / Conus Albers, the panel's grid projection
GRID_M = 800

HPMS_URL = "https://geo.dot.gov/server/rest/services/Hosted/HPMS_FULL_{state}_{year}/FeatureServer/0"
HPMS_FIELDS = ["objectid", "route_id", "begin_point", "end_point", "f_system", "urban_id", "ownership",
               "facility_type", "through_lanes", "speed_limit", "access_control", "aadt",
               "aadt_single_unit", "aadt_combination", "sectionlength"]
HPMS_RURAL_URBAN_ID = 99999
HPMS_F_SYSTEM = {1: "interstate", 2: "freeway_expressway", 3: "principal_arterial", 4: "minor_arterial",
                 5: "major_collector", 6: "minor_collector", 7: "local"}

NLCD_URL = "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/Annual_NLCD_LndCov_{year}_CU_C1V1.zip"
NLCD_GROUPS = {"water": (11, 12), "developed": (21, 22, 23, 24), "barren": (31,), "forest": (41, 42, 43),
               "shrub": (52,), "grass": (71,), "agriculture": (81, 82), "wetland": (90, 95)}
NLCD_NODATA = 250
NLCD_BUFFER_M = 200

GBIF_API = "https://api.gbif.org/v1"
# ODOT ANIMAL label -> (column suffix, GBIF taxon key, taxon). Keys from
# /species/match. "Wild Game - Small", pets, farm animals and "Bridge" have
# no taxon and are not pulled.
GBIF_TAXA = {
    "Deer": ("deer", 2440964, "Odocoileus (genus)"),
    "Elk": ("elk", 8600904, "Cervus canadensis"),
    "Bear": ("bear", 2433407, "Ursus americanus"),
    "Cougar": ("cougar", 2435099, "Puma concolor"),
    "Antelope": ("antelope", 2440902, "Antilocapra americana"),
    "Mt Goat": ("mt_goat", 2441151, "Oreamnos americanus"),
    "Big Horn Sheep": ("bighorn_sheep", 2441119, "Ovis canadensis"),
    "Owl": ("owl", 1450, "Strigiformes (order)"),
    "Hawk": ("hawk", 2877, "Accipitridae (family)"),
    "Bald Eagle": ("bald_eagle", 2480446, "Haliaeetus leucocephalus"),
    "Golden Eagle": ("golden_eagle", 2480506, "Aquila chrysaetos"),
}
GBIF_EXCLUDED_DATASETS = {"4fa7b334-ce0d-4e88-aaae-2e0c138d049e": "eBird Observation Dataset"}
GBIF_PAGE = 300
GBIF_MAX_OFFSET = 100_000
GBIF_SLICE = 3000  # records per year/month slice; keeps page offsets shallow
GBIF_WORKERS = 4
GBIF_RADIUS_M = 5000
GBIF_MAX_UNCERTAINTY_M = 10_000
OREGON_WKT = "POLYGON((-124.7 41.9,-116.4 41.9,-116.4 46.3,-124.7 46.3,-124.7 41.9))"

TRAIN_YEARS = (2015, 2020)
TEST_YEARS = (2021, 2025)


# --- Panel -> grid cells -> segments -------------------------------------------

def grid_id(x, y):
    """The panel's grid_id for EPSG:5070 coordinates: floor(x/800)_floor(y/800)."""
    return (np.floor(np.asarray(x) / GRID_M).astype(int).astype(str) + "_"
            + np.floor(np.asarray(y) / GRID_M).astype(int).astype(str))


def cell_centers(ids):
    """EPSG:5070 center of each grid_id."""
    parts = pd.Series(ids, dtype="string").str.split("_", expand=True).astype(float)
    return np.column_stack([(parts[0] + 0.5) * GRID_M, (parts[1] + 0.5) * GRID_M])


STATIC_AGG = {"elevation_mean_m": "mean", "elevation_min_m": "min", "elevation_max_m": "max",
              "slope_mean_deg": "mean", "slope_max_deg": "max", "pop_density_per_km2": "mean"}


def panel_by_cell(panel):
    """Collapse the segment x month panel to its grid cells. Returns (static,
    weather): static terrain/population per cell, and precipitation and
    temperature per cell and month."""
    first = panel.drop_duplicates("segment_id")
    static = first.groupby("grid_id").agg(**{c: (c, f) for c, f in STATIC_AGG.items()},
                                          panel_segments=("segment_id", "size")).reset_index()
    weather = panel.groupby(["grid_id", "year", "month"], as_index=False)[["ppt_mean", "tmean_mean"]].mean()
    return static, weather


def segment_cells(segments, step_m=100):
    """Grid cells under each segment, weighted by points placed every step_m
    along it (so a cell's weight is the share of the segment inside it)."""
    proj = segments.to_crs(GRID_CRS)
    rows = []
    for sid, geom in zip(proj["segment_id"], proj.geometry):
        if geom is None or geom.is_empty:
            continue
        n = max(2, int(geom.length // step_m) + 1)
        pts = shapely.line_interpolate_point(geom, np.linspace(0, geom.length, n))
        ids = grid_id(shapely.get_x(pts), shapely.get_y(pts))
        for g, k in zip(*np.unique(ids, return_counts=True)):
            rows.append((sid, g, k / n))
    return pd.DataFrame(rows, columns=["segment_id", "grid_id", "weight"])


def cover_segments(cells, segments, known):
    """Keep the cells that have panel data. A segment with none borrows the
    cell nearest its midpoint. Returns (cells, coverage) where coverage has
    panel_coverage (weight on cells with data) and panel_borrow_m (distance
    to the borrowed cell, 0 when nothing was borrowed)."""
    known = pd.Index(pd.unique(known))
    has = cells[cells["grid_id"].isin(known)]
    cov = pd.DataFrame({"segment_id": segments["segment_id"].values})
    cov["panel_coverage"] = cov["segment_id"].map(has.groupby("segment_id")["weight"].sum()).fillna(0.0).values
    cov["panel_borrow_m"] = 0.0
    missing = cov["panel_coverage"] <= 0
    if missing.any():
        mid = segments.to_crs(GRID_CRS).geometry.interpolate(0.5, normalized=True)
        xy = np.column_stack([mid.x.values, mid.y.values])[missing.values]
        dist, idx = cKDTree(cell_centers(known)).query(xy)
        borrowed = pd.DataFrame({"segment_id": cov.loc[missing, "segment_id"].values,
                                 "grid_id": known[idx], "weight": 1.0})
        has = pd.concat([has, borrowed], ignore_index=True)
        cov.loc[missing, "panel_borrow_m"] = dist
    return has, cov


def weighted_mean(df, col, by="segment_id", w="weight"):
    ok = df[df[col].notna()]
    return (ok[col] * ok[w]).groupby(ok[by]).sum() / ok.groupby(by)[w].sum()


def transfer_static(cells, static):
    """Terrain and population per segment from the cells it crosses."""
    j = cells.merge(static, on="grid_id")
    out = pd.DataFrame({"segment_id": pd.unique(cells["segment_id"])})
    for col, how in STATIC_AGG.items():
        val = weighted_mean(j, col) if how == "mean" else j.groupby("segment_id")[col].agg(how)
        out[col] = out["segment_id"].map(val).values
    return out


WINTER, SUMMER = (12, 1, 2), (6, 7, 8)


def transfer_weather(cells, weather):
    """Climate summaries per segment from the panel's monthly weather: monthly
    precipitation mean/min/max and mean annual total, mean temperature with
    its coldest and warmest month, and winter and summer means."""
    j = cells.merge(weather, on="grid_id")
    for c in ("ppt_mean", "tmean_mean"):
        j[c] = j[c] * j["weight"]
    g = j.groupby(["segment_id", "year", "month"]).agg(ppt=("ppt_mean", "sum"), t=("tmean_mean", "sum"),
                                                       w=("weight", "sum")).reset_index()
    g["ppt"], g["t"] = g["ppt"] / g["w"], g["t"] / g["w"]
    by = g.groupby("segment_id")
    out = pd.DataFrame({"segment_id": by.size().index})
    out = out.set_index("segment_id")
    out["ppt_month_mean_mm"] = by["ppt"].mean()
    out["ppt_month_min_mm"] = by["ppt"].min()
    out["ppt_month_max_mm"] = by["ppt"].max()
    full = g.groupby(["segment_id", "year"]).agg(ppt=("ppt", "sum"), n=("month", "size")).reset_index()
    out["ppt_annual_mm"] = full[full["n"] == 12].groupby("segment_id")["ppt"].mean()
    out["tmean_c"] = by["t"].mean()
    out["tmean_month_min_c"] = by["t"].min()
    out["tmean_month_max_c"] = by["t"].max()
    for name, months in (("winter", WINTER), ("summer", SUMMER)):
        season = g[g["month"].isin(months)].groupby("segment_id")
        out[f"tmean_{name}_c"] = season["t"].mean()
        out[f"ppt_{name}_month_mm"] = season["ppt"].mean()
    out["weather_months"] = by.size()
    return out.reset_index()


# --- Collisions ---------------------------------------------------------------

def collision_years(assigned):
    """Assigned records per segment, year, and animal (long)."""
    ok = assigned[assigned["status"] == "assigned"]
    return ok.groupby(["segment_id", "year", "ANIMAL"]).size().rename("n").reset_index()


def collision_windows(assigned, segments, target_species, windows):
    """Counts and per-year rates per segment for each (start, end) window,
    for all animals and for the target species."""
    out = segments[["segment_id"]].copy()
    ok = assigned[assigned["status"] == "assigned"]
    for start, end in windows:
        w = ok[ok["year"].between(start, end)]
        years = end - start + 1
        for label, sub in (("all", w), ("deer_elk", w[w["ANIMAL"].isin(target_species)])):
            n = out["segment_id"].map(sub.groupby("segment_id").size()).fillna(0).astype(int)
            out[f"coll_{label}_{start}_{end}"] = n.values
            out[f"coll_{label}_{start}_{end}_per_yr"] = (n / years).values
    return out


# --- HPMS ---------------------------------------------------------------------

def hpms_intervals(hpms):
    """Mainline milepost intervals from HPMS sections. Oregon's route_id is the
    8-character LRM_KEY; roadway "I" is the inventory direction (the mainline
    the segments are built on)."""
    rid = hpms["route_id"].astype("string")
    m = (rid.str.len() == 8) & (rid.str[3:5] == "00") & (rid.str[5] == "I")
    h = hpms.loc[m]
    out = pd.DataFrame({"hwy": rid[m].str[:3].values, "mkey": rid[m].str[6:8].values,
                        "begin": h["begin_point"].astype(float).values, "end": h["end_point"].astype(float).values})
    for c in ("aadt", "f_system", "urban_id", "through_lanes", "speed_limit", "access_control",
              "aadt_single_unit", "aadt_combination"):
        out[c] = pd.to_numeric(h[c], errors="coerce").values
    # HPMS codes "not collected" as 0 for these; treat as missing.
    for c in ("aadt", "speed_limit", "through_lanes", "access_control", "f_system"):
        out.loc[out[c] <= 0, c] = np.nan
    out["rural"] = (out["urban_id"] == HPMS_RURAL_URBAN_ID).astype(float)
    out["truck_aadt"] = out["aadt_single_unit"].fillna(0) + out["aadt_combination"].fillna(0)
    return out[out["end"] - out["begin"] > 1e-9]


def interval_mode(segments, intervals, col):
    """Value of col covering the most of each segment (length-weighted mode)."""
    j = segments[["segment_id", "hwy", "mkey", "begin_mp", "end_mp"]].merge(intervals, on=["hwy", "mkey"])
    j["overlap"] = np.minimum(j["end_mp"], j["end"]) - np.maximum(j["begin_mp"], j["begin"])
    j = j[(j["overlap"] > 1e-9) & j[col].notna()]
    best = j.groupby(["segment_id", col])["overlap"].sum().reset_index() \
            .sort_values(["segment_id", "overlap"], ascending=[True, False]).drop_duplicates("segment_id")
    return segments["segment_id"].map(best.set_index("segment_id")[col])


def hpms_features(segments, hpms):
    """One row per segment: hpms_coverage, hpms_aadt, hpms_log_aadt,
    hpms_truck_share, hpms_lanes, hpms_speed_limit, hpms_rural_share,
    hpms_f_system, hpms_functional_class, hpms_access_control."""
    ints = hpms_intervals(hpms)
    num = ["aadt", "truck_aadt", "through_lanes", "speed_limit", "rural"]
    j = interval_join(segments, ints, num)
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


# --- NLCD ---------------------------------------------------------------------

def nlcd_crop(zip_path, out_path, bounds_4326, pad_m=5000):
    """Write the part of the CONUS Annual NLCD GeoTIFF (inside the MRLC zip)
    covering bounds_4326 (minx, miny, maxx, maxy), padded, to out_path."""
    with zipfile.ZipFile(zip_path) as z:
        tif = next(n for n in z.namelist() if n.lower().endswith(".tif"))
    with rasterio.open(f"zip+file://{zip_path}!{tif}") as src:
        b = gpd.GeoSeries([shapely.box(*bounds_4326)], crs="EPSG:4326").to_crs(src.crs).total_bounds
        win = from_bounds(b[0] - pad_m, b[1] - pad_m, b[2] + pad_m, b[3] + pad_m, src.transform) \
            .round_offsets().round_lengths()
        data = src.read(1, window=win)
        meta = src.meta.copy()
        meta.update(height=data.shape[0], width=data.shape[1], transform=src.window_transform(win),
                    compress="lzw", driver="GTiff")
        meta.pop("blockxsize", None), meta.pop("blockysize", None)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data, 1)
    return out_path


def nlcd_shares(segments, nlcd_path, buffer_m=NLCD_BUFFER_M):
    """Share of each NLCD group within buffer_m of the segment, the dominant
    group, and the number of pixels counted."""
    with rasterio.open(nlcd_path) as src:
        crs = src.crs
    geoms = segments.to_crs(crs).geometry.buffer(buffer_m)
    stats = zonal_stats(geoms, nlcd_path, categorical=True, nodata=NLCD_NODATA)
    out = segments[["segment_id"]].copy()
    counts = pd.DataFrame(stats).fillna(0)
    counts.columns = [int(c) for c in counts.columns]
    total = counts.sum(axis=1)
    for name, codes in NLCD_GROUPS.items():
        present = [c for c in codes if c in counts.columns]
        out[f"nlcd_{name}"] = (counts[present].sum(axis=1) / total.where(total > 0)).values
    shares = out[[f"nlcd_{n}" for n in NLCD_GROUPS]]
    out["nlcd_dominant"] = shares.idxmax(axis=1).str.replace("nlcd_", "").where(total.values > 0)
    out["nlcd_pixels"] = total.astype(int).values
    return out


# --- GBIF ---------------------------------------------------------------------

def gbif_get(path, params, attempts=6):
    for i in range(attempts):
        try:
            resp = requests.get(f"{GBIF_API}/{path}", params=params, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if i == attempts - 1:
                raise
            print(f"  retrying after error: {exc}")
            time.sleep(min(60, 2 ** (i + 1)))


def gbif_base_params(taxon_key, geometry=OREGON_WKT):
    return {"taxonKey": taxon_key, "geometry": geometry, "hasCoordinate": "true",
            "hasGeospatialIssue": "false", "occurrenceStatus": "PRESENT"}


def gbif_datasets(taxon_key, geometry=OREGON_WKT):
    """Dataset keys holding records for the taxon in the area, minus the
    excluded ones, with their record counts."""
    facets = gbif_get("occurrence/search", {**gbif_base_params(taxon_key, geometry), "limit": 0,
                                            "facet": "datasetKey", "facetLimit": 1000})["facets"]
    counts = facets[0]["counts"] if facets else []
    return {c["name"]: c["count"] for c in counts if c["name"] not in GBIF_EXCLUDED_DATASETS}


KEEP = ["key", "species", "scientificName", "taxonKey", "decimalLatitude", "decimalLongitude",
        "coordinateUncertaintyInMeters", "year", "month", "basisOfRecord", "datasetKey"]


def gbif_slices(params, years=None, months=None):
    """Split the search into (params, count) slices of at most GBIF_SLICE
    records, by year range and then by month, because GBIF's search slows to
    minutes per page at deep offsets. Searches small enough for one slice are
    not filtered by year, so they keep undated records; split searches lose
    them (about 0.1% of records)."""
    p = dict(params)
    if years:
        p["year"] = f"{years[0]},{years[1]}"
    if months:
        p["month"] = f"{months[0]},{months[1]}"
    count = gbif_get("occurrence/search", {**p, "limit": 0})["count"]
    if count == 0:
        return []
    if count <= GBIF_SLICE:
        return [(p, count)]
    years = years or (1900, 2030)
    if years[0] < years[1]:
        mid = (years[0] + years[1]) // 2
        return gbif_slices(params, (years[0], mid)) + gbif_slices(params, (mid + 1, years[1]))
    lo, hi = months or (1, 12)
    if lo < hi:
        mid = (lo + hi) // 2
        return gbif_slices(params, years, (lo, mid)) + gbif_slices(params, years, (mid + 1, hi))
    if count > GBIF_MAX_OFFSET:
        raise RuntimeError(f"{count} records in one month exceed GBIF's paging limit")
    return [(p, count)]


def gbif_pages(params):
    """All records for the search, fetched in shallow pages over year/month slices."""
    slices = gbif_slices(params)
    total = sum(c for _, c in slices)
    jobs = [(p, offset) for p, c in slices for offset in range(0, c, GBIF_PAGE)]

    def page(job):
        p, offset = job
        got = gbif_get("occurrence/search", {**p, "limit": GBIF_PAGE, "offset": offset}).get("results", [])
        return [{k: r.get(k) for k in KEEP} for r in got]

    rows = []
    with ThreadPoolExecutor(GBIF_WORKERS) as pool:
        for got in pool.map(page, jobs):
            rows += got
            print(f"\r  {len(rows):,} / {total:,} ({len(slices)} slices)", end="", flush=True)
    print()
    return rows


def download_gbif(taxon_key, geometry=OREGON_WKT):
    """Records for the taxon in the area from all datasets except the excluded
    ones, as a DataFrame. Returns an empty frame when there are none."""
    datasets = gbif_datasets(taxon_key, geometry)
    print(f"  {sum(datasets.values()):,} records in {len(datasets)} datasets (excluding "
          f"{', '.join(GBIF_EXCLUDED_DATASETS.values())})")
    if not datasets:
        return pd.DataFrame(columns=KEEP)
    rows = gbif_pages({**gbif_base_params(taxon_key, geometry), "datasetKey": list(datasets)})
    df = pd.DataFrame(rows, columns=KEEP)
    if df["key"].duplicated().any():
        df = df.drop_duplicates("key")
    return df


def gbif_counts(segments, records, radius_m=GBIF_RADIUS_M, max_uncertainty_m=GBIF_MAX_UNCERTAINTY_M):
    """Records within radius_m of each segment, per taxon. `records` maps a
    column suffix to a frame of GBIF records; records with a stated
    coordinate uncertainty above max_uncertainty_m are ignored."""
    proj = segments[["segment_id", "geometry"]].to_crs(GRID_CRS)
    buffers = gpd.GeoDataFrame(proj[["segment_id"]], geometry=proj.geometry.buffer(radius_m), crs=GRID_CRS)
    out = segments[["segment_id"]].copy()
    for name, df in records.items():
        df = df[df["decimalLatitude"].notna() & df["decimalLongitude"].notna()]
        unc = pd.to_numeric(df["coordinateUncertaintyInMeters"], errors="coerce")
        df = df[unc.isna() | (unc <= max_uncertainty_m)]
        pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df["decimalLongitude"], df["decimalLatitude"]),
                               crs="EPSG:4326").to_crs(GRID_CRS)
        hits = gpd.sjoin(pts, buffers, predicate="within", how="inner")
        out[f"gbif_{name}"] = out["segment_id"].map(hits.groupby("segment_id").size()).fillna(0).astype(int).values
    return out


# --- Assembly -----------------------------------------------------------------

def segment_base(segments):
    """Identifiers, mileposts, length, and centroid lat/lon per segment."""
    out = segments[["segment_id", "hwy", "mkey", "mile", "begin_mp", "end_mp", "length_mi", "is_short"]].copy()
    mid = segments.to_crs(GRID_CRS).geometry.interpolate(0.5, normalized=True).to_crs("EPSG:4326")
    out["lat"], out["lon"] = mid.y.values, mid.x.values
    return out


def assemble(parts):
    """Merge one-row-per-segment frames on segment_id, checking each is unique."""
    table = parts[0]
    for p in parts[1:]:
        table = table.merge(p, on="segment_id", how="left", validate="one_to_one")
    return table


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
