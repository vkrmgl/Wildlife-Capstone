"""Species occurrence counts from GBIF: records within GBIF_RADIUS_M of each
segment, one column per species group in capstone.collisions. eBird is
excluded because its checklists are most bird records and mark birding
sites, not roads. Counts are observations, not animals: they rise with
observer density.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import pandas as pd
import requests

from capstone.data import RAW

GBIF_API = "https://api.gbif.org/v1"
GBIF_DIR = RAW / "gbif"
# group -> (GBIF taxon key, taxon). Keys from /species/match.
GBIF_TAXA = {
    "deer": (2440964, "Odocoileus (genus)"),
    "elk": (8600904, "Cervus canadensis"),
    "moose": (2440940, "Alces alces"),
    "bear": (2433407, "Ursus americanus"),
    "cougar": (2435099, "Puma concolor"),
    "pronghorn": (2440902, "Antilocapra americana"),
    "mt_goat": (2441151, "Oreamnos americanus"),
    "bighorn_sheep": (2441119, "Ovis canadensis"),
    "owl": (1450, "Strigiformes (order)"),
    "hawk": (2877, "Accipitridae (family)"),
    "bald_eagle": (2480446, "Haliaeetus leucocephalus"),
    "golden_eagle": (2480506, "Aquila chrysaetos"),
}
GBIF_EXCLUDED_DATASETS = {"4fa7b334-ce0d-4e88-aaae-2e0c138d049e": "eBird Observation Dataset"}
GBIF_PAGE = 300
GBIF_MAX_OFFSET = 100_000
GBIF_SLICE = 3000  # records per year/month slice; keeps page offsets shallow
GBIF_WORKERS = 4
GBIF_RADIUS_M = 5000
GBIF_MAX_UNCERTAINTY_M = 10_000
GBIF_PAD_DEG = 0.1  # records just outside the state count for border segments
KEEP = ["key", "species", "scientificName", "taxonKey", "decimalLatitude", "decimalLongitude",
        "coordinateUncertaintyInMeters", "year", "month", "basisOfRecord", "datasetKey"]


def bbox_wkt(bounds):
    """Counter-clockwise WKT box, as the GBIF API requires."""
    minx, miny, maxx, maxy = (round(v, 3) for v in bounds)
    return f"POLYGON(({minx} {miny},{maxx} {miny},{maxx} {maxy},{minx} {maxy},{minx} {miny}))"


def gbif_path(state, group):
    return GBIF_DIR / state.code.lower() / f"gbif_{group}.parquet"


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


def gbif_base_params(taxon_key, geometry):
    return {"taxonKey": taxon_key, "geometry": geometry, "hasCoordinate": "true",
            "hasGeospatialIssue": "false", "occurrenceStatus": "PRESENT"}


def gbif_datasets(taxon_key, geometry):
    """Dataset keys holding records for the taxon in the area, minus the
    excluded ones, with their record counts."""
    facets = gbif_get("occurrence/search", {**gbif_base_params(taxon_key, geometry), "limit": 0,
                                            "facet": "datasetKey", "facetLimit": 1000})["facets"]
    counts = facets[0]["counts"] if facets else []
    return {c["name"]: c["count"] for c in counts if c["name"] not in GBIF_EXCLUDED_DATASETS}


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


def download_gbif(taxon_key, geometry):
    """Records for the taxon inside the WKT geometry from all datasets except
    the excluded ones. Returns an empty frame when there are none."""
    datasets = gbif_datasets(taxon_key, geometry)
    print(f"  {sum(datasets.values()):,} records in {len(datasets)} datasets (excluding "
          f"{', '.join(GBIF_EXCLUDED_DATASETS.values())})")
    if not datasets:
        return pd.DataFrame(columns=KEEP)
    rows = gbif_pages({**gbif_base_params(taxon_key, geometry), "datasetKey": list(datasets)})
    df = pd.DataFrame(rows, columns=KEEP)
    return df.drop_duplicates("key")


def gbif_counts(segments, records, radius_m=GBIF_RADIUS_M, max_uncertainty_m=GBIF_MAX_UNCERTAINTY_M):
    """Records within radius_m of each segment, per group. `records` maps a
    group to a frame of GBIF records; records with a stated coordinate
    uncertainty above max_uncertainty_m are ignored."""
    proj = segments[["segment_id", "geometry"]].to_crs("EPSG:5070")
    buffers = gpd.GeoDataFrame(proj[["segment_id"]], geometry=proj.geometry.buffer(radius_m), crs="EPSG:5070")
    out = segments[["segment_id"]].copy()
    for name, df in records.items():
        df = df[df["decimalLatitude"].notna() & df["decimalLongitude"].notna()]
        unc = pd.to_numeric(df["coordinateUncertaintyInMeters"], errors="coerce")
        df = df[unc.isna() | (unc <= max_uncertainty_m)]
        pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df["decimalLongitude"], df["decimalLatitude"]),
                               crs="EPSG:4326").to_crs("EPSG:5070")
        hits = gpd.sjoin(pts, buffers, predicate="within", how="inner")
        out[f"gbif_{name}"] = out["segment_id"].map(hits.groupby("segment_id").size()).fillna(0).astype(int).values
    return out
