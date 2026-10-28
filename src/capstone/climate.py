"""Climate from PRISM monthly 4 km grids (precipitation and mean
temperature, Oregon State University). One grid per variable and month for
the lower 48 states, sampled at points along each segment and summarized
over CLIMATE_YEARS.
"""

import io
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import shapely

from capstone.data import RAW

# The archive serves one zip per month; the web service is the fallback and
# allows each file to be fetched only twice a day.
PRISM_ARCHIVE_URL = "https://data.prism.oregonstate.edu/time_series/us/an/4km/{var}/monthly/{year}/prism_{var}_us_25m_{yyyymm}.zip"
PRISM_SERVICE_URL = "https://services.nacse.org/prism/data/get/us/4km/{var}/{yyyymm}?format=bil"
PRISM_DIR = RAW / "prism"
VARIABLES = ("ppt", "tmean")
CLIMATE_YEARS = (2015, 2025)
SAMPLE_STEP_M = 200
WINTER, SUMMER = (12, 1, 2), (6, 7, 8)


def months(years=CLIMATE_YEARS):
    return [(y, m) for y in range(years[0], years[1] + 1) for m in range(1, 13)]


RASTER_EXT = ("tif", "bil")  # the archive ships GeoTIFF, the web service ESRI BIL


def grid_path(var, year, month, prism_dir=PRISM_DIR):
    """The month's raster file, whichever format it was fetched in (the .tif
    path when neither exists yet)."""
    stem = prism_dir / var / f"prism_{var}_{year}{month:02d}"
    for ext in RASTER_EXT:
        if stem.with_suffix(f".{ext}").exists():
            return stem.with_suffix(f".{ext}")
    return stem.with_suffix(".tif")


def download_prism(var, year, month, prism_dir=PRISM_DIR):
    """Fetch one monthly grid; returns False when PRISM has no grid for that
    month yet. Only the raster files are kept (the station list is not)."""
    path = grid_path(var, year, month, prism_dir)
    if path.exists():
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    yyyymm = f"{year}{month:02d}"
    content = None
    for url in (PRISM_ARCHIVE_URL.format(var=var, year=year, yyyymm=yyyymm),
                PRISM_SERVICE_URL.format(var=var, yyyymm=yyyymm)):
        resp = requests.get(url, timeout=300)
        if resp.status_code == 404:
            continue
        resp.raise_for_status()
        if resp.content[:2] == b"PK":
            content = resp.content
            break
        print(f"  PRISM {var} {yyyymm}: {resp.text[:120].strip()}")
    if content is None:
        return False
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for name in z.namelist():
            ext = name.rsplit(".", 1)[-1].lower()
            if ext in (*RASTER_EXT, "hdr", "prj", "stx"):
                path.with_suffix(f".{ext}").write_bytes(z.read(name))
    return True


def sample_points(segments, step_m=SAMPLE_STEP_M):
    """Points every step_m along each segment, as (lon, lat, segment index)."""
    proj = segments.to_crs("EPSG:5070")
    xs, ys, idx = [], [], []
    for i, geom in enumerate(proj.geometry):
        n = max(2, int(geom.length // step_m) + 1)
        pts = shapely.line_interpolate_point(geom, np.linspace(0, geom.length, n))
        xs.append(shapely.get_x(pts)), ys.append(shapely.get_y(pts)), idx.append(np.full(n, i))
    pts = gpd.GeoSeries(shapely.points(np.concatenate(xs), np.concatenate(ys)), crs="EPSG:5070")
    g = pts.to_crs("EPSG:4326")
    return g.x.to_numpy(), g.y.to_numpy(), np.concatenate(idx)


def sample_grid(path, lon, lat):
    """Grid values at lon/lat (NaN for nodata or outside the grid)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        rows, cols = rasterio.transform.rowcol(src.transform, lon, lat)
        rows, cols = np.asarray(rows), np.asarray(cols)
        inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        out = np.full(len(lon), np.nan)
        out[inside] = arr[rows[inside], cols[inside]]
        if src.nodata is not None:
            out[out == src.nodata] = np.nan
    return out


def monthly_means(segments, prism_dir=PRISM_DIR, years=CLIMATE_YEARS, variables=VARIABLES):
    """Long frame: segment_id, year, month, and one column per variable
    (mean over the segment's sample points). Months without a grid are left out."""
    lon, lat, idx = sample_points(segments)
    sids = segments["segment_id"].to_numpy()
    frames = []
    for year, month in months(years):
        row = {"segment_id": sids, "year": year, "month": month}
        have = True
        for var in variables:
            path = grid_path(var, year, month, prism_dir)
            if not path.exists():
                have = False
                break
            vals = sample_grid(path, lon, lat)
            sums = np.bincount(idx, weights=np.nan_to_num(vals), minlength=len(sids))
            counts = np.bincount(idx, weights=np.isfinite(vals), minlength=len(sids))
            row[var] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        if have:
            frames.append(pd.DataFrame(row))
    return pd.concat(frames, ignore_index=True)


def climate_summaries(monthly):
    """Per segment from the monthly values: monthly precipitation mean, min
    and max, mean annual total (complete years only), winter and summer
    monthly means; mean temperature with its coldest and warmest month and
    winter and summer means; and weather_months, the months available."""
    by = monthly.groupby("segment_id")
    out = pd.DataFrame(index=by.size().index)
    out["ppt_month_mean_mm"] = by["ppt"].mean()
    out["ppt_month_min_mm"] = by["ppt"].min()
    out["ppt_month_max_mm"] = by["ppt"].max()
    full = monthly.groupby(["segment_id", "year"]).agg(ppt=("ppt", "sum"), n=("month", "size")).reset_index()
    out["ppt_annual_mm"] = full[full["n"] == 12].groupby("segment_id")["ppt"].mean()
    out["tmean_c"] = by["tmean"].mean()
    out["tmean_month_min_c"] = by["tmean"].min()
    out["tmean_month_max_c"] = by["tmean"].max()
    for name, season in (("winter", WINTER), ("summer", SUMMER)):
        s = monthly[monthly["month"].isin(season)].groupby("segment_id")
        out[f"tmean_{name}_c"] = s["tmean"].mean()
        out[f"ppt_{name}_month_mm"] = s["ppt"].mean()
    out["weather_months"] = by.size()
    return out.rename_axis("segment_id").reset_index()
