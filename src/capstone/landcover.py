"""Land cover from USGS Annual NLCD (National Land Cover Database, MRLC):
the share of each cover group within NLCD_BUFFER_M of the segment. The
CONUS mosaic (1.4 GB) is downloaded once and cropped per state.
"""

import os
import time
import zipfile

import geopandas as gpd
import pandas as pd
import rasterio
import requests
import shapely
from rasterio.windows import from_bounds
from rasterstats import zonal_stats

from capstone.data import RAW

NLCD_YEAR = 2024
NLCD_URL = "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/Annual_NLCD_LndCov_{year}_CU_C1V1.zip"
NLCD_SCIENCEBASE = "https://www.sciencebase.gov/catalog/item/6810c1a4d4be022940554075"
NLCD_DIR = RAW / "nlcd"
NLCD_GROUPS = {"water": (11, 12), "developed": (21, 22, 23, 24), "barren": (31,), "forest": (41, 42, 43),
               "shrub": (52,), "grass": (71,), "agriculture": (81, 82), "wetland": (90, 95)}
NLCD_NODATA = 250
NLCD_BUFFER_M = 200


def nlcd_zip():
    return NLCD_DIR / f"Annual_NLCD_LndCov_{NLCD_YEAR}_CU_C1V1.zip"


def nlcd_path(state):
    return NLCD_DIR / f"nlcd_{NLCD_YEAR}_{state.code.lower()}.tif"


def download_nlcd(attempts=4):
    """Download the CONUS mosaic zip unless it is already present. The MRLC
    download server is sometimes unavailable (HTTP 503) for hours, so failures
    are retried with backoff and the final error says how to get the file
    another way. CAPSTONE_NLCD_URL overrides the source URL."""
    path = nlcd_zip()
    if path.exists():
        return path
    NLCD_DIR.mkdir(parents=True, exist_ok=True)
    url = os.environ.get("CAPSTONE_NLCD_URL") or NLCD_URL.format(year=NLCD_YEAR)
    print(f"NLCD {NLCD_YEAR} (1.4 GB): {url}")
    part = path.with_suffix(".part")
    for i in range(attempts):
        try:
            with requests.get(url, stream=True, timeout=600) as r, open(part, "wb") as f:
                r.raise_for_status()
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            part.rename(path)
            return path
        except requests.RequestException as exc:
            if i == attempts - 1:
                raise RuntimeError(
                    f"Could not download {url} ({exc}).\n"
                    "The MRLC download server is down at times for hours. Any of these works instead:\n"
                    f"  - copy data/raw/nlcd/nlcd_{NLCD_YEAR}_<st>.tif from a teammate into {NLCD_DIR}/;"
                    " the zip is then not needed;\n"
                    f"  - put {path.name} (from a teammate, or from ScienceBase {NLCD_SCIENCEBASE},"
                    f" item 'Collection 1.1 Land Cover') at {path};\n"
                    "  - set CAPSTONE_NLCD_URL to another copy of the zip."
                ) from exc
            print(f"  retrying after error: {exc}")
            time.sleep(min(60, 2 ** (i + 2)))


def nlcd_crop(zip_path, out_path, bounds_4326, pad_m=5000):
    """Write the part of the CONUS GeoTIFF (inside the MRLC zip) covering
    bounds_4326 (minx, miny, maxx, maxy), padded, to out_path."""
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


def nlcd_shares(segments, nlcd_file, buffer_m=NLCD_BUFFER_M):
    """Share of each NLCD group within buffer_m of the segment, the dominant
    group, and the number of pixels counted."""
    with rasterio.open(nlcd_file) as src:
        crs = src.crs
    geoms = segments.to_crs(crs).geometry.buffer(buffer_m)
    stats = zonal_stats(geoms, nlcd_file, categorical=True, nodata=NLCD_NODATA)
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
