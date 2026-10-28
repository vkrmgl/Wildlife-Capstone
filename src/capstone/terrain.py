"""Elevation and slope from USGS 3DEP 1 arc-second (about 30 m) DEM tiles.

Tiles are 1 x 1 degree GeoTIFFs on the USGS staged-products bucket, named by
their north-west corner (n45w123 covers latitude 44-45, longitude -123 to
-122). Each segment's statistics are taken over the DEM pixels within
BUFFER_M of its line; slope is computed on the tile window around it.
"""

import math

import numpy as np
import pandas as pd
import rasterio
import requests
import shapely
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds

from capstone.data import RAW

TILE_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/current/{tile}/USGS_1_{tile}.tif"
TILE_DIR = RAW / "3dep"
BUFFER_M = 30
M_PER_DEG = 111_320.0


def tile_name(lat_floor, lon_floor):
    """Tile covering [lat_floor, lat_floor + 1] x [lon_floor, lon_floor + 1]
    (western hemisphere, northern latitudes)."""
    return f"n{lat_floor + 1:02d}w{-lon_floor:03d}"


def tiles_for(bounds_4326):
    """Tiles intersecting (minx, miny, maxx, maxy) in degrees."""
    minx, miny, maxx, maxy = bounds_4326
    return [tile_name(lat, lon) for lat in range(math.floor(miny), math.ceil(maxy))
            for lon in range(math.floor(minx), math.ceil(maxx))]


def download_tiles(tiles, tile_dir=TILE_DIR):
    """Fetch missing tiles (about 55 MB each). Tiles that do not exist (open
    water) are recorded as empty marker files so they are not retried."""
    tile_dir.mkdir(parents=True, exist_ok=True)
    for i, tile in enumerate(tiles, 1):
        path = tile_dir / f"USGS_1_{tile}.tif"
        missing = tile_dir / f"USGS_1_{tile}.missing"
        if path.exists() or missing.exists():
            continue
        url = TILE_URL.format(tile=tile)
        with requests.get(url, stream=True, timeout=300) as r:
            if r.status_code == 404:
                missing.touch()
                print(f"  [{i}/{len(tiles)}] {tile}: no tile (water)")
                continue
            r.raise_for_status()
            part = path.with_suffix(".part")
            with open(part, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            part.rename(path)
        print(f"  [{i}/{len(tiles)}] {tile}: {path.stat().st_size / 1e6:.0f} MB")


def slope_degrees(elev, res_deg, lat):
    """Slope of a DEM window in degrees. Pixel size is converted to metres
    using the latitude of the window; nodata must already be NaN."""
    dy = res_deg * M_PER_DEG
    dx = res_deg * M_PER_DEG * math.cos(math.radians(lat))
    gy, gx = np.gradient(elev, dy, dx)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def window_stats(src, polygon, pad_px=3):
    """(elevation values, slope values) of the pixels under polygon (EPSG:4326)
    in one open tile; empty arrays when the polygon misses the tile."""
    minx, miny, maxx, maxy = polygon.bounds
    full = from_bounds(minx, miny, maxx, maxy, src.transform)
    win = Window(math.floor(full.col_off) - pad_px, math.floor(full.row_off) - pad_px,
                 math.ceil(full.width) + 2 * pad_px + 1, math.ceil(full.height) + 2 * pad_px + 1)
    win = win.intersection(Window(0, 0, src.width, src.height))
    if win.width <= 2 or win.height <= 2:
        return np.array([]), np.array([])
    elev = src.read(1, window=win).astype("float64")
    if src.nodata is not None:
        elev[elev == src.nodata] = np.nan
    transform = src.window_transform(win)
    slope = slope_degrees(elev, src.res[0], (miny + maxy) / 2)
    inside = geometry_mask([polygon], elev.shape, transform, invert=True, all_touched=True)
    ok = inside & np.isfinite(elev)
    return elev[ok], slope[ok]


def terrain_features(segments, tile_dir=TILE_DIR, buffer_m=BUFFER_M):
    """One row per segment: elevation_mean_m, elevation_min_m, elevation_max_m,
    slope_mean_deg, slope_max_deg, terrain_pixels. Segments are buffered by
    buffer_m (in EPSG:5070) and read from every tile they touch."""
    polys = segments.to_crs("EPSG:5070").geometry.buffer(buffer_m).to_crs("EPSG:4326")
    open_tiles = {}

    def tile(name):
        if name not in open_tiles:
            path = tile_dir / f"USGS_1_{name}.tif"
            open_tiles[name] = rasterio.open(path) if path.exists() else None
        return open_tiles[name]

    rows = []
    for sid, poly in zip(segments["segment_id"], polys):
        elev, slope = [], []
        for name in tiles_for(poly.bounds):
            src = tile(name)
            if src is None:
                continue
            e, s = window_stats(src, poly)
            elev.append(e), slope.append(s)
        elev, slope = np.concatenate(elev) if elev else np.array([]), np.concatenate(slope) if slope else np.array([])
        if len(elev):
            rows.append((sid, elev.mean(), elev.min(), elev.max(), np.nanmean(slope), np.nanmax(slope), len(elev)))
        else:
            rows.append((sid, np.nan, np.nan, np.nan, np.nan, np.nan, 0))
    for src in open_tiles.values():
        if src is not None:
            src.close()
    out = pd.DataFrame(rows, columns=["segment_id", "elevation_mean_m", "elevation_min_m", "elevation_max_m",
                                      "slope_mean_deg", "slope_max_deg", "terrain_pixels"])
    return out
