"""Download every input for the per-segment table to data/raw/. Files that
already exist are skipped, so the script can be rerun after an interruption.

Per state: collision records (ODOT or IDFG), HPMS sections with geometry,
3DEP elevation tiles (about 55 MB each, 50-65 per state), PRISM monthly
grids (shared across states), Census tracts and ACS population, the NLCD
mosaic (1.4 GB, shared) cropped to the state, and GBIF records per species
group. Everything is logged in data/raw/manifest.json.

Usage: uv run python scripts/download_sources.py [OR ID ...]   (default: all states)
"""

import sys

import geopandas as gpd

from capstone.climate import CLIMATE_YEARS, VARIABLES, download_prism, months
from capstone.collisions import download_collisions
from capstone.data import RAW, record_download
from capstone.gbif import GBIF_PAD_DEG, GBIF_TAXA, bbox_wkt, download_gbif, gbif_path
from capstone.landcover import download_nlcd, nlcd_crop, nlcd_path
from capstone.network import download_hpms, hpms_path
from capstone.population import download_population
from capstone.states import STATES, download_states, get_state, state_bounds
from capstone.terrain import download_tiles, tiles_for

RAW.mkdir(parents=True, exist_ok=True)
states = [get_state(c) for c in sys.argv[1:]] or list(STATES.values())

print("== shared sources")
download_states()
nlcd_zip = download_nlcd()
missing_months = []
for var in VARIABLES:
    for year, month in months(CLIMATE_YEARS):
        if not download_prism(var, year, month):
            missing_months.append(f"{var} {year}-{month:02d}")
print(f"PRISM {CLIMATE_YEARS[0]}-{CLIMATE_YEARS[1]}: {len(VARIABLES) * len(months(CLIMATE_YEARS)) - len(missing_months)} grids"
      + (f", not yet published: {', '.join(missing_months)}" if missing_months else ""))

for state in states:
    print(f"\n== {state.name}")
    download_collisions(state)
    download_hpms(state)
    download_population(state)

    bounds = state_bounds(state, pad_deg=0.05)
    sections = gpd.read_parquet(hpms_path(state))
    tiles = tiles_for(sections.total_bounds)
    print(f"3DEP: {len(tiles)} tiles for the network extent")
    download_tiles(tiles)

    crop = nlcd_path(state)
    if crop.exists():
        print(f"{crop.name} exists, skipping")
    else:
        print(f"NLCD: cropping to {state.name}")
        nlcd_crop(nlcd_zip, crop, sections.total_bounds)
        record_download(crop.stem, crop, 0, 0, source=nlcd_zip.name)

    geometry = bbox_wkt(state_bounds(state, pad_deg=GBIF_PAD_DEG))
    for group, (key, taxon) in GBIF_TAXA.items():
        path = gbif_path(state, group)
        if path.exists():
            print(f"{path.parent.name}/{path.name} exists, skipping")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"GBIF {state.code} {group} ({taxon}, taxonKey {key})")
        df = download_gbif(key, geometry)
        df.to_parquet(path, index=False)
        record_download(f"gbif_{state.code.lower()}_{group}", path, len(df), len(df),
                        source=f"https://api.gbif.org/v1/occurrence/search?taxonKey={key}&geometry={geometry}")
        print(f"  {len(df):,} records")
