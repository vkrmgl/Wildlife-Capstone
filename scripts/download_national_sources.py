"""Download the nationally available covariate sources for Oregon to data/raw/:
HPMS road sections (attributes only), the Annual NLCD land cover mosaic
(cropped to Oregon), and GBIF occurrence records per animal type. Files that
already exist are skipped.

Usage: uv run python scripts/download_national_sources.py
"""

import geopandas as gpd
import requests

from capstone.data import PROCESSED, RAW, download_layer, record_download
from capstone.national import GBIF_TAXA, HPMS_FIELDS, HPMS_URL, NLCD_URL, download_gbif, nlcd_crop

STATE, HPMS_YEAR, NLCD_YEAR = "OR", 2024, 2024

# --- HPMS: state-owned sections (ownership 1 = state highway agency) ---------
hpms_path = RAW / f"hpms_{STATE.lower()}_{HPMS_YEAR}.parquet"
if hpms_path.exists():
    print(f"{hpms_path.name} exists, skipping")
else:
    url = HPMS_URL.format(state=STATE, year=HPMS_YEAR)
    print(f"HPMS {STATE} {HPMS_YEAR}: {url}")
    df, total = download_layer(url, where="ownership=1", id_field="objectid", out_fields=",".join(HPMS_FIELDS))
    df.to_parquet(hpms_path, index=False)
    record_download(hpms_path.stem, hpms_path, len(df), total, source=f"{url} where ownership=1")
    print(f"  {len(df):,} sections")

# --- NLCD: CONUS zip from MRLC (1.4 GB), then an Oregon crop ----------------
nlcd_dir = RAW / "nlcd"
nlcd_dir.mkdir(exist_ok=True)
zip_path = nlcd_dir / f"Annual_NLCD_LndCov_{NLCD_YEAR}_CU_C1V1.zip"
crop_path = nlcd_dir / f"nlcd_{NLCD_YEAR}_{STATE.lower()}.tif"
if crop_path.exists():
    print(f"{crop_path.name} exists, skipping")
else:
    if not zip_path.exists():
        url = NLCD_URL.format(year=NLCD_YEAR)
        print(f"NLCD {NLCD_YEAR}: {url}")
        with requests.get(url, stream=True, timeout=120) as r, open(zip_path, "wb") as f:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    segments = gpd.read_parquet(PROCESSED / "or_segments.parquet")
    print(f"Cropping {zip_path.name} to the segment extent...")
    nlcd_crop(zip_path, crop_path, segments.total_bounds)
    record_download(crop_path.stem, crop_path, 0, 0, source=NLCD_URL.format(year=NLCD_YEAR))
    print(f"  wrote {crop_path.name}")

# --- GBIF: one file per animal type -----------------------------------------
gbif_dir = RAW / "gbif"
gbif_dir.mkdir(exist_ok=True)
for animal, (name, key, taxon) in GBIF_TAXA.items():
    path = gbif_dir / f"gbif_{name}.parquet"
    if path.exists():
        print(f"{path.name} exists, skipping")
        continue
    print(f"GBIF {animal} ({taxon}, taxonKey {key})")
    df = download_gbif(key)
    df.to_parquet(path, index=False)
    record_download(path.stem, path, len(df), len(df), source=f"https://api.gbif.org/v1/occurrence/search?taxonKey={key}")
    print(f"  {len(df):,} records")
