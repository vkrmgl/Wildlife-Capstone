"""Build the per-segment modeling table with nationally available covariates.

Reads data/processed/master_panel.parquet (the segment x month panel), the
one-mile segments and assigned collisions from build_dataset.py, and the
sources from download_national_sources.py. Writes
data/processed/or_segment_table.parquet (one row per one-mile segment),
data/processed/or_segment_year_counts.parquet (collisions per segment, year,
and animal), and results/segment_table_coverage.csv.

Usage: uv run python scripts/build_segment_table.py
"""

import geopandas as gpd
import pandas as pd

from capstone.data import PROCESSED, RAW, RESULTS, TARGET_SPECIES, assign_collisions, prepare_collisions
from capstone.national import (GBIF_TAXA, TEST_YEARS, TRAIN_YEARS, assemble, collision_windows, collision_years,
                               cover_segments, gbif_counts, hpms_features, nlcd_shares, panel_by_cell,
                               segment_base, segment_cells, transfer_static, transfer_weather)

seg = gpd.read_parquet(PROCESSED / "or_segments.parquet")
print(f"{len(seg):,} one-mile segments")

# --- Target: every animal type assigned by milepost, like build_dataset.py but
# without the deer/elk filter.
raw = pd.read_parquet(RAW / "odot_wildlife_collisions.parquet")
prepared = prepare_collisions(raw).assign(is_target_species=True)
assigned = assign_collisions(prepared, seg)
n_ok = int((assigned["status"] == "assigned").sum())
print(f"collisions: {n_ok:,} / {len(assigned):,} records assigned to a segment")
collision_years(assigned).to_parquet(PROCESSED / "or_segment_year_counts.parquet", index=False)
target = collision_windows(assigned, seg, TARGET_SPECIES, [TRAIN_YEARS, TEST_YEARS])

# --- Terrain, population, climate from the panel via its grid cells --------
panel = pd.read_parquet(PROCESSED / "master_panel.parquet",
                        columns=["segment_id", "grid_id", "year", "month", "ppt_mean", "tmean_mean",
                                 "elevation_mean_m", "elevation_min_m", "elevation_max_m",
                                 "slope_mean_deg", "slope_max_deg", "pop_density_per_km2"])
static, weather = panel_by_cell(panel)
print(f"panel: {panel['segment_id'].nunique():,} segments in {len(static):,} grid cells")
cells, coverage = cover_segments(segment_cells(seg), seg, static["grid_id"])
terrain = transfer_static(cells, static)
climate = transfer_weather(cells, weather)
print(f"  {int((coverage['panel_coverage'] > 0).sum()):,} segments cross panel cells, "
      f"{int((coverage['panel_borrow_m'] > 0).sum())} borrow the nearest cell")

# --- HPMS, NLCD, GBIF -------------------------------------------------------
hpms = hpms_features(seg, pd.read_parquet(RAW / "hpms_or_2024.parquet"))
print(f"HPMS: {100 * (hpms['hpms_coverage'] > 0).mean():.1f}% of segments covered, "
      f"{100 * hpms['hpms_aadt'].notna().mean():.1f}% with AADT")
nlcd = nlcd_shares(seg, RAW / "nlcd" / "nlcd_2024_or.tif")
print(f"NLCD: dominant group counts\n{nlcd['nlcd_dominant'].value_counts().to_string()}")
records = {name: pd.read_parquet(RAW / "gbif" / f"gbif_{name}.parquet") for name, _, _ in GBIF_TAXA.values()}
gbif = gbif_counts(seg, records)
print("GBIF: records within radius, mean per segment\n"
      + gbif.drop(columns="segment_id").mean().round(1).to_string())

table = assemble([segment_base(seg), target, coverage, terrain, climate, hpms, nlcd, gbif])
table.to_parquet(PROCESSED / "or_segment_table.parquet", index=False)
print(f"\nwrote or_segment_table.parquet: {len(table):,} rows x {len(table.columns)} columns")

cov = {
    "segments": len(table),
    "collision_records": len(assigned),
    "collision_records_assigned": n_ok,
    "pct_segments_panel_covered": round(100 * (coverage["panel_coverage"] > 0).mean(), 2),
    "segments_panel_borrowed": int((coverage["panel_borrow_m"] > 0).sum()),
    "pct_segments_hpms_covered": round(100 * (hpms["hpms_coverage"] > 0).mean(), 2),
    "pct_segments_hpms_aadt": round(100 * hpms["hpms_aadt"].notna().mean(), 2),
    "pct_segments_hpms_speed_limit": round(100 * hpms["hpms_speed_limit"].notna().mean(), 2),
    "pct_segments_nlcd": round(100 * (nlcd["nlcd_pixels"] > 0).mean(), 2),
}
for c in table.columns:
    if c.startswith("gbif_"):
        cov[f"{c}_records"] = int(len(records[c[5:]]))
        cov[f"{c}_pct_segments_nonzero"] = round(100 * (table[c] > 0).mean(), 2)
pd.Series(cov, name="value").rename_axis("metric").to_csv(RESULTS / "segment_table_coverage.csv")
print(pd.Series(cov).to_string())
