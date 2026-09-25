"""Build segments, assign collisions, and join traffic features.

Writes data/processed/or_segments.parquet, or_collisions_assigned.parquet,
or_traffic_features.parquet, and results/conservation.csv and
results/traffic_coverage.csv.

Usage: uv run python scripts/build_dataset.py
"""

import geopandas as gpd
import pandas as pd

from capstone.data import (PROCESSED, RAW, RESULTS, assign_collisions, build_segments,
                           conservation_report, mainline_pieces, prepare_collisions)
from capstone.features import traffic_features

PROCESSED.mkdir(parents=True, exist_ok=True)

highways = gpd.read_parquet(RAW / "odot_state_highways.parquet")
seg = build_segments(mainline_pieces(highways))
seg.to_parquet(PROCESSED / "or_segments.parquet", index=False)
print(f"{len(seg):,} segments, {seg['length_mi'].sum():,.1f} miles, {int(seg['is_short'].sum())} under 0.25 mi")

raw = pd.read_parquet(RAW / "odot_wildlife_collisions.parquet")
assigned = assign_collisions(prepare_collisions(raw), seg)
cols = ["OBJECTID", "ANIMAL", "DATE", "year", "LRM_KEY", "hwy", "sfx", "direction", "mkey", "mp",
        "LAT", "LONGTD", "is_target_species", "status", "segment_id"]
assigned[cols].to_parquet(PROCESSED / "or_collisions_assigned.parquet", index=False)
report = conservation_report(assigned)
pd.Series(report, name="value").rename_axis("metric").to_csv(RESULTS / "conservation.csv")
print(report)

tf = traffic_features(seg, pd.read_parquet(RAW / "odot_aadt.parquet"), pd.read_parquet(RAW / "odot_posted_speed.parquet"),
                      pd.read_parquet(RAW / "odot_lanes.parquet"), highways)
tf.to_parquet(PROCESSED / "or_traffic_features.parquet", index=False)

coverage = {
    "segments": len(tf),
    "pct_segments_with_aadt_overlap": round(100 * (tf["aadt_coverage"] > 0).mean(), 2),
    "pct_segments_aadt_fully_covered": round(100 * (tf["aadt_coverage"] >= 0.99).mean(), 2),
    "pct_segments_aadt_imputed": round(100 * tf["aadt_imputed"].mean(), 2),
    "segments_aadt_imputed": int(tf["aadt_imputed"].sum()),
    "pct_segments_with_truck_pct": round(100 * tf["truck_pct"].notna().mean(), 2),
    "pct_segments_with_speed": round(100 * (tf["speed_coverage"] > 0).mean(), 2),
    "pct_segments_with_lanes": round(100 * (tf["lanes_coverage"] > 0).mean(), 2),
    "pct_segments_divided_any": round(100 * (tf["divided_share"] > 0).mean(), 2),
}
for col in ("aadt", "log_aadt", "truck_pct", "posted_speed", "lanes"):
    for p, v in tf[col].quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).items():
        coverage[f"{col}_p{int(p * 100):02d}"] = round(float(v), 3)
pd.Series(coverage, name="value").rename_axis("metric").to_csv(RESULTS / "traffic_coverage.csv")
print(f"AADT coverage {coverage['pct_segments_with_aadt_overlap']}%, imputed {coverage['segments_aadt_imputed']} segments")
