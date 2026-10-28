"""Build the per-segment modeling table for each state from the sources in
data/raw/ (see download_sources.py).

Writes, per state (prefix = lower-case state code):
  data/processed/{st}_segments.parquet              one-mile segments with geometry
  data/processed/{st}_collisions_assigned.parquet   every collision record with its segment
  data/processed/{st}_segment_year_counts.parquet   assigned records per segment, year, species group
  data/processed/{st}_segment_table.parquet         one row per segment: target windows and covariates
  data/processed/{st}_segment_table_dictionary.csv  column descriptions (tracked)
  results/{st}_segment_table_coverage.csv           coverage and conservation checks (tracked)

Usage: uv run python scripts/build_segment_table.py [OR ID ...]   (default: all states)
"""

import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from capstone.climate import climate_summaries, monthly_means
from capstone.collisions import (assign_collisions, collision_windows, collision_years, conservation_report,
                                 prepare_collisions)
from capstone.data import PROCESSED, RESULTS
from capstone.gbif import GBIF_TAXA, gbif_counts, gbif_path
from capstone.landcover import nlcd_path, nlcd_shares
from capstone.network import build_segments, geometry_check, hpms_features, hpms_path, mainline_sections, orient_sections
from capstone.population import population_features, tract_density
from capstone.states import STATES, get_state
from capstone.table import assemble, check_dictionary, dictionary, segment_base
from capstone.terrain import terrain_features

PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
states = [get_state(c) for c in sys.argv[1:]] or list(STATES.values())

for state in states:
    st = state.code.lower()
    print(f"\n== {state.name}")

    # --- Segments from HPMS mainline sections ---------------------------------
    hpms = gpd.read_parquet(hpms_path(state))
    sections = orient_sections(mainline_sections(hpms, state))
    ratio = geometry_check(sections)
    seg = build_segments(sections, state)
    seg.to_parquet(PROCESSED / f"{st}_segments.parquet", index=False)
    print(f"segments: {len(seg):,} on {seg['route_id'].nunique()} routes, {seg['length_mi'].sum():,.1f} miles, "
          f"{int(seg['is_short'].sum())} under 0.25 mi; drawn/milepost length ratio median {np.median(ratio):.3f}")

    # --- Target -----------------------------------------------------------------
    records = prepare_collisions(state, seg)
    assigned = assign_collisions(records, seg, state)
    assigned.to_parquet(PROCESSED / f"{st}_collisions_assigned.parquet", index=False)
    report = conservation_report(assigned)
    print(f"collisions: {report['assigned']:,} / {report['raw_records']:,} records assigned "
          f"({report['assigned_milepost']:,} by milepost, {report['assigned_coordinates']:,} by coordinates)")
    collision_years(assigned).to_parquet(PROCESSED / f"{st}_segment_year_counts.parquet", index=False)
    target = collision_windows(assigned, seg)

    # --- Covariates -------------------------------------------------------------
    terrain = terrain_features(seg)
    print(f"terrain: {100 * (terrain['terrain_pixels'] > 0).mean():.1f}% of segments with DEM pixels")
    monthly = monthly_means(seg)
    climate = climate_summaries(monthly)
    print(f"climate: {monthly.groupby('segment_id').size().max()} months per segment")
    population = population_features(seg, tract_density(state))
    print(f"population: {100 * population['pop_density_per_km2'].notna().mean():.1f}% of segments with a tract")
    traffic = hpms_features(seg, sections)
    print(f"HPMS: {100 * traffic['hpms_aadt'].notna().mean():.1f}% of segments with AADT, "
          f"{100 * traffic['hpms_speed_limit'].notna().mean():.1f}% with a speed limit")
    landcover = nlcd_shares(seg, nlcd_path(state))
    print(f"NLCD: dominant groups {landcover['nlcd_dominant'].value_counts().to_dict()}")
    records_by_group = {g: pd.read_parquet(gbif_path(state, g)) for g in GBIF_TAXA}
    gbif = gbif_counts(seg, records_by_group)
    print("GBIF: mean records within 5 km " + ", ".join(f"{c[5:]} {v:.0f}" for c, v in gbif.drop(columns='segment_id').mean().items()))

    # --- Table, dictionary, coverage ---------------------------------------------
    table = assemble([segment_base(seg, state), target, terrain, climate, population, traffic, landcover, gbif])
    dic = dictionary(state)
    check_dictionary(table, dic)
    table.to_parquet(PROCESSED / f"{st}_segment_table.parquet", index=False)
    dic.to_csv(PROCESSED / f"{st}_segment_table_dictionary.csv", index=False)
    print(f"wrote {st}_segment_table.parquet: {len(table):,} rows x {len(table.columns)} columns")

    cov = {
        "segments": len(table),
        "routes": int(seg["route_id"].nunique()),
        "network_miles": round(float(seg["length_mi"].sum()), 1),
        "pct_sections_drawn_within_10pct_of_milepost_length": round(100 * float(np.mean(np.abs(ratio - 1) <= 0.1)), 2),
        **{f"collision_{k}": v for k, v in report.items()},
        "collision_snap_m_median": round(float(assigned["snap_m"].median()), 1) if assigned["snap_m"].notna().any() else None,
        "pct_segments_terrain": round(100 * (terrain["terrain_pixels"] > 0).mean(), 2),
        "weather_months": int(climate["weather_months"].max()),
        "pct_segments_population": round(100 * population["pop_density_per_km2"].notna().mean(), 2),
        "pct_segments_hpms_aadt": round(100 * traffic["hpms_aadt"].notna().mean(), 2),
        "pct_segments_hpms_speed_limit": round(100 * traffic["hpms_speed_limit"].notna().mean(), 2),
        "pct_segments_nlcd": round(100 * (landcover["nlcd_pixels"] > 0).mean(), 2),
    }
    for g in GBIF_TAXA:
        cov[f"gbif_{g}_records"] = int(len(records_by_group[g]))
        cov[f"gbif_{g}_pct_segments_nonzero"] = round(100 * (table[f"gbif_{g}"] > 0).mean(), 2)
    pd.Series(cov, name="value").rename_axis("metric").to_csv(RESULTS / f"{st}_segment_table_coverage.csv")
