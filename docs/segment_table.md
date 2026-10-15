# Per-segment table with national covariates

`data/processed/or_segment_table.parquet`: one row per one-mile segment (the
segments from `build_dataset.py`), built by `scripts/build_segment_table.py`.
Every covariate comes from a source that exists for every state, so the same
table can be built elsewhere by changing the state code and the collision
source.

The segment x month panel in `data/processed/master_panel.parquet` (half-mile
OR-Trans segments, 2015-2025, built in `notebooks/wvc_capstone_pipeline.ipynb`)
is the source of terrain, population, and climate. The time dimension is
collapsed: the modeling question is where collisions happen, not when.

## Sources

| Covariates | Source | National? | How it reaches a segment |
|---|---|---|---|
| Elevation, slope, population density | Panel (USGS 3DEP, Census tracts) | yes | 800 m grid cells the segment crosses, weighted by length in each |
| Precipitation, temperature | Panel (PRISM monthly, 2015-2025) | lower 48 | Same cells; monthly values summarized per segment |
| Traffic, lanes, speed limit, functional class, rural/urban | FHWA HPMS 2024, Oregon, state-owned sections | yes | Milepost overlap: Oregon's `route_id` is ODOT's `LRM_KEY` |
| Land cover | USGS Annual NLCD 2024 (MRLC) | lower 48 | Share of each class within 200 m of the segment |
| Species occurrences | GBIF, all datasets except eBird | yes | Records within 5 km of the segment, per animal type |
| Collisions (target) | ODOT animal incident records | no | Assigned by milepost, all animal types |

Dropped from the panel because they have no national equivalent: OCAMP
connectivity areas, ODFW mule deer migration corridors, the 1992 Oregon
vegetation map, and the station-based AADT and functional class.

## Columns

Identifiers: `segment_id`, `hwy`, `mkey`, `mile`, `begin_mp`, `end_mp`,
`length_mi`, `is_short`, `lat`, `lon` (segment midpoint).

Target, for the train window 2015-2020 and the test window 2021-2025:
`coll_all_<start>_<end>` (all animal types), `coll_deer_elk_<start>_<end>`,
and each as `_per_yr`. `or_segment_year_counts.parquet` has the counts per
segment, year, and `ANIMAL` for other windows or species.

Terrain and population: `elevation_mean_m`, `elevation_min_m`,
`elevation_max_m`, `slope_mean_deg`, `slope_max_deg`, `pop_density_per_km2`.
`panel_coverage` is the share of the segment on cells with panel data;
`panel_borrow_m` is nonzero when a segment crossed no such cell and took the
nearest one.

Climate (from monthly values): `ppt_month_mean_mm`, `ppt_month_min_mm`,
`ppt_month_max_mm`, `ppt_annual_mm` (mean of complete years),
`ppt_winter_month_mm` and `ppt_summer_month_mm` (Dec-Feb, Jun-Aug monthly
means), `tmean_c`, `tmean_month_min_c`, `tmean_month_max_c`, `tmean_winter_c`,
`tmean_summer_c`, `weather_months`.

HPMS: `hpms_aadt`, `hpms_log_aadt`, `hpms_truck_share` (single-unit plus
combination trucks over AADT), `hpms_lanes`, `hpms_speed_limit`,
`hpms_rural_share` (share of the segment outside an urban area),
`hpms_f_system` (1 interstate .. 7 local) with `hpms_functional_class`,
`hpms_access_control` (1 full, 2 partial, 3 none), `hpms_coverage`. Values are
length-weighted over overlapping sections; codes take the class covering the
most of the segment. HPMS reports 0 where an item was not collected, which is
treated as missing. Speed limit and access control are sample-panel items in
HPMS and cover fewer sections than AADT.

NLCD: `nlcd_water`, `nlcd_developed`, `nlcd_barren`, `nlcd_forest`,
`nlcd_shrub`, `nlcd_grass`, `nlcd_agriculture`, `nlcd_wetland` (shares summing
to 1), `nlcd_dominant`, `nlcd_pixels`. The 200 m buffer includes the road
itself, which NLCD classes as developed, so `nlcd_developed` has a floor of
roughly 0.1 on every segment and is a measure of the surroundings only above
that.

GBIF: `gbif_deer`, `gbif_elk`, `gbif_bear`, `gbif_cougar`, `gbif_antelope`,
`gbif_mt_goat`, `gbif_bighorn_sheep`, `gbif_owl`, `gbif_hawk`,
`gbif_bald_eagle`, `gbif_golden_eagle`. Taxa are in `capstone.national.GBIF_TAXA`
(`Deer` is the genus Odocoileus, `Owl` the order Strigiformes, `Hawk` the
family Accipitridae). Records with a stated coordinate uncertainty over 10 km
are ignored. Counts are observations, not animals: they rise with observer
density, so they are strongest as relative measures next to
`pop_density_per_km2`. ODOT's `Wild Game - Small`, pets, farm animals, and
`Bridge` have no taxon and no column.

## Coverage (Oregon, 7,573 segments)

From `results/segment_table_coverage.csv`.

| | |
|---|---|
| Collision records assigned to a segment (all animals) | 102,796 of 104,528 (98.3%) |
| Deer/elk counts | identical to `build_dataset.py` for both windows |
| Segments crossing panel cells | 99.96%; 3 borrow the nearest cell (at most 640 m away) |
| Segments with HPMS sections | 100%; 99.6% with AADT; speed limit 100% |
| Segments with NLCD pixels | 100% (about 830 pixels each) |
| GBIF records (all taxa, non-eBird) | 76,376; deer within 5 km of 65% of segments, elk 30%, bear 20% |

HPMS is ODOT's own submission to FHWA: per segment, HPMS AADT matches the
ODOT AADT feature from `build_dataset.py` (Spearman 1.00, 99.8% within 20%),
lanes 0.95, and speed limit 0.995. HPMS covers every segment because it is
stored per road section rather than per counting station, which is why the
panel's station-based AADT reached only half the segments.
