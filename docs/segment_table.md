# Per-segment tables with national covariates

`data/processed/or_segment_table.parquet` and `id_segment_table.parquet`: one
row per one-mile segment of state highway in Oregon and Idaho, built by
`scripts/download_sources.py` and `scripts/build_segment_table.py`. Every
covariate comes from a source that exists for every state; only the collision
records (the target) come from the state agency. Adding a state means adding
an entry to `src/capstone/states.py` (where its collision records come from
and how its HPMS route_id marks the mainline) and nothing else.

The modeling question is where collisions happen, not when, so there is no
time dimension: climate is summarized over 2015-2025 and collisions are
counted in a train window (2015-2020) and a test window (2021-2025).

## Sources

| Covariates | Source | How it reaches a segment |
|---|---|---|
| Segments, mileposts | FHWA HPMS 2024 (state-owned sections with geometry) | Mainline sections cut at integer mileposts along `route_id` |
| Traffic, lanes, speed limit, functional class, rural/urban | FHWA HPMS 2024 | Length-weighted over the sections overlapping the segment |
| Elevation, slope | USGS 3DEP 1 arc-second DEM (about 30 m) | Pixels within 30 m of the segment; slope from the DEM around it |
| Precipitation, temperature | PRISM monthly 4 km grids, 2015-2025 | Sampled every 200 m along the segment, averaged, then summarized |
| Population density | Census TIGER 2023 tracts + ACS 2019-2023 total population | Tract density, length-weighted along the segment |
| Land cover | USGS Annual NLCD 2024 | Share of each class within 200 m of the segment |
| Species occurrences | GBIF, all datasets except eBird | Records within 5 km of the segment, per species group |
| Collisions (target) | ODOT animal incidents; IDFG roadkill observations | By milepost or by coordinates (see below) |

None of these need an API key. The downloads are about 8 GB in total,
most of it the 3DEP tiles (about 55 MB each, 54 for Oregon and 63 for Idaho)
and the NLCD CONUS mosaic (1.4 GB, shared). Files already present are
skipped, so an interrupted download can be rerun.

## Segments

HPMS `route_id` is state-defined. Oregon submits ODOT's 8-character LRM_KEY
(highway, suffix, roadway, mileage key), so the mainline is suffix `00` and
roadway `I` (the inventory direction). Idaho submits an 11-character ITD key
(route number, roadway `A`/`D`, signing `IN`/`US`/`SH`, number), so the
mainline is roadway `A` with HPMS facility type 1 or 2 (the second roadway
of a divided highway is facility type 6 in Idaho but a one-way roadway in
Oregon, which is why the rule is per state). Ramps and non-mainline pieces
are dropped in both.

HPMS does not record which way a section's line was digitized, so each
section is oriented by its neighbours on the route before it is cut at
mileposts. `segment_id` is `route_id` plus the starting milepost; `route` is
a readable label (`001` or `001Z1` in Oregon, `US-95` in Idaho). Oregon's
segments match the earlier ODOT-based segmentation one for one (7,574
against 7,573, 7,337.7 miles in both).

## Collisions

Records are put in a common form (species label, species group, date,
coordinates, and the HPMS route_id plus milepost when the record has one)
and assigned to a segment in two ways:

- **by milepost**: route_id plus milepost, exact. ODOT locates records this
  way and derives coordinates from the milepost, so Oregon tries it first.
- **by coordinates**: the nearest segment within 100 m. IDFG records are
  pinned on a map, and their highway/milepost fields are free text that newer
  reports do not fill in, so Idaho tries this first. The highway field is
  used for the few records without coordinates when it is a standard label
  (`US-95`, `I-84`, `ID-21`).

Species groups are shared across states (`capstone.collisions.GROUP_RULES`):
deer, elk, moose, bear, cougar, pronghorn, mt_goat, bighorn_sheep, owl,
hawk, bald_eagle, golden_eagle, other. The target columns count all animals,
deer and elk together, and deer, elk, and moose separately.

| | Oregon | Idaho |
|---|---|---|
| Records | 104,528 | 70,057 (after dropping 580 double submissions) |
| Assigned by milepost | 102,796 | 594 |
| Assigned by coordinates | 1,413 | 55,526 |
| Off the state network | 319 | 13,502 |
| No usable location | 0 | 435 |
| Share of located records assigned | 99.7% | 80.6% |
| Median snap distance (coordinate assignment) | 47 m | 1 m |

Idaho's unassigned records are not near a state highway: half are more than
1.3 km from the nearest segment, and only about 1,000 more would be assigned
at a 200 m limit. They are reports from county and city roads. Idaho's
assigned share also falls over time (89% in 2012, 64% in 2025) as public
reporting of roads off the state system grows; year-to-year changes in
Idaho counts reflect reporting effort as much as collisions.

## Columns

Identifiers: `segment_id`, `state`, `route_id`, `route`, `mile`, `begin_mp`,
`end_mp`, `length_mi`, `is_short`, `lat`, `lon` (segment midpoint).

Target, for 2015-2020 and 2021-2025: `coll_{group}_{start}_{end}` and
`_per_yr` for `all`, `deer_elk`, `deer`, `elk`, `moose`.
`{st}_segment_year_counts.parquet` has counts per segment, year, and group
for other windows; `{st}_collisions_assigned.parquet` has every record with
its segment, assignment status, and snap distance.

Terrain: `elevation_mean_m`, `elevation_min_m`, `elevation_max_m`,
`slope_mean_deg`, `slope_max_deg`, `terrain_pixels`.

Climate: `ppt_month_mean_mm`, `ppt_month_min_mm`, `ppt_month_max_mm`,
`ppt_annual_mm` (mean of complete years), `ppt_winter_month_mm`,
`ppt_summer_month_mm` (Dec-Feb, Jun-Aug monthly means), `tmean_c`,
`tmean_month_min_c`, `tmean_month_max_c`, `tmean_winter_c`,
`tmean_summer_c`, `weather_months` (132 when every month is present).

Population: `pop_density_per_km2`.

HPMS: `hpms_aadt`, `hpms_log_aadt`, `hpms_truck_share`, `hpms_lanes`,
`hpms_speed_limit`, `hpms_rural_share`, `hpms_f_system` (1 interstate .. 7
local) with `hpms_functional_class`, `hpms_access_control` (1 full, 2
partial, 3 none), `hpms_coverage`. HPMS reports 0 where an item was not
collected, which is treated as missing; speed limit and access control are
sample-panel items and cover fewer sections than AADT.

NLCD: `nlcd_water`, `nlcd_developed`, `nlcd_barren`, `nlcd_forest`,
`nlcd_shrub`, `nlcd_grass`, `nlcd_agriculture`, `nlcd_wetland` (shares
summing to 1), `nlcd_dominant`, `nlcd_pixels`. The 200 m buffer includes the
road itself, which NLCD classes as developed, so `nlcd_developed` has a floor
of roughly 0.1 on every segment.

GBIF: `gbif_{group}` for the twelve species groups above (taxa in
`capstone.gbif.GBIF_TAXA`). Records with a stated coordinate uncertainty over
10 km are ignored. Counts are observations, not animals: they rise with
observer density, so they are strongest as relative measures next to
`pop_density_per_km2`.

The full list with sources and plain-language descriptions is in
`data/processed/{st}_segment_table_dictionary.csv` (tracked); coverage and
conservation checks are in `results/{st}_segment_table_coverage.csv`.

## Checks

- Oregon's table was compared with the earlier one built from the
  segment x month panel (`notebooks/wvc_capstone_pipeline.ipynb`). Collision
  counts are identical on 99% of segments, HPMS values on all, and elevation,
  climate, population, land cover, and GBIF counts correlate at 0.97-1.00.
  Slope differs (correlation 0.72, mean 5.9 against 4.1 degrees): it is now
  computed within 30 m of the road instead of averaged over the 800 m grid
  cells the panel used.
- Drawn section length agrees with milepost length within 10% on 90% of
  Oregon sections and 89% of Idaho sections; the rest are short pieces at
  interchanges and mileage equations.
- Every segment in both states has DEM pixels, 132 months of climate, a
  census tract, HPMS traffic, and NLCD pixels (`*_segment_table_coverage.csv`).
- `tests/` cover the mainline rules, section orientation, milepost cutting,
  both assignment methods, conservation, and each covariate on synthetic
  rasters and polygons.
