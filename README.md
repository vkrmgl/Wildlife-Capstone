# Wildlife-Vehicle Collision Risk on Oregon State Highways

UC Berkeley MIDS W210 Capstone. Ewura Mensah, Perla Meza, Vikram Magal, Gabrielle Doran.

State agencies choose where to build wildlife crossings and fencing mostly by ranking
road segments by past collision counts. We test whether a model built on road and
traffic features can rank where future deer and elk collisions will occur, and how it
compares to that historical-count ranking.

## Data

All from open ODOT ArcGIS layers (downloaded 2026-09-24):

| Layer | Use |
|---|---|
| Wildlife collisions (104,528 records, 2009-2025) | Outcome: deer and elk carcass records |
| State highways | 1-mile segments on the milepost system |
| AADT (2024), posted speed and lanes (2025) | Traffic and road features |

## Approach

- Oregon state highways are split into 7,573 one-mile segments (7,338 miles).
  98.4% of deer and elk records are assigned to a segment.
- **B0** ranks segments by collisions per mile per year in the four years before the cutoff.
- **B1** is a binomial model of collision counts with log AADT, truck share,
  posted speed, lanes, and divided share, offset by segment length.
- Temporal evaluation: history through 2020, test period 2021-2025 (validation:
  history through 2017, target 2018-2020).
- The main metric is **capture@K**: the share of test-period collisions on the
  top-ranked K% of network miles.

## Results (test period 2021-2025)

| Model | History | capture@1% | capture@5% | capture@10% | Spearman |
|---|---|---|---|---|---|
| B0 historical counts | yes | 0.098 | 0.282 | 0.428 | 0.72 |
| B1 traffic/road | no | 0.015 | 0.072 | 0.170 | 0.44 |
| Random (expected) | no | 0.010 | 0.050 | 0.100 | 0 |
| Ceiling (actual future rate) | n/a | 0.114 | 0.330 | 0.492 | 1.00 |

Collision locations persist strongly from year to year, and B0 is within 5 points of
the ceiling. B1 ranks corridors better than random but does not separate miles within
a highway well. AADT, speed, and lanes are current snapshots, so B1 is a spatial
ranking with present-day road attributes. See [docs/methods.md](docs/methods.md).

## Per-segment table with national covariates

`data/processed/master_panel.parquet` (from `notebooks/wvc_capstone_pipeline.ipynb`)
is a segment x month panel on half-mile OR-Trans segments with weather, terrain,
population, habitat, connectivity, and traffic joined in. `scripts/build_segment_table.py`
collapses it to one row per one-mile segment and keeps only covariates with a
national source, so the same table can be built for any state:

- terrain, population, and climate summaries carried over from the panel through
  its 800 m grid cells;
- traffic, lanes, speed limit, functional class, and rural/urban from FHWA HPMS,
  joined by milepost;
- land cover shares from USGS Annual NLCD;
- GBIF occurrence counts within 5 km for each animal type in the ODOT records;
- collision counts for all animals and for deer/elk, 2015-2020 and 2021-2025.

Output: `data/processed/or_segment_table.parquet`. Columns and sources are in
[docs/segment_table.md](docs/segment_table.md); coverage in
`results/segment_table_coverage.csv`.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/download_data.py   # ODOT layers -> data/raw/
uv run python scripts/build_dataset.py   # segments, assignment, features -> data/processed/
uv run python scripts/run_models.py      # B0, B1 -> results/
uv run python scripts/diagnostics.py     # B0 vs B1 diagnostics -> results/
uv run python scripts/download_national_sources.py  # HPMS, NLCD (1.4 GB), GBIF -> data/raw/
uv run python scripts/build_segment_table.py        # per-segment table -> data/processed/
uv run pytest
```

`data/` is not tracked except `master_panel.parquet`. Result tables in `results/`
are small and tracked.

`notebooks/EDA.ipynb` is an exploratory analysis of Idaho roadkill records (IDFG),
which is not yet part of the modeling pipeline.
