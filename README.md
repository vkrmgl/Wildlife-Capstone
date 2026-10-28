# Wildlife-Vehicle Collision Risk on State Highways

UC Berkeley MIDS W210 Capstone. Ewura Mensah, Perla Meza, Vikram Magal, Gabrielle Doran.

State agencies choose where to build wildlife crossings and fencing mostly by ranking
road segments by past collision counts. We test whether a model built on road,
traffic, terrain, climate, land cover, and wildlife features can rank where future
deer and elk collisions will occur, and how it compares to that historical-count
ranking. The pipeline runs on Oregon and Idaho and is built so that any state with
collision records can be added.

## Main datasets

`data/processed/or_segment_table.parquet` and `id_segment_table.parquet`: one row per
one-mile segment of state highway (7,574 in Oregon, 5,050 in Idaho), with collision
counts for a train window (2015-2020) and a test window (2021-2025) and 51 covariates.
Every covariate comes from a national source, so the same table can be built for any
state; only the collision records come from the state agency.

| | Source | Per segment |
|---|---|---|
| Segments, traffic, lanes, speed limit, road class | FHWA HPMS 2024 | Sections cut at integer mileposts; attributes length-weighted |
| Elevation, slope | USGS 3DEP (30 m) | Within 30 m of the road |
| Precipitation, temperature | PRISM monthly 4 km, 2015-2025 | Monthly, annual, winter, summer summaries |
| Population density | Census tracts, ACS 2019-2023 | Length-weighted along the road |
| Land cover | USGS Annual NLCD 2024 | Shares within 200 m |
| Species occurrences | GBIF (eBird excluded) | Records within 5 km, per species group |
| Collisions | ODOT animal incidents (OR), IDFG roadkill observations (ID) | By milepost (OR) or nearest segment within 100 m (ID) |

Columns, sources, and checks are in [docs/segment_table.md](docs/segment_table.md);
plain-language column descriptions in `data/processed/{st}_segment_table_dictionary.csv`;
coverage and conservation in `results/{st}_segment_table_coverage.csv`.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/). No API keys. The downloads are about 8 GB
(mostly elevation tiles and the land cover mosaic) and are skipped when already present.

```bash
uv sync
uv run python scripts/download_sources.py        # all sources for OR and ID -> data/raw/
uv run python scripts/build_segment_table.py     # segments, collisions, covariates -> data/processed/
uv run pytest
```

Both scripts take state codes as arguments (`... OR` builds Oregon only). To add a
state, add a `State` to `src/capstone/states.py`: where its collision records come
from (a loader in `src/capstone/collisions.py`) and how its HPMS `route_id` marks the
mainline. Everything else is shared.

Layout: `src/capstone/` holds the pipeline (`states`, `network`, `collisions`,
`terrain`, `climate`, `population`, `landcover`, `gbif`, `table`); `scripts/` the
entry points; `tests/` unit tests on synthetic data. `data/` is not tracked except the
data dictionaries. Small summary tables in `results/` are tracked.

## XGBoost model

`notebooks/xgboost_model.ipynb` fits a Poisson-count XGBoost model of deer and elk
collisions per mile-year on the covariates above except bird sightings (`src/capstone/xgb.py`), trains on
2015-2020 and scores 2021-2025 with the same capture@K metric as the baseline, within
each state and across states. Results are in `results/xgb_results.csv` and feature
importance in `results/xgb_importance.csv`.

| Tested on | Model | capture@10% | Spearman |
|---|---|---|---|
| Oregon | B0 history rate | 0.433 | 0.75 |
| Oregon | XGB covariates | 0.418 | 0.74 |
| Oregon | XGB covariates + history | 0.436 | 0.77 |
| Oregon | XGB trained on Idaho | 0.217 | 0.48 |
| Idaho | B0 history rate | 0.385 | 0.55 |
| Idaho | XGB covariates | 0.364 | 0.58 |
| Idaho | XGB covariates + history | 0.392 | 0.58 |
| Idaho | XGB trained on Oregon | 0.222 | 0.40 |

Covariates alone rank segments nearly as well as collision history within a state,
far better than the traffic-only B1, but a model does not transfer from one state to
the other without being fit there.

## Oregon baseline (B0 vs B1)

The first experiment, on ODOT layers only, is kept as a reference:

- Oregon state highways are split into 7,573 one-mile segments (7,338 miles).
  98.4% of deer and elk records are assigned to a segment.
- **B0** ranks segments by collisions per mile per year in the four years before the cutoff.
- **B1** is a binomial model of collision counts with log AADT, truck share,
  posted speed, lanes, and divided share, offset by segment length.
- Temporal evaluation: history through 2020, test period 2021-2025 (validation:
  history through 2017, target 2018-2020).
- The main metric is **capture@K**: the share of test-period collisions on the
  top-ranked K% of network miles.

| Model | History | capture@1% | capture@5% | capture@10% | Spearman |
|---|---|---|---|---|---|
| B0 historical counts | yes | 0.098 | 0.282 | 0.428 | 0.72 |
| B1 traffic/road | no | 0.015 | 0.072 | 0.170 | 0.44 |
| Random (expected) | no | 0.010 | 0.050 | 0.100 | 0 |
| Ceiling (actual future rate) | n/a | 0.114 | 0.330 | 0.492 | 1.00 |

Collision locations persist strongly from year to year, and B0 is within 5 points of
the ceiling. B1 ranks corridors better than random but does not separate miles within
a highway well. See [docs/methods.md](docs/methods.md).

```bash
uv run python scripts/download_data.py   # ODOT layers -> data/raw/
uv run python scripts/build_dataset.py   # ODOT segments, assignment, traffic -> data/processed/odot_*
uv run python scripts/run_models.py      # B0, B1 -> results/
uv run python scripts/diagnostics.py     # B0 vs B1 diagnostics -> results/
```

## Earlier work

`notebooks/wvc_capstone_pipeline.ipynb` built a segment x month panel for Oregon in
Colab (`data/processed/master_panel.parquet`) with terrain, climate, population,
habitat, connectivity, and traffic. It was the starting point for the covariate set
but is not an input to the pipeline: its Oregon-only layers (OCAMP connectivity, ODFW
migration corridors, the 1992 habitat map, station-based AADT) were replaced by
national sources, and its remaining covariates are now computed by the scripts.
`notebooks/EDA.ipynb` is an exploratory analysis of the Idaho records.
