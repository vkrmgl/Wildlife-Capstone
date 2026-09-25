# Methods

## Data

| Layer (ODOT ArcGIS REST) | Rows | Vintage |
|---|---|---|
| `data_layers/wildlife_collisions/MapServer/0` | 104,528 | 2009-08-04 to 2025-12-31 |
| `data_layers/state_highways/MapServer/0` | 21,376 pieces | 2025 |
| `transgis/catalog/MapServer/155` (AADT, state) | 6,544 intervals | 2024 |
| `transgis/catalog/MapServer/158` (posted speed) | 2,532 intervals | 2025 |
| `transgis/catalog/MapServer/126` (lanes) | 6,036 intervals | 2025 |

Base URL `https://gis.odot.state.or.us/arcgis1006/rest/services/`. Downloads are
logged in `data/raw/manifest.json` with row counts and checksums.

Collision records and network layers share ODOT's linear reference key (`LRM_KEY`):
highway (3 characters), suffix (2; `00` = mainline, others are ramps, connections,
frontage roads), roadway (1; `I` = add roadway, `D` = second roadway of a divided
highway), and mileage key (2; `00` = regular, `Z1`..`Z7` = overlapping mileage from a
mileage equation).

## Segments

- Base network: mainline (`00`), `RDWY_TYP = reg`, roadway 1. The second roadway of a
  divided highway shares mileposts, so it is counted once and kept as the
  `divided_share` feature.
- Ramps, connections, and frontage roads are excluded.
- A segment is the network between mileposts k and k+1 on one highway and mileage
  key. Z-mileage is segmented separately (IDs like `014Z1_25`) because it repeats
  milepost numbers from the regular mileage.
- Segment length is the covered length, so terminal miles and miles broken by
  milepost gaps are shorter than 1.0. All are kept; length is the model offset.

Result: 7,573 segments, 7,337.7 miles; 114 under 0.25 mi; 29 Z-mileage segments.

## Collisions

Target species are ODOT `ANIMAL` values `Deer` and `Elk` (Oregon has no moose
category). A record is assigned to mile `floor(MP)` on its highway and mileage key;
`D`-roadway records go to the same mainline mile.

| Outcome | Records |
|---|---|
| Raw | 104,528 |
| Not deer/elk | 10,390 |
| Deer/elk | 94,138 |
| On ramps/frontage roads (unassigned) | 1,527 |
| Milepost with no segment (unassigned) | 2 |
| Assigned | 92,609 (98.38%) |

## Traffic and road features

Traffic layers give values over milepost intervals. Each segment takes the
length-weighted mean of the intervals it overlaps (roadway 1).

| Feature | Definition | Coverage |
|---|---|---|
| `log_aadt` | log of length-weighted AADT | 99.76% direct; 18 Z-mileage segments take the nearest regular-mileage interval (`aadt_imputed = 1`) |
| `truck_pct` | truck share of AADT | 100% |
| `posted_speed` | length-weighted posted speed | 100% |
| `lanes` | roadway 1 lanes plus roadway 2 lanes times its covered share | 100% |
| `divided_share` | share of the segment with a second roadway | 100% |

**Time.** AADT is 2024 and speed and lanes are 2025, applied to every year. Collision
history is strictly before each cutoff, but the road attributes are current. B1 is
therefore a spatial ranking with present-day road attributes, and any bias from the
snapshot favors B1. ODOT's Highway Reports tool lists AADT for 2019-2025 only, which
could not cover the validation fold.

## Folds

| Fold | History | Target | B1 fit window |
|---|---|---|---|
| Validation | 2014-2017 | 2018-2020 | 2015-2017 |
| Test | 2017-2020 | 2021-2025 | 2015-2020 |

2009 is excluded (records start in August). History and target windows never
overlap, and every segment appears in every fold.

## Models

**B0.** Collisions per mile per year in the history window. A second variant uses all
years from 2010 to the cutoff; the 4-year window scored higher on validation and is
the reference.

**B1.**

```
deer_elk_count ~ log_aadt + truck_pct + posted_speed + lanes + divided_share
offset = log(length_mi * years)
```

Fit on one row per segment, with the count pooled over the fit window, because the
covariates do not change over time and repeated rows of one segment are not
independent. Standard errors are clustered by highway. Poisson was overdispersed
(Pearson chi-squared / df = 5.6 and 9.3), so B1 is negative binomial (alpha = 1.58
and 1.43).

Test-fit coefficients (log scale, clustered SE): log_aadt 0.73 (0.05), truck_pct
0.016 (0.006), posted_speed 0.010 (0.005), lanes -0.22 (0.12), divided_share -0.53
(0.31). `results/b1_coefficients.csv` has both fits.

## Evaluation

- **capture@K**: share of target-period collisions on the top-ranked segments filling
  K% of network miles (K = 1, 5, 10). Tied scores are one block; a partly selected
  block contributes its collisions per mile times the remaining miles (the expected
  value of random tie-breaking).
- **Spearman** correlation between the score and the target rate per mile-year.
- **Ceiling**: ranking by the actual target rate, the maximum capture for each K.
- **Random**: expected capture equals K.

All metrics were recomputed independently (random tie-breaking, 400 draws) and
matched to within 0.0001.

## Results

| Fold | Model | capture@1% | capture@5% | capture@10% | Spearman |
|---|---|---|---|---|---|
| Validation | B0 | 0.088 | 0.274 | 0.423 | 0.700 |
| Validation | B1 | 0.019 | 0.087 | 0.190 | 0.451 |
| Validation | Ceiling | 0.116 | 0.336 | 0.504 | 1.000 |
| Test | B0 | 0.098 | 0.282 | 0.428 | 0.723 |
| Test | B1 | 0.015 | 0.072 | 0.170 | 0.442 |
| Test | Ceiling | 0.114 | 0.330 | 0.492 | 1.000 |

Diagnostics on the test fold (`results/diagnostics.csv`, `results/history_groups.csv`):

- The 12% of miles with the highest 2017-2020 rates hold 49% of 2021-2025 collisions.
  The 37% of miles with no 2017-2020 record hold 7%.
- 80% of the variance in B1's score is between highways; 69% of the variance in future
  collision rates is within highways. On the 50 highways with at least 50 segments, B0
  has the higher within-highway Spearman on 90%.
- On segments with no history, B1 captures 6.7% of their future collisions in its top
  5% of their miles (5% expected at random).

## Limitations

- Traffic and road attributes are current snapshots (see Time).
- Collisions on ramps and frontage roads (1.6% of deer/elk records) are outside the network.
- Records are carcasses reported by ODOT maintenance crews; reporting effort may vary
  by district and year. Annual records fall from about 7,000 in 2018 to about 5,000 in
  2021-2025, for unknown reasons.
- Segment geometry assumes mileposts are proportional to distance and is used only
  for mapping.
