"""Score B0 and B1 on the validation and test folds.

B1 is fit on one pooled window per scored fold (one row per segment): 2015-2017
for validation, 2015-2020 for test. The ceiling ranks by the actual target
rate and is a reference line only.

Writes results/model_results.csv, baselines.csv, b1_coefficients.csv, and
data/processed/or_scores_{fold}.parquet (used by diagnostics.py).

Usage: uv run python scripts/run_models.py
"""

import pandas as pd

from capstone.data import PROCESSED, RESULTS
from capstone.evaluate import evaluate
from capstone.features import build_fold, target_table
from capstone.models import coefficients, fit_b1, predict_rate

# fold: (history start, history end, target start, target end)
FOLDS = {"validation": (2014, 2017, 2018, 2020), "test": (2017, 2020, 2021, 2025)}
B1_FIT_WINDOWS = {"validation": (2015, 2017), "test": (2015, 2020)}
FIRST_FULL_YEAR = 2010  # records start in August 2009

seg = pd.read_parquet(PROCESSED / "or_segments.parquet", columns=["segment_id", "hwy", "length_mi"])
assigned = pd.read_parquet(PROCESSED / "or_collisions_assigned.parquet")
traffic = pd.read_parquet(PROCESSED / "or_traffic_features.parquet")

results, coefs = [], []
for fold_name, window in FOLDS.items():
    hs, he, ts, te = window
    fold = build_fold(assigned, seg, *window).merge(traffic, on="segment_id", validate="one_to_one")
    all_years = build_fold(assigned, seg, FIRST_FULL_YEAR, he, ts, te)

    fit_start, fit_end = B1_FIT_WINDOWS[fold_name]
    assert fit_end < ts, "B1 fit window must end before the target period"
    train = target_table(assigned, seg, fit_start, fit_end).merge(traffic, on="segment_id", validate="one_to_one")
    b1, info = fit_b1(train, groups=seg["hwy"])
    coefs.append(coefficients(b1).assign(fold=fold_name, fit_window=f"{fit_start}-{fit_end}", **info))

    scores = {
        "B0_history_4yr": (fold["history_rate"].to_numpy(), hs),
        "B0_history_all": (all_years["history_rate"].to_numpy(), FIRST_FULL_YEAR),
        "B1_traffic": (predict_rate(b1, fold), hs),
        "ceiling": (fold["deer_elk_rate"].to_numpy(), hs),
    }
    for model, (score, start) in scores.items():
        results.append({"fold": fold_name, "model": model, "history_start": start, "history_end": he,
                        "target_start": ts, "target_end": te, **evaluate(score, fold)})
        fold[f"score_{model}"] = score
    fold.to_parquet(PROCESSED / f"or_scores_{fold_name}.parquet", index=False)

results = pd.DataFrame(results)
results.to_csv(RESULTS / "model_results.csv", index=False)
pd.concat(coefs).to_csv(RESULTS / "b1_coefficients.csv", index=False)

# Canonical comparison, with the expected capture of a random ranking (= K).
labels = {"B0_history_4yr": ("B0", "collisions per mile-year, 4-year history", "yes"),
          "B1_traffic": ("B1", "log AADT, truck %, posted speed, lanes, divided share", "no"),
          "ceiling": ("Ceiling", "actual target-window rate", "n/a")}
base = results[results["model"].isin(labels)].copy()
base[["model", "features", "history"]] = base["model"].map(labels).tolist()
random = pd.DataFrame({"fold": list(FOLDS), "model": "Random", "features": "none (expected value)", "history": "no",
                       "capture_at_1": 0.01, "capture_at_5": 0.05, "capture_at_10": 0.10, "spearman": 0.0})
cols = ["fold", "model", "features", "history", "capture_at_1", "capture_at_5", "capture_at_10", "spearman"]
pd.concat([base[cols], random]).sort_values("fold", kind="stable").to_csv(RESULTS / "baselines.csv", index=False)

print(results.to_string(index=False))
