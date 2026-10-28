"""XGBoost rate model on the per-segment tables.

Deer and elk collisions in a window are modeled as Poisson counts with
log(miles x years) as the exposure offset, so the prediction is a collision
rate per mile-year that can be ranked like B0 and B1. The number of trees is
chosen by cross-validation with whole routes held out, because neighbouring
miles are near-duplicates and a random split would overfit.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from capstone.collisions import TEST_YEARS, TRAIN_YEARS
from capstone.data import PROCESSED
from capstone.gbif import GBIF_TAXA
from capstone.landcover import NLCD_GROUPS

TERRAIN = ["elevation_mean_m", "elevation_min_m", "elevation_max_m", "slope_mean_deg", "slope_max_deg"]
CLIMATE = ["ppt_month_mean_mm", "ppt_month_min_mm", "ppt_month_max_mm", "ppt_annual_mm", "ppt_winter_month_mm",
           "ppt_summer_month_mm", "tmean_c", "tmean_month_min_c", "tmean_month_max_c", "tmean_winter_c",
           "tmean_summer_c"]
ROAD = ["hpms_log_aadt", "hpms_truck_share", "hpms_lanes", "hpms_speed_limit", "hpms_rural_share",
        "hpms_f_system", "hpms_access_control"]
# Bird sightings (owl, hawk, eagles) are not plausible drivers of deer and elk
# collisions and mostly mark where people record observations, so only the
# mammal taxa are used as features.
BIRDS = {"owl", "hawk", "bald_eagle", "golden_eagle"}
WILDLIFE = [f"gbif_{g}" for g in GBIF_TAXA if g not in BIRDS]
FEATURES = (TERRAIN + CLIMATE + ["pop_density_per_km2"] + ROAD
            + [f"nlcd_{g}" for g in NLCD_GROUPS] + WILDLIFE)
HISTORY = "history_rate"
TARGET = "deer_elk"

PARAMS = {"objective": "count:poisson", "eta": 0.03, "max_depth": 4, "min_child_weight": 5,
          "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "poisson-nloglik", "seed": 0}
MAX_ROUNDS = 3000
EARLY_STOP = 50
FOLDS = 5


def load_table(state_code):
    return pd.read_parquet(PROCESSED / f"{state_code.lower()}_segment_table.parquet")


def history_rate(state_code, start, end, groups=("deer", "elk")):
    """Collisions per mile-year over start..end from the per-year counts,
    indexed by segment_id (segments with none are missing here)."""
    counts = pd.read_parquet(PROCESSED / f"{state_code.lower()}_segment_year_counts.parquet")
    ok = counts["group"].isin(groups) & counts["year"].between(start, end)
    return counts[ok].groupby("segment_id")["n"].sum() / (end - start + 1)


def make_fold(table, target_window, history_window=None):
    """One row per segment with the target window's count, exposure, rate,
    the features, and (optionally) the collision rate of an earlier window."""
    start, end = target_window
    out = table[["segment_id", "state", "route", "length_mi", *FEATURES]].copy()
    out["deer_elk_count"] = table[f"coll_{TARGET}_{start}_{end}"].to_numpy()
    out["target_years"] = end - start + 1
    out["deer_elk_rate"] = out["deer_elk_count"] / out["length_mi"] / out["target_years"]
    if history_window is not None:
        hist = history_rate(table["state"].iloc[0], *history_window)
        out[HISTORY] = (out["segment_id"].map(hist).fillna(0.0) / out["length_mi"]).to_numpy()
    return out


def route_folds(routes, k=FOLDS, seed=0):
    """(train index, test index) pairs that keep every mile of a route together."""
    unique = np.array(sorted(pd.unique(routes)))
    fold_of_route = dict(zip(unique, np.random.default_rng(seed).permutation(len(unique)) % k))
    fold = np.array([fold_of_route[r] for r in routes])
    idx = np.arange(len(routes))
    return [(idx[fold != f], idx[fold == f]) for f in range(k)]


def dmatrix(fold, features, with_offset=True):
    X = fold[features].astype(float)
    y = fold["deer_elk_count"].to_numpy()
    offset = np.log(fold["length_mi"].to_numpy() * fold["target_years"].to_numpy())
    return xgb.DMatrix(X, label=y, base_margin=offset if with_offset else None)


def fit(train, features=FEATURES, params=PARAMS):
    """Choose the number of trees by route-grouped CV, then fit on all of
    train. Returns (booster, info)."""
    dtrain = dmatrix(train, features)
    cv = xgb.cv(params, dtrain, num_boost_round=MAX_ROUNDS, folds=route_folds(train["route"].to_numpy()),
                early_stopping_rounds=EARLY_STOP, verbose_eval=False, seed=params.get("seed", 0))
    rounds = len(cv)
    booster = xgb.train(params, dtrain, num_boost_round=rounds)
    info = {"rounds": rounds, "cv_poisson_nloglik": round(float(cv["test-poisson-nloglik-mean"].iloc[-1]), 4),
            "n_train": len(train)}
    return booster, info


def predict_rate(booster, fold, features=FEATURES):
    """Predicted collisions per mile-year (no offset, so the margin is the rate)."""
    return booster.predict(xgb.DMatrix(fold[features].astype(float)))


def importance(booster, features=FEATURES):
    gain = booster.get_score(importance_type="gain")
    out = pd.DataFrame({"feature": features, "gain": [gain.get(f, 0.0) for f in features]})
    out["share"] = out["gain"] / out["gain"].sum()
    return out.sort_values("gain", ascending=False).reset_index(drop=True)


def windows():
    """(train target, test target) windows; history for the test fold is the train window."""
    return TRAIN_YEARS, TEST_YEARS
