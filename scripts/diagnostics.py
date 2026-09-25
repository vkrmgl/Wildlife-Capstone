"""Descriptive comparison of B0 and B1 on the validation and test folds.

Writes results/diagnostics.csv and results/history_groups.csv. Nothing here
is used to choose or tune a model.

Usage: uv run python scripts/diagnostics.py (after run_models.py)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from capstone.data import PROCESSED, RESULTS
from capstone.evaluate import capture_at_k


def rho(a, b):
    return float(spearmanr(a, b).statistic)


def top_miles(score, miles, k):
    order = np.argsort(-score, kind="stable")
    mask = np.zeros(len(score), bool)
    mask[order[np.cumsum(miles[order]) <= k * miles.sum()]] = True
    return mask


seg = pd.read_parquet(PROCESSED / "or_segments.parquet", columns=["segment_id", "hwy"])
metrics, groups = [], []
for fold_name in ("validation", "test"):
    t = pd.read_parquet(PROCESSED / f"or_scores_{fold_name}.parquet").merge(seg, on="segment_id")
    b0, b1 = t["score_B0_history_4yr"].to_numpy(), t["score_B1_traffic"].to_numpy()
    fut, mi, y = t["deer_elk_rate"].to_numpy(), t["length_mi"].to_numpy(), t["deer_elk_count"].to_numpy()
    years = t["target_years"].iloc[0]
    zero = t["history_count"].to_numpy() == 0
    b1_pct = pd.Series(b1).rank(pct=True).to_numpy()
    top_b0, top_b1 = top_miles(b0, mi, 0.05), top_miles(b1, mi, 0.05)

    m = {
        "spearman_history_vs_future": rho(b0, fut),
        "spearman_B1_vs_future": rho(b1, fut),
        "spearman_B0_vs_B1": rho(b0, b1),
        "pearson_log1p_history_vs_future": float(np.corrcoef(np.log1p(b0), np.log1p(fut))[0, 1]),
        "B0_top5_median_B1_percentile": float(np.median(b1_pct[top_b0])),
        "B0_top5_share_also_in_B1_top5_miles": float(mi[top_b0 & top_b1].sum() / mi[top_b0].sum()),
        "zero_history_segments": int(zero.sum()),
        "zero_history_share_of_miles": float(mi[zero].sum() / mi.sum()),
        "zero_history_share_of_future_collisions": float(y[zero].sum() / y.sum()),
        "zero_history_future_rate": float(y[zero].sum() / mi[zero].sum() / years),
        "nonzero_history_future_rate": float(y[~zero].sum() / mi[~zero].sum() / years),
        "within_zero_history_spearman_B1": rho(b1[zero], fut[zero]),
        "within_zero_history_capture5_B1": capture_at_k(b1[zero], y[zero], mi[zero], 0.05),
        "within_nonzero_history_spearman_B0": rho(b0[~zero], fut[~zero]),
        "within_nonzero_history_spearman_B1": rho(b1[~zero], fut[~zero]),
    }
    # How much of each quantity varies between highways rather than along them.
    for name, v in (("log_B1", np.log(b1)), ("log1p_future_rate", np.log1p(fut)), ("log1p_B0", np.log1p(b0))):
        s = pd.Series(v)
        m[f"share_variance_between_highways_{name}"] = float(s.groupby(t["hwy"]).transform("mean").var() / s.var())

    # Within-highway ranking on highways with at least 50 segments.
    by_hwy = pd.DataFrame([{"B0": rho(d["score_B0_history_4yr"], d["deer_elk_rate"]),
                            "B1": rho(d["score_B1_traffic"], d["deer_elk_rate"])}
                           for _, d in t.groupby("hwy") if len(d) >= 50 and d["deer_elk_count"].sum() > 0])
    m.update(by_highway_highways=len(by_hwy), by_highway_median_spearman_B0=by_hwy["B0"].median(),
             by_highway_median_spearman_B1=by_hwy["B1"].median(),
             by_highway_share_B0_higher=(by_hwy["B0"] > by_hwy["B1"]).mean())
    metrics += [{"fold": fold_name, "metric": k, "value": round(float(v), 4)} for k, v in m.items()]

    # Future collisions by history group: zero history, then quintiles of the rest.
    grp = pd.Series("zero history", index=t.index)
    grp[~zero] = pd.qcut(pd.Series(b0[~zero]).rank(method="first"), 5,
                         labels=[f"nonzero Q{i}" for i in range(1, 6)]).astype(str).values
    g = pd.DataFrame({"group": grp, "mi": mi, "y": y, "b1_pct": b1_pct}).groupby("group").agg(
        segments=("y", "size"), miles=("mi", "sum"), future_collisions=("y", "sum"), mean_B1_percentile=("b1_pct", "mean"))
    g["share_of_miles"] = g["miles"] / g["miles"].sum()
    g["share_of_future_collisions"] = g["future_collisions"] / g["future_collisions"].sum()
    g["future_rate_per_mile_year"] = g["future_collisions"] / g["miles"] / years
    groups.append(g.round(4).reset_index().assign(fold=fold_name))

metrics = pd.DataFrame(metrics)
metrics.to_csv(RESULTS / "diagnostics.csv", index=False)
pd.concat(groups).to_csv(RESULTS / "history_groups.csv", index=False)
print(metrics.pivot(index="metric", columns="fold", values="value").to_string())
