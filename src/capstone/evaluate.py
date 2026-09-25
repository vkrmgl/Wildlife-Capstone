"""Ranking metrics for segment risk scores."""

import pandas as pd
from scipy.stats import spearmanr

CAPTURE_KS = (0.01, 0.05, 0.10)


def capture_at_k(score, collisions, miles, k):
    """Share of collisions on the top-scored segments that fill k of network miles.

    Tied scores form one block. A block that does not fit in the remaining
    budget contributes its collisions per mile times the remaining miles,
    which is the expected value of breaking the tie at random.
    """
    df = pd.DataFrame({"score": score, "y": collisions, "mi": miles})
    blocks = df.groupby("score").agg(y=("y", "sum"), mi=("mi", "sum")).iloc[::-1]
    remaining, captured = k * df["mi"].sum(), 0.0
    for b in blocks.itertuples():
        if remaining <= 0:
            break
        take = min(1.0, remaining / b.mi)
        captured += take * b.y
        remaining -= take * b.mi
    return captured / df["y"].sum()


def evaluate(score, fold):
    """capture@1/5/10% of miles and Spearman correlation with the target rate."""
    mi = fold["length_mi"].to_numpy()
    y = fold["deer_elk_count"].to_numpy()
    out = {"n_segments": len(fold), "network_miles": round(float(mi.sum()), 1), "future_collisions": int(y.sum())}
    for k in CAPTURE_KS:
        out[f"capture_at_{round(k * 100)}"] = round(capture_at_k(score, y, mi, k), 4)
    out["spearman"] = round(float(spearmanr(score, fold["deer_elk_rate"]).statistic), 4)
    return out
