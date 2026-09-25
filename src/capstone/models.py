"""B0 (historical-count ranking) and B1 (traffic/road count model).

Scores are predicted deer/elk collisions per mile per year; higher ranks first.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

B1_FEATURES = ["log_aadt", "truck_pct", "posted_speed", "lanes", "divided_share"]
# Pearson chi2 / df above this switches B1 from Poisson to negative binomial.
OVERDISPERSION_THRESHOLD = 1.5


def offset(table):
    """log(segment miles * target years): segment length is the exposure offset.
    AADT enters as a covariate so risk need not scale in proportion to traffic."""
    return np.log(table["length_mi"].to_numpy() * table["target_years"].to_numpy())


def design(table):
    X = table[B1_FEATURES].astype(float)
    if X.isna().any().any():
        raise ValueError("Missing B1 feature values")
    return sm.add_constant(X, has_constant="add")


def fit_b1(train, groups):
    """Fit Poisson; refit as NB2 if overdispersed. train must have one row per
    segment. Standard errors are clustered by `groups` (highway) because
    neighboring segments are not independent; this does not change the fit.
    Returns (model, summary dict)."""
    if not train["segment_id"].is_unique:
        raise ValueError("fit_b1 expects one row per segment")
    X, y, off = design(train), train["deer_elk_count"].to_numpy(), offset(train)
    cluster = {"cov_type": "cluster", "cov_kwds": {"groups": pd.factorize(pd.Series(groups))[0]}}

    pois = sm.GLM(y, X, family=sm.families.Poisson(), offset=off).fit(**cluster)
    dispersion = float(pois.pearson_chi2 / pois.df_resid)
    info = {"n_train": len(y), "poisson_dispersion": round(dispersion, 3), "poisson_aic": round(float(pois.aic), 1)}
    if dispersion <= OVERDISPERSION_THRESHOLD:
        return pois, {**info, "family": "poisson"}

    nb = sm.NegativeBinomial(y, X, offset=off, loglike_method="nb2").fit(disp=0, maxiter=500, **cluster)
    return nb, {**info, "family": "negative_binomial", "nb_alpha": round(float(nb.params["alpha"]), 4),
                "nb_aic": round(float(nb.aic), 1), "nb_converged": bool(nb.mle_retvals["converged"])}


def predict_rate(model, table):
    """Predicted collisions per mile per year."""
    X = design(table)
    params = model.params.drop("alpha", errors="ignore")
    return np.exp(X.to_numpy() @ params[X.columns].to_numpy())


def coefficients(model):
    return pd.DataFrame({"term": model.params.index, "estimate": model.params.values,
                         "std_error": model.bse.values, "p_value": model.pvalues.values})
