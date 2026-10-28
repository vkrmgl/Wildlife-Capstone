"""Population density from Census tracts: TIGER/Line tract boundaries for
the state and total population (table B01003) from the ACS 5-year summary
file, which needs no API key.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import shapely

from capstone.data import RAW

TIGER_YEAR = 2023
ACS_YEAR = 2023
TRACT_URL = "https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{fips}_tract.zip"
ACS_URL = ("https://www2.census.gov/programs-surveys/acs/summary_file/{year}/table-based-SF/data/5YRData/"
           "acsdt5y{year}-b01003.dat")
ACS_PATH = RAW / f"acsdt5y{ACS_YEAR}-b01003.dat"
TRACT_PREFIX = "1400000US"


def tract_path(state):
    return RAW / f"tl_{TIGER_YEAR}_{state.fips}_tract.zip"


def _fetch(url, path):
    if path.exists():
        print(f"{path.name} exists, skipping")
        return path
    print(f"{path.name}: {url}")
    with requests.get(url, stream=True, timeout=300) as r, open(path, "wb") as f:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    return path


def download_population(state):
    _fetch(TRACT_URL.format(year=TIGER_YEAR, fips=state.fips), tract_path(state))
    _fetch(ACS_URL.format(year=ACS_YEAR), ACS_PATH)


def tract_density(state):
    """Tracts of the state with population, land area, and people per km2."""
    tracts = gpd.read_file(tract_path(state))[["GEOID", "ALAND", "geometry"]]
    acs = pd.read_csv(ACS_PATH, sep="|", dtype={"GEO_ID": "string"}, usecols=["GEO_ID", "B01003_E001"])
    acs = acs[acs["GEO_ID"].str.startswith(TRACT_PREFIX)]
    acs["GEOID"] = acs["GEO_ID"].str[len(TRACT_PREFIX):]
    tracts = tracts.merge(acs[["GEOID", "B01003_E001"]], on="GEOID", how="left")
    tracts["population"] = pd.to_numeric(tracts["B01003_E001"], errors="coerce")
    tracts["pop_density_per_km2"] = tracts["population"] / (tracts["ALAND"] / 1e6)
    tracts.loc[tracts["ALAND"] <= 0, "pop_density_per_km2"] = np.nan
    return tracts[["GEOID", "population", "ALAND", "pop_density_per_km2", "geometry"]]


def population_features(segments, tracts):
    """One row per segment: pop_density_per_km2, the density of the tracts the
    segment runs through weighted by the length in each. A segment touching
    no tract (just over the state line) takes the nearest tract."""
    seg = segments[["segment_id", "geometry"]].to_crs("EPSG:5070")
    tr = tracts[["pop_density_per_km2", "geometry"]].to_crs("EPSG:5070").dropna(subset=["pop_density_per_km2"])
    j = gpd.sjoin(seg, tr, how="inner", predicate="intersects")
    j["len"] = shapely.length(shapely.intersection(j.geometry.values, tr.geometry.loc[j["index_right"]].values))
    j = j[j["len"] > 0]
    dens = (j["pop_density_per_km2"] * j["len"]).groupby(j["segment_id"]).sum() / j.groupby("segment_id")["len"].sum()
    out = segments[["segment_id"]].copy()
    out["pop_density_per_km2"] = out["segment_id"].map(dens).values
    missing = out["pop_density_per_km2"].isna()
    if missing.any():
        near = gpd.sjoin_nearest(seg[missing.values], tr, how="left").drop_duplicates("segment_id")
        out.loc[missing, "pop_density_per_km2"] = near.set_index("segment_id")["pop_density_per_km2"] \
            .reindex(out.loc[missing, "segment_id"]).values
    return out
