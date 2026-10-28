"""The states the pipeline runs on.

Every covariate comes from a national source, so adding a state means saying
where its collision records come from and how its HPMS route_id marks the
mainline. HPMS route_id is state-defined: Oregon submits ODOT's 8-character
LRM_KEY, Idaho an 11-character ITD key. The rest of the pipeline is the same
for every state.
"""

from dataclasses import dataclass
from typing import Callable

import geopandas as gpd
import pandas as pd
import requests

from capstone.data import RAW

STATES_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip"
STATES_ZIP = RAW / "cb_2023_us_state_500k.zip"


# --- Oregon: route_id = ODOT LRM_KEY -----------------------------------------
# highway (3), suffix (2, "00" = mainline), roadway (1, "I" = inventory
# direction, "D" = second roadway of a divided highway), mileage key (2).

def odot_mainline(hpms):
    rid = hpms["route_id"].astype("string")
    return ((rid.str.len() == 8) & (rid.str[3:5] == "00") & (rid.str[5] == "I")).fillna(False).to_numpy()


def odot_route_label(route_id):
    """'001' for regular mileage, '001Z1' for an overlapping mileage key."""
    rid = pd.Series(route_id, dtype="string")
    mkey = rid.str[6:8]
    return (rid.str[:3] + mkey.where(mkey != "00", "")).to_numpy()


# --- Idaho: route_id = ITD key ------------------------------------------------
# route number (5), roadway (1, "A" = inventory direction, "D" = the other
# roadway of a divided highway), signing (2: IN, US, SH, OH), number (3).
# Divided highways' second roadways are facility_type 6 (non-inventory
# direction); ramps are 4 and non-mainline pieces 5.

ITD_SIGNING = {"IN": "I", "US": "US", "SH": "ID", "OH": "OH"}


def itd_mainline(hpms):
    rid = hpms["route_id"].astype("string")
    ok = (rid.str.len() == 11) & (rid.str[5] == "A")
    return (ok.fillna(False) & hpms["facility_type"].isin([1, 2])).to_numpy()


def itd_route_label(route_id):
    """'US-95', 'I-84', 'ID-21'. Spurs and couplets share the label of the
    signed route they belong to; segment_id uses the full route_id."""
    rid = pd.Series(route_id, dtype="string")
    return (rid.str[6:8].map(ITD_SIGNING).fillna(rid.str[6:8]) + "-" + rid.str[8:11].str.lstrip("0")).to_numpy()


@dataclass(frozen=True)
class State:
    code: str
    fips: str
    name: str
    collisions: str            # loader name in capstone.collisions.SOURCES
    assign_order: tuple        # which location to trust first: "milepost" or "coordinates"
    mainline: Callable         # HPMS sections -> boolean mask of inventory-direction mainline
    route_label: Callable      # route_id values -> human-readable route labels


STATES = {
    "OR": State("OR", "41", "Oregon", "odot", ("milepost", "coordinates"), odot_mainline, odot_route_label),
    "ID": State("ID", "16", "Idaho", "idfg", ("coordinates", "milepost"), itd_mainline, itd_route_label),
}


def get_state(code):
    try:
        return STATES[code.upper()]
    except KeyError:
        raise SystemExit(f"Unknown state {code!r}; known: {', '.join(STATES)}") from None


def download_states():
    """Census cartographic state boundaries (1:500k), shared by every state."""
    if STATES_ZIP.exists():
        return STATES_ZIP
    RAW.mkdir(parents=True, exist_ok=True)
    with requests.get(STATES_URL, stream=True, timeout=120) as r, open(STATES_ZIP, "wb") as f:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    return STATES_ZIP


def state_boundary(state):
    """The state's polygon as a GeoSeries in EPSG:4326."""
    states = gpd.read_file(download_states())
    return states.loc[states["STUSPS"] == state.code, "geometry"].reset_index(drop=True)


def state_bounds(state, pad_deg=0.0):
    """(minx, miny, maxx, maxy) of the state in degrees, padded on every side."""
    minx, miny, maxx, maxy = state_boundary(state).total_bounds
    return minx - pad_deg, miny - pad_deg, maxx + pad_deg, maxy + pad_deg
