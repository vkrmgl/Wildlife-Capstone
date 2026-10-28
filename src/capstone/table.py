"""Assembly of the per-segment table and its data dictionary."""

import pandas as pd

from capstone.collisions import TARGET_GROUPS, TEST_YEARS, TRAIN_YEARS
from capstone.gbif import GBIF_TAXA
from capstone.landcover import NLCD_GROUPS


def segment_base(segments, state):
    """Identifiers, mileposts, length, and midpoint lat/lon per segment."""
    out = segments[["segment_id", "route_id", "route", "mile", "begin_mp", "end_mp", "length_mi", "is_short"]].copy()
    out.insert(1, "state", state.code)
    mid = segments.to_crs("EPSG:5070").geometry.interpolate(0.5, normalized=True).to_crs("EPSG:4326")
    out["lat"], out["lon"] = mid.y.values, mid.x.values
    return out


def assemble(parts):
    """Merge one-row-per-segment frames on segment_id, checking each is unique."""
    table = parts[0]
    for p in parts[1:]:
        table = table.merge(p, on="segment_id", how="left", validate="one_to_one")
    return table


# --- Data dictionary ----------------------------------------------------------

SOURCES = {
    "hpms": ("FHWA HPMS", "Road sections every state reports to the Federal Highway Administration, "
                          "with mileposts, traffic counts, lanes, and speed limits"),
    "odot": ("ODOT animal incidents", "Carcass pickups logged by ODOT crews with species and milepost"),
    "idfg": ("IDFG roadkill observations", "Roadkill reports from the public, ITD, and IDFG with species and a map location"),
    "3dep": ("USGS 3DEP", "National elevation model at about 30 m"),
    "prism": ("PRISM monthly climate", "Monthly precipitation and temperature grids at 4 km, 2015 to 2025"),
    "census": ("Census tracts (ACS 5-year)", "Tract boundaries and total population"),
    "nlcd": ("USGS Annual NLCD", "National land cover map at 30 m"),
    "gbif": ("GBIF", "Wildlife sightings and specimens from museums, agencies, and iNaturalist (eBird excluded)"),
}

GROUP_WORDS = {"all": "Animals of any type", "deer_elk": "Deer and elk", "deer": "Deer", "elk": "Elk", "moose": "Moose"}
NLCD_WORDS = {"water": "water", "developed": "developed land (roads, buildings)", "barren": "bare rock or sand",
              "forest": "forest", "shrub": "shrubland", "grass": "grassland", "agriculture": "farmland",
              "wetland": "wetland"}


def dictionary(state):
    """Rows of (column, source, source_description, column_description)."""
    coll = state.collisions
    rows = [
        ("segment_id", "hpms", "Segment ID: HPMS route ID plus the starting milepost"),
        ("state", "hpms", "Two-letter state code"),
        ("route_id", "hpms", "Route ID as the state reports it to HPMS"),
        ("route", "hpms", "Readable route label (Oregon highway number, or US-95 style in Idaho)"),
        ("mile", "hpms", "Starting milepost of the segment"),
        ("begin_mp", "hpms", "First milepost covered by road in this segment"),
        ("end_mp", "hpms", "Last milepost covered by road in this segment"),
        ("length_mi", "hpms", "Miles of road in the segment. Under 1 at route ends and gaps"),
        ("is_short", "hpms", "True if the segment is under a quarter mile"),
        ("lat", "hpms", "Latitude of the segment midpoint"),
        ("lon", "hpms", "Longitude of the segment midpoint"),
    ]
    for start, end in (TRAIN_YEARS, TEST_YEARS):
        years = end - start + 1
        for label in TARGET_GROUPS:
            rows.append((f"coll_{label}_{start}_{end}", coll, f"{GROUP_WORDS[label]} hit on this segment in {start} to {end}"))
            rows.append((f"coll_{label}_{start}_{end}_per_yr", coll, f"Same count divided by {years} years"))
    rows += [
        ("elevation_mean_m", "3dep", "Average elevation along the road, metres"),
        ("elevation_min_m", "3dep", "Lowest elevation along the road, metres"),
        ("elevation_max_m", "3dep", "Highest elevation along the road, metres"),
        ("slope_mean_deg", "3dep", "Average ground slope within 30 m of the road, degrees"),
        ("slope_max_deg", "3dep", "Steepest ground slope within 30 m of the road, degrees"),
        ("terrain_pixels", "3dep", "Number of elevation pixels used"),
        ("ppt_month_mean_mm", "prism", "Average monthly rainfall, mm"),
        ("ppt_month_min_mm", "prism", "Driest month's rainfall, mm"),
        ("ppt_month_max_mm", "prism", "Wettest month's rainfall, mm"),
        ("ppt_annual_mm", "prism", "Average rainfall per year, mm"),
        ("ppt_winter_month_mm", "prism", "Average monthly rainfall in December to February, mm"),
        ("ppt_summer_month_mm", "prism", "Average monthly rainfall in June to August, mm"),
        ("tmean_c", "prism", "Average temperature, C"),
        ("tmean_month_min_c", "prism", "Coldest month's average temperature, C"),
        ("tmean_month_max_c", "prism", "Warmest month's average temperature, C"),
        ("tmean_winter_c", "prism", "Average temperature in December to February, C"),
        ("tmean_summer_c", "prism", "Average temperature in June to August, C"),
        ("weather_months", "prism", "Number of months of climate data behind the averages"),
        ("pop_density_per_km2", "census", "People per square km in the census tracts the road runs through"),
        ("hpms_coverage", "hpms", "Share of the segment covered by HPMS sections"),
        ("hpms_aadt", "hpms", "Average daily traffic, vehicles per day"),
        ("hpms_log_aadt", "hpms", "Natural log of daily traffic"),
        ("hpms_truck_share", "hpms", "Share of daily traffic that is trucks"),
        ("hpms_lanes", "hpms", "Number of through lanes"),
        ("hpms_speed_limit", "hpms", "Posted speed limit, mph"),
        ("hpms_rural_share", "hpms", "Share of the segment outside an urban area"),
        ("hpms_f_system", "hpms", "Road class code: 1 interstate to 7 local"),
        ("hpms_functional_class", "hpms", "Road class name"),
        ("hpms_access_control", "hpms", "1 full access control (freeway), 2 partial, 3 none"),
    ]
    for name in NLCD_GROUPS:
        rows.append((f"nlcd_{name}", "nlcd", f"Share of {NLCD_WORDS[name]} within 200 m of the road"))
    rows += [("nlcd_dominant", "nlcd", "Land cover group with the largest share"),
             ("nlcd_pixels", "nlcd", "Number of land cover pixels counted")]
    for name, (_, taxon) in GBIF_TAXA.items():
        rows.append((f"gbif_{name}", "gbif", f"{taxon} sightings within 5 km of the road"))
    return pd.DataFrame([(c, SOURCES[s][0], SOURCES[s][1], d) for c, s, d in rows],
                        columns=["column", "source", "source_description", "column_description"])


def check_dictionary(table, dictionary_frame):
    """Raise if the table and dictionary columns differ."""
    missing = set(table.columns) - set(dictionary_frame["column"])
    extra = set(dictionary_frame["column"]) - set(table.columns)
    if missing or extra:
        raise AssertionError(f"dictionary out of date: missing {sorted(missing)}, extra {sorted(extra)}")
