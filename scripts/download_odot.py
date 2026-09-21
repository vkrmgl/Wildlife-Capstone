"""Download the complete Oregon DOT Wildlife Collisions layer and save it locally.

Pages through the ArcGIS REST query endpoint (MaxRecordCount = 1000), converts
the epoch-millisecond DATE field to datetime, writes CSV + Parquet to
data/raw/, and prints a validation summary.

Usage: python download_odot.py
"""

import math
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

QUERY_URL = (
    "https://gis.odot.state.or.us/arcgis1006/rest/services/"
    "data_layers/wildlife_collisions/MapServer/0/query"
)
PAGE_SIZE = 1000
PAGE_DELAY_S = 0.25  # polite pause between pages
TIMEOUT_S = 60
MAX_ATTEMPTS = 6
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 60.0
RETRY_STATUSES = {429, 500, 502, 503, 504}

# A year is flagged when its count is below this fraction of its neighbors' mean.
LOW_YEAR_RATIO = 0.5

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
CSV_PATH = OUT_DIR / "oregon_wildlife_collisions.csv"
PARQUET_PATH = OUT_DIR / "oregon_wildlife_collisions.parquet"


def fetch_json(session, params):
    """GET the query endpoint, retrying with exponential backoff + jitter."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        retry_after = None
        try:
            resp = session.get(QUERY_URL, params=params, timeout=TIMEOUT_S)
            if resp.status_code in RETRY_STATUSES:
                retry_after = resp.headers.get("Retry-After")
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
            # ArcGIS reports failures as HTTP 200 with an "error" body.
            if "error" in data:
                err = data["error"]
                raise RuntimeError(f"ArcGIS error {err.get('code')}: {err.get('message')}")
            return data
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Request failed after {MAX_ATTEMPTS} attempts ({params}): {exc}"
                ) from exc
            wait = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2 ** (attempt - 1))
            if retry_after and retry_after.isdigit():
                wait = max(wait, float(retry_after))
            wait += random.uniform(0, 1)
            print(
                f"\n  attempt {attempt}/{MAX_ATTEMPTS} failed: {exc} -- retrying in {wait:.1f}s",
                flush=True,
            )
            time.sleep(wait)


def get_total_count(session):
    data = fetch_json(session, {"where": "1=1", "returnCountOnly": "true", "f": "json"})
    return int(data["count"])


def download_all(session, total):
    rows = []
    offset = 0
    page = 0
    # Guard against a server that never signals the end of the result set.
    max_pages = math.ceil(total / PAGE_SIZE) + 5
    start = time.monotonic()

    while page < max_pages:
        data = fetch_json(
            session,
            {
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "false",
                "orderByFields": "OBJECTID",
                "resultOffset": offset,
                "resultRecordCount": PAGE_SIZE,
                "f": "json",
            },
        )
        features = data.get("features", [])
        if not features:
            break

        rows.extend(f["attributes"] for f in features)
        offset += len(features)
        page += 1

        pct = 100 * len(rows) / total if total else 100
        print(
            f"\r  page {page:>3} | {len(rows):>7,} / {total:,} records ({pct:5.1f}%) "
            f"| {time.monotonic() - start:5.0f}s",
            end="",
            flush=True,
        )

        # exceededTransferLimit=true means more records remain. The server omits
        # the flag on the final page, so a short page without it is the end; a
        # full page without it costs one extra (empty) request to confirm.
        if not data.get("exceededTransferLimit", False) and len(features) < PAGE_SIZE:
            break
        time.sleep(PAGE_DELAY_S)

    print()
    return rows


def build_dataframe(rows):
    df = pd.DataFrame(rows)
    # ArcGIS dates are epoch milliseconds in UTC; DATE_TXT is kept for cross-checking.
    df["DATE"] = pd.to_datetime(df["DATE"], unit="ms")
    return df


def flag_low_years(per_year):
    """Return {year: reason} for years far below their neighbors or missing entirely."""
    flags = {}
    years = per_year.index.tolist()
    for year in range(min(years), max(years) + 1):
        if year not in per_year.index:
            flags[year] = "no records at all"
            continue
        neighbors = [per_year[y] for y in (year - 1, year + 1) if y in per_year.index]
        if not neighbors:
            continue
        baseline = sum(neighbors) / len(neighbors)
        if per_year[year] < LOW_YEAR_RATIO * baseline:
            flags[year] = f"{per_year[year] / baseline:.0%} of neighboring-year mean ({baseline:,.0f})"
    return flags


def print_section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def print_summary(df, expected_total):
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print_section("Row counts")
    print(f"Total rows downloaded : {len(df):,}")
    print(f"Server-reported count : {expected_total:,}")
    if len(df) != expected_total:
        print(f"!! MISMATCH: {len(df) - expected_total:+,} rows vs. server count")
    dupes = int(df["OBJECTID"].duplicated().sum())
    print(f"Duplicate OBJECTIDs   : {dupes:,}" + ("  !! paging overlap" if dupes else ""))

    # YEAR is served as a string field.
    year = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    print_section("Year coverage")
    print(f"Min year: {year.min()}   Max year: {year.max()}   (null/unparseable YEAR: {int(year.isna().sum()):,})")
    print(f"DATE range: {df['DATE'].min():%Y-%m-%d} to {df['DATE'].max():%Y-%m-%d}")

    per_year = year.dropna().astype(int).value_counts().sort_index()
    flags = flag_low_years(per_year)
    date_span = df.groupby(year)["DATE"].agg(["min", "max"])

    print_section("Records per year")
    print(f"{'year':<6}{'count':>8}  {'first date':<12}{'last date':<12}flag")
    for y in range(per_year.index.min(), per_year.index.max() + 1):
        if y not in per_year.index:
            print(f"{y:<6}{0:>8}  {'-':<12}{'-':<12}<-- {flags[y]}")
            continue
        first, last = date_span.loc[y, "min"], date_span.loc[y, "max"]
        note = f"<-- LOW: {flags[y]}" if y in flags else ""
        print(f"{y:<6}{per_year[y]:>8,}  {first:%Y-%m-%d}  {last:%Y-%m-%d}  {note}")

    print_section("Coverage check (service description claims data ends in 2019)")
    max_year = int(per_year.index.max())
    if max_year > 2019:
        post = int(per_year[per_year.index > 2019].sum())
        print(f"Records exist beyond 2019: {post:,} rows in 2020-{max_year}. The description is stale.")
    else:
        print(f"Latest year is {max_year}; consistent with the service description.")
    if flags:
        print("Years flagged as possibly incomplete:")
        for y, reason in flags.items():
            print(f"  {y}: {reason}")
    else:
        print("No year is dramatically lower than its neighbors.")

    print_section("Cross-checks")
    txt_date = pd.to_datetime(df["DATE_TXT"], format="%m/%d/%Y", errors="coerce")
    both = df["DATE"].notna() & txt_date.notna()
    print(f"DATE vs DATE_TXT mismatches : {int((df['DATE'].dt.normalize()[both] != txt_date[both]).sum()):,}")
    has_year = df["DATE"].notna() & year.notna()
    print(f"DATE year vs YEAR mismatches: {int((df['DATE'].dt.year[has_year] != year[has_year]).sum()):,}")
    print(f"Null DATE                   : {int(df['DATE'].isna().sum()):,}")

    print_section(f"ANIMAL: {df['ANIMAL'].nunique()} unique values (top 20)")
    print(df["ANIMAL"].value_counts(dropna=False).head(20).to_string())

    print_section("Records per SEASON")
    print(df["SEASON"].value_counts(dropna=False).to_string())

    print_section("Null counts")
    for col in ["HWYNUMB", "MP", "LAT", "LONGTD"]:
        print(f"{col:<8}: {int(df[col].isna().sum()):,}")


def main():
    with requests.Session() as session:
        session.headers["User-Agent"] = "odot-wildlife-collisions-downloader/1.0 (python-requests)"

        print(f"Endpoint: {QUERY_URL}")
        total = get_total_count(session)
        print(f"Total records reported by service: {total:,}")

        print(f"Downloading in pages of {PAGE_SIZE}...")
        rows = download_all(session, total)

    if not rows:
        sys.exit("No records downloaded.")

    df = build_dataframe(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    df.to_parquet(PARQUET_PATH, index=False)
    print(f"Wrote {CSV_PATH} ({CSV_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"Wrote {PARQUET_PATH} ({PARQUET_PATH.stat().st_size / 1e6:.1f} MB)")

    print_summary(df, total)


if __name__ == "__main__":
    main()
