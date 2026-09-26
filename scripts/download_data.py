"""Download the five ODOT layers to data/raw/ and log them in data/raw/manifest.json.

Usage: uv run python scripts/download_data.py
"""

import pandas as pd

from capstone.data import LAYERS, RAW, download_layer, record_download

RAW.mkdir(parents=True, exist_ok=True)
for name, url in LAYERS.items():
    print(f"{name}: {url}")
    # Only the highway network needs geometry; traffic layers join by milepost.
    df, server_count = download_layer(url, geometry=(name == "odot_state_highways"))
    if name == "odot_wildlife_collisions":
        df["DATE"] = pd.to_datetime(df["DATE"], unit="ms")  # epoch milliseconds
        print(df["YEAR"].astype(int).value_counts().sort_index().to_string())
    path = RAW / f"{name}.parquet"
    df.to_parquet(path, index=False)
    record_download(name, path, len(df), server_count)
    print(f"  {len(df):,} rows (server reports {server_count:,})")
