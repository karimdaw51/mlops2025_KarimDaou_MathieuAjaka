from __future__ import annotations

import numpy as np
import pandas as pd


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # time features
    if "pickup_datetime" in df.columns:
        dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        df["pickup_hour"] = dt.dt.hour.fillna(0).astype(int)
        df["pickup_dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
        df["pickup_month"] = dt.dt.month.fillna(0).astype(int)
        df["is_weekend"] = (df["pickup_dayofweek"] >= 5).astype(int)
        df["is_rush_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    # distance feature
    req = {"pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"}
    if req.issubset(df.columns):
        dist = _haversine_km(
            df["pickup_latitude"].to_numpy(),
            df["pickup_longitude"].to_numpy(),
            df["dropoff_latitude"].to_numpy(),
            df["dropoff_longitude"].to_numpy(),
        )
        df["distance_km"] = np.clip(dist, 0, 200)

    # drop id + leakage columns if present
    for col in ["id", "dropoff_datetime"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df

