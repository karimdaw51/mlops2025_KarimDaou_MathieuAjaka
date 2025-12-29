from __future__ import annotations

import pandas as pd


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates()

    if "trip_duration" in df.columns:
        td = pd.to_numeric(df["trip_duration"], errors="coerce")
        df = df[td > 0].copy()

    if "passenger_count" in df.columns:
        pc = pd.to_numeric(df["passenger_count"], errors="coerce")
        df = df[(pc.isna()) | ((pc >= 1) & (pc <= 8))].copy()

    for col in ["pickup_datetime", "dropoff_datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        else:
            if df[col].isna().any():
                df[col] = df[col].fillna("UNKNOWN")

    return df

