#!/usr/bin/env python3
"""
Day 11 — Feature Engineering for log data.

Input : data/cleaned.csv   (expected columns: timestamp,event,user,ip)
Output: data/feature_engineered.csv

Run:
  python scripts/feature_demo.py \
      --in_csv data/cleaned.csv \
      --out_csv data/feature_engineered.csv
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

# --------- IPv4 validation regex (0-255 per octet) ----------
IPV4_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)$"
)

def is_ipv4(x: str) -> bool:
    if not isinstance(x, str): return False
    x = x.strip()
    return bool(IPV4_RE.match(x))

def add_basic_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # robust parse
    df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["date"]     = df["timestamp_parsed"].dt.date
    df["hour"]     = df["timestamp_parsed"].dt.hour
    df["minute"]   = df["timestamp_parsed"].dt.minute
    df["weekday"]  = df["timestamp_parsed"].dt.weekday  # Mon=0
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    return df

def add_ip_quality_and_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ip_clean"] = df["ip"].astype(str).str.strip()
    df["is_valid_ipv4"] = df["ip_clean"].apply(is_ipv4).astype(int)

    # Per-IP aggregates
    df["count_per_ip"] = (
        df.groupby("ip_clean")["ip_clean"].transform("size").astype(int)
    )
    if "user" in df.columns:
        df["unique_users_per_ip"] = (
            df.groupby("ip_clean")["user"].transform("nunique").astype(int)
        )
    else:
        df["unique_users_per_ip"] = 1
    return df

def add_ip_hour_density(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "hour" not in df.columns:
        raise ValueError("Hour missing; call add_basic_time_parts first.")
    # Count events per (ip, hour)
    grp = df.groupby(["ip_clean", "hour"])["hour"].transform("size").astype(int)
    df["events_per_ip_hour"] = grp
    return df

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per IP:
      - seconds since previous/next event
      - rolling counts in past/next 10 minutes (excluding current row)
    """
    df = df.copy()
    if "timestamp_parsed" not in df.columns:
        raise ValueError("timestamp_parsed missing; add_basic_time_parts first.")

    # Sort so diffs make sense
    df = df.sort_values(["ip_clean", "timestamp_parsed"], kind="mergesort")

    # Deltas (seconds)
    df["secs_since_prev_event"] = (
        df.groupby("ip_clean")["timestamp_parsed"]
          .diff()
          .dt.total_seconds()
    )
    df["secs_to_next_event"] = (
        df.groupby("ip_clean")["timestamp_parsed"]
          .diff(-1)
          .abs()
          .dt.total_seconds()
    )

    # Rolling in 10 minutes (per IP)
    def _rolling_counts(g: pd.DataFrame) -> pd.DataFrame:
        g = g.set_index("timestamp_parsed")
        ones = pd.Series(1, index=g.index)

        # Past 10 minutes including current, then exclude current
        prev = ones.rolling("10min", closed="both").sum() - 1
        # Next 10 minutes: reverse time, do the same trick, then reverse back
        rev = ones.sort_index(ascending=False)
        next_ = rev.rolling("10min", closed="both").sum() - 1
        next_ = next_.sort_index()

        g["events_prev_10min"] = prev.clip(lower=0).astype(int)
        g["events_next_10min"] = next_.clip(lower=0).astype(int)
        return g.reset_index()

    df = (
        df.groupby("ip_clean", group_keys=False)
          .apply(_rolling_counts)
          .sort_values(["ip_clean","timestamp_parsed"], kind="mergesort")
          .reset_index(drop=True)
    )
    return df

def add_peak_hour_flag(df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    df = df.copy()
    hour_counts = df["hour"].value_counts().sort_values(ascending=False)
    peak_hours = set(hour_counts.head(top_k).index.tolist())
    df["is_peak_hour"] = df["hour"].isin(peak_hours).astype(int)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    needed = {"timestamp","event","user","ip"}
    if not needed.issubset(df.columns):
        missing = needed - set(df.columns)
        raise ValueError(f"Missing columns in input: {sorted(missing)}")

    df = add_basic_time_parts(df)
    df = add_ip_quality_and_aggregates(df)
    df = add_ip_hour_density(df)
    df = add_temporal_features(df)
    df = add_peak_hour_flag(df)

    # Housekeeping: replace NaNs in numeric cols with 0
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)

    df.to_csv(out_path, index=False)
    print(f"✅ Saved feature file -> {out_path.resolve()}")

if __name__ == "__main__":
    main()
