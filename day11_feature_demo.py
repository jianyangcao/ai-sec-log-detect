'''
feature_demo.py

Day 11 — Feature Engineering for log data.

Input  : data/cleaned.csv   (expected columns: timestamp,event,user,ip)
Output : data/feature_engineered.csv

What it adds:
  - Parsed time parts: date, hour, minute, weekday, is_weekend
  - IP quality flag: is_valid_ipv4
  - Per-IP aggregates: count_per_ip, unique_users_per_ip
  - Per-(ip,hour) traffic: events_per_ip_hour
  - Temporal features per IP:
        secs_since_prev_event, secs_to_next_event,
        events_prev_10min, events_next_10min
  - Global “peak hour” flag (top-3 frequent hours in the day)
'''
from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd

IPV4_RE = re.compile(
    r'^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}'
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$'
)

def is_ipv4I(s: str) -> bool:
    if not isinstance(s,str):
        return False
    return IPV4_RE.match(s.strip()) is not None

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description='Frature engineering for log data.')
    p.add_argument('--in_csv', type=Path, default=Path('reports/day9_cleaned_sample.csv'))
    p.add_argument('--out_csv', type=Path, default=Path('reports/day11_feature_engineered.csv'))
    return p.parse_args()

def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    # Robust timestamp parsing (handles 10/21/25 10:00, ISO, etc.)
    ts=pd.to_datetime(df['timestamp'],errors='coerce', infer_datetime_format=True)
    df=df.assign(
        timestamp_parsed=ts,
        date=ts.dt.date.astype('string'),
        hour=ts.dt.hour,
        minute=ts.dt.minute,
        weekday=ts.dt.dayofweek,  # 0=Mon, 6=Sun
        is_weekend=(ts.dt.dayofweek >=5).astype(int),
    )
    return df

def add_ip_quality_and_fill(df: pd.DataFrame) -> pd.DataFrame: 
    # Flag valid IPv4; anything else becomes NaN for grouping safety
    valid=df['ip'].apply(is_ipv4I)
    df=df.assign(
        is_valid_ipv4=valid.astype(int),
        ip_clean=np.where(valid, df['ip'], np.nan)
    )
    return df

def add_basic_group_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    # Per-IP totals and unique users
    df['count_per_ip']=df.groupby('ip_clean')['event'].transform
    df['unique_users_per_ip']=df.groupby('ip_clean')['user'].transform('nunique')
    # Events per (ip, hour) slice 
    df['events_per_ip_hour']=df.groupby(['ip_clean','hour'])['event'].transform('count')
    return df  

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    # Sort within each IP by time; compute time deltas and rolling windows
    df=df.sort_values(['ip_clean','timestamp_parsed'],kind='mergesort')
    # Deltas to previous/next event (per IP)
    df['secs_since_prev_event']=(
        df.groupby('ip_clean')['timestamp_parsed'].diff().dt.total_seconds()
    )
    df['ses_to_next_event']=(
        df.groupby('ip_clean')['timestamp_parsed'].diff(-1).abs().dt.total_seconds()
    )
    # Rolling counts of events in 10-minute windows (per IP)
    #need a time index per group for time_based rolling
    def _rolling_counts(g: pd.DataFrame) -> pd.DataFrame:
        g=g.set_index('timestamp_parsed')
        #past 10 minutes up to current event
        g['events_prev_10min']=(
            g['event'].rolling('10min').count().astype(int)
        )
        #future 10 minutes from current event (use reverse trick)
        g_rev=g.iloc[::-1].copy()
        g_rev['events_next_10min']=(
            g_rev['event'].rolling('10min').count().astype(int)
        )
        g['events_next 10min']=g_rev.iloc[::-1]['events_next_10min'].values
        return g.reset_index()
    df=(
        df.groupby('ip_clean',group_keys=False).apply(_rolling_counts)
    )

    return df

def add_peak_hours_flag(df: pd.DataFrame) -> pd.DataFrame:
    # Find top-3 frequent hours globally; flag events in those hours
    hour_counts=df['hour'].value_counts(dropna=True)
    top_hours=set(hour_counts.head(3).index.tolist())
    df['is_peak_hour']=df['hour'].isin(top_hours).astype(int)
    return df

def main():
    args = parse_args()

    if not args.in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.in_csv}")

    df = pd.read_csv(args.in_csv)

    required_cols = {"timestamp", "event", "user", "ip"}
    missing = required_cols - set(map(str, df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- Build features ---
    df = add_time_parts(df)
    df = add_ip_quality_and_fill(df)
    df = add_basic_group_aggregates(df)
    df = add_temporal_features(df)
    df = add_peak_hours_flag(df)

    # Order columns for readability
    front_cols = [
        "timestamp", "event", "user", "ip",
        "timestamp_parsed", "date", "hour", "minute", "weekday", "is_weekend",
        "is_valid_ipv4", "ip_clean",
        "count_per_ip", "unique_users_per_ip", "events_per_ip_hour",
        "secs_since_prev_event", "secs_to_next_event",
        "events_prev_10min", "events_next_10min",
        "is_peak_hour",
    ]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print("✅ Feature engineering complete.")
    print(f"   Input : {args.in_csv}")
    print(f"   Output: {args.out_csv}")
    print("   Preview:")
    with pd.option_context("display.max_columns", 0, "display.width", 120):
        print(df.head(8))


if __name__ == "__main__":
    main()
