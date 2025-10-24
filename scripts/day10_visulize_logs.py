"""
visualize_logs.py
Day 10: Visualization
Generate bar and line plots to show event frequencies from cleaned logs.

Run:
    python scripts/visualize_logs.py --in_csv data/cleaned.csv
Outputs:
    reports/figures/bar.png
    reports/figures/line.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_event_frequency(df: pd.DataFrame, out_dir:Path):
    freq=df['event'].value_counts()
    print(freq)

    plt.figure(figsize=(10,5))
    freq.plot(kind='bar', color='skyblue')
    plt.title('Event Frequency - Bar Plot')
    plt.xlabel('Event Type')
    plt.ylabel('Count')
    plt.tight_layout()
    out_path=out_dir/'day10_bar.png'
    plt.savefig(out_path)
    print(f'saved bar chart to {out_path}')
    plt.close()

def plot_events_overtime(df: pd.DataFrame, out_dir:Path):
    if 'timestamp' not in df.columns:
        print("No 'timestamp' column found in data.")
        return
    
    df['timestamp']=pd.to_datetime(df['timestamp'],errors='coerce')
    df=df.dropna(subset=['timestamp'])

    mintely=df.groupby(df['timestamp'].dt.floor('min'))['event'].count()
    
    plt.figure(figsize=(10,5))
    mintely.plot(kind='line', marker='o')
    plt.title('Events Over Time - Line Plot')
    plt.xlabel('Date')
    plt.ylabel('Number of Events')
    plt.tight_layout()
    out_path=out_dir/'day10_line.png'
    plt.savefig(out_path)
    print(f'saved line chart to {out_path}')
    plt.close()

def main():
    parser=argparse.ArgumentParser(description='Visulaize event frequecies from cleaned logs.')
    parser.add_argument('--in_csv', required=True, help='Path to cleaned CSV file.')
    args=parser.parse_args()

    csv_path=Path(args.in_csv)
    out_dir=Path('reports/figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading cleaned data from {csv_path}')
    df=pd.read_csv(csv_path)
    print(f'loaded {len(df)} rows with columns: {list(df.columns)}')

    plot_event_frequency(df, out_dir)
    plot_events_overtime(df, out_dir)

if __name__=='__main__':
    main()