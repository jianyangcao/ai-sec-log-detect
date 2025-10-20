"""
basic_python.py
Day3 refresh: variables, lists/dicts, loops, functions, file I/O, plotting.
Read a CSV of (timestamp, event), computes frequecy per event, saves summary,
and plots a bar chart to reports/figrues/day3_event_frequency.png

Run:
    python scripts/basic_python.py --in_csv data/day3_sample.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

#--------------1) Pure-Python helpers (no pandas)----------------
def read_event(csv_path):
    """
    Read a CSV with headers 'timestamp,event' and return a list of events(strings).
    Uses python's csv library so you practice file I/O directly.
    """
    events=[]
    with open(csv_path,"r",newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        for row in reader:
            events.append(row['event'])
    return events

def count_frequency(items):
    '''
    Count occurrences of strings in 'items' using a dict.
    '''
    freq=defaultdict(int)
    for x in items: #loop
        freq[x]+=1
    return dict(freq)

def write_summary(summary_dict, out_csv):
    '''
    write a 2 column CSV: event count
    '''
    out_csv.parent.mkdir(parents=True,exist_ok=True)
    with open(out_csv,'w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['event','count'])
        for k, v in sorted(summary_dict.items()):
            writer.writerow([k,v])

def plot_bar(summary_dict, out_png, title='Envent Frequency(Day 3)'):
    '''
    Save a simple bar cahrt with matplotlib(no seaborn).
    '''
    out_png.parent.mkdir(parents=True,exist_ok=True)
    labels=list(summary_dict.keys())
    values=[summary_dict[k] for k in labels]

    plt.figure() #single,clean figure
    plt.bar(labels,values)
    plt.title(title)
    plt.ylabel('Count')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

#--------------2) CLI entry point(argparse)----------------
def parse_args():
    parser=argparse.ArgumentParser(description='Day 3 Python refresh')
    parser.add_argument(
        '--in_csv',
        type=str,
        required=True,
        help='Path to input CSV with columns timestamp,event'
    )
    parser.add_argument(
        '--out_dir_figures',
        type=str,
        default='reports/figures',
        help='Dirrectory to write outputs(PNG)'
    )  
    parser.add_argument(
        '--out_dir_csv',
        type=str,
        default='reports',
        help='Dirrectory to write outputs(CVS)'
    )
    return parser.parse_args()


def main():
    args=parse_args()
    in_path=Path(args.in_csv)
    out_dir_figures=Path(args.out_dir_figures)
    out_dir_csv=Path(args.out_dir_csv)

    #1) read events from CSV
    events=read_event(in_path)

    #2) count state
    summary=count_frequency(events)

    #3) write summary CSV
    out_csv=out_dir_csv/'day3_event_summary.csv'
    write_summary(summary,out_csv)

    #4) plot bar chart
    out_png=out_dir_figures/'day3_event_frequency.png'
    plot_bar(summary,out_png)

    print('Doen.')
    print(f'Read from: {in_path}')
    print(f'Wrote csv to: {out_csv}')
    print(f'Wrote png to: {out_png}')

if __name__=='__main__':
    main()