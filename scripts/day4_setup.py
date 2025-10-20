'''
Day 4: Verify scientific Python stack and produce simple outputs.

Outputs:
- prints library versions
- creates a small DataFrame and summary
- saves summary to reports/day4_summary.csv
- saves a histogram to reports/figures/day4_hist.png
- fits a tiny sklearn model and prints accurac
'''

from pathlib import Path #Pathlib: handles file paths in a platform-independent way.

import numpy as np #NumPy (np): generates numerical data (random numbers).
import pandas as pd #Pandas (pd): creates and analyzes DataFrames.
import matplotlib.pyplot as plt #Matplotlib (plt): for plotting and saving figures.
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #Scikit-learn (sklearn): builds and tests a simple ML model.

def ensure_dirs(): #make sure output directories exist
    Path('reports').mkdir(exist_ok=True)
    Path('reports/figures').mkdir(parents=True, exist_ok=True)

def print_versions(): #print library versions
    print('Versions:')
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Matplotlib version: {plt.matplotlib.__version__}")
    import sklearn #scikit-learn version lives in the module
    print(f"Scikit-learn version: {sklearn.__version__}")

def make_dataframe(n=100, seed=0): #create a tiny synthetic dataset for practice
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        'value': rng.normal(loc=0.0, scale=1.0, size=n),
        'category': rng.choice(['A', 'B', 'C'], size=n, replace=True)
    })
    return df

def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame: #return a compct summary table
    #per-column baisic summary
    basic= df.describe(include='all')
    #group-by example: mean value per category
    group_mean = df.groupby('category',observed=True)['value'].mean().rename('mean_value')
    summary = pd.concat(
        {'basic_describe': basic, 'mean_per_category': group_mean}, 
        axis=1
    )
    return summary

def save_histogram(df: pd.DataFrame, out_path='reports/figures/day4_hist.png'):
    #plot and save a histogram oof the 'value' column
    plt.figure()
    df['value'].hist(bins=20) #simple histogram
    plt.title('Distribution of Values')
    plt.xlabel('value')
    plt.ylabel('count')
    plt.tight_layout
    plt.savefig(out_path,dpi=150)
    plt.close()
    print(f'[saved] {out_path}')

def tiny_model_demo(seed=0):
    #train a tiny logistic regression model just ot verify sklearn
    X, y =make_classification(
    n_samples=400, n_features=5, n_informative=3, n_redundant=0,
    n_clusters_per_class=1, random_state=seed
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )
    clf=LogisticRegression(max_iter=1000) #higher max_iter to avoid convergence
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    acc= accuracy_score(y_test, y_pred)
    print(f'Logistic Regression accuracy: {acc:.3f}')

def main():
    ensure_dirs()
    print_versions()

    df=make_dataframe(n=120, seed=42)
    print('\nHead of DataFrame:')
    print(df.head())

    summary=summarize_dataframe(df)
    out_csv=Path('reports/day4_summary.csv')
    summary.to_csv(out_csv)
    print(f'\n[saved] {out_csv}')

    save_histogram(df)
    tiny_model_demo(seed=42)

if __name__ == '__main__':
    main()