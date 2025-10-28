#!/usr/bin/env python3
"""
train_logreg.py — Day 12: Logistic Regression baseline

Input :
  data/feature_engineered.csv   (from Day 11)

Output:
  models/baseline_logreg.pkl
  reports/metrics/logreg_metrics.json
  reports/figures/logreg_confusion.png
  reports/figures/logreg_roc.png
  reports/figures/logreg_pr.png

Usage:
  python scripts/train_logreg.py \
      --in_csv data/feature_engineered.csv \
      --label is_attack
  # If no true label exists, omit --label and it will build a proxy
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay
)

NUMERIC_CANDIDATES = [
    # common features created on Day 11 (safe to ignore missing ones)
    "count_per_ip", "unique_users_per_ip", "events_per_ip_hour",
    "is_valid_ipv4", "hour", "weekday", "is_weekend",
    "secs_since_prev_event", "secs_to_next_event",
    "events_prev_10min", "events_next_10min",
]

def pick_existing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

def ensure_label(df: pd.DataFrame, label_col: str | None) -> tuple[pd.DataFrame, str]:
    """
    If label_col provided and exists -> use it.
    Else, create a proxy 'label_proxy' where event in {error, admin} -> 1 else 0.
    """
    if label_col and label_col in df.columns:
        return df, label_col

    if "event" not in df.columns:
        raise ValueError("No label column and no 'event' column to build a proxy label from.")

    positives = {"error", "admin"}
    y_proxy = df["event"].astype(str).str.lower().isin(positives).astype(int)
    df = df.copy()
    df["label_proxy"] = y_proxy
    return df, "label_proxy"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, required=True)
    ap.add_argument("--label", type=str, default=None, help="Name of the true label column if available")
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    # --- Load ---
    df = pd.read_csv(in_path)

    # --- Target ---
    df, label_col = ensure_label(df, args.label)
    y = df[label_col].astype(int).values

    # --- Features ---
    feat_cols = pick_existing(df, NUMERIC_CANDIDATES)
    if not feat_cols:
        raise ValueError("No numeric Day-11 features found. Re-run Day 11 or check column names.")
    X = df[feat_cols].fillna(0.0).values

    # --- Train/Val split (stratified) ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # --- Pipeline: scale + logistic regression ---
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",   # helpful for imbalanced logs
            solver="liblinear"         # stable on small feature sets
        ))
    ])

    # --- Cross-validated predictions on train for sanity (AUC, etc.) ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    y_tr_prob = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]

    # Fit on full train
    pipe.fit(X_tr, y_tr)

    # --- Evaluate on test ---
    y_prob = pipe.predict_proba(X_te)[:, 1]
    y_hat  = (y_prob >= 0.5).astype(int)

    metrics = {
        "features_used": feat_cols,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "class_balance_test": {
            "negatives": int((y_te == 0).sum()),
            "positives": int((y_te == 1).sum()),
        },
        "roc_auc_test": float(roc_auc_score(y_te, y_prob)),
        "pr_auc_test": float(average_precision_score(y_te, y_prob)),
        "classification_report_test": classification_report(y_te, y_hat, output_dict=True),
        "confusion_matrix_test": confusion_matrix(y_te, y_hat).tolist(),
        "cv_roc_auc_train": float(roc_auc_score(y_tr, y_tr_prob)),
        "cv_pr_auc_train": float(average_precision_score(y_tr, y_tr_prob)),
    }

    # --- Save model & metrics ---
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports/metrics").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, "models/baseline_logreg.pkl")
    with open("reports/metrics/logreg_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # --- Plots ---
    fig1, ax1 = plt.subplots()
    RocCurveDisplay.from_predictions(y_te, y_prob, ax=ax1)
    ax1.set_title("LogReg ROC (test)")
    fig1.tight_layout()
    fig1.savefig("reports/figures/logreg_roc.png", dpi=150)

    fig2, ax2 = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_te, y_prob, ax=ax2)
    ax2.set_title("LogReg Precision-Recall (test)")
    fig2.tight_layout()
    fig2.savefig("reports/figures/logreg_pr.png", dpi=150)

    fig3, ax3 = plt.subplots()
    cm = confusion_matrix(y_te, y_hat)
    im = ax3.imshow(cm, interpolation="nearest")
    ax3.set_title("Confusion Matrix (test)")
    ax3.set_xlabel("Predicted"); ax3.set_ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        ax3.text(j, i, str(v), ha="center", va="center")
    fig3.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    fig3.tight_layout()
    fig3.savefig("reports/figures/logreg_confusion.png", dpi=150)

    print("✅ Trained and saved baseline logistic regression.")
    print("   Model:   models/baseline_logreg.pkl")
    print("   Metrics: reports/metrics/logreg_metrics.json")
    print("   Figures: reports/figures/logreg_*.png")

if __name__ == "__main__":
    main()
