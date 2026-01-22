import argparse
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

LABELS = ["ANTI", "NEUTRAL", "PRO"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--text_col", default="window_text")
    ap.add_argument("--label_col", default="stance_label")
    ap.add_argument("--group_col", default="GovtrackID")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.train_csv, encoding="utf-8")

    # Only keep the 3 stance labels for training
    df = df[df[args.label_col].isin(LABELS)].copy()

    X_text = df[args.text_col].astype(str).tolist()
    y = df[args.label_col].astype(str).tolist()

    # Group split by politician to reduce leakage
    if args.group_col in df.columns:
        groups = df[args.group_col].astype(str).tolist()
    else:
        groups = list(range(len(df)))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, val_idx = next(gss.split(X_text, y, groups=groups))

    X_train = [X_text[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val = [X_text[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    Xtr = vec.fit_transform(X_train)
    Xva = vec.transform(X_val)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )
    clf.fit(Xtr, y_train)

    pred = clf.predict(Xva)
    report = classification_report(y_val, pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_val, pred, labels=LABELS)

    joblib.dump(vec, os.path.join(args.outdir, "tfidf_vectorizer.joblib"))
    joblib.dump(clf, os.path.join(args.outdir, "logreg_model.joblib"))

    with open(os.path.join(args.outdir, "eval_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n\nConfusion matrix (rows=true, cols=pred) label order: ANTI, NEUTRAL, PRO\n")
        f.write(np.array2string(cm))

    print("Saved model to:", args.outdir)
    print("Wrote eval_report.txt")

if __name__ == "__main__":
    main()
