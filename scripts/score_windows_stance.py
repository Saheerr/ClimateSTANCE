import argparse
import os
import joblib
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--text_col", default="window_text")
    args = ap.parse_args()

    vec_path = os.path.join(args.model_dir, "tfidf_vectorizer.joblib")
    clf_path = os.path.join(args.model_dir, "logreg_model.joblib")

    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"Missing vectorizer: {vec_path}")
    if not os.path.exists(clf_path):
        raise FileNotFoundError(f"Missing model: {clf_path}")

    vec = joblib.load(vec_path)
    clf = joblib.load(clf_path)

    df = pd.read_csv(args.infile, encoding="utf-8")
    if args.text_col not in df.columns:
        raise ValueError(f"Missing column in infile: {args.text_col}")

    texts = df[args.text_col].astype(str).tolist()

    X = vec.transform(texts)
    probs = clf.predict_proba(X)
    classes = list(clf.classes_)

    def get_prob(label: str):
        if label in classes:
            return probs[:, classes.index(label)]
        return 0.0

    df["p_pro"] = get_prob("PRO")
    df["p_neutral"] = get_prob("NEUTRAL")
    df["p_anti"] = get_prob("ANTI")
    df["pred_label"] = clf.predict(X)

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    df.to_csv(args.outfile, index=False, encoding="utf-8")
    print("Wrote:", args.outfile)
    print("Rows:", len(df))

if __name__ == "__main__":
    main()
