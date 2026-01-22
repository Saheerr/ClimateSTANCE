import argparse
import os
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans

def clean_text(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample_size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_per_repmonth", type=int, default=5)
    ap.add_argument("--n_clusters", type=int, default=60)
    args = ap.parse_args()

    df = pd.read_csv(args.windows)

    required = {"GovtrackID","BioID","Year","Month","window_text","climate_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["window_text"] = df["window_text"].astype(str).map(clean_text)

    df["repmonth"] = (
        df["GovtrackID"].astype(str) + "_" +
        df["Year"].astype(str) + "_" +
        df["Month"].astype(str)
    )

    # Cap per rep-month to avoid domination
    df = df.sort_values("climate_prob", ascending=False).copy()
    df = df.groupby("repmonth", group_keys=False).head(args.max_per_repmonth).copy()

    texts = df["window_text"].tolist()

    vec = TfidfVectorizer(
        min_df=3,
        max_df=0.90,
        ngram_range=(1, 2),
        stop_words="english"
    )
    X = vec.fit_transform(texts)

    svd = TruncatedSVD(n_components=100, random_state=args.seed)
    Xr = svd.fit_transform(X)

    n_clusters = min(args.n_clusters, max(10, len(df) // 20))
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=args.seed, batch_size=2048)
    df["cluster_id"] = km.fit_predict(Xr)

    rng = np.random.default_rng(args.seed)

    years = sorted(df["Year"].unique().tolist())
    clusters = sorted(df["cluster_id"].unique().tolist())

    # Stratify primarily by year, then by cluster inside year
    per_year = max(1, args.sample_size // max(1, len(years)))
    picked = []

    for y in years:
        dyy = df[df["Year"] == y]
        if len(dyy) == 0:
            continue

        per_cluster = max(1, per_year // max(1, len(clusters)))

        for c in clusters:
            dcc = dyy[dyy["cluster_id"] == c]
            if len(dcc) == 0:
                continue
            take = min(per_cluster, len(dcc))
            idx = rng.choice(dcc.index.to_numpy(), size=take, replace=False)
            picked.append(df.loc[idx])

    out_df = pd.concat(picked, ignore_index=True) if picked else df.sample(
        n=min(args.sample_size, len(df)), random_state=args.seed
    )

    if len(out_df) > args.sample_size:
        out_df = out_df.sample(n=args.sample_size, random_state=args.seed)

    out_df = out_df.copy()
    out_df["stance_label"] = ""
    out_df["notes"] = ""

    keep = [
        "GovtrackID","BioID","Year","Month",
        "chunk_id","window_id","climate_prob",
        "cluster_id","window_text","stance_label","notes"
    ]
    for c in keep:
        if c not in out_df.columns:
            out_df[c] = ""

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df[keep].to_csv(args.out, index=False)

    print(f"Wrote {len(out_df)} rows to {args.out}")
    print(f"Clusters used: {out_df['cluster_id'].nunique()}")

if __name__ == "__main__":
    main()
