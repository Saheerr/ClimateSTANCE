# scripts/make_labeling_sample_more_anti.py
# Generates a new labeling sample enriched for ANTI by using TF-IDF similarity to your labeled ANTI texts.
# Assumes your labeled CSV is UTF-8 (you already exported it as UTF-8).

import argparse
import os
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def norm_text(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True, help="Full candidate windows CSV (filtered + window-level climate prob)")
    ap.add_argument("--labeled", required=True, help="Your labeled CSV (UTF-8)")
    ap.add_argument("--out", required=True, help="Output CSV for additional labeling")
    ap.add_argument("--sample_size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_per_repmonth", type=int, default=3)
    ap.add_argument("--max_per_person", type=int, default=12)
    ap.add_argument("--anti_share", type=float, default=0.85)
    args = ap.parse_args()

    win = pd.read_csv(args.windows)
    lab = pd.read_csv(args.labeled)

    # Required columns
    for c in ["GovtrackID", "BioID", "Year", "Month", "window_text"]:
        if c not in win.columns:
            raise ValueError(f"windows file missing column: {c}")
    for c in ["window_text", "stance_label"]:
        if c not in lab.columns:
            raise ValueError(f"labeled file missing column: {c}")

    # Normalize
    win["window_text"] = win["window_text"].astype(str).map(norm_text)
    lab["window_text"] = lab["window_text"].astype(str).map(norm_text)
    lab["stance_label"] = lab["stance_label"].astype(str).str.strip().str.upper()

    # Unique ANTI and PRO seed texts
    anti_seed = (
        lab[lab["stance_label"] == "ANTI"]["window_text"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    pro_seed = (
        lab[lab["stance_label"] == "PRO"]["window_text"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    print(f"ANTI seed texts (unique): {len(anti_seed)}")
    print(f"PRO seed texts (unique):  {len(pro_seed)}")

    if len(anti_seed) < 1:
        raise ValueError("No ANTI examples found in labeled file after normalization.")

    # Exclude texts you've already labeled
    labeled_texts = set(lab["window_text"].dropna().tolist())
    cand = win[~win["window_text"].isin(labeled_texts)].copy()

    # Deduplicate candidates by exact text
    cand = cand.drop_duplicates(subset=["window_text"], keep="first").copy()

    # Cap per rep-month and per person for diversity
    cand["repmonth"] = (
        cand["GovtrackID"].astype(str) + "_" +
        cand["Year"].astype(str) + "_" +
        cand["Month"].astype(str)
    )
    cand = cand.sample(frac=1.0, random_state=args.seed).copy()
    cand = cand.groupby("repmonth", group_keys=False).head(args.max_per_repmonth).copy()
    cand = cand.groupby("GovtrackID", group_keys=False).head(args.max_per_person).copy()

    # TF-IDF over candidates + seeds
    texts_all = cand["window_text"].tolist() + anti_seed + pro_seed
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.95)
    X = vec.fit_transform(texts_all)

    n_c = len(cand)
    n_a = len(anti_seed)
    n_p = len(pro_seed)

    X_c = X[:n_c]
    X_a = X[n_c:n_c + n_a]
    X_p = X[n_c + n_a:n_c + n_a + n_p] if n_p > 0 else None

    # Similarity to ANTI (IMPORTANT: convert centroid to ndarray to avoid np.matrix errors)
    anti_centroid = np.asarray(X_a.mean(axis=0))
    cand["sim_anti"] = cosine_similarity(X_c, anti_centroid).ravel()

    # Similarity to PRO (avoid pro-like text) if enough PRO seeds exist
    if X_p is not None and n_p >= 3:
        pro_centroid = np.asarray(X_p.mean(axis=0))
        cand["sim_pro"] = cosine_similarity(X_c, pro_centroid).ravel()
        cand["anti_priority"] = cand["sim_anti"] - 0.5 * cand["sim_pro"]
    else:
        cand["sim_pro"] = 0.0
        cand["anti_priority"] = cand["sim_anti"]

    cand_sorted = cand.sort_values("anti_priority", ascending=False).copy()

    n_anti = int(round(args.sample_size * args.anti_share))
    n_rest = args.sample_size - n_anti

    anti_part = cand_sorted.head(min(n_anti, len(cand_sorted))).copy()
    rem = cand_sorted.iloc[len(anti_part):].copy()

    # Remaining rows: random sample for diversity
    if len(rem) > 0 and n_rest > 0:
        rest_part = rem.sample(n=min(n_rest, len(rem)), random_state=args.seed).copy()
    else:
        rest_part = pd.DataFrame(columns=cand.columns)

    out_df = pd.concat([anti_part, rest_part], ignore_index=True)

    # Top up if short
    if len(out_df) < args.sample_size:
        need = args.sample_size - len(out_df)
        extra = cand_sorted.iloc[len(out_df):len(out_df) + need].copy()
        out_df = pd.concat([out_df, extra], ignore_index=True)

    # Trim if long
    if len(out_df) > args.sample_size:
        out_df = out_df.head(args.sample_size).copy()

    out_df["stance_label"] = ""
    out_df["notes"] = ""

    keep = [
        "GovtrackID", "BioID", "Year", "Month",
        "chunk_id", "window_id",
        "window_text",
        "sim_anti", "sim_pro", "anti_priority",
        "stance_label", "notes"
    ]
    for c in keep:
        if c not in out_df.columns:
            out_df[c] = ""

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df[keep].to_csv(args.out, index=False)

    print(f"Wrote {len(out_df)} rows to {args.out}")
    print(out_df["anti_priority"].describe().to_string())


if __name__ == "__main__":
    main()
