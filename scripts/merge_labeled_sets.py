import argparse
import os
import re
import pandas as pd

VALID = {"PRO", "NEUTRAL", "ANTI", "UNKNOWN"}

def norm_label(x) -> str:
    if pd.isna(x):
        return ""
    t = str(x).strip().upper()
    t = re.sub(r"\s+", "_", t)
    t = t.replace("-", "_")
    return t

def norm_text(x) -> str:
    t = "" if pd.isna(x) else str(x)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def read_utf8(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    dfs = []
    for p in args.inputs:
        df = read_utf8(p)
        if "window_text" not in df.columns or "stance_label" not in df.columns:
            raise ValueError(f"{p} missing window_text or stance_label columns")
        df["window_text"] = df["window_text"].map(norm_text)
        df["stance_label"] = df["stance_label"].map(norm_label)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # Keep only valid labels and non-empty text
    all_df = all_df[all_df["window_text"].astype(str).str.len() > 0].copy()
    all_df = all_df[all_df["stance_label"].isin(VALID)].copy()

    # Deduplicate exact duplicates (same text and same label)
    before = len(all_df)
    all_df = all_df.drop_duplicates(subset=["window_text", "stance_label"], keep="first")
    after = len(all_df)

    # Training set excludes UNKNOWN
    train_df = all_df[all_df["stance_label"].isin({"PRO", "NEUTRAL", "ANTI"})].copy()

    out_merged = os.path.join(args.outdir, "labeled_merged_dedup.csv")
    out_train  = os.path.join(args.outdir, "stance_train.csv")
    out_report = os.path.join(args.outdir, "label_counts.txt")

    all_df.to_csv(out_merged, index=False, encoding="utf-8")
    train_df.to_csv(out_train, index=False, encoding="utf-8")

    with open(out_report, "w", encoding="utf-8") as f:
        f.write("Merged labeled rows (after exact dedup):\n")
        f.write(f"{after}\n")
        f.write(f"Duplicates removed: {before - after}\n\n")
        f.write("Counts (merged, includes UNKNOWN):\n")
        f.write(all_df["stance_label"].value_counts().to_string())
        f.write("\n\nCounts (training set, excludes UNKNOWN):\n")
        f.write(train_df["stance_label"].value_counts().to_string())
        f.write("\n\nUnique ANTI texts in training:\n")
        f.write(str(train_df[train_df["stance_label"]=="ANTI"]["window_text"].nunique()))
        f.write("\n")

    print("Wrote:")
    print(out_merged)
    print(out_train)
    print(out_report)

if __name__ == "__main__":
    main()
