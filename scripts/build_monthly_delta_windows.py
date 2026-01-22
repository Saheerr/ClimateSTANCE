import argparse
import os
import re
import hashlib
import pandas as pd


def norm_text(t: str) -> str:
    t = "" if pd.isna(t) else str(t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def ym_int(year, month) -> int:
    return int(year) * 100 + int(month)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="outputs_clean/evidence_windows_with_stance_probs.csv")
    ap.add_argument("--out_windows", required=True, help="outputs_clean/delta_windows.csv")
    ap.add_argument("--out_monthly", required=True, help="outputs_clean/rep_monthly_stance_delta.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.infile, encoding="utf-8")

    need = ["GovtrackID", "BioID", "Year", "Month", "window_text", "p_pro", "p_neutral", "p_anti"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in infile: {missing}")

    df["norm"] = df["window_text"].map(norm_text)
    df["text_hash"] = df["norm"].map(sha1_hex)
    df["ym"] = df.apply(lambda r: ym_int(r["Year"], r["Month"]), axis=1)

    df = df.sort_values(["GovtrackID", "BioID", "ym"]).copy()

    kept = []
    for (gid, bid), g in df.groupby(["GovtrackID", "BioID"], sort=False):
        prev_hashes = set()
        for ym, gm in g.groupby("ym", sort=True):
            cur_hashes = set(gm["text_hash"].tolist())
            new_hashes = cur_hashes - prev_hashes
            if new_hashes:
                dm = gm[gm["text_hash"].isin(new_hashes)].copy()
                kept.append(dm)
            prev_hashes = cur_hashes

    if kept:
        ddf = pd.concat(kept, ignore_index=True)
    else:
        ddf = df.iloc[0:0].copy()

    ddf["window_score"] = ddf["p_pro"] - ddf["p_anti"]

    monthly = ddf.groupby(["GovtrackID", "BioID", "Year", "Month"]).agg(
        stance_score=("window_score", "mean"),
        evidence_count=("window_score", "size"),
        pro_prob_mean=("p_pro", "mean"),
        anti_prob_mean=("p_anti", "mean"),
        neutral_prob_mean=("p_neutral", "mean"),
    ).reset_index()

    os.makedirs(os.path.dirname(args.out_windows), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_monthly), exist_ok=True)

    ddf.to_csv(args.out_windows, index=False, encoding="utf-8")
    monthly.to_csv(args.out_monthly, index=False, encoding="utf-8")

    print("Wrote delta windows:", args.out_windows, "rows:", len(ddf))
    print("Wrote monthly delta stance:", args.out_monthly, "rows:", len(monthly))


if __name__ == "__main__":
    main()
