import argparse
import os
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--anti_threshold", type=float, default=0.30)  # lower threshold helps recall
    ap.add_argument("--min_evidence", type=int, default=1)
    args = ap.parse_args()

    df = pd.read_csv(args.infile, encoding="utf-8")

    needed = ["GovtrackID", "BioID", "Year", "Month", "p_pro", "p_neutral", "p_anti"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Window stance score: positive = pro, negative = anti
    df["window_score"] = df["p_pro"] - df["p_anti"]
    df["anti_signal"] = (df["p_anti"] >= args.anti_threshold).astype(int)

    gcols = ["GovtrackID", "BioID", "Year", "Month"]
    out = df.groupby(gcols).agg(
        stance_score=("window_score", "mean"),
        evidence_count=("window_score", "size"),
        anti_signal_count=("anti_signal", "sum"),
        pro_prob_mean=("p_pro", "mean"),
        anti_prob_mean=("p_anti", "mean"),
        neutral_prob_mean=("p_neutral", "mean"),
    ).reset_index()

    # Flag low evidence months
    out["low_evidence_flag"] = (out["evidence_count"] < args.min_evidence).astype(int)

    # Simple month label for convenience (do not treat as ground truth)
    out["stance_label_month"] = np.where(out["stance_score"] > 0.10, "PRO",
                                 np.where(out["stance_score"] < -0.10, "ANTI", "NEUTRAL"))

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    out.to_csv(args.outfile, index=False, encoding="utf-8")
    print("Wrote:", args.outfile)
    print("Rows:", len(out))

if __name__ == "__main__":
    main()
