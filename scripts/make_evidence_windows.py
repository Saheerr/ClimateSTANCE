import argparse
import os
import re
import pandas as pd

def simple_sentence_split(text: str):
    """
    Conservative splitter: better to under-split than over-split.
    Works decently on punctuation-separated text.
    """
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)

    # Split on . ! ? followed by a space, but avoid common abbreviations a bit
    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="archive-only chunks CSV")
    ap.add_argument("--out", required=True, help="output evidence windows CSV")
    ap.add_argument("--min_prob", type=float, default=0.8, help="min climate_prob to include")
    ap.add_argument("--max_windows_per_repmonth", type=int, default=25, help="cap to avoid domination")
    args = ap.parse_args()

    df = pd.read_csv(args.chunks)

    required = {"GovtrackID","BioID","Year","Month","chunk_id","chunk_text","climate_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in chunks file: {sorted(missing)}")

    # Accuracy gate: if provenance column exists, enforce archive only
    if "provenance_chosen" in df.columns:
        bad = df[df["provenance_chosen"] != "archive"]
        if len(bad) > 0:
            raise ValueError(
                f"Found non-archive rows in archive-only input: {len(bad)}. "
                "Do not proceed. Fix your input first."
            )

    df = df[df["climate_prob"] >= args.min_prob].copy()

    rows = []
    for _, r in df.iterrows():
        sentences = simple_sentence_split(r["chunk_text"])

        # If we cannot reliably split, keep the whole chunk as one window (marked)
        if len(sentences) < 2:
            rows.append({
                "GovtrackID": r["GovtrackID"],
                "BioID": r["BioID"],
                "Year": int(r["Year"]),
                "Month": int(r["Month"]),
                "chunk_id": r["chunk_id"],
                "window_id": 0,
                "window_text": str(r["chunk_text"])[:1200],
                "window_note": "unsplit_chunk",
                "climate_prob": float(r["climate_prob"]),
            })
            continue

        for i in range(len(sentences)):
            left = sentences[i-1] if i-1 >= 0 else ""
            mid = sentences[i]
            right = sentences[i+1] if i+1 < len(sentences) else ""
            window = " ".join([x for x in [left, mid, right] if x])

            # Keep windows reasonably sized
            window = window.strip()
            if len(window) < 40:
                continue
            if len(window) > 1200:
                window = window[:1200]

            rows.append({
                "GovtrackID": r["GovtrackID"],
                "BioID": r["BioID"],
                "Year": int(r["Year"]),
                "Month": int(r["Month"]),
                "chunk_id": r["chunk_id"],
                "window_id": i,
                "window_text": window,
                "window_note": "",
                "climate_prob": float(r["climate_prob"]),
            })

    out = pd.DataFrame(rows)

    # Cap windows per rep-month so one member-month does not dominate labeling
    out["repmonth"] = out["GovtrackID"].astype(str) + "_" + out["Year"].astype(str) + "_" + out["Month"].astype(str)
    out = out.groupby("repmonth", group_keys=False).head(args.max_windows_per_repmonth).copy()
    out.drop(columns=["repmonth"], inplace=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} evidence windows to {args.out}")

if __name__ == "__main__":
    main()
