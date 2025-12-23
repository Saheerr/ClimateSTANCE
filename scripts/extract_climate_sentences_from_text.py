
from pathlib import Path
from typing import List, Dict

import pandas as pd
import re

BASE_DIR = Path(".")
OUT_DIR = BASE_DIR / "outputs"
MONTHLY_TEXT_PATH = OUT_DIR / "monthly_website_text_2017_2023.csv"

SENT_SPLIT_RE = re.compile(r"(?<=[\.\?!])\s+|\n+")


def sent_split(text: str) -> List[str]:
    parts = SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if len(p.strip()) > 10]


def main():
    if not MONTHLY_TEXT_PATH.exists():
        raise FileNotFoundError(f"Cannot find monthly text file: {MONTHLY_TEXT_PATH}")

    print(f"[info] Loading monthly website text from: {MONTHLY_TEXT_PATH}")
    df = pd.read_csv(MONTHLY_TEXT_PATH, dtype=str)

    # Ensure required columns exist
    for col in ["GovtrackID", "BioID", "Year", "Month", "text", "ok"]:
        if col not in df.columns:
            df[col] = ""

    # We only care about rows with ok == "1" and non-empty text
    df["ok"] = df["ok"].fillna("0")
    n_total = len(df)
    df_good = df[(df["ok"] == "1") & df["text"].notna() & (df["text"].str.strip() != "")]
    n_good = len(df_good)

    print(f"[info] Total rows in monthly text: {n_total}")
    print(f"[info] Rows with ok == '1' and non-empty text: {n_good}")
    print()

    years = list(range(2017, 2024))

    for year in years:
        year_str = str(year)
        df_y = df_good[df_good["Year"] == year_str].copy()
        n_rows_y = len(df_y)

        print("==================================================")
        print(f"[info] Year {year}: rows with text: {n_rows_y}")
        print("==================================================")

        out_rows: List[Dict[str, str]] = []
        processed = 0
        kept_sentences = 0

        for idx, row in df_y.iterrows():
            text = str(row.get("text", "") or "").strip()
            if not text:
                continue

            sentences = sent_split(text)
            for s in sentences:
                out_rows.append(
                    {
                        "GovtrackID": str(row.get("GovtrackID", "") or "").strip(),
                        "BioID": str(row.get("BioID", "") or "").strip(),
                        "Year": year_str,
                        "Month": str(row.get("Month", "") or "").strip(),
                        "sentence": s,
                    }
                )
                kept_sentences += 1

            processed += 1
            if processed % 1000 == 0:
                print(
                    f"[progress] Year {year}: processed {processed} / {n_rows_y} rows, "
                    f"collected {kept_sentences} sentences so far..."
                )

        out_path = OUT_DIR / f"all_sentences_from_text_{year}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        import csv
        with out_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = ["GovtrackID", "BioID", "Year", "Month", "sentence"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in out_rows:
                w.writerow(r)

        print(f"[done] Year {year}: wrote {len(out_rows)} sentences to {out_path}")
        print()

    print("[done] All years 2017–2023 processed from monthly text.")


if __name__ == "__main__":
    main()
