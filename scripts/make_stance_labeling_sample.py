#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_stance_labeling_sample.py

Builds a stance-labeling CSV sample from:
  outputs/monthly_text_chunks_2017_2023_with_climateprob.csv

Filters to climate_prob >= climate_min_prob, then samples evenly across years.

Writes:
  outputs/stance_labeling_sample_{N}_minprob{P}.csv

You will manually fill stance_label as:
  PRO / NEUTRAL / ANTI
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

IN_PATH = Path("outputs/monthly_text_chunks_2017_2023_with_climateprob.csv")
OUT_DIR = Path("outputs")

# Change this to 0.80 as requested
CLIMATE_MIN_PROB = 0.80

TOTAL_SAMPLE = 3000
SEED = 42

# Allow big fields just in case
csv.field_size_limit(10_000_000)


def safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input: {IN_PATH}")

    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[info] input: {IN_PATH}")
    print(f"[info] climate_min_prob: {CLIMATE_MIN_PROB}")
    print(f"[info] total_sample: {TOTAL_SAMPLE}")
    print(f"[info] seed: {SEED}")

    # First pass: collect eligible rows by year
    eligible_by_year: dict[str, list[dict]] = defaultdict(list)

    with IN_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header.")

        for row in reader:
            year = (row.get("Year", "") or "").strip()
            if year not in {"2017", "2018", "2019", "2020", "2021", "2022", "2023"}:
                continue

            cp = safe_float((row.get("climate_prob", "") or "").strip())
            if cp < CLIMATE_MIN_PROB:
                continue

            txt = (row.get("chunk_text", "") or "").strip()
            if not txt:
                continue

            eligible_by_year[year].append(row)

    years = ["2017", "2018", "2019", "2020", "2021", "2022", "2023"]

    # Compute per-year targets (as even as possible)
    base = TOTAL_SAMPLE // len(years)
    remainder = TOTAL_SAMPLE % len(years)
    targets = {}
    for i, y in enumerate(years):
        targets[y] = base + (1 if i < remainder else 0)

    print("[info] target per year:")
    for y in years:
        print(f"  {y}: {targets[y]}")

    print("\n[info] eligible climate chunks per year (cp >= min):")
    sampled_rows: list[dict] = []
    for y in years:
        pool = eligible_by_year.get(y, [])
        need = targets[y]
        take = min(len(pool), need)
        print(f"  {y}: {len(pool)} eligible, sampled {take}")

        if take == 0:
            continue

        sampled = random.sample(pool, k=take) if len(pool) >= take else pool
        sampled_rows.extend(sampled)

    # If we are short (some years lacked enough eligible rows), top up from all eligible
    if len(sampled_rows) < TOTAL_SAMPLE:
        all_pool = []
        for y in years:
            all_pool.extend(eligible_by_year.get(y, []))

        # Avoid duplicates when topping up
        seen_ids = set()
        for r in sampled_rows:
            # Try to create a stable unique key
            key = (
                (r.get("GovtrackID", "") or "")
                + "|"
                + (r.get("Year", "") or "")
                + "|"
                + (r.get("Month", "") or "")
                + "|"
                + (r.get("chunk_id", "") or "")
            )
            seen_ids.add(key)

        topup = []
        for r in all_pool:
            key = (
                (r.get("GovtrackID", "") or "")
                + "|"
                + (r.get("Year", "") or "")
                + "|"
                + (r.get("Month", "") or "")
                + "|"
                + (r.get("chunk_id", "") or "")
            )
            if key in seen_ids:
                continue
            topup.append(r)

        need_more = TOTAL_SAMPLE - len(sampled_rows)
        if need_more > 0 and len(topup) > 0:
            add = random.sample(topup, k=min(need_more, len(topup)))
            sampled_rows.extend(add)

    # Shuffle final sample
    random.shuffle(sampled_rows)

    out_path = OUT_DIR / f"stance_labeling_sample_{len(sampled_rows)}_minprob{CLIMATE_MIN_PROB}.csv"
    fieldnames = list(sampled_rows[0].keys()) if sampled_rows else []

    # Ensure labeling columns exist
    if "stance_label" not in fieldnames:
        fieldnames.append("stance_label")
    if "notes" not in fieldnames:
        fieldnames.append("notes")

    with out_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for r in sampled_rows:
            if "stance_label" not in r:
                r["stance_label"] = ""
            if "notes" not in r:
                r["notes"] = ""
            writer.writerow(r)

    print(f"\n[done] wrote: {out_path}")
    print("[done] Open this CSV and start labeling stance_label as PRO / NEUTRAL / ANTI")


if __name__ == "__main__":
    main()
