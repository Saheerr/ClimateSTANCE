import argparse
import os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="outputs/master_manifest_2017_2023_with_quality.csv")
    ap.add_argument("--monthly", required=True, help="outputs_clean/rep_monthly_stance_delta.csv")
    ap.add_argument("--outfile", required=True, help="outputs_clean/rep_monthly_stance_delta_FULLGRID.csv")
    args = ap.parse_args()

    grid = pd.read_csv(args.grid, encoding="utf-8")
    monthly = pd.read_csv(args.monthly, encoding="utf-8")

    need_grid = ["GovtrackID", "BioID", "Year", "Month", "provenance_chosen", "html_quality_flag"]
    missing = [c for c in need_grid if c not in grid.columns]
    if missing:
        raise ValueError(f"Grid file missing columns: {missing}")

    base = grid[need_grid].copy()

    out = base.merge(monthly, on=["GovtrackID", "BioID", "Year", "Month"], how="left")

    out["reason_code"] = np.where(
        out["html_quality_flag"] != "good", "no_good_html",
        np.where(out["provenance_chosen"] != "archive", "no_archive",
        np.where(out["evidence_count"].isna(), "no_new_climate_evidence", "ok"))
    )

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    out.to_csv(args.outfile, index=False, encoding="utf-8")

    print("Wrote full grid:", args.outfile, "rows:", len(out))
    print("Reason code counts:")
    print(out["reason_code"].value_counts().to_string())


if __name__ == "__main__":
    main()
