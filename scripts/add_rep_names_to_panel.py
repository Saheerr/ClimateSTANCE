import argparse
import os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--excel", required=True)
    ap.add_argument("--sheet", default="District panel - LCV score and ")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    panel = pd.read_csv(args.panel, encoding="utf-8")
    roster = pd.read_excel(args.excel, sheet_name=args.sheet)

    lookup = roster[["BioID", "Member of Congress"]].drop_duplicates()
    lookup = lookup.rename(columns={"Member of Congress": "rep_name"})

    out = panel.merge(lookup, on="BioID", how="left")

    # move rep_name right after BioID
    cols = out.columns.tolist()
    cols.remove("rep_name")
    idx = cols.index("BioID") + 1
    cols = cols[:idx] + ["rep_name"] + cols[idx:]
    out = out[cols]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8")

    print("Wrote:", args.out)
    print("Missing rep_name:", int(out["rep_name"].isna().sum()))

if __name__ == "__main__":
    main()
