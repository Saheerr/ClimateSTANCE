
import argparse
from pathlib import Path
import pandas as pd

def clean_member_name(name: str) -> str:
    """Convert 'Last, First' to 'First Last' (same as before)."""
    if pd.isna(name):
        return name
    parts = [p.strip() for p in str(name).split(",")]
    if len(parts) == 2:
        return f"{parts[1]} {parts[0]}"
    return str(name)

def main():
    parser = argparse.ArgumentParser(description="Build monthly panel skeleton from roster (2017–2023).")
    parser.add_argument("--roster", default="../outputs/reps_roster_2017_2023.csv",
                        help="Path to reps_roster_2017_2023.csv")
    parser.add_argument("--outdir", default="../outputs",
                        help="Output directory (default: ../outputs)")
    parser.add_argument("--start", type=int, default=2017, help="Start year (default 2017)")
    parser.add_argument("--end",   type=int, default=2023, help="End year (default 2023)")
    args = parser.parse_args()

    roster_path = Path(args.roster)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not roster_path.exists():
        raise FileNotFoundError(f"Roster not found: {roster_path}")

    
    df = pd.read_csv(roster_path)

    
    keep_cols = [
        "Year",
        "District",
        "Member of Congress",
        "GovtrackID",
        "BioID",
        "Yearly LCV Score",
    ]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Roster missing required columns: {missing}")

    df = df[keep_cols].copy()

    df = df[(df["Year"] >= args.start) & (df["Year"] <= args.end)].copy()

    
    if "MemberClean" not in df.columns:
        df["MemberClean"] = df["Member of Congress"].apply(clean_member_name)

   
    roster = df.drop_duplicates(subset=["Year", "District"]).copy()

    
    coverage = roster.groupby("Year")["District"].nunique().reset_index(name="num_districts")
    print("District coverage by year:")
    print(coverage.to_string(index=False))


    monthly_rows = []
    for _, r in roster.iterrows():
        for m in range(1, 13):  # 1..12
            monthly_rows.append({
                "Year": int(r["Year"]),
                "Month": int(m),
                "District": r["District"],
                "Member of Congress": r["Member of Congress"],
                "MemberClean": r["MemberClean"],
                "GovtrackID": r["GovtrackID"],
                "BioID": r["BioID"],
                "Yearly LCV Score": r["Yearly LCV Score"],
             
                "twitter_handle": pd.NA,
                "official_website": pd.NA,
                "stance_label": pd.NA,
                "tweets_count": pd.NA,
                "site_statements_count": pd.NA,
            })

    panel = pd.DataFrame(monthly_rows)

 
    panel["MonthStart"] = pd.to_datetime(
        panel["Year"].astype(str) + "-" + panel["Month"].astype(str) + "-01",
        format="%Y-%m-%d"
    ).dt.date.astype(str)

   
    out_path = out_dir / "rep_monthly_panel_skeleton_2017_2023.csv"
    panel.to_csv(out_path, index=False)
    print(f"\nSaved monthly panel skeleton to: {out_path.resolve()}")

    
    unique_y_d = roster[["Year", "District"]].drop_duplicates().shape[0]
    expected_rows = unique_y_d * 12
    total_rows = len(panel)

    print("\nSummary")
    print(f"- Unique Year×District in roster: {unique_y_d}")
    print(f"- Monthly panel rows (should be 12 × Year×District): {total_rows}")
    if total_rows != expected_rows:
        print("  [WARN] Row count mismatch — check roster for duplicates or year filter.")

    
    print("\nSample rows:")
    print(panel.head(24).to_string(index=False))

if __name__ == "__main__":
    main()
