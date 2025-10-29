

import pandas as pd
from pathlib import Path


DATA_DIR   = Path("data")
OUT_DIR    = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)  


EXCEL_PATH = DATA_DIR / "Activism and Political Outcomes  - all data (1).xlsx"
SHEET_NAME = "District panel - LCV score and " 


START_YEAR = 2017
END_YEAR   = 2023


ROSTER_CSV = OUT_DIR / "reps_roster_2017_2023.csv"
PANEL_CSV  = OUT_DIR / "rep_halfyear_panel_skeleton_2017_2023.csv"

def clean_member_name(name: str) -> str:
    """Convert 'Last, First' to 'First Last'."""
    if pd.isna(name):
        return name
    parts = [p.strip() for p in str(name).split(",")]
    if len(parts) == 2:
        return f"{parts[1]} {parts[0]}"
    return str(name)

def main():
    
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    
    keep_cols = [
        "Year",
        "District",
        "Member of Congress",
        "GovtrackID",
        "BioID",
        "Yearly LCV Score",
    ]
    df = df[keep_cols].copy()

   
    df = df[(df["Year"] >= START_YEAR) & (df["Year"] <= END_YEAR)].copy()

    
    df["MemberClean"] = df["Member of Congress"].apply(clean_member_name)

    
    roster = df.drop_duplicates(subset=["Year", "District"]).copy()

    
    coverage = roster.groupby("Year")["District"].nunique().reset_index(name="num_districts")
    print("District coverage by year:")
    print(coverage.to_string(index=False))

   
    roster.to_csv(ROSTER_CSV, index=False)
    print(f"\nSaved roster to: {ROSTER_CSV.resolve()}")

    
    half_rows = []
    for _, r in roster.iterrows():
        for half in (1, 2):
            half_rows.append({
                "Year": int(r["Year"]),
                "Half": int(half),
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

    panel = pd.DataFrame(half_rows)

    
    panel.to_csv(PANEL_CSV, index=False)
    print(f"Saved half-year panel skeleton to: {PANEL_CSV.resolve()}")

    
    total_rows = len(panel)
    unique_y_d = roster[["Year", "District"]].drop_duplicates().shape[0]
    print("\nSummary")
    print(f"- Unique Year×District rows: {unique_y_d}")
    print(f"- Half-year panel rows (should be 2 × Year×District): {total_rows}")

    
    print("\nSample rows:")
    print(panel.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
