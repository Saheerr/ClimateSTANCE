

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

try:
    import requests
except Exception:
    requests = None



BASE = "https://unitedstates.github.io/congress-legislators"
URLS = {
    "current_csv":    f"{BASE}/legislators-current.csv",
    "historical_csv": f"{BASE}/legislators-historical.csv",
    "social_json":    f"{BASE}/legislators-social-media.json",
}

DEFAULT_ROSTER = "../outputs/reps_roster_2017_2023.csv"
DEFAULT_PANEL  = "../outputs/rep_halfyear_panel_skeleton_2017_2023.csv"

OUT_DIR = Path("outputs")
SRC_DIR = Path("source_congress_legislators")


def log(msg: str) -> None:
    print(msg, flush=True)

def norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def download(url: str, dest: Path, retries: int = 3, timeout: int = 30) -> bool:
    if requests is None:
        log(f"[warn] 'requests' not installed. Skipping download: {url}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and r.content:
                dest.write_bytes(r.content)
                log(f"[ok] downloaded {url} -> {dest}")
                return True
            log(f"[warn] HTTP {r.status_code} for {url} (attempt {i})")
        except Exception as e:
            log(f"[warn] {e} (attempt {i})")
        time.sleep(1.0 * i)
    log(f"[fail] could not download {url}")
    return False

def ensure_sources() -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    cur = SRC_DIR / "legislators-current.csv"
    his = SRC_DIR / "legislators-historical.csv"
    soc = SRC_DIR / "legislators-social-media.json"

    if not cur.exists(): download(URLS["current_csv"], cur)
    if not his.exists(): download(URLS["historical_csv"], his)
    if not soc.exists(): download(URLS["social_json"], soc)

    return (cur if cur.exists() else None,
            his if his.exists() else None,
            soc if soc.exists() else None)

def read_csv_loose(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])


def build_legislators_ref(cur_csv: Optional[Path], his_csv: Optional[Path], soc_json: Optional[Path]) -> pd.DataFrame:
    frames = []
    if cur_csv is not None:
        dcur = read_csv_loose(cur_csv)
        dcur["__src"] = "current"
        frames.append(dcur)
    if his_csv is not None:
        dhis = read_csv_loose(his_csv)
        dhis["__src"] = "historical"
        frames.append(dhis)
    if not frames:
        return pd.DataFrame(columns=["govtrack_id","bioguide_id","official_website","twitter_handle","party"])

    people = pd.concat(frames, ignore_index=True)

    # Detect columns
    cols = {c.lower(): c for c in people.columns}
    gov_col   = cols.get("govtrack", cols.get("govtrack_id"))
    bio_col   = cols.get("bioguide", cols.get("bioguide_id"))
    url_col   = cols.get("url", cols.get("official_website", cols.get("website")))
    party_col = cols.get("party")

    people["govtrack_id"]      = norm(people[gov_col]) if gov_col else ""
    people["bioguide_id"]      = norm(people[bio_col]) if bio_col else ""
    people["official_website"] = norm(people[url_col]) if url_col else ""
    people["party"]            = norm(people[party_col]) if party_col else ""

    # Social media JSON
    if soc_json is not None and soc_json.exists():
        try:
            j = json.loads(soc_json.read_text(encoding="utf-8"))
            rows = []
            for item in j:
                bioguide = (item.get("id") or {}).get("bioguide", "")
                twitter  = (item.get("social") or {}).get("twitter", "")
                if bioguide and twitter:
                    rows.append({"bioguide_id": str(bioguide).strip(),
                                 "twitter_handle": str(twitter).strip().lstrip("@")})
            soc_df = pd.DataFrame(rows)
            ref = people.merge(soc_df, on="bioguide_id", how="left")
        except Exception as e:
            log(f"[warn] could not parse social JSON: {e}")
            people["twitter_handle"] = ""
            ref = people
    else:
        people["twitter_handle"] = ""
        ref = people

   
    score = (
        (ref["official_website"] != "").astype(int)
        + (ref["twitter_handle"].fillna("") != "").astype(int)
        + (ref["party"] != "").astype(int)
    )
    ref = ref.assign(__score=score).sort_values(
        ["govtrack_id","bioguide_id","__score"], ascending=[True, True, False]
    )

    by_gov = ref.drop_duplicates("govtrack_id", keep="first")
    by_bio = ref.drop_duplicates("bioguide_id", keep="first")
    ref = pd.concat([
        by_gov[["govtrack_id","bioguide_id","official_website","twitter_handle","party"]],
        by_bio[["govtrack_id","bioguide_id","official_website","twitter_handle","party"]],
    ], ignore_index=True).drop_duplicates(subset=["govtrack_id","bioguide_id"], keep="first")

    for c in ["govtrack_id","bioguide_id","official_website","twitter_handle","party"]:
        ref[c] = ref[c].astype(str)

    return ref


def enrich_roster_panel(roster_in: Path, panel_in: Path, ref_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    roster = pd.read_csv(roster_in, dtype=str)
    panel  = pd.read_csv(panel_in, dtype=str)

    names_r = roster["MemberClean"].astype(str).copy()
    names_p = panel["MemberClean"].astype(str).copy()
    len_r0, len_p0 = len(roster), len(panel)

   
    for col in ["official_website","twitter_handle","party"]:
        if col not in roster.columns:
            roster[col] = pd.NA
    for col in ["official_website","twitter_handle","party"]:
        if col not in panel.columns:
            panel[col] = pd.NA

    roster["GovtrackID_str"] = norm(roster["GovtrackID"])
    roster["BioID_str"]      = norm(roster["BioID"])
    panel["GovtrackID_str"]  = norm(panel["GovtrackID"])
    panel["BioID_str"]       = norm(panel["BioID"])

    ref_g = ref_df.loc[ref_df["govtrack_id"] != ""].drop_duplicates("govtrack_id", keep="first")
    ref_b = ref_df.loc[ref_df["bioguide_id"] != ""].drop_duplicates("bioguide_id", keep="first")

    r = roster.merge(ref_g, left_on="GovtrackID_str", right_on="govtrack_id", how="left", suffixes=("","_gov"))
    r = r.merge(ref_b.rename(columns={
        "official_website":"official_website_bio",
        "twitter_handle":"twitter_handle_bio",
        "party":"party_bio"
    }), left_on="BioID_str", right_on="bioguide_id", how="left")

    r["official_website"] = r["official_website"].combine_first(r.get("official_website_bio")).combine_first(roster["official_website"])
    r["twitter_handle"]   = r["twitter_handle"].combine_first(r.get("twitter_handle_bio")).combine_first(roster["twitter_handle"])
    r["party"]            = r["party"].combine_first(r.get("party_bio")).combine_first(roster["party"])

    roster_cols = [c for c in roster.columns if c not in ("GovtrackID_str","BioID_str")]
    roster_enriched = r[roster_cols].copy()

   
    p = panel.merge(ref_g, left_on="GovtrackID_str", right_on="govtrack_id", how="left", suffixes=("","_gov"))
    p = p.merge(ref_b.rename(columns={
        "official_website":"official_website_bio",
        "twitter_handle":"twitter_handle_bio",
        "party":"party_bio"
    }), left_on="BioID_str", right_on="bioguide_id", how="left")

    p["official_website"] = p["official_website"].combine_first(p.get("official_website_bio")).combine_first(panel["official_website"])
    p["twitter_handle"]   = p["twitter_handle"].combine_first(p.get("twitter_handle_bio")).combine_first(panel["twitter_handle"])
    p["party"]            = p["party"].combine_first(p.get("party_bio")).combine_first(panel["party"])

    panel_cols = [c for c in panel.columns if c not in ("GovtrackID_str","BioID_str")]
    panel_enriched = p[panel_cols].copy()

   
    assert len(roster_enriched) == len_r0
    assert len(panel_enriched)  == len_p0
    assert names_r.equals(roster_enriched["MemberClean"].astype(str))
    assert names_p.equals(panel_enriched["MemberClean"].astype(str))

    unmatched_roster = roster_enriched[
        roster_enriched[["official_website","twitter_handle","party"]].isna().all(axis=1)
    ][["Year","District","Member of Congress","MemberClean","GovtrackID","BioID"]].copy()

    unmatched_cols = [c for c in ["Year","Half","Month","District","Member of Congress","MemberClean","GovtrackID","BioID"] if c in panel_enriched.columns]
    unmatched_panel = panel_enriched[
        panel_enriched[["official_website","twitter_handle","party"]].isna().all(axis=1)
    ][unmatched_cols].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    roster_enriched.to_csv(out_dir / "reps_roster_2017_2023_enriched.csv", index=False)
    panel_name = Path(panel_in).name
    panel_out  = out_dir / panel_name.replace(".csv", "_enriched.csv")
    panel_enriched.to_csv(panel_out, index=False)

    unmatched_roster.to_csv(out_dir / "unmatched_roster.csv", index=False)
    unmatched_panel.to_csv(out_dir / "unmatched_panel.csv", index=False)
    ref_df.to_csv(out_dir / "legislators_ref.csv", index=False)

    lines = [
        "Enrichment succeeded.",
        f"Roster rows: {len(roster_enriched)} (input {len_r0})",
        f"Panel  rows: {len(panel_enriched)} (input {len_p0})",
        f"Unmatched roster rows: {len(unmatched_roster)}",
        f"Unmatched panel rows: {len(unmatched_panel)}",
        "Data source: congress-legislators (CC0 / public domain) via unitedstates.github.io.",
        "Matching: strict by GovtrackID then BioID.",
    ]
    (out_dir / "enrichment_report.txt").write_text("\n".join(lines))
    log("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Enrich roster/panel using congress-legislators (CC0)")
    parser.add_argument("--roster", default=DEFAULT_ROSTER)
    parser.add_argument("--panel",  default=DEFAULT_PANEL)
    parser.add_argument("--outdir", default=str(OUT_DIR))
    args = parser.parse_args()

    roster_in = Path(args.roster)
    panel_in  = Path(args.panel)
    out_dir   = Path(args.outdir)

    if not roster_in.exists() or not panel_in.exists():
        log(f"[fail] Missing inputs.\n  roster={roster_in.exists()} path={roster_in}\n  panel={panel_in.exists()} path={panel_in}")
        sys.exit(2)

    cur_csv, his_csv, soc_json = ensure_sources()
    if cur_csv is None and his_csv is None:
        log("[fail] Could not access legislators CSVs.")
        sys.exit(3)

    ref = build_legislators_ref(cur_csv, his_csv, soc_json)
    if ref.empty:
        log("[fail] Reference is empty.")
        sys.exit(4)

    enrich_roster_panel(roster_in, panel_in, ref, out_dir)

if __name__ == "__main__":
    main()
