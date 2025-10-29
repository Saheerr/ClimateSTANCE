
"""
Collect all months (2017–2023) in one continuous run.

Requires:
- wayback_fetch.py  (with collect_dual)
- rep_monthly_panel_skeleton_2017_2023_enriched.csv

Output:
- outputs/master_manifest_2017_2023.csv   (one combined CSV)
- HTML files stored under outputs/html/YYYY_MM/
"""

import pandas as pd
from pathlib import Path
import time
from wayback_fetch import collect_dual


PANEL_PATH = Path("outputs/rep_monthly_panel_skeleton_2017_2023_enriched.csv")
OUT_DIR = Path("outputs")
HTML_BASE = OUT_DIR / "html"
OUT_DIR.mkdir(exist_ok=True)
HTML_BASE.mkdir(exist_ok=True)

START_YEAR, END_YEAR = 2017, 2023
RATE_SLEEP_SEC = 0.5 
MASTER_CSV = OUT_DIR / "master_manifest_2017_2023.csv"


df = pd.read_csv(PANEL_PATH, dtype=str)
df = df[df["official_website"].notna()]


if not MASTER_CSV.exists():
    pd.DataFrame(columns=[
        "GovtrackID","BioID","Year","Month",
        "source_url_live","source_url_archive","archive_snapshot_ts",
        "status_live","status_archive",
        "provenance_chosen","chosen_reason",
        "html_path_chosen","html_path_archive","html_path_live",
        "text_len_archive","text_len_live"
    ]).to_csv(MASTER_CSV, index=False)


existing = pd.read_csv(MASTER_CSV, dtype=str)
existing_keys = set(existing[["GovtrackID","BioID","Year","Month"]]
                    .astype(str).agg("|".join, axis=1))


total_rows = 0
for year in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):
        print(f"\n=== Processing {year}-{month:02d} ===")
        sub = df[(df["Year"] == str(year)) & (df["Month"] == str(month))]
        if sub.empty:
            print("No rows for this period.")
            continue

        manifest_rows = []
        for _, r in sub.iterrows():
            gid = str(r.get("GovtrackID","")).strip()
            bid = str(r.get("BioID","")).strip()
            key = f"{gid}|{bid}|{year}|{month}"
            if key in existing_keys:
                continue

            url = str(r.get("official_website","")).strip()
            if not url:
                continue

            res = collect_dual(url=url, year=year, month=month)
            base = f"{gid or bid or 'unknown'}_{year:04d}_{month:02d}"
            html_dir = HTML_BASE / f"{year:04d}_{month:02d}"
            html_dir.mkdir(parents=True, exist_ok=True)
            archive_path = html_dir / f"{base}_archive.html"
            live_path = html_dir / f"{base}_live.html"

            if res.get("archive_html"):
                archive_path.write_text(res["archive_html"], encoding="utf-8", errors="ignore")
            if res.get("live_html"):
                live_path.write_text(res["live_html"], encoding="utf-8", errors="ignore")

            chosen = res.get("chosen_provenance","none")
            chosen_path = str(archive_path) if chosen=="archive" else (
                str(live_path) if chosen=="live" else "")

            manifest_rows.append({
                "GovtrackID": gid,
                "BioID": bid,
                "Year": str(year),
                "Month": str(month),
                "source_url_live": url,
                "source_url_archive": res.get("archive_url",""),
                "archive_snapshot_ts": res.get("snapshot_ts",""),
                "status_live": res.get("status_live",""),
                "status_archive": res.get("status_archive",""),
                "provenance_chosen": chosen,
                "chosen_reason": res.get("reason",""),
                "html_path_chosen": chosen_path,
                "html_path_archive": str(archive_path) if res.get("archive_html") else "",
                "html_path_live": str(live_path) if res.get("live_html") else "",
                "text_len_archive": str(len(res.get("archive_html") or "")),
                "text_len_live": str(len(res.get("live_html") or "")),
            })

            existing_keys.add(key)
            total_rows += 1
            time.sleep(RATE_SLEEP_SEC)

        if manifest_rows:
            pd.DataFrame(manifest_rows).to_csv(
                MASTER_CSV, mode="a", header=False, index=False
            )
            print(f"Added {len(manifest_rows)} rows.")
        else:
            print("Nothing new added for this month.")

print(f"\nAll done. Total new rows added: {total_rows}")
print(f"Combined manifest: {MASTER_CSV}")
