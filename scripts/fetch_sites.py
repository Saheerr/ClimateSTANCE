

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse

from wayback_fetch import collect_dual


def read_csv_dicts(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
        return rows, r.fieldnames or []


def write_csv_dicts(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_url(u: str) -> str:
    """
    Ensure a scheme and strip a trailing '/' if it is the only path segment.
    """
    u = (u or "").strip()
    if not u:
        return ""
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    parsed = urlparse(u)
    if parsed.path == "/":
        u = u[:-1]
    return u


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collect website HTML for a given month (archive-first; keep live in parallel)."
    )
    ap.add_argument("--panel", required=True,
                    help="Path to monthly panel CSV (the ENRICHED file with official_website).")
    ap.add_argument("--out_dir", required=True,
                    help="Directory where HTML and manifest will be written.")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--website_col", default="official_website",
                    help="Column name in panel for the website URL (default: official_website).")
    ap.add_argument("--govtrack_col", default="GovtrackID",
                    help="GovtrackID column name (default: GovtrackID).")
    ap.add_argument("--bioid_col", default="BioID",
                    help="BioID column name (default: BioID).")
    ap.add_argument("--rate_sleep_ms", type=int, default=500,
                    help="Sleep between requests in milliseconds (default: 500).")

    args = ap.parse_args()
    panel_path = Path(args.panel)
    out_dir = Path(args.out_dir)
    html_dir = out_dir / "html"
    ensure_dir(out_dir)
    ensure_dir(html_dir)

    
    if not panel_path.exists():
        print(f"[error] panel not found: {panel_path}", file=sys.stderr)
        sys.exit(2)
    panel_rows, panel_fields = read_csv_dicts(panel_path)

    
    required_cols = [args.website_col, "Year", "Month", args.govtrack_col]
    missing = [c for c in required_cols if c not in panel_fields]
    if missing:
        print(f"[error] panel missing required columns: {missing}", file=sys.stderr)
        sys.exit(2)

    Y, M = args.year, args.month
    manifest_rows: List[Dict[str, str]] = []
    addon_rows: List[Dict[str, str]] = []

    processed = 0
    for row in panel_rows:
        
        try:
            y = int(row.get("Year", "0"))
            m = int(row.get("Month", "0"))
        except ValueError:
            continue
        if y != Y or m != M:
            continue

        website_raw = (row.get(args.website_col) or "").strip()
        website = normalize_url(website_raw)
        gid = (row.get(args.govtrack_col) or "").strip()
        bid = (row.get(args.bioid_col) or "").strip()

       
        safe_gid = gid if gid else (bid or "unknown")
        base = f"{safe_gid}_{Y:04d}_{M:02d}"
        archive_path = html_dir / f"{base}_archive.html"
        live_path = html_dir / f"{base}_live.html"

       
        result = collect_dual(url=website, year=Y, month=M)

        
        archive_html = result.get("archive_html") or ""
        live_html = result.get("live_html") or ""
        if archive_html:
            archive_path.write_text(archive_html, encoding="utf-8", errors="ignore")
        if live_html:
            live_path.write_text(live_html, encoding="utf-8", errors="ignore")

        chosen = result.get("chosen_provenance", "none")
        chosen_path = ""
        if chosen == "archive" and archive_html:
            chosen_path = str(archive_path)
        elif chosen == "live" and live_html:
            chosen_path = str(live_path)

        len_archive = str(len(archive_html))
        len_live = str(len(live_html))

        
        manifest_rows.append({
            "GovtrackID": gid,
            "BioID": bid,
            "Year": str(Y),
            "Month": str(M),
            "source_url_live": website,
            "source_url_archive": result.get("archive_url", ""),
            "archive_snapshot_ts": result.get("snapshot_ts", ""),
            "status_live": result.get("status_live", ""),
            "status_archive": result.get("status_archive", ""),
            "provenance_chosen": chosen,
            "chosen_reason": result.get("reason", ""),
            "html_path_chosen": chosen_path,
            "html_path_archive": str(archive_path) if archive_html else "",
            "html_path_live": str(live_path) if live_html else "",
            "text_len_archive": len_archive,
            "text_len_live": len_live,
        })

       
        addon_rows.append({
            "GovtrackID": gid,
            "BioID": bid,
            "Year": str(Y),
            "Month": str(M),
            "source_url_live": website,
            "source_url_archive": result.get("archive_url", ""),
            "archive_snapshot_ts": result.get("snapshot_ts", ""),
            "provenance_chosen": chosen,
            "html_path_chosen": chosen_path,
            "text_len_archive": len_archive,
            "text_len_live": len_live,
        })

        processed += 1
        if args.rate_sleep_ms > 0:
            import time as _t
            _t.sleep(args.rate_sleep_ms / 1000.0)

    
    manifest_fields = [
        "GovtrackID", "BioID", "Year", "Month",
        "source_url_live", "source_url_archive", "archive_snapshot_ts",
        "status_live", "status_archive",
        "provenance_chosen", "chosen_reason",
        "html_path_chosen", "html_path_archive", "html_path_live",
        "text_len_archive", "text_len_live"
    ]
    manifest_path = out_dir / f"manifest_{Y:04d}_{M:02d}.csv"
    write_csv_dicts(manifest_path, manifest_fields, manifest_rows)
    print(f"[ok] wrote manifest: {manifest_path} ({len(manifest_rows)} rows)")

    
    addon_fields = [
        "GovtrackID", "BioID", "Year", "Month",
        "source_url_live", "source_url_archive", "archive_snapshot_ts",
        "provenance_chosen", "html_path_chosen",
        "text_len_archive", "text_len_live"
    ]
    addon_path = out_dir / f"panel_addon_{Y:04d}_{M:02d}.csv"
    write_csv_dicts(addon_path, addon_fields, addon_rows)
    print(f"[ok] wrote panel add-on: {addon_path} ({len(addon_rows)} rows)")

    print(f"[done] processed {processed} rows for {Y}-{M:02d}")


if __name__ == "__main__":
    main()
