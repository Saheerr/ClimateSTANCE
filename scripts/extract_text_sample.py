#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_text_sample.py — sample-friendly website text extractor (standard library only)

Purpose:
Quickly confirm that you can extract readable text for a subset of your data
before running any ML scoring. Designed to run a small 2017 sample by default.

Key traits:
- Original code, no third-party dependencies.
- Reads your master manifest CSV that points to saved HTML.
- Filters by --year (default 2017) and optional --month.
- --limit controls how many rows to process (default 250).
- Produces a compact CSV with text you can feed to an ML script.

Usage examples:
# Default quick sample: year=2017, limit=250
python extract_text_sample.py \
  --manifest outputs/master_manifest_2017_2023.csv \
  --out outputs/sample_text_2017.csv

# Smaller or bigger sample:
python extract_text_sample.py --manifest outputs/master_manifest_2017_2023.csv --out outputs/sample_text_2017_50.csv --limit 50

# A specific month of 2017:
python extract_text_sample.py --manifest outputs/master_manifest_2017_2023.csv --out outputs/sample_text_2017_05.csv --month 5

Output columns:
ok, err, GovtrackID, BioID, Year, Month, html_path, text_len, text
"""

import argparse
import csv
import sys
import re
import html as _html
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

_IGNORE_TAGS = {
    "script", "style", "noscript", "iframe", "svg", "canvas",
    "nav", "footer", "header", "aside", "form", "meta", "link"
}

_BLOCK_TAGS = {
    "p", "div", "section", "article", "main", "ul", "ol", "li",
    "table", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6",
    "br", "hr", "blockquote", "pre"
}

class _TagAwareStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._buf: List[str] = []
        self._stack: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str]]) -> None:
        t = tag.lower()
        self._stack.append(t)
        if t in _IGNORE_TAGS:
            self._skip_depth += 1
        if self._skip_depth > 0:
            return
        if t in _BLOCK_TAGS:
            self._buf.append(" ")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i] == t:
                del self._stack[i]
                break
        if t in _IGNORE_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if self._skip_depth > 0:
            return
        if t in _BLOCK_TAGS:
            self._buf.append(" ")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0 or not data:
            return
        self._buf.append(data)

    def text(self) -> str:
        raw = "".join(self._buf)
        raw = _html.unescape(raw)
        raw = re.sub(r"\s+", " ", raw)
        return raw.strip()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def _write_csv(path: Path, rows: Iterable[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _choose_path(row: Dict[str, str]) -> str:
    for key in ("html_path_chosen", "html_path_archive", "html_path_live"):
        p = (row.get(key) or "").strip()
        if p:
            return p
    return ""


def _extract_html_text(path: Path) -> (str, str):
    try:
        html = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return "", "read_fail"
    try:
        parser = _TagAwareStripper()
        parser.feed(html)
        text = parser.text()
        return text, ""
    except Exception:
        return "", "parse_fail"


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample-friendly HTML→text extractor (standard library only).")
    ap.add_argument("--manifest", required=True, help="Path to master_manifest_2017_2023.csv")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--year", type=int, default=2017, help="Filter Year (default 2017)")
    ap.add_argument("--month", type=int, default=0, help="Optional: filter Month (1..12)")
    ap.add_argument("--limit", type=int, default=250, help="Max rows to process (default 250)")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    out_path = Path(args.out)

    rows = _read_csv(manifest)
    for r in rows:
        for c in ("GovtrackID", "BioID", "Year", "Month",
                  "html_path_chosen", "html_path_archive", "html_path_live"):
            r.setdefault(c, "")

    # Filter by year/month
    filt = []
    for r in rows:
        y = str(r.get("Year", "")).strip()
        m = str(r.get("Month", "")).strip()
        if y != str(args.year):
            continue
        if args.month and m != str(args.month):
            continue
        filt.append(r)

    if not filt:
        print("[warn] No rows matched the selected year/month.", file=sys.stderr)

    if args.limit and args.limit > 0:
        filt = filt[:args.limit]

    out_fields = ["ok", "err", "GovtrackID", "BioID", "Year", "Month", "html_path", "text_len", "text"]
    out_rows: List[Dict[str, str]] = []

    total = len(filt)
    processed = 0
    ok_count = fail_read = fail_parse = no_path = 0

    for r in filt:
        html_path = _choose_path(r)
        if not html_path:
            no_path += 1
            out_rows.append({
                "ok": "0", "err": "no_path",
                "GovtrackID": r["GovtrackID"],
                "BioID": r["BioID"],
                "Year": r["Year"],
                "Month": r["Month"],
                "html_path": "",
                "text_len": "0",
                "text": ""
            })
        else:
            text, err = _extract_html_text(Path(html_path))
            if err == "read_fail":
                fail_read += 1
            elif err == "parse_fail":
                fail_parse += 1
            ok = (err == "" and len(text) > 0)
            ok_count += 1 if ok else 0
            out_rows.append({
                "ok": "1" if ok else "0",
                "err": err,
                "GovtrackID": r["GovtrackID"],
                "BioID": r["BioID"],
                "Year": r["Year"],
                "Month": r["Month"],
                "html_path": html_path,
                "text_len": str(len(text)),
                "text": text
            })

        processed += 1
        if processed % 100 == 0 or processed == total:
            print(f"[progress] {processed}/{total} rows", file=sys.stderr)

    _write_csv(out_path, out_rows, out_fields)
    print(f"[done] wrote {len(out_rows)} rows to {out_path}")
    print(f"  ok rows: {ok_count}")
    print(f"  read failures: {fail_read}")
    print(f"  parse failures: {fail_parse}")
    print(f"  missing path: {no_path}")

if __name__ == "__main__":
    main()
