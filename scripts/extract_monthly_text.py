import argparse
import csv
import sys
import re
import html as _html
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

OUT_DIR = Path("outputs")
MANIFEST_PATH = OUT_DIR / "master_manifest_2017_2023_with_quality.csv"
OUT_PATH = OUT_DIR / "monthly_website_text_2017_2023.csv"


_IGNORE_TAGS = {
    "script", "style", "noscript", "iframe", "svg", "canvas",
    "nav", "footer", "header", "aside", "form", "meta", "link"
}

_BLOCK_TAGS = {
    "p", "div", "section", "article", "main", "ul", "ol", "li",
    "table", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6",
    "br", "hr", "blockquote", "pre"
}

class TagAwareStripper(HTMLParser):
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


def html_to_text(path: Path) -> Tuple[str, str]:
    """
    Read HTML file and return (text, error_code).
    err = "" if ok; "read_fail" or "parse_fail" otherwise.
    """
    try:
        html = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return "", "read_fail"
    try:
        parser = TagAwareStripper()
        parser.feed(html)
        txt = parser.text()
        return txt, ""
    except Exception:
        return "", "parse_fail"


def main():
    if not MANIFEST_PATH.exists():
        print(f"[error] Manifest with quality not found: {MANIFEST_PATH}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(MANIFEST_PATH, dtype=str)

    # Ensure needed columns exist
    for col in [
        "GovtrackID", "BioID", "Year", "Month",
        "provenance_chosen", "html_path_chosen", "html_quality_flag"
    ]:
        if col not in df.columns:
            df[col] = ""

    
    good = df[df["html_quality_flag"] == "good"].copy()
    n_good = len(good)
    print(f"[info] Total rows in manifest: {len(df)}")
    print(f"[info] Rows with html_quality_flag == 'good': {n_good}")

    rows_out: List[Dict[str, str]] = []
    processed = 0
    ok_count = 0
    fail_read = 0
    fail_parse = 0

    for idx, row in good.iterrows():
        if processed % 200 == 0:
            print(f"[progress] Processed {processed} / {n_good} good rows...")

        gid = str(row.get("GovtrackID", "")).strip()
        bid = str(row.get("BioID", "")).strip()
        year = str(row.get("Year", "")).strip()
        month = str(row.get("Month", "")).strip()
        prov = str(row.get("provenance_chosen", "")).strip()
        path_str = str(row.get("html_path_chosen", "")).strip()

        if not path_str:
            text = ""
            err = "no_path"
        else:
            text, err = html_to_text(Path(path_str))

        text_len = len(text)
        ok = (err == "" and text_len > 0)
        if ok:
            ok_count += 1
        elif err == "read_fail":
            fail_read += 1
        elif err == "parse_fail":
            fail_parse += 1

        rows_out.append({
            "ok": "1" if ok else "0",
            "err": err,
            "GovtrackID": gid,
            "BioID": bid,
            "Year": year,
            "Month": month,
            "provenance_chosen": prov,
            "html_path": path_str,
            "text_len": str(text_len),
            "text": text,
        })

        processed += 1

    print(f"[done] HTML→text processed rows: {processed}")
    print(f"  ok rows: {ok_count}")
    print(f"  read failures: {fail_read}")
    print(f"  parse failures: {fail_parse}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_PATH
    with out_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "ok", "err",
            "GovtrackID", "BioID", "Year", "Month",
            "provenance_chosen", "html_path",
            "text_len", "text",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(f"[done] wrote {len(rows_out)} rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()
