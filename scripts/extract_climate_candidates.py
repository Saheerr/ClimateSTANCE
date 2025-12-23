

import argparse
import csv
import html as _html
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Tuple, Dict, Iterable


ANCHORS = {
    "climate", "climate change", "global warming", "emissions", "carbon",
    "greenhouse", "methane", "epa", "paris", "clean energy", "renewable",
    "renewables", "pollution", "decarbon", "net zero",
}


SECTION_HINTS = {"press", "press-releases", "media", "media-center", "news", "statements"}

DATE_RE = re.compile(
    r"(?:\b20[01]\d\b|\b202[0-9]\b)|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}"
)
SENT_SPLIT_RE = re.compile(r'(?<=[\.\?!])\s+|\n+')


def has_anchor(text: str) -> bool:
    t = text.lower()
    return any(a in t for a in ANCHORS)


def sent_split(text: str) -> List[str]:
    parts = SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if len(p.strip()) > 10]


class _Collector(HTMLParser):
    """
    Minimal HTML collector for titles, headings, and link texts.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.meta_title = ""
        self.h1: List[str] = []
        self.h2: List[str] = []
        self.h3: List[str] = []
        self.links: List[Tuple[str, str]] = []  # (text, href)
        self._buf: List[str] = []
        self._stack: List[str] = []
        self._skip = 0
        self._in_title = False
        self._cur_link_href = None

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        self._stack.append(t)
        if t in {"script", "style", "noscript", "iframe"}:
            self._skip += 1
        if t == "title":
            self._in_title = True
        if t == "meta":
            attrd = {k.lower(): (v or "") for k, v in attrs}
            name = attrd.get("name", "").lower()
            prop = attrd.get("property", "").lower()
            content = attrd.get("content", "")
            if (name in {"title", "og:title"} or prop in {"og:title"}) and content:
                self.meta_title = content.strip()
        if t == "a":
            for k, v in attrs:
                if k.lower() == "href":
                    self._cur_link_href = v or ""
                    break

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in {"h1", "h2", "h3"} and self._buf:
            text = _html.unescape("".join(self._buf)).strip()
            text = re.sub(r"\s+", " ", text)
            if text:
                if t == "h1":
                    self.h1.append(text)
                elif t == "h2":
                    self.h2.append(text)
                else:
                    self.h3.append(text)
            self._buf.clear()
        if t == "a":
            txt = _html.unescape("".join(self._buf)).strip()
            txt = re.sub(r"\s+", " ", txt)
            if txt and self._cur_link_href:
                self.links.append((txt, self._cur_link_href))
            self._buf.clear()
            self._cur_link_href = None
        if t == "title":
            self._in_title = False
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i] == t:
                del self._stack[i]
                break
        if t in {"script", "style", "noscript", "iframe"} and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip > 0 or not data:
            return
        if self._in_title:
            self.title += data
        if self._stack and self._stack[-1] in {"h1", "h2", "h3", "a"}:
            self._buf.append(data)


def read_html(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def extract_titles_and_links(html: str):
    p = _Collector()
    p.feed(html)
    page_title = p.meta_title or p.title.strip()

    def uniq(lst: List[str]) -> List[str]:
        seen, out = set(), []
        for x in lst:
            if x and x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return page_title, uniq(p.h1), uniq(p.h2), uniq(p.h3), p.links


def choose_path(row: Dict[str, str]) -> str:
    """
    Pick the best HTML path for this row.
    Prefer html_path_chosen, fall back to archive/live if needed.
    """
    for key in ("html_path_chosen", "html_path_archive", "html_path_live"):
        v = (row.get(key) or "").strip()
        if v:
            return v
    return ""


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, str]], fields: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    rows = read_manifest(manifest)

    filt = []
    skipped_bad_quality = 0
    for r in rows:
        y = (r.get("Year", "") or "").strip()
        if y != str(args.year):
            continue

        qflag = (r.get("html_quality_flag") or "").strip().lower()
        if qflag and qflag != "good":
            skipped_bad_quality += 1
            continue

        p = choose_path(r)
        if not p:
            continue
        r["_html_path"] = p
        filt.append(r)
        if args.limit and len(filt) >= args.limit:
            break

    print(f"[info] Year {args.year}: total manifest rows: {len(rows)}")
    print(f"[info] Year {args.year}: kept rows with usable HTML (and good quality if flagged): {len(filt)}")
    if skipped_bad_quality:
        print(f"[info] Year {args.year}: skipped rows with non-good html_quality_flag: {skipped_bad_quality}")

    headline_out: List[Dict[str, str]] = []
    sent_out: List[Dict[str, str]] = []
    seen_headline = set()

    for r in filt:
        html = read_html(Path(r["_html_path"]))
        if not html:
            continue

        page_title, h1, h2, h3, links = extract_titles_and_links(html)

        
        candidates = []
        if page_title:
            candidates.append(("title", page_title, ""))
        for t in h1:
            candidates.append(("h1", t, ""))
        for t in h2:
            candidates.append(("h2", t, ""))
        for t in h3:
            candidates.append(("h3", t, ""))
        for txt, href in links:
            href_lc = (href or "").lower()
            if any(hint in href_lc for hint in SECTION_HINTS):
                candidates.append(("link", txt, href))
        for kind, txt, href in candidates:
            clean = re.sub(r"\s+", " ", (txt or "")).strip()
            if not clean:
                continue
            if has_anchor(clean):
                key = (r.get("GovtrackID", ""), r.get("Year", ""), r.get("Month", ""), kind, clean)
                if key in seen_headline:
                    continue
                seen_headline.add(key)
                headline_out.append({
                    "GovtrackID": r.get("GovtrackID", ""),
                    "BioID": r.get("BioID", ""),
                    "Year": r.get("Year", ""),
                    "Month": r.get("Month", ""),
                    "kind": kind,
                    "text": clean,
                    "href": href,
                    "has_date_pattern": "1" if DATE_RE.search(clean) else "0",
                })

        body = re.sub(r"(?is)<(script|style|noscript|iframe).*?>.*?</\\1>", " ", html)
        body = re.sub(r"(?is)<[^>]+>", " ", body)
        body = _html.unescape(body)
        body = re.sub(r"\s+", " ", body).strip()
        sents = sent_split(body)
        for s in sents:
            if has_anchor(s):
                sent_out.append({
                    "GovtrackID": r.get("GovtrackID", ""),
                    "BioID": r.get("BioID", ""),
                    "Year": r.get("Year", ""),
                    "Month": r.get("Month", ""),
                    "sentence": s,
                })

    out_dir = Path("outputs")
    hl_path = out_dir / f"climate_headlines_{args.year}.csv"
    st_path = out_dir / f"climate_sentences_{args.year}.csv"
    write_csv(
        hl_path,
        headline_out,
        ["GovtrackID", "BioID", "Year", "Month", "kind", "text", "href", "has_date_pattern"],
    )
    write_csv(
        st_path,
        sent_out,
        ["GovtrackID", "BioID", "Year", "Month", "sentence"],
    )

    print(f"[done] Year {args.year}: headlines -> {hl_path}  ({len(headline_out)} rows)")
    print(f"[done] Year {args.year}: sentences -> {st_path}  ({len(sent_out)} rows)")


if __name__ == "__main__":
    main()
