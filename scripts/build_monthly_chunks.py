import csv
import sys
import re
from pathlib import Path

csv.field_size_limit(10_000_000)


IN_PATH = Path("outputs/monthly_website_text_2017_2023.csv")
OUT_PATH = Path("outputs/monthly_text_chunks_2017_2023.csv")

PARA_SPLIT_RE = re.compile(r"\n\s*\n+")
WS_RE = re.compile(r"\s+")

MAX_CHARS = 1800
MIN_CHARS = 200


def norm(s: str) -> str:
    return WS_RE.sub(" ", (s or "").strip())


def chunk_text(text: str):
    if not text or not text.strip():
        return []

    text = text.strip()

    paras = [norm(p) for p in PARA_SPLIT_RE.split(text)]
    paras = [p for p in paras if len(p) >= MIN_CHARS]

   
    if not paras:
        clean = norm(text)
        out = []
        i = 0
        while i < len(clean):
            j = min(len(clean), i + MAX_CHARS)
            ch = clean[i:j].strip()
            if len(ch) >= MIN_CHARS:
                out.append(ch)
            i = j
        return out

   
    out = []
    for p in paras:
        if len(p) <= MAX_CHARS:
            out.append(p)
        else:
            i = 0
            while i < len(p):
                j = min(len(p), i + MAX_CHARS)
                ch = p[i:j].strip()
                if len(ch) >= MIN_CHARS:
                    out.append(ch)
                i = j
    return out


def main():
    print("[info] build_monthly_chunks starting...")

    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input: {IN_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with IN_PATH.open("r", encoding="utf-8", newline="") as f_in, OUT_PATH.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = ["GovtrackID", "BioID", "Year", "Month", "chunk_id", "chunk_len", "chunk_text"]
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()

        total = 0
        written = 0

        for row in reader:
            total += 1
            ok = str(row.get("ok", "") or "").strip()
            text = row.get("text", "") or ""
            if ok != "1" or not text.strip():
                continue

            chunks = chunk_text(text)
            for i, ch in enumerate(chunks):
                w.writerow({
                    "GovtrackID": (row.get("GovtrackID", "") or "").strip(),
                    "BioID": (row.get("BioID", "") or "").strip(),
                    "Year": (row.get("Year", "") or "").strip(),
                    "Month": (row.get("Month", "") or "").strip(),
                    "chunk_id": str(i),
                    "chunk_len": str(len(ch)),
                    "chunk_text": ch,
                })
                written += 1

            if total % 2000 == 0:
                print(f"[progress] processed {total} monthly rows, wrote {written} chunks...")

    print(f"[done] processed {total} monthly rows")
    print(f"[done] wrote {written} chunks to {OUT_PATH}")


if __name__ == "__main__":
    main()
