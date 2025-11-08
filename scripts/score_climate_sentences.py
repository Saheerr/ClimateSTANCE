#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_climate_sentences.py — score stance of already-filtered climate sentences

Input:
  outputs/climate_sentences_2017.csv
Output:
  outputs/website_scores_2017.csv
"""

import csv, re
from pathlib import Path
from statistics import mean

IN_PATH  = Path("outputs") / "climate_sentences_2017.csv"
OUT_PATH = Path("outputs") / "website_scores_2017.csv"

PRO_PHRASES = {
    "clean energy","renewable energy","emissions reduction","reduce emissions",
    "net zero","carbon neutral","paris agreement","epa standards","energy efficiency",
    "climate resilience","green jobs","electric vehicle","ev charging","methane rule",
    "environmental protection"
}
ANTI_PHRASES = {
    "climate hoax","end the epa","abolish the epa","drill baby drill","expand drilling",
    "war on coal","withdraw from paris","repeal climate regulations",
    "job-killing regulation","cap and trade is a tax","rollback regulations","roll back regulations"
}
PRO_WORDS  = {"renewable","solar","wind","geothermal","decarbonize","resilience",
              "efficiency","ev","electric","methane","standards"}
ANTI_WORDS = {"fracking","drilling","coal","deregulation","rollback","hoax","burden"}
NEGATIONS  = {"not","no","never","oppose","opposes","opposed","against","reject","repeal"}

TOKEN_RE = re.compile(r"[a-z0-9']+")

def tokenize(s: str):
    return TOKEN_RE.findall(s.lower())

def count_hits(tokens):
    text = " ".join(tokens)
    pro = sum(1 for p in PRO_PHRASES if p in text) + sum(1 for t in tokens if t in PRO_WORDS)
    anti = sum(1 for p in ANTI_PHRASES if p in text) + sum(1 for t in tokens if t in ANTI_WORDS)
    for i, tok in enumerate(tokens):
        if tok in NEGATIONS:
            if i+1 < len(tokens) and tokens[i+1] in PRO_WORDS:
                pro = max(0, pro - 1)
            if i+1 < len(tokens) and tokens[i+1] in ANTI_WORDS:
                anti = max(0, anti - 1)
    return pro, anti

def sentence_score(sent):
    toks = tokenize(sent)
    pro, anti = count_hits(toks)
    tot = pro + anti
    if tot == 0:
        return 0.0
    return (pro - anti) / tot

def trimmed_mean(vals, trim=0.1):
    if not vals:
        return 0.0
    n = len(vals)
    k = int(n * trim)
    vals = sorted(vals)
    vals = vals[k:n-k] if n > 20 and k > 0 else vals
    return sum(vals)/len(vals)

def main():
    if not IN_PATH.exists():
        raise SystemExit(f"Input not found: {IN_PATH}")

    rows = list(csv.DictReader(open(IN_PATH, encoding="utf-8")))
    bucket = {}
    for row in rows:
        s = (row.get("sentence") or "").strip()
        if not s:
            continue
        sc = sentence_score(s)
        key = (row.get("GovtrackID",""), row.get("BioID",""), row.get("Year",""), row.get("Month",""))
        b = bucket.setdefault(key, {"scores": [], "n": 0})
        b["scores"].append(sc)
        b["n"] += 1

    out = []
    for (gt,bio,y,m),b in bucket.items():
        score = trimmed_mean(b["scores"])
        out.append({
            "GovtrackID": gt,
            "BioID": bio,
            "Year": y,
            "Month": m,
            "website_score": f"{score:.4f}",
            "n_passages": b["n"]
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["GovtrackID","BioID","Year","Month","website_score","n_passages"])
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"[done] wrote {len(out)} rows -> {OUT_PATH}")

if __name__ == "__main__":
    main()
