import argparse
import os
import re
import pandas as pd

# Patterns that indicate navigation / boilerplate / page chrome
BOILER_PATTERNS = [
    r"\bskip to main content\b",
    r"\bsign up\b",
    r"\be-?newsletter\b",
    r"\bnewsletter signup\b",
    r"\bcontact\b",
    r"\bflag requests\b",
    r"\bstudent resources\b",
    r"\bvisiting dc\b",
    r"\bread more\b",
    r"\blearn more\b",
    r"\bclick here\b",
    r"\bpress release\b",
    r"\bissues:\b",
    r"\btweets by\b",
    r"\bflickr\b",
    r"\byoutube channel\b",
    r"\bfacebook\b",
    r"\binstagram\b",
    r"\bsite map\b",
    r"\bsearch\b",
    r"\bbill search\b",
    r"\brecent votes\b",
    r"\broll call\b",
    r"\bprivacy\b",
    r"\bterms\b",
]

def looks_like_boilerplate(text: str) -> bool:
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t).strip()

    # Too short to contain a real stance statement
    if len(t) < 120:
        return True

    # If it is mostly navigation-like tokens (many capitalized labels in your data become words here)
    # Heuristic: many occurrences of these generic UI words
    ui_hits = sum(t.count(w) for w in ["sign up", "read more", "learn more", "contact", "skip", "search", "newsletter"])
    if ui_hits >= 2:
        return True

    # Regex patterns
    for pat in BOILER_PATTERNS:
        if re.search(pat, t):
            return True

    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.infile)

    if "window_text" not in df.columns:
        raise ValueError("Expected a column named window_text")

    df["is_boilerplate"] = df["window_text"].astype(str).map(looks_like_boilerplate)

    kept = df[df["is_boilerplate"] == False].copy()
    dropped = int(df["is_boilerplate"].sum())

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    kept.to_csv(args.outfile, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Dropped boilerplate: {dropped}")
    print(f"Kept rows: {len(kept)}")
    print(f"Wrote: {args.outfile}")

if __name__ == "__main__":
    main()
