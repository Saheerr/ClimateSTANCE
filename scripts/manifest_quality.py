import pandas as pd
from pathlib import Path


OUT_DIR = Path("outputs")
MANIFEST_PATH = OUT_DIR / "master_manifest_2017_2023.csv"
OUT_PATH = OUT_DIR / "master_manifest_2017_2023_with_quality.csv"


def to_int_safe(x):
    """
    Convert value to int. Return 0 if missing or invalid.
    """
    try:
        return int(str(x).strip())
    except Exception:
        return 0


def col_str(row, col: str) -> str:
    """
    Safely get a string value from a row.
    Returns "" for NaN or None.
    """
    val = row.get(col, "")
    if pd.isna(val):
        return ""
    return str(val).strip()


def classify_quality(html_text: str, length: int) -> str:
    """
    Simple quality classifier for chosen HTML.
    You can tighten the rules later if needed.
    """
    if length <= 0 or not html_text:
        return "missing"

    low = html_text.lower()

    error_patterns = [
        "404 not found",
        "page not found",
        "the page you are looking for",
        "error 404",
        "403 forbidden",
        "access denied",
        "bad gateway",
        "maintenance",
        "temporarily unavailable",
    ]
    for pat in error_patterns:
        if pat in low:
            return "error_like"
    if length < 400:
        return "tiny"
    elif length < 1500:
        return "short"
    else:
        return "good"


if __name__ == "__main__":
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH, dtype=str)

    for col in [
        "GovtrackID",
        "BioID",
        "Year",
        "Month",
        "provenance_chosen",
        "html_path_chosen",
        "text_len_archive",
        "text_len_live",
    ]:
        if col not in df.columns:
            df[col] = ""

    n_total = len(df)
    n_archive = (df["provenance_chosen"] == "archive").sum()
    n_live = (df["provenance_chosen"] == "live").sum()
    n_none = (df["provenance_chosen"].isin(["none", "", None])).sum()

    print("=== Manifest coverage summary ===")
    print(f"Total rows: {n_total}")
    print(f"Chosen archive: {n_archive}")
    print(f"Chosen live:    {n_live}")
    print(f"Chosen none/empty: {n_none}")
    print()

    chosen_len = []
    quality = []

    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processed {idx} / {n_total} rows...")

        prov = col_str(row, "provenance_chosen")
        path = col_str(row, "html_path_chosen")
        if prov == "archive":
            length = to_int_safe(row.get("text_len_archive", 0))
        elif prov == "live":
            length = to_int_safe(row.get("text_len_live", 0))
        else:
            length = 0

        html_text = ""
        if path and length > 0:
            try:
                html_text = Path(path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                html_text = ""

        q = classify_quality(html_text, length)
        chosen_len.append(length)
        quality.append(q)

    df["html_len_chosen"] = chosen_len
    df["html_quality_flag"] = quality

  
    print()
    print("=== Quality flag counts ===")
    print(df["html_quality_flag"].value_counts(dropna=False).to_string())
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved updated manifest with quality flags to: {OUT_PATH.resolve()}")
