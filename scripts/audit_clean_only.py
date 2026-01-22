import argparse
import os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    chunks = pd.read_csv(args.chunks)

    key = ["GovtrackID", "BioID", "Year", "Month"]
    needed = key + ["provenance_chosen"]
    missing = [c for c in needed if c not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    manifest_keyed = manifest[key + ["provenance_chosen"]].copy()

    joined = chunks.merge(manifest_keyed, on=key, how="left", validate="many_to_one")

    total = len(joined)
    live = int((joined["provenance_chosen"] == "live").sum())
    archive = int((joined["provenance_chosen"] == "archive").sum())
    none = int(joined["provenance_chosen"].isna().sum())

    clean = joined[joined["provenance_chosen"] == "archive"].copy()

    out_clean = os.path.join(args.outdir, "monthly_text_chunks_2017_2023_with_climateprob_archiveonly.csv")
    clean.to_csv(out_clean, index=False)

    out_audit = os.path.join(args.outdir, "audit_summary.txt")
    with open(out_audit, "w", encoding="utf-8") as f:
        f.write("AUDIT SUMMARY\n")
        f.write(f"total_chunks={total}\n")
        f.write(f"archive_chunks={archive}\n")
        f.write(f"live_chunks={live}\n")
        f.write(f"missing_provenance={none}\n")
        f.write("\n")
        f.write("Clean output contains only provenance_chosen == archive.\n")

    print("Done.")
    print(f"Archive-only chunks: {out_clean}")
    print(f"Audit: {out_audit}")

if __name__ == "__main__":
    main()
