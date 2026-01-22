import argparse
import csv
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME_DEFAULT = "climatebert/distilroberta-base-climate-detector"
csv.field_size_limit(10_000_000)

def pick_climate_label_id(id2label: dict) -> int:
    if not id2label:
        return 1
    for k, v in id2label.items():
        name = str(v).lower()
        if "climate" in name and "non" not in name and "not" not in name:
            return int(k)
        if name in {"climate", "climate-related", "climate_related", "related"}:
            return int(k)
    return 1

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--text_col", default="window_text")
    ap.add_argument("--model", default=MODEL_NAME_DEFAULT)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    in_path = Path(args.infile)
    out_path = Path(args.outfile)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")
    print(f"[info] loading model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()

    id2label = model.config.id2label or {}
    climate_id = pick_climate_label_id(id2label)
    print(f"[info] climate_label_id used: {climate_id}")
    print(f"[info] reading: {in_path}")
    print(f"[info] writing: {out_path}")
    print(f"[info] text_col: {args.text_col}")
    print(f"[info] batch_size: {args.batch_size}")

    total = 0
    wrote = 0

    with in_path.open("r", encoding="utf-8", newline="") as f_in, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header.")

        out_fields = list(reader.fieldnames) + ["window_climate_prob", "window_pred_id", "window_pred_label"]
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()

        batch_rows = []
        batch_texts = []

        def flush_batch():
            nonlocal wrote
            if not batch_rows:
                return

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)  # [B, num_labels]

            for i, row in enumerate(batch_rows):
                p = probs[i]
                pred_id = int(torch.argmax(p).item())
                pred_label = str(id2label.get(pred_id, f"LABEL_{pred_id}"))
                climate_prob = float(p[climate_id].item())

                row["window_climate_prob"] = f"{climate_prob:.6f}"
                row["window_pred_id"] = str(pred_id)
                row["window_pred_label"] = pred_label
                writer.writerow(row)
                wrote += 1

            batch_rows.clear()
            batch_texts.clear()

        for row in reader:
            total += 1
            text = (row.get(args.text_col, "") or "").strip()
            if not text:
                row["window_climate_prob"] = ""
                row["window_pred_id"] = ""
                row["window_pred_label"] = ""
                writer.writerow(row)
                wrote += 1
            else:
                batch_rows.append(row)
                batch_texts.append(text)
                if len(batch_rows) >= args.batch_size:
                    flush_batch()

            if total % 5000 == 0:
                print(f"[progress] processed {total} rows...")

        flush_batch()

    print(f"[done] processed {total} rows")
    print(f"[done] wrote {wrote} rows to: {out_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted by user", file=sys.stderr)
        raise
