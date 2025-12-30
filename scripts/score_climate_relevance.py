#climatebert/distilroberta-base-climate-detector


import csv
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

IN_PATH = Path("outputs/monthly_text_chunks_2017_2023.csv")
OUT_PATH = Path("outputs/monthly_text_chunks_2017_2023_with_climateprob.csv")

MODEL_NAME = "climatebert/distilroberta-base-climate-detector"


csv.field_size_limit(10_000_000)


def pick_climate_label_id(id2label: dict) -> int:
    """
    Try to infer which label corresponds to climate-related.
    If unsure, default to 1 (common for binary classifiers).
    """
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
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input CSV: {IN_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    print(f"[info] loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    id2label = model.config.id2label or {}
    climate_id = pick_climate_label_id(id2label)
    print(f"[info] id2label: {id2label}")
    print(f"[info] climate_label_id used: {climate_id}")
    print(f"[info] reading: {IN_PATH}")
    print(f"[info] writing: {OUT_PATH}")

    total = 0
    wrote = 0

    with IN_PATH.open("r", encoding="utf-8", newline="") as f_in, OUT_PATH.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header/fieldnames.")

        out_fields = list(reader.fieldnames) + ["climate_prob", "pred_id", "pred_label"]
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()

        for row in reader:
            total += 1
            text = (row.get("chunk_text", "") or "").strip()
            if not text:
                row["climate_prob"] = ""
                row["pred_id"] = ""
                row["pred_label"] = ""
                writer.writerow(row)
                wrote += 1
                continue

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits  
                probs = torch.softmax(logits, dim=-1).squeeze(0)  # [num_labels]

            pred_id = int(torch.argmax(probs).item())
            pred_label = str(id2label.get(pred_id, f"LABEL_{pred_id}"))
            climate_prob = float(probs[climate_id].item())

            row["climate_prob"] = f"{climate_prob:.6f}"
            row["pred_id"] = str(pred_id)
            row["pred_label"] = pred_label
            writer.writerow(row)
            wrote += 1

            if total % 2000 == 0:
                print(f"[progress] scored {total} chunks...")

    print(f"[done] scored {total} chunks")
    print(f"[done] wrote {wrote} rows to: {OUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] interrupted by user", file=sys.stderr)
        raise
