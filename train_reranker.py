"""Fine-tune a binary legal query/passage cross-encoder from labeled JSONL pairs."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def read_pairs(path: Path) -> tuple[list[dict], set[str]]:
    rows, query_ids, pair_ids = [], set(), set()
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            needed = ("query_id", "doc_id", "query", "passage", "label")
            if any(key not in row for key in needed):
                raise ValueError(f"{path}:{line_no}: require {', '.join(needed)}")
            if not all(isinstance(row[key], str) and row[key].strip()
                       for key in needed[:4]) or type(row["label"]) not in (int, float) or row["label"] not in (0, 1):
                raise ValueError(f"{path}:{line_no}: strings must be nonempty; label must be 0 or 1")
            pair = (row["query_id"], row["doc_id"])
            if pair in pair_ids:
                raise ValueError(f"{path}:{line_no}: duplicate query/document pair {pair}")
            pair_ids.add(pair)
            query_ids.add(row["query_id"])
            rows.append(row)
    if not rows or {row["label"] for row in rows} != {0, 1}:
        raise ValueError(f"{path}: require both positive and negative labeled pairs")
    return rows, query_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Training pair JSONL")
    parser.add_argument("--validation", type=Path, required=True, help="Separate query IDs, never the test set")
    parser.add_argument("--output", type=Path, default=Path("example-checkpoint"))
    parser.add_argument("--base-model", default="nlpaueb/legal-bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.max_length) < 1:
        parser.error("epochs, batch-size, and max-length must be positive")
    train_rows, train_ids = read_pairs(args.train)
    valid_rows, valid_ids = read_pairs(args.validation)
    if train_ids & valid_ids:
        raise ValueError(f"Training/validation query IDs overlap: {sorted(train_ids & valid_ids)[:5]}")
    random.seed(args.seed)
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
        from sentence_transformers import CrossEncoder, InputExample
        from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before training") from exc
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    examples = [InputExample(texts=[r["query"], r["passage"]], label=float(r["label"]))
                for r in train_rows]
    validation = [InputExample(texts=[r["query"], r["passage"]], label=int(r["label"]))
                  for r in valid_rows]
    model = CrossEncoder(args.base_model, num_labels=1, max_length=args.max_length)
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(validation, name="validation")
    model.fit(
        train_dataloader=loader, evaluator=evaluator, epochs=args.epochs,
        warmup_steps=math.ceil(len(loader) * args.epochs * .1),
        evaluation_steps=len(loader), output_path=str(args.output),
        save_best_model=True, show_progress_bar=True,
    )
    if not (args.output / "config.json").exists():
        raise RuntimeError("Training ended without a saved checkpoint")
    print(f"Saved best validation checkpoint to {args.output}")


if __name__ == "__main__":
    main()
