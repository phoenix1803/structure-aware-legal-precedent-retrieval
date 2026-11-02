"""Compute case-level retrieval metrics from ranked runs and relevance labels."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def metrics(ranked: list[str], relevant: dict[str, int], cutoff: int) -> dict[str, float]:
    if cutoff < 20 or not relevant or not any(value > 0 for value in relevant.values()):
        raise ValueError("Require cutoff >= 20 and at least one relevant case")
    if len(ranked) != len(set(ranked)):
        raise ValueError("Ranked case IDs must be unique")
    positives = {case_id for case_id, grade in relevant.items() if grade > 0}
    at_k = ranked[:cutoff]
    hits = [case_id in positives for case_id in at_k]
    def recall(k: int) -> float:
        return sum(hits[:k]) / len(positives)
    first = next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
    precision_sum = sum(sum(hits[:rank]) / rank for rank, hit in enumerate(hits, 1) if hit)
    gains = [2 ** relevant.get(case_id, 0) - 1 for case_id in ranked[:5]]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal_gains = sorted((2 ** grade - 1 for grade in relevant.values()), reverse=True)[:5]
    ideal = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, 1))
    return {
        "P@5": sum(hits[:5]) / 5,
        f"MRR@{cutoff}": first,
        "nDCG@5": dcg / ideal,
        "Recall@5": recall(5),
        "Recall@10": recall(10),
        "Recall@20": recall(20),
        f"MAP@{cutoff}": precision_sum / len(positives),
    }


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = row.get("query_id")
            if not isinstance(query_id, str) or not query_id or query_id in rows:
                raise ValueError(f"{path}:{line_no}: missing or duplicate query_id")
            rows[query_id] = row
    if not rows:
        raise ValueError(f"Empty input: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, help="Output JSONL from infer.py")
    parser.add_argument("qrels", type=Path, help='JSONL: query_id and relevant {"case-id": grade}')
    parser.add_argument("--cutoff", type=int, default=20, help="Max returned rank; >= 20")
    parser.add_argument("--per-query", type=Path, help="Write per-query metric JSONL")
    args = parser.parse_args()
    if args.cutoff < 20:
        parser.error("--cutoff must be >= 20")
    runs, qrels = read_jsonl(args.runs), read_jsonl(args.qrels)
    if runs.keys() != qrels.keys():
        raise ValueError(f"Run/qrels query IDs differ; missing labels: {sorted(runs.keys() - qrels.keys())[:5]}; missing runs: {sorted(qrels.keys() - runs.keys())[:5]}")
    per_query = []
    for query_id, run in runs.items():
        labels = qrels[query_id].get("relevant")
        if not isinstance(labels, dict) or not labels or any(
            not isinstance(key, str) or type(value) is not int or value < 0
            for key, value in labels.items()
        ):
            raise ValueError(f"Invalid graded relevance for {query_id}")
        results = run.get("results")
        if not isinstance(results, list) or any(not isinstance(item, dict) or not isinstance(item.get("doc_id"), str) for item in results):
            raise ValueError(f"Invalid results for {query_id}")
        if len(results) < args.cutoff:
            raise ValueError(f"{query_id}: only {len(results)} results; need {args.cutoff} to compute Recall@20")
        scores = metrics([item["doc_id"] for item in results], labels, args.cutoff)
        per_query.append({"query_id": query_id, **scores})
    names = list(per_query[0].keys())[1:]
    aggregate = {name: sum(row[name] for row in per_query) / len(per_query) for name in names}
    if args.per_query:
        args.per_query.parent.mkdir(parents=True, exist_ok=True)
        with args.per_query.open("w", encoding="utf-8") as stream:
            for row in per_query:
                stream.write(json.dumps(row) + "\n")
    print(json.dumps({"queries": len(per_query), "cutoff": args.cutoff, "macro_average": aggregate}, indent=2))


if __name__ == "__main__":
    main()
