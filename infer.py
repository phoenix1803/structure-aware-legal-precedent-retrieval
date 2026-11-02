"""Run batched legal retrieval, optionally with a locally trained cross-encoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag import SearchEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("queries", type=Path, help="JSONL: query_id, query, optional before_year/exclude_doc_id")
    parser.add_argument("output", type=Path, help="Per-query ranked results JSONL")
    parser.add_argument("--reranker", help="Local cross-encoder directory, e.g. example-checkpoint")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--include-text", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    engine = SearchEngine(args.index, args.reranker)
    seen = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.queries.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id, query = row.get("query_id"), row.get("query")
            if not isinstance(query_id, str) or not query_id or not isinstance(query, str) or not query.strip():
                raise ValueError(f"{args.queries}:{line_no}: require query_id and query strings")
            if query_id in seen:
                raise ValueError(f"Duplicate query ID: {query_id}")
            seen.add(query_id)
            before = row.get("before_year")
            if before is not None and type(before) is not int:
                raise ValueError(f"{args.queries}:{line_no}: before_year must be an integer")
            ranked = engine.search(query, args.top_k, before, row.get("exclude_doc_id"))
            results = []
            for rank, hit in enumerate(ranked, 1):
                result = {key: hit[key] for key in ("doc_id", "year", "role", "score", "source")}
                result["rank"] = rank
                if "rerank_score" in hit:
                    result["rerank_score"] = hit["rerank_score"]
                if args.include_text:
                    result["passage"] = hit["text"]
                results.append(result)
            target.write(json.dumps({"query_id": query_id, "results": results}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(seen)} query runs to {args.output}")


if __name__ == "__main__":
    main()
