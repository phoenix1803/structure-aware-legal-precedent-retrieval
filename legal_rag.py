"""Minimal, reusable implementation of the paper's retrieval approach.

Input: JSONL judgments with id, year, text, and optional cited case IDs.
The parser is intentionally adaptable to a jurisdiction's headings/citation IDs.
No benchmark results or evaluation labels are bundled with this module.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer


ROLE_PATTERN = re.compile(
    r"^\s*(FACTS?|ISSUES?|ARGUMENTS?|ANALYSIS|REASONS?|HELD|HOLDING|RATIO|"
    r"JUDGMENT|ORDER)\s*[:.\-]?\s*$", re.I | re.M
)
REASONING = {"ANALYSIS", "REASONS", "HELD", "HOLDING", "RATIO", "JUDGMENT"}
TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")


def tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def split_sections(text: str) -> list[tuple[str, str]]:
    matches = list(ROLE_PATTERN.finditer(text))
    if not matches:
        return [("FULL_TEXT", text)]
    sections = []
    if text[: matches[0].start()].strip():
        sections.append(("PREAMBLE", text[: matches[0].start()].strip()))
    for i, heading in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[heading.end() : end].strip()
        if body:
            sections.append((heading.group(1).upper(), body))
    return sections or [("FULL_TEXT", text)]


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    step = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, len(words), step)
            if words[i : i + size]]


def make_chunks(doc: dict, size: int, overlap: int) -> list[dict]:
    chunks = []
    for role, section in split_sections(doc["text"]):
        for part in chunk_words(section, size, overlap):
            chunks.append({
                "id": f"{doc['id']}:{len(chunks)}", "doc_id": doc["id"],
                "year": int(doc["year"]), "role": role, "text": part,
                "citations": list(doc.get("citations", [])),
            })
    return chunks


class Encoder:
    def __init__(self, checkpoint: str):
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        rows = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = self.tokenizer(
                    texts[start : start + batch_size], padding=True, truncation=True,
                    max_length=512, return_tensors="pt"
                )
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                rows.append(pooled.cpu().numpy().astype("float32"))
        return np.vstack(rows)


def build(source: Path, destination: Path, model: str, size: int, overlap: int) -> None:
    with source.open(encoding="utf-8") as stream:
        docs = [json.loads(line) for line in stream if line.strip()]
    if not docs or len({d["id"] for d in docs}) != len(docs):
        raise ValueError("Input needs nonempty judgments with unique IDs")
    aliases = {}
    for doc in docs:
        aliases[str(doc["id"]).casefold()] = doc["id"]
        for alias in doc.get("aliases", []):
            aliases[str(alias).casefold()] = doc["id"]
    for doc in docs:
        doc["citations"] = [
            aliases.get(str(cite).casefold(), str(cite))
            for cite in doc.get("citations", [])
        ]
    destination.mkdir(parents=True, exist_ok=True)
    chunks = [c for doc in docs for c in make_chunks(doc, size, overlap)]
    if not chunks:
        raise ValueError("No text chunks produced")
    encoder = Encoder(model)
    vectors = encoder.encode([c["text"] for c in chunks])
    index = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = 128
    index.add(vectors)
    faiss.write_index(index, str(destination / "dense.faiss"))
    with (destination / "chunks.jsonl").open("w", encoding="utf-8") as out:
        for chunk in chunks:
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    (destination / "config.json").write_text(
        json.dumps({"model": model, "chunk_size_words": size, "overlap_words": overlap,
                    "hnsw_m": 32, "ef_search": 128}, indent=2), encoding="utf-8"
    )
    print(f"Indexed {len(docs)} judgments and {len(chunks)} chunks in {destination}")


def normalize(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    return {i: (value - lo) / (hi - lo) if hi > lo else 0.0
            for i, value in values.items()}


def search(directory: Path, query: str, k: int, before_year: int | None,
           reranker_checkpoint: str | None) -> list[dict]:
    cfg = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    chunks = [json.loads(line) for line in
              (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    index = faiss.read_index(str(directory / "dense.faiss"))
    index.hnsw.efSearch = int(cfg["ef_search"])
    encoder = Encoder(cfg["model"])
    bm25 = BM25Okapi([tokens(c["text"]) for c in chunks])
    lexical = bm25.get_scores(tokens(query))
    lexical_ids = np.argsort(lexical)[::-1][: max(50, k * 10)]
    distances, dense_ids = index.search(encoder.encode([query]), max(50, k * 10))
    dense = {int(i): float(score) for i, score in zip(dense_ids[0], distances[0]) if i >= 0}
    lex = {int(i): float(lexical[i]) for i in lexical_ids}
    dn, ln = normalize(dense), normalize(lex)
    by_doc: dict[str, dict] = {}
    for i in set(dn) | set(ln):
        c = chunks[i]
        if before_year is not None and c["year"] >= before_year:
            continue
        role = 1.0 if c["role"] in REASONING else 0.5
        score = .45 * dn.get(i, 0.0) + .30 * ln.get(i, 0.0) + .15 * role
        if c["doc_id"] not in by_doc or score > by_doc[c["doc_id"]]["score"]:
            by_doc[c["doc_id"]] = {**c, "score": score, "source": "retrieval"}
    initial = sorted(by_doc.values(), key=lambda c: c["score"], reverse=True)
    # Resolve explicitly supplied citation IDs; a corpus-specific citation extractor
    # may populate this field during preprocessing.
    by_id = defaultdict(list)
    for c in chunks:
        if c["role"] in REASONING:
            by_id[c["doc_id"]].append(c)
    cited = []
    for c in initial[:3]:
        cited.extend(c["citations"])
    for doc_id in dict.fromkeys(cited):
        if doc_id in by_doc or not by_id.get(doc_id):
            continue
        c = by_id[doc_id][0]
        if before_year is not None and c["year"] >= before_year:
            continue
        by_doc[doc_id] = {**c, "score": .10, "source": "citation"}
    candidates = sorted(by_doc.values(), key=lambda c: c["score"], reverse=True)
    if reranker_checkpoint:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(reranker_checkpoint)
        top = candidates[:50]
        scores = reranker.predict([(query, c["text"]) for c in top])
        for c, score in zip(top, scores):
            c["rerank_score"] = float(score)
        candidates[:50] = sorted(top, key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("corpus", type=Path, help="JSONL: id, year, text, citations[]")
    b.add_argument("index", type=Path)
    b.add_argument("--model", default="nlpaueb/legal-bert-base-uncased")
    b.add_argument("--chunk-size", type=int, default=512, help="Words, not model tokens")
    b.add_argument("--overlap", type=int, default=50, help="Words")
    s = sub.add_parser("search")
    s.add_argument("index", type=Path)
    s.add_argument("query")
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--before-year", type=int)
    s.add_argument("--reranker", help="Compatible CrossEncoder checkpoint")
    args = parser.parse_args()
    if args.command == "build":
        build(args.corpus, args.index, args.model, args.chunk_size, args.overlap)
    else:
        for result in search(args.index, args.query, args.top_k,
                             args.before_year, args.reranker):
            print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
