"""Synthetic checks for ranking metrics and the retrieval interface."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

from evaluate import metrics
from legal_rag import SearchEngine
from train_reranker import read_pairs


class WorkflowChecks(unittest.TestCase):
    def test_case_level_metrics(self):
        ranked = ["a", "b", "c"] + [f"f{i}" for i in range(17)]
        scores = metrics(ranked, {"a": 2, "c": 1}, 20)
        self.assertAlmostEqual(scores["P@5"], 2 / 5)
        self.assertAlmostEqual(scores["MRR@20"], 1)
        self.assertAlmostEqual(scores["Recall@20"], 1)
        self.assertAlmostEqual(scores["MAP@20"], (1 + 2 / 3) / 2)
        self.assertAlmostEqual(scores["nDCG@5"], (3 + 1 / math.log2(4)) / (3 + 1 / math.log2(3)))

    def test_training_pair_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "pairs.jsonl"
            rows = [
                {"query_id": "q1", "doc_id": "a", "query": "question", "passage": "yes", "label": 1},
                {"query_id": "q1", "doc_id": "b", "query": "question", "passage": "no", "label": 0},
            ]
            source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            parsed, ids = read_pairs(source)
            self.assertEqual((len(parsed), ids), (2, {"q1"}))
            source.write_text("\n".join(json.dumps(row) for row in rows + [rows[0]]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_pairs(source)

    def test_temporal_and_self_exclusion(self):
        class FakeEncoder:
            def __init__(self, checkpoint):
                pass
            def encode(self, texts):
                return np.tile(np.array([[1.0, 0.0]], dtype="float32"), (len(texts), 1))

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            chunks = [
                {"doc_id": f"case-{i}", "year": 2017 + i % 4, "role": "HELD",
                 "text": f"privacy principle {i}", "citations": [], "id": f"case-{i}:0"}
                for i in range(24)
            ]
            (directory / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in chunks), encoding="utf-8")
            (directory / "config.json").write_text(json.dumps({"model": "unused", "ef_search": 32}), encoding="utf-8")
            index = faiss.IndexHNSWFlat(2, 4, faiss.METRIC_INNER_PRODUCT)
            index.add(np.tile(np.array([[1.0, 0.0]], dtype="float32"), (len(chunks), 1)))
            faiss.write_index(index, str(directory / "dense.faiss"))
            with patch("legal_rag.Encoder", FakeEncoder):
                results = SearchEngine(directory).search("privacy", k=20, before_year=2019, exclude_doc_id="case-0")
            self.assertTrue(results)
            self.assertTrue(all(r["year"] < 2019 and r["doc_id"] != "case-0" for r in results))


if __name__ == "__main__":
    unittest.main()
