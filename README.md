# Structure-Aware Legal Precedent Retrieval

A reference implementation of **Structure-Aware Retrieval for Legal Precedent
Discovery at Scale**. It parses judgment headings, chunks sections, builds a
Legal-BERT/FAISS index, combines dense and BM25 scores, expands cited cases,
and reranks candidates with a trained cross-encoder.

## Install

```bash
python -m pip install -r requirements.txt
```

## Prepare the corpus

Provide one judgment per line in `judgments.jsonl`. Citation IDs must match a
judgment `id` or one of its `aliases`.

```json
{"id":"case-001","year":2018,"text":"FACTS\n...\nHELD\n...","aliases":["AIR 2018 SC 123"],"citations":["case-000"]}
```

```bash
python legal_rag.py build judgments.jsonl index
python legal_rag.py search index "land acquisition compensation" --top-k 10
```

The default encoder is `nlpaueb/legal-bert-base-uncased`. The section parser
recognizes common English headings and uses `FULL_TEXT` when none are found.
Chunks use 512 words with a 50-word overlap; the model input is truncated to
512 tokens. Citation extraction and alias resolution for raw PDFs must be done
during corpus preparation.

## Train the reranker

Prepare `train_pairs.jsonl` and `validation_pairs.jsonl` using distinct
`query_id` values. Each line is a query/passage pair with a binary relevance
label. Both files need positive and negative examples.

```json
{"query_id":"q-001","doc_id":"case-000","query":"land acquisition compensation","passage":"The holding of the candidate case...","label":1}
```

```bash
python train_reranker.py --train train_pairs.jsonl --validation validation_pairs.jsonl --output example-checkpoint
```

The best validation checkpoint is saved in `example-checkpoint/`. Supply that
directory to either inference command:

```bash
python legal_rag.py search index "land acquisition compensation" --top-k 20 --reranker example-checkpoint
python infer.py index queries.jsonl runs.jsonl --reranker example-checkpoint --top-k 20
```

`queries.jsonl` contains `query_id`, `query`, and optional `before_year` and
`exclude_doc_id` fields. `infer.py` writes ordered case results for every query
and loads the encoder and reranker once for the full batch:

```json
{"query_id":"q-001","query":"land acquisition compensation","before_year":2018,"exclude_doc_id":"case-001"}
```

## Evaluate a labeled run

Provide a separate `qrels.jsonl` with one `query_id` and a mapping of relevant
case IDs to integer grades per line. A grade above zero is relevant; grades
also determine nDCG gain. Keep the test queries distinct from training and
validation queries.

```json
{"query_id":"q-001","relevant":{"case-000":2,"case-002":1}}
```

```bash
python evaluate.py runs.jsonl qrels.jsonl --per-query per_query.jsonl
```

The evaluator reports P@5, MRR@20, nDCG@5, Recall@5/10/20, and MAP@20,
along with per-query output. The `@20` suffix makes the run cutoff explicit.
Use `--top-k` and `--cutoff` together when evaluating a deeper run.
