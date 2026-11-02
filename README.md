# Structure-Aware Legal Precedent Retrieval

A small reference implementation of the paper's retrieval approach: heading-aware
chunking, Legal-BERT embeddings, BM25/dense fusion, citation expansion, and an
optional cross-encoder reranker. It can be adapted to another judgment collection
and evaluation protocol.

The paper is **Structure-Aware Retrieval for Legal Precedent Discovery at Scale**.
The manuscript is maintained separately. Its numerical results came from a
separate expert run and are not produced by this repository. No tests,
benchmarks, or relevance judgments are bundled.

## Input

One JSON object per line, with a unique ID, year, full text, and optional citations.
Citation identifiers should match case IDs or an alias in the input.

    {"id":"case-001","year":2018,"text":"FACTS\n...\nHELD\n...","aliases":["AIR 2018 SC 123"],"citations":["case-000"]}

## Use

    python -m pip install -r requirements.txt
    python legal_rag.py build judgments.jsonl index
    python legal_rag.py search index "land acquisition compensation" --top-k 10
    python legal_rag.py search index "right to privacy" --before-year 2020 --reranker YOUR_CHECKPOINT

The default encoder is nlpaueb/legal-bert-base-uncased with explicit masked mean
pooling and normalized vectors. Chunks use 512 **words** with 50-word overlap;
the encoder truncates to 512 model tokens. Adjust these choices for your corpus.
Section extraction recognizes common English headings and falls back to FULL_TEXT.
The optional citations field is an already resolved graph edge list. Supply a
jurisdiction-specific citation extractor and aliases if starting from raw PDFs.

The reranker is optional because a trained cross-encoder checkpoint is not included.
Without one, the code returns the weighted first-stage ranking. The paper's reported
scores and statistical claims should not be inferred from this reference code.

Use your own labeled query set and metrics to evaluate this code. The published
paper's scores should be cited only from the final expert-verified manuscript.
