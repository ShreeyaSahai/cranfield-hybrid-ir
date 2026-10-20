
# TF-IDF and BM25 Retrieval

## Overview

This module implements two lexical information retrieval methods
for the Cranfield dataset:

- **TF-IDF:** Uses term weighting and cosine similarity to rank documents.
- **BM25:** Uses term frequency, inverse document frequency, and
  document-length normalization.

Both methods return a list of `(document_id, score)` tuples.

## Implementation

The implementation is available in `src/tfidf_bm25.py`.

It reuses the project's `TextPreprocessor` and `IndexAPI`.

## Run the Demo

Activate the project's virtual environment and run:

```bash
python -m tests.demo_person2
```

Enter a search query when prompted.

## Run Tests

Run the retrieval module tests:

```bash
python -m pytest tests/test_tfidf_bm25.py -v
```

Run the full project test suite:

```bash
python -m pytest -v
```

## Current Test Coverage

Tests cover TF-IDF retrieval, BM25 retrieval, empty queries,
unknown queries, result limits, invalid BM25 parameters,
and descending score order.
