
from pathlib import Path

import pandas as pd

from src.data_loader import load_cranfield
from src.preprocessing import TextPreprocessor
from src.indexing import InvertedIndex
from src.search_api import IndexAPI
from src.tfidf_bm25 import LexicalRetriever
from src.semantic_search import SemanticRetriever
from src.hybrid_search import HybridRetriever


def dcg(relevances):
    """Discounted cumulative gain using graded relevance."""
    import math

    return sum(
        (2 ** max(0, rel) - 1) / math.log2(rank + 1)
        for rank, rel in enumerate(relevances, start=1)
    )


def evaluate_query(ranked_ids, qrel_map, total_relevant):
    """Calculate P@10, R@100, AP, and nDCG@10 for one query."""
    ranked_ids = [str(doc_id) for doc_id in ranked_ids]

    # Unknown/unjudged documents are not counted as relevant.
    binary = [int(qrel_map.get(doc_id, 0) > 0) for doc_id in ranked_ids]
    graded = [max(0, qrel_map.get(doc_id, 0)) for doc_id in ranked_ids]

    p10 = sum(binary[:10]) / 10
    r100 = (
        sum(binary[:100]) / total_relevant
        if total_relevant
        else 0.0
    )

    hits = 0
    precision_sum = 0.0
    for rank, is_relevant in enumerate(binary, start=1):
        if is_relevant:
            hits += 1
            precision_sum += hits / rank

    average_precision = (
        precision_sum / total_relevant if total_relevant else 0.0
    )

    ideal_grades = sorted(
        [max(0, relevance) for relevance in qrel_map.values()],
        reverse=True,
    )[:10]
    ideal_dcg = dcg(ideal_grades)
    ndcg10 = dcg(graded[:10]) / ideal_dcg if ideal_dcg else 0.0

    return {
        "Precision@10": p10,
        "Recall@100": r100,
        "MAP_component": average_precision,
        "nDCG@10": ndcg10,
    }


def main():
    print("Loading Cranfield dataset...")
    documents, queries, qrels = load_cranfield()

    # -1 means no interest; 1–4 are graded relevant judgments.
    qrel_maps = {}
    for row in qrels.itertuples(index=False):
        qrel_maps.setdefault(str(row.query_id), {})[
            str(row.doc_id)
        ] = int(row.relevance)

    preprocessor = TextPreprocessor()
    inverted_index = InvertedIndex()
    inverted_index.build(documents, preprocessor)
    index_api = IndexAPI(inverted_index)

    lexical = LexicalRetriever(
        documents=documents,
        preprocessor=preprocessor,
        index_api=index_api,
    )

    print("Loading Sentence Transformer and encoding documents...")
    semantic = SemanticRetriever(documents=documents)
    hybrid = HybridRetriever(lexical, semantic)

    systems = {
        "TF-IDF": lambda query: lexical.search_tfidf(
            query, top_k=len(documents)
        ),
        "BM25": lambda query: lexical.search_bm25(
            query, top_k=len(documents)
        ),
        "Sentence Transformer": lambda query: semantic.semantic_search(
            query, top_k=len(documents)
        ),
        "Hybrid RRF": lambda query: hybrid.search(
            query,
            top_k=len(documents),
            candidate_k=len(documents),
            rrf_k=60,
        ),
    }

    per_query_rows = []
    total_queries = len(queries)

    for query_number, row in enumerate(
        queries.itertuples(index=False), start=1
    ):
        query_id = str(row.query_id)
        query_text = row.text
        judgments = qrel_maps.get(query_id, {})
        total_relevant = sum(
            1 for relevance in judgments.values() if relevance > 0
        )

        for system_name, search in systems.items():
            results = search(query_text)
            ranked_ids = [doc_id for doc_id, _score in results]

            metrics = evaluate_query(
                ranked_ids, judgments, total_relevant
            )
            per_query_rows.append({
                "query_id": query_id,
                "system": system_name,
                **metrics,
            })

        if query_number % 25 == 0 or query_number == total_queries:
            print(f"Evaluated {query_number}/{total_queries} queries")

    per_query = pd.DataFrame(per_query_rows)

    summary = (
        per_query.groupby("system", sort=False)
        .agg(
            **{
                "Precision@10": ("Precision@10", "mean"),
                "Recall@100": ("Recall@100", "mean"),
                "MAP": ("MAP_component", "mean"),
                "nDCG@10": ("nDCG@10", "mean"),
            }
        )
        .reset_index()
    )

    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)

    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    per_query.to_csv(output_dir / "per_query_metrics.csv", index=False)

    # Precision-recall curve data, aggregated across all queries.
    from collections import defaultdict

    pr_totals = defaultdict(lambda: {"relevant": 0, "retrieved": 0})
    pr_rows = []

    for query_row in queries.itertuples(index=False):
        query_id = str(query_row.query_id)
        judgments = qrel_maps.get(query_id, {})
        total_relevant = sum(
            1 for relevance in judgments.values() if relevance > 0
        )

        if total_relevant == 0:
            continue

        for system_name, search in systems.items():
            results = search(query_row.text)
            ranked_ids = [str(doc_id) for doc_id, _score in results]

            relevant_found = 0
            for rank, doc_id in enumerate(ranked_ids, start=1):
                if judgments.get(doc_id, 0) > 0:
                    relevant_found += 1

                precision = relevant_found / rank
                recall = relevant_found / total_relevant

                pr_rows.append({
                    "system": system_name,
                    "query_id": query_id,
                    "rank": rank,
                    "precision": precision,
                    "recall": recall,
                })

    pd.DataFrame(pr_rows).to_csv(
        output_dir / "precision_recall_curve.csv", index=False
    )

    print("\n=== Evaluation Summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
