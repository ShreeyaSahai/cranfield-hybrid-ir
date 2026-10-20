
from typing import List, Tuple

from src.rrf import reciprocal_rank_fusion


SearchResult = Tuple[str, float]


class HybridRetriever:
    """Combine lexical and semantic search using Reciprocal Rank Fusion."""

    def __init__(self, lexical_retriever, semantic_retriever):
        self.lexical_retriever = lexical_retriever
        self.semantic_retriever = semantic_retriever

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 100,
        rrf_k: int = 60,
    ) -> List[SearchResult]:
        """Fuse BM25, TF-IDF, and semantic rankings."""

        if not query or not query.strip():
            return []

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k < 0
        ):
            raise ValueError("top_k must be a non-negative integer.")

        if (
            isinstance(candidate_k, bool)
            or not isinstance(candidate_k, int)
            or candidate_k < 0
        ):
            raise ValueError("candidate_k must be a non-negative integer.")

        if top_k == 0 or candidate_k == 0:
            return []

        bm25_results = self.lexical_retriever.search_bm25(
            query, top_k=candidate_k
        )
        tfidf_results = self.lexical_retriever.search_tfidf(
            query, top_k=candidate_k
        )
        semantic_results = self.semantic_retriever.semantic_search(
            query, top_k=candidate_k
        )

        return reciprocal_rank_fusion(
            [bm25_results, tfidf_results, semantic_results],
            top_k=top_k,
            rrf_k=rrf_k,
        )
