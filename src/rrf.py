
from typing import Dict, List, Sequence, Tuple


SearchResult = Tuple[str, float]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]],
    top_k: int = 10,
    rrf_k: int = 60,
) -> List[SearchResult]:
    """
    Combine ranked search results using Reciprocal Rank Fusion.

    Args:
        rankings: Ranked result lists, each containing (doc_id, score).
        top_k: Maximum number of fused results to return.
        rrf_k: Constant used in the RRF formula.

    Returns:
        A list of (doc_id, fused_score) pairs, sorted by fused score.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer.")

    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer.")

    if top_k == 0:
        return []

    fused_scores: Dict[str, float] = {}

    for ranking in rankings:
        seen_in_ranking = set()

        for rank, result in enumerate(ranking, start=1):
            doc_id = str(result[0])

            # A document contributes at most once per ranking list.
            if doc_id in seen_in_ranking:
                continue

            seen_in_ranking.add(doc_id)
            fused_scores[doc_id] = (
                fused_scores.get(doc_id, 0.0)
                + 1.0 / (rrf_k + rank)
            )

    fused_results = sorted(
        fused_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return fused_results[:top_k]
