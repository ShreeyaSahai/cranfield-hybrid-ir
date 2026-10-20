
import pytest

from src.hybrid_search import HybridRetriever


class FakeLexicalRetriever:
    def search_bm25(self, query, top_k=10):
        return [("1", 5.0), ("2", 3.0)][:top_k]

    def search_tfidf(self, query, top_k=10):
        return [("2", 0.9), ("3", 0.7)][:top_k]


class FakeSemanticRetriever:
    def semantic_search(self, query, top_k=10):
        return [("3", 0.8), ("1", 0.6)][:top_k]


@pytest.fixture
def retriever():
    return HybridRetriever(
        lexical_retriever=FakeLexicalRetriever(),
        semantic_retriever=FakeSemanticRetriever(),
    )


def test_hybrid_search_combines_all_three_rankings(retriever):
    results = retriever.search("aircraft design", top_k=3)

    ids = [doc_id for doc_id, score in results]

    assert len(results) == 3
    # Each document appears in multiple retrieval methods.
    assert set(ids) == {"1", "2", "3"}


def test_hybrid_search_returns_fused_scores(retriever):
    results = retriever.search("aircraft design", top_k=3)

    # RRF scores are positive and lower than 1 for the default constant.
    assert all(score > 0 for _, score in results)
    assert all(score < 1 for _, score in results)


def test_empty_query_returns_empty_results(retriever):
    assert retriever.search("") == []
    assert retriever.search("   ") == []


def test_zero_top_k_returns_empty_results(retriever):
    assert retriever.search("aircraft design", top_k=0) == []


def test_zero_candidate_k_returns_empty_results(retriever):
    assert retriever.search("aircraft design", candidate_k=0) == []


@pytest.mark.parametrize("top_k", [-1, 1.5, True, "2"])
def test_invalid_top_k_is_rejected(retriever, top_k):
    with pytest.raises(ValueError):
        retriever.search("aircraft design", top_k=top_k)


@pytest.mark.parametrize("candidate_k", [-1, 1.5, True, "2"])
def test_invalid_candidate_k_is_rejected(retriever, candidate_k):
    with pytest.raises(ValueError):
        retriever.search("aircraft design", candidate_k=candidate_k)
