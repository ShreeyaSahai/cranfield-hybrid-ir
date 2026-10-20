import pytest

from src.rrf import reciprocal_rank_fusion


def test_fuses_rankings_and_prioritizes_consensus():
    ranking_a = [("1", 0.9), ("2", 0.8), ("3", 0.7)]
    ranking_b = [("2", 0.95), ("1", 0.6), ("4", 0.5)]

    results = reciprocal_rank_fusion([ranking_a, ranking_b])

    # Documents ranked highly by both systems should rank well.
    assert results[0][0] == "1"
    assert results[1][0] == "2"
    assert len(results) == 4


def test_returns_only_top_k_results():
    ranking = [("1", 0.9), ("2", 0.8), ("3", 0.7)]

    results = reciprocal_rank_fusion([ranking], top_k=2)

    assert len(results) == 2
    assert [doc_id for doc_id, _ in results] == ["1", "2"]


def test_empty_rankings_return_empty_results():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_zero_top_k_returns_empty_results():
    assert reciprocal_rank_fusion([[("1", 0.9)]], top_k=0) == []


@pytest.mark.parametrize("top_k", [-1, 1.5, True, "2"])
def test_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], top_k=top_k)


@pytest.mark.parametrize("rrf_k", [0, -1, 1.5, True, "60"])
def test_rejects_invalid_rrf_k(rrf_k):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], rrf_k=rrf_k)


def test_duplicate_document_in_one_ranking_counts_once():
    ranking = [("1", 0.9), ("1", 0.8), ("2", 0.7)]

    results = reciprocal_rank_fusion([ranking])

    scores = dict(results)

    # The repeated document contributes only at its first rank.
    assert scores["1"] == pytest.approx(1 / 61)
    assert scores["2"] == pytest.approx(1 / 63)