
import numpy as np
import pandas as pd
import pytest

from src.semantic_search import SemanticRetriever


class FakeModel:
    """A lightweight model replacement for unit tests."""

    def encode(self, texts, **kwargs):
        embeddings = {
            "aircraft wing ": [1.0, 0.0],
            "engine design ": [0.0, 1.0],
            "aircraft design ": [0.9, 0.1],
            "aircraft design": [0.9, 0.1],
            "engine design": [0.0, 1.0],
        }
        return np.array([embeddings[text] for text in texts])


@pytest.fixture
def retriever():
    documents = pd.DataFrame(
        {
            "doc_id": ["1", "2", "3"],
            "title": ["aircraft wing", "engine design", "aircraft design"],
            "text": ["", "", ""],
        }
    )

    instance = SemanticRetriever.__new__(SemanticRetriever)
    instance.documents = documents
    instance.doc_ids = documents["doc_id"].astype(str).tolist()
    instance.document_texts = (
        documents["title"] + " " + documents["text"]
    ).tolist()
    instance.model = FakeModel()
    instance.document_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
        ]
    )
    return instance


def test_semantic_search_returns_ranked_results(retriever):
    results = retriever.semantic_search("aircraft design", top_k=2)

    assert len(results) == 2
    assert results[0][0] == "3"
    assert results[0][1] > results[1][1]


def test_semantic_search_empty_query_returns_empty_list(retriever):
    assert retriever.semantic_search("") == []
    assert retriever.semantic_search("   ") == []


def test_semantic_search_zero_top_k_returns_empty_list(retriever):
    assert retriever.semantic_search("aircraft design", top_k=0) == []


@pytest.mark.parametrize("top_k", [-1, 1.5, True, "2"])
def test_semantic_search_rejects_invalid_top_k(retriever, top_k):
    with pytest.raises(ValueError):
        retriever.semantic_search("aircraft design", top_k=top_k)


def test_semantic_search_unknown_query_embedding(retriever):
    # The fake model should only accept the known query below.
    results = retriever.semantic_search("engine design", top_k=1)

    assert results[0][0] == "2"
