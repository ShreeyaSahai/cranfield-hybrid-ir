
import pandas as pd
import pytest
from collections import Counter, defaultdict

from src.preprocessing import TextPreprocessor
from src.tfidf_bm25 import LexicalRetriever


class FakeIndexAPI:
    def __init__(self, documents, preprocessor):
        self.frequencies = {}
        self.lengths = {}
        self.postings = defaultdict(list)

        for _, row in documents.iterrows():
            doc_id = str(row["doc_id"])
            tokens = preprocessor.preprocess(row["text"])
            counts = Counter(tokens)

            self.frequencies[doc_id] = counts
            self.lengths[doc_id] = len(tokens)

            for term in counts:
                self.postings[term].append(doc_id)

    def get_postings(self, term):
        return self.postings.get(term, [])

    def get_tf(self, doc_id, term):
        return self.frequencies[str(doc_id)].get(term, 0)

    def get_df(self, term):
        return len(self.get_postings(term))

    def get_doc_length(self, doc_id):
        return self.lengths[str(doc_id)]

    def get_all_documents(self):
        return list(self.frequencies.keys())

    def get_vocabulary(self):
        return list(self.postings.keys())

    def number_of_documents(self):
        return len(self.frequencies)


@pytest.fixture
def retriever():
    documents = pd.DataFrame([
        {"doc_id": "1", "text": "aerodynamic characteristics of aircraft wings"},
        {"doc_id": "2", "text": "aircraft wing design and aerodynamic performance"},
        {"doc_id": "3", "text": "heat transfer in high speed aircraft engines"},
    ])

    preprocessor = TextPreprocessor()
    api = FakeIndexAPI(documents, preprocessor)

    return LexicalRetriever(
        documents=documents,
        preprocessor=preprocessor,
        index_api=api,
    )


def test_tfidf_search(retriever):
    results = retriever.tfidf_search("aerodynamic wings")
    assert results
    assert results[0][0] == "1"
    assert all(score > 0 for _, score in results)


def test_bm25_search(retriever):
    results = retriever.bm25_search("aerodynamic wings")
    assert results
    assert results[0][0] == "1"
    assert all(score > 0 for _, score in results)


def test_empty_queries(retriever):
    assert retriever.tfidf_search("") == []
    assert retriever.bm25_search("") == []


def test_unknown_query(retriever):
    assert retriever.tfidf_search("quasar xyzzy") == []
    assert retriever.bm25_search("quasar xyzzy") == []


def test_top_k_limit(retriever):
    assert len(retriever.tfidf_search("aircraft", top_k=1)) <= 1
    assert len(retriever.bm25_search("aircraft", top_k=1)) <= 1


def test_invalid_top_k(retriever):
    assert retriever.tfidf_search("aircraft", top_k=0) == []
    assert retriever.bm25_search("aircraft", top_k=0) == []


def test_invalid_k1(retriever):
    documents = pd.DataFrame([
        {"doc_id": "1", "text": "aerodynamic aircraft wings"},
    ])

    with pytest.raises(ValueError):
        LexicalRetriever(
            documents=documents,
            preprocessor=retriever.preprocessor,
            index_api=retriever.index_api,
            k1=0,
        )


def test_invalid_b(retriever):
    documents = pd.DataFrame([
        {"doc_id": "1", "text": "aerodynamic aircraft wings"},
    ])

    with pytest.raises(ValueError):
        LexicalRetriever(
            documents=documents,
            preprocessor=retriever.preprocessor,
            index_api=retriever.index_api,
            b=1.5,
        )

def test_results_sorted_by_score(retriever):
    for search in [
        retriever.tfidf_search,
        retriever.bm25_search,
    ]:
        results = search("aerodynamic aircraft wings", top_k=3)
        scores = [score for _, score in results]

        assert scores == sorted(scores, reverse=True)

def test_whitespace_query(retriever):
    assert retriever.tfidf_search("   ") == []
    assert retriever.bm25_search("   ") == []


def test_negative_top_k(retriever):
    assert retriever.tfidf_search("aircraft", top_k=-1) == []
    assert retriever.bm25_search("aircraft", top_k=-1) == []

def test_documents_with_no_searchable_terms():
    documents = pd.DataFrame([
        {"doc_id": "1", "text": "the and is"},
        {"doc_id": "2", "text": "a an the"},
    ])

    preprocessor = TextPreprocessor()
    api = FakeIndexAPI(documents, preprocessor)

    retriever = LexicalRetriever(
        documents=documents,
        preprocessor=preprocessor,
        index_api=api,
    )

    assert retriever.tfidf_search("aircraft") == []

def test_retriever_with_real_inverted_index():
    from src.indexing import InvertedIndex
    from src.search_api import IndexAPI

    
    documents = pd.DataFrame([
    {
        "doc_id": "1",
        "title": "Aerodynamic wings",
        "author": "",
        "bib": "",
        "text": "aerodynamic aircraft wings",
    },
    {
        "doc_id": "2",
        "title": "Aircraft engine",
        "author": "",
        "bib": "",
        "text": "aircraft engine design",
    },
])


    preprocessor = TextPreprocessor()
    index = InvertedIndex()
    index.build(documents, preprocessor)

    retriever = LexicalRetriever(
        documents=documents,
        preprocessor=preprocessor,
        index_api=IndexAPI(index),
    )

    assert retriever.tfidf_search("aerodynamic wings")
    assert retriever.bm25_search("aerodynamic wings")
