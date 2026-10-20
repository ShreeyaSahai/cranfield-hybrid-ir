
import math
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import TextPreprocessor
from src.search_api import IndexAPI


SearchResult = Tuple[str, float]


class LexicalRetriever:
    """
    Classical lexical retrieval for the Cranfield collection.

    Supports:
    1. TF-IDF with cosine similarity
    2. BM25 with document-length normalization

    Both methods return:
        [(document_id, score), ...]
    """

    def __init__(
        self,
        documents,
        preprocessor: TextPreprocessor,
        index_api: IndexAPI,
        k1: float = 1.5,
        b: float = 0.75,
        id_column: str = None,
        text_column: str = None,
    ):
        if documents.empty:
            raise ValueError("The documents collection is empty.")

        if k1 <= 0:
            raise ValueError("k1 must be greater than zero.")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        self.preprocessor = preprocessor
        self.index_api = index_api
        self.k1 = k1
        self.b = b

        # Identify the document ID column.
        id_candidates = ["doc_id", "document_id", "id"]
        if id_column is None:
            id_column = next(
                (c for c in id_candidates if c in documents.columns),
                None,
            )

        # Identify the document text column.
        text_candidates = [
            "text", "content", "abstract", "document", "body"
        ]
        if text_column is None:
            text_column = next(
                (c for c in text_candidates if c in documents.columns),
                None,
            )

        if text_column is None:
            raise ValueError(
                "Cannot identify the text column. "
                "Pass its name using text_column."
            )

        if id_column is not None:
            self.doc_ids = documents[id_column].astype(str).tolist()
        else:
            self.doc_ids = documents.index.astype(str).tolist()

        self.doc_ids = [
            doc_id for doc_id in self.doc_ids
        ]

        # Preprocess all documents using the team's pipeline.
        self.processed_documents = [
            " ".join(preprocessor.preprocess(str(text)))
            for text in documents[text_column].fillna("")
        ]

        # Use already-processed tokens without processing them again.
        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.processed_documents
        )

        self.avg_doc_length = self._calculate_average_length()

    def _calculate_average_length(self):
        """Calculate average indexed document length."""
        doc_ids = self.index_api.get_all_documents()

        if not doc_ids:
            return 0.0

        total_length = sum(
            self.index_api.get_doc_length(doc_id)
            for doc_id in doc_ids
        )

        return total_length / len(doc_ids)

    def search_tfidf(
        self, query: str, top_k: int = 10
    ) -> List[SearchResult]:
        """Rank documents using TF-IDF cosine similarity."""
        if top_k <= 0:
            return []

        query_tokens = self.preprocessor.preprocess(query)
        processed_query = " ".join(query_tokens)

        if not processed_query or not self.doc_ids:
            return []

        query_vector = self.vectorizer.transform([processed_query])
        scores = cosine_similarity(
            query_vector, self.tfidf_matrix
        ).flatten()

        ranked_indices = np.argsort(-scores, kind="stable")[:top_k]

        return [
            (self.doc_ids[i], float(scores[i]))
            for i in ranked_indices
            if scores[i] > 0
        ]

    def search_bm25(
        self, query: str, top_k: int = 10
    ) -> List[SearchResult]:
        """Rank documents using BM25 and the existing inverted index."""
        if top_k <= 0:
            return []

        query_tokens = self.preprocessor.preprocess(query)

        if not query_tokens:
            return []

        total_docs = self.index_api.number_of_documents()
        if total_docs == 0:
            return []

        avg_length = self.avg_doc_length
        scores = {}

        # Repeated query terms contribute repeatedly.
        for term in query_tokens:
            df = self.index_api.get_df(term)

            if df == 0:
                continue

            idf = math.log(
                1 + (total_docs - df + 0.5) / (df + 0.5)
            )

            for doc_id in self.index_api.get_postings(term):
                tf = self.index_api.get_tf(doc_id, term)
                doc_length = self.index_api.get_doc_length(doc_id)

                length_norm = (
                    1 - self.b + self.b * doc_length / avg_length
                    if avg_length > 0
                    else 1.0
                )

                denominator = tf + self.k1 * length_norm

                if denominator == 0:
                    continue

                score = (
                    idf
                    * tf
                    * (self.k1 + 1)
                    / denominator
                )

                scores[doc_id] = scores.get(doc_id, 0.0) + score

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], str(item[0])),
        )

        return [
            (str(doc_id), float(score))
            for doc_id, score in ranked[:top_k]
        ]

    # Convenient aliases for callers.
    def tfidf_search(
        self, query: str, top_k: int = 10
    ) -> List[SearchResult]:
        return self.search_tfidf(query, top_k)

    def bm25_search(
        self, query: str, top_k: int = 10
    ) -> List[SearchResult]:
        return self.search_bm25(query, top_k)
