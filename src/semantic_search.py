
from typing import List, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


SearchResult = Tuple[str, float]


class SemanticRetriever:
    """Search Cranfield documents using sentence embeddings."""

    def __init__(
        self,
        documents,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ):
        if documents.empty:
            raise ValueError("The documents collection is empty.")

        self.documents = documents.reset_index(drop=True).copy()
        self.doc_ids = self.documents["doc_id"].astype(str).tolist()

        self.document_texts = (
            self.documents["title"].fillna("").astype(str)
            + " "
            + self.documents["text"].fillna("").astype(str)
        ).tolist()

        # Load the pre-trained model.
        self.model = SentenceTransformer(model_name)

        # Encode all documents once during initialization.
        self.document_embeddings = self.model.encode(
            self.document_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def semantic_search(
        self, query: str, top_k: int = 10
    ) -> List[SearchResult]:

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer.")

        if top_k == 0 or not query or not query.strip():
            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        scores = cosine_similarity(
            query_embedding,
            self.document_embeddings,
        ).ravel()

        results = [
            (doc_id, float(score))
            for doc_id, score in zip(self.doc_ids, scores)
        ]

        results.sort(key=lambda item: (-item[1], item[0]))
        return results[:top_k]
