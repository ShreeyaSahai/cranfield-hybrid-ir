from src.indexing import InvertedIndex


class IndexAPI:
    """
    Public interface for accessing the inverted index.

    Member 2 can use this class for TF-IDF and BM25
    without directly modifying the indexing implementation.
    """

    def __init__(self, index):
        self.index = index

    def get_postings(self, term):
        """Return documents containing the term."""
        return self.index.get_documents(term)

    def get_tf(self, doc_id, term):
        """Return term frequency of a term in a document."""
        return self.index.get_term_frequency(doc_id, term)

    def get_df(self, term):
        """Return document frequency of a term."""
        return self.index.get_document_frequency(term)

    def get_doc_length(self, doc_id):
        """Return number of indexed terms in a document."""
        return self.index.get_document_length(doc_id)

    def get_all_documents(self):
        """Return all indexed document IDs."""
        return self.index.get_all_documents()

    def get_vocabulary(self):
        """Return the complete vocabulary."""
        return self.index.get_vocabulary()

    def number_of_documents(self):
        """Return total number of indexed documents."""
        return len(self.index.get_all_documents())