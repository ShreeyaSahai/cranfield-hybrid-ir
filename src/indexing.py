from collections import defaultdict, Counter

class InvertedIndex:

    def __init__(self):
        # term -> set of document IDs
        self.index = defaultdict(set)

        # document ID -> number of tokens
        self.doc_lengths = {}

        # document ID -> term frequency Counter
        self.term_frequencies = {}

    def add_document(self, doc_id, tokens):
        """
        Add a single document to the inverted index.
        """

        # Store document length
        self.doc_lengths[doc_id] = len(tokens)

        # Count terms in the document
        term_counts = Counter(tokens)

        self.term_frequencies[doc_id] = term_counts

        # Add document ID to posting list for every term
        for term in term_counts:
            self.index[term].add(doc_id)

    def build(self, documents, preprocessor):
        """
        Build the inverted index from a collection of documents.
        """

        for _, document in documents.iterrows():

            # Combine title and abstract
            text = (
                document["title"]
                + " "
                + document["text"]
            )

            tokens = preprocessor.preprocess(text)

            self.add_document(
                document["doc_id"],
                tokens
            )

    def get_documents(self, term):
        """
        Return documents containing a term.
        """

        return self.index.get(term, set())

    def get_vocabulary(self):
        """
        Return all indexed terms.
        """

        return set(self.index.keys())

    def __len__(self):
        """
        Number of unique terms in vocabulary.
        """

        return len(self.index)

    def get_term_frequency(self, doc_id, term):
        """
        Return the number of times a term occurs in a document.
        """

        return self.term_frequencies.get(
            doc_id,
            {}
        ).get(term, 0)

    def get_document_frequency(self, term):
        """
        Return the number of documents containing a term.
        """

        return len(
            self.index.get(term, set())
        )

    def get_document_length(self, doc_id):
        """
        Return the number of processed tokens in a document.
        """

        return self.doc_lengths.get(doc_id, 0)

    def get_all_documents(self):
        """
        Return all document IDs in the index.
        """

        return set(self.doc_lengths.keys())