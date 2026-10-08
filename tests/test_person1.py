from src.data_loader import load_cranfield
from src.preprocessing import TextPreprocessor
from src.indexing import InvertedIndex


def main():

    print("Loading Cranfield dataset...")

    documents, queries, qrels = load_cranfield()

    print(f"Documents: {len(documents)}")
    print(f"Queries: {len(queries)}")
    print(f"Qrels: {len(qrels)}")

    print("\nCreating preprocessor...")

    preprocessor = TextPreprocessor()

    sample_text = documents.iloc[0]["text"]

    processed = preprocessor.preprocess(sample_text)

    print("\nOriginal text:")
    print(sample_text[:500])

    print("\nProcessed tokens:")
    print(processed[:30])

    print("\nBuilding inverted index...")

    index = InvertedIndex()

    index.build(
        documents,
        preprocessor
    )

    print("\nIndex successfully built.")

    print(f"Vocabulary size: {len(index)}")

    # Test some terms
    test_terms = [
        "aircraft",
        "aerodynam",
        "wing",
        "flow"
    ]

    print("\nPosting list examples:")

    for term in test_terms:

        docs = index.get_documents(term)

        print(
            f"{term:15} -> "
            f"{len(docs)} documents"
        )


if __name__ == "__main__":
    main()