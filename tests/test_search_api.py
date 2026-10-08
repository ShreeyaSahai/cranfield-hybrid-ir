from src.data_loader import load_cranfield
from src.preprocessing import TextPreprocessor
from src.indexing import InvertedIndex
from src.search_api import IndexAPI


def main():
    print("Loading dataset...")
    documents, _, _ = load_cranfield()

    print("Building index...")
    preprocessor = TextPreprocessor()
    index = InvertedIndex()
    index.build(documents, preprocessor)

    api = IndexAPI(index)

    print("\nTesting Index API...")

    term = "wing"

    print(f"Documents containing '{term}':")
    print(len(api.get_postings(term)))

    print(f"\nDF('{term}'):")
    print(api.get_df(term))

    doc_id = "1"

    print(f"\nTF('{term}', document {doc_id}):")
    print(api.get_tf(doc_id, term))

    print(f"\nDocument length ({doc_id}):")
    print(api.get_doc_length(doc_id))

    print("\nTotal documents:")
    print(api.number_of_documents())

    print("\nVocabulary size:")
    print(len(api.get_vocabulary()))

    print("\nAPI test successful!")


if __name__ == "__main__":
    main()