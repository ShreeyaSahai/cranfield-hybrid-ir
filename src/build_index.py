from pathlib import Path

from src.data_loader import load_cranfield
from src.preprocessing import TextPreprocessor
from src.indexing import InvertedIndex


def main():
    print("Loading Cranfield dataset...")
    documents, _, _ = load_cranfield()

    print("Creating preprocessor...")
    preprocessor = TextPreprocessor()

    print("Building inverted index...")
    index = InvertedIndex()
    index.build(documents, preprocessor)

    Path("data").mkdir(exist_ok=True)

    index_path = "data/inverted_index.pkl"
    index.save(index_path)

    print("\nIndex saved successfully.")
    print(f"File: {index_path}")
    print(f"Documents indexed: {len(index.get_all_documents())}")
    print(f"Vocabulary size: {len(index)}")


if __name__ == "__main__":
    main()