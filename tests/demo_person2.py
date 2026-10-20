
from src.data_loader import load_cranfield
from src.preprocessing import TextPreprocessor
from src.indexing import InvertedIndex
from src.search_api import IndexAPI
from src.tfidf_bm25 import LexicalRetriever


def main():
    documents, _, _ = load_cranfield()

    preprocessor = TextPreprocessor()

    print("Building inverted index...")
    index = InvertedIndex()
    index.build(documents, preprocessor)

    api = IndexAPI(index)

    print("Preparing TF-IDF and BM25...")
    retriever = LexicalRetriever(
        documents=documents,
        preprocessor=preprocessor,
        index_api=api,
    )

    query = input("\nEnter a search query: ").strip()

    if not query:
        print("Please enter a non-empty query.")
        return

    for method, search in [
        ("TF-IDF", retriever.tfidf_search),
        ("BM25", retriever.bm25_search),
    ]:
        print(f"\n{'=' * 55}")
        print(f"{method} RESULTS")
        print(f"{'=' * 55}")

        results = search(query, top_k=10)

        if not results:
            print("No matching documents found.")
            continue

        for rank, (doc_id, score) in enumerate(results, start=1):
            matching_docs = documents[
                documents["doc_id"].astype(str) == str(doc_id)
            ]

            title = ""
            if not matching_docs.empty:
                title = str(matching_docs.iloc[0]["title"])

            print(f"\nRank: {rank}")
            print(f"Document ID: {doc_id}")
            print(f"Score: {score:.4f}")
            if title:
                print(f"Title: {title}")


if __name__ == "__main__":
    main()
