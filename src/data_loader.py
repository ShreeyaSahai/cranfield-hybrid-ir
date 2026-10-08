import ir_datasets
import pandas as pd
from pathlib import Path

def load_cranfield():
    """
    Load the Cranfield Information Retrieval dataset.

    Returns:
        documents: pandas DataFrame
        queries: pandas DataFrame
        qrels: pandas DataFrame
    """

    dataset = ir_datasets.load("cranfield")

    # -------------------------
    # Load documents
    # -------------------------
    documents = []

    for doc in dataset.docs_iter():
        documents.append({
            "doc_id": str(doc.doc_id),
            "title": doc.title or "",
            "author": doc.author or "",
            "bib": doc.bib or "",
            "text": doc.text or ""
        })

    documents = pd.DataFrame(documents)

    # -------------------------
    # Load queries
    # -------------------------
    queries = []

    for query in dataset.queries_iter():
        queries.append({
            "query_id": str(query.query_id),
            "text": query.text
        })

    queries = pd.DataFrame(queries)

    # -------------------------
    # Load relevance judgments
    # -------------------------
    qrels = []

    for qrel in dataset.qrels_iter():
        qrels.append({
            "query_id": str(qrel.query_id),
            "doc_id": str(qrel.doc_id),
            "relevance": int(qrel.relevance),
            "iteration": str(qrel.iteration)
        })

    qrels = pd.DataFrame(qrels)

    return documents, queries, qrels

def save_dataset(documents, queries, qrels):
    """
    Save Cranfield data as CSV files.
    """

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    documents.to_csv(
        data_dir / "documents.csv",
        index=False
    )

    queries.to_csv(
        data_dir / "queries.csv",
        index=False
    )

    qrels.to_csv(
        data_dir / "qrels.csv",
        index=False
    )

    print("Dataset saved successfully.")

if __name__ == "__main__":

    documents, queries, qrels = load_cranfield()

    print("=" * 50)
    print("CRANFIELD DATASET")
    print("=" * 50)

    print(f"Documents : {len(documents)}")
    print(f"Queries   : {len(queries)}")
    print(f"Qrels     : {len(qrels)}")

    save_dataset(documents, queries, qrels)