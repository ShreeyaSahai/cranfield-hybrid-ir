"""
Separate integration server for the Cranfield Hybrid IR frontend.

This file is an adapter only: it imports and calls the existing src modules.
It does not edit any existing backend file and uses Python's standard library
for HTTP serving, so Flask/FastAPI is not required.
Run from the repository root with: python frontend_server.py
"""
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from src.data_loader import load_cranfield
from src.preprocessing import TextPreprocessor
from src.indexing import InvertedIndex
from src.search_api import IndexAPI
from src.tfidf_bm25 import LexicalRetriever
from src.hybrid_search import HybridRetriever

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"

print("Loading Cranfield collection...")
documents, _, _ = load_cranfield()
documents = documents.reset_index(drop=True)
documents_by_id = {
    str(row["doc_id"]): row for _, row in documents.iterrows()
}

print("Building the existing inverted index...")
preprocessor = TextPreprocessor()
index = InvertedIndex()
index.build(documents, preprocessor)
index_api = IndexAPI(index)

print("Preparing TF-IDF and BM25 retriever...")
lexical_retriever = LexicalRetriever(
    documents=documents,
    preprocessor=preprocessor,
    index_api=index_api,
)

# Load the sentence-transformer model only when semantic/hybrid search is first used.
semantic_retriever = None
hybrid_retriever = None


def get_semantic_retriever():
    global semantic_retriever
    if semantic_retriever is None:
        print("Loading semantic model (first use may download model files)...")
        from src.semantic_search import SemanticRetriever
        semantic_retriever = SemanticRetriever(documents=documents)
    return semantic_retriever


def get_hybrid_retriever():
    global hybrid_retriever
    if hybrid_retriever is None:
        hybrid_retriever = HybridRetriever(
            lexical_retriever=lexical_retriever,
            semantic_retriever=get_semantic_retriever(),
        )
    return hybrid_retriever


class FrontendHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def send_json(self, status_code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            self.send_json(200, {
                "status": "ok",
                "document_count": int(len(documents)),
                "methods": ["tfidf", "bm25", "semantic", "hybrid"],
            })
            return
        if path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/search":
            self.send_json(404, {"error": "Endpoint not found."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                self.send_json(400, {"error": "Request body is missing or too large."})
                return

            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            query = str(payload.get("query", "")).strip()
            method = str(payload.get("method", "hybrid")).lower()
            top_k = int(payload.get("top_k", 10))

            if not query:
                self.send_json(400, {"error": "Please enter a non-empty search query."})
                return
            if method not in {"tfidf", "bm25", "semantic", "hybrid"}:
                self.send_json(400, {"error": "Unknown retrieval method."})
                return
            if not 1 <= top_k <= 50:
                self.send_json(400, {"error": "Results count must be between 1 and 50."})
                return

            if method == "tfidf":
                ranked = lexical_retriever.search_tfidf(query, top_k=top_k)
            elif method == "bm25":
                ranked = lexical_retriever.search_bm25(query, top_k=top_k)
            elif method == "semantic":
                ranked = get_semantic_retriever().semantic_search(query, top_k=top_k)
            else:
                ranked = get_hybrid_retriever().search(query, top_k=top_k)

            results = []
            for doc_id, score in ranked:
                doc_id = str(doc_id)
                row = documents_by_id.get(doc_id)
                if row is None:
                    continue
                full_text = str(row.get("text", "") or "")
                results.append({
                    "doc_id": doc_id,
                    "score": float(score),
                    "title": str(row.get("title", "") or ""),
                    "author": str(row.get("author", "") or ""),
                    "bib": str(row.get("bib", "") or ""),
                    "snippet": full_text[:850] + ("…" if len(full_text) > 850 else ""),
                })

            self.send_json(200, {
                "query": query,
                "method": method,
                "count": len(results),
                "results": results,
            })
        except json.JSONDecodeError:
            self.send_json(400, {"error": "Invalid JSON request."})
        except Exception as exc:
            print(f"Search error: {exc}")
            self.send_json(500, {
                "error": f"Search failed: {type(exc).__name__}: {exc}"
            })


if __name__ == "__main__":
    if not FRONTEND_DIR.is_dir():
        raise SystemExit(f"Frontend folder not found: {FRONTEND_DIR}")
    server = ThreadingHTTPServer(("127.0.0.1", 8000), FrontendHandler)
    print("\nCranfield frontend is ready.")
    print("Open: http://127.0.0.1:8000")
    print("Press Ctrl+C to stop the server.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping frontend server...")
    finally:
        server.server_close()
