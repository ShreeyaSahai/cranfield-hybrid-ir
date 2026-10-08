from src.data_loader import load_cranfield
from src.preprocessing import TextPreprocessor
from src.indexing import InvertedIndex
from src.search_api import IndexAPI


def build_test_index():
    documents, queries, qrels = load_cranfield()

    preprocessor = TextPreprocessor()
    index = InvertedIndex()
    index.build(documents, preprocessor)

    return documents, queries, qrels, index


def test_dataset_counts():
    documents, queries, qrels, _ = build_test_index()

    assert len(documents) == 1400
    assert len(queries) == 225
    assert len(qrels) == 1837


def test_preprocessing():
    processor = TextPreprocessor()

    text = "The AERODYNAMICS of Wings!"

    tokens = processor.preprocess(text)

    assert isinstance(tokens, list)
    assert "the" not in tokens
    assert "aerodynam" in tokens
    assert "wing" in tokens


def test_inverted_index():
    _, _, _, index = build_test_index()

    assert len(index.get_all_documents()) == 1400
    assert len(index) == 4292

    assert len(index.get_documents("wing")) == 226


def test_index_term_statistics():
    _, _, _, index = build_test_index()

    assert index.get_document_frequency("wing") == 226
    assert index.get_term_frequency("1", "wing") == 4
    assert index.get_document_length("1") == 84


def test_search_api():
    _, _, _, index = build_test_index()

    api = IndexAPI(index)

    assert api.number_of_documents() == 1400
    assert api.get_df("wing") == 226
    assert api.get_tf("1", "wing") == 4
    assert api.get_doc_length("1") == 84