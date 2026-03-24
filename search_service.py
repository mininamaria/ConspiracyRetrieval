import time

from corpus import CorpusProcessor, load_corpus_data
from search_engine import SearchEngine


ALLOWED_MODELS = ("bm25", "word2vec", "fasttext")


def _build_selected_model(engine, model_name):
    if model_name == "bm25":
        engine.build_bm25()
        return
    if model_name == "word2vec":
        engine.build_word2vec()
        return
    if model_name == "fasttext":
        engine.build_fasttext()
        return
    raise ValueError(f"Unknown model: {model_name}")


def run_search(query, model="bm25", limit=1197, top_k=5, verbose=False, data_path="data/reddit_ct.csv"):
    # общий сценарий поиска для CLI и веба
    model = (model or "").lower().strip()
    query = (query or "").strip()

    if model not in ALLOWED_MODELS:
        raise ValueError(f"Unknown model: {model}. Allowed: {', '.join(ALLOWED_MODELS)}")
    if not query:
        raise ValueError("Query must not be empty")
    if limit < 1:
        raise ValueError("Data limit must be >= 1")
    if top_k < 1:
        raise ValueError("Number of matching items must be >= 1")

    processor = CorpusProcessor()

    load_start = time.perf_counter()
    data = load_corpus_data(data_path, limit=limit, verbose=False)
    documents = processor.create_documents(data, "title")
    load_time = time.perf_counter() - load_start

    if not documents:
        raise ValueError("No documents available after preprocessing.")

    query_tokens = processor.preprocess_text(query)
    if not query_tokens:
        raise ValueError("Query is empty after preprocessing. Try different words.")

    engine = SearchEngine(documents)

    build_start = time.perf_counter()
    _build_selected_model(engine, model)
    build_time = time.perf_counter() - build_start

    search_start = time.perf_counter()
    results = engine.search(model, query_tokens, top_k=top_k)
    search_time = time.perf_counter() - search_start

    stats = None
    if verbose:
        avg_tokens = sum(len(doc.tokens) for doc in documents) / len(documents) if documents else 0.0
        stats = {
            "model": model,
            "rows_loaded": len(data),
            "documents_created": len(documents),
            "unique_words": len(processor.unique_words),
            "avg_tokens_doc": avg_tokens,
            "query_tokens": len(query_tokens),
            "load_time": load_time,
            "build_time": build_time,
            "search_time": search_time,
        }

    return {
        "results": results,
        "search_time": search_time,
        "stats": stats,
    }