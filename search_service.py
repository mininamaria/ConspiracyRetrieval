import time
import logging

from corpus import (
    CorpusProcessor,
    load_corpus_data,
    save_preprocessed_cache,
    try_load_preprocessed_cache,
)
from search_engine import SearchEngine


ALLOWED_MODELS = ("bm25", "word2vec", "fasttext")


logger = logging.getLogger(__name__)


def _build_selected_model(engine, model_name):
    logger.info("Starting phase 5: build selected model (%s)", model_name)
    if model_name == "bm25":
        engine.build_bm25()
        logger.info("Finished phase 5: build selected model (%s)", model_name)
        return
    if model_name == "word2vec":
        engine.build_word2vec()
        logger.info("Finished phase 5: build selected model (%s)", model_name)
        return
    if model_name == "fasttext":
        engine.build_fasttext()
        logger.info("Finished phase 5: build selected model (%s)", model_name)
        return
    raise ValueError(f"Unknown model: {model_name}")


def run_search(query, model="bm25", limit=1197, top_k=5, verbose=False, data_path="data/reddit_ct.csv"):
    # общий сценарий поиска для CLI и веба
    logger.info("Starting phase 0: validate search input")
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
    logger.info("Finished phase 0: validate search input")

    logger.info("Starting phase 1: initialize processor")
    processor = CorpusProcessor()
    logger.info("Finished phase 1: initialize processor")

    logger.info("Starting phase 2: load and preprocess corpus")
    load_start = time.perf_counter()
    rows_loaded = 0
    cache_payload = try_load_preprocessed_cache(data_path=data_path, limit=limit)

    if cache_payload is not None:
        documents = cache_payload["documents"]
        processor.unique_words = set(cache_payload["unique_words"])
        rows_loaded = cache_payload["rows_loaded"]
    else:
        data = load_corpus_data(data_path, limit=limit, verbose=False)
        rows_loaded = len(data)
        documents = processor.create_documents(data, "title")
        save_preprocessed_cache(
            data_path=data_path,
            limit=limit,
            documents=documents,
            unique_words=processor.unique_words,
            rows_loaded=rows_loaded,
        )

    load_time = time.perf_counter() - load_start
    logger.info("Finished phase 2: load and preprocess corpus")

    if not documents:
        raise ValueError("No documents available after preprocessing.")

    logger.info("Starting phase 3: preprocess query")
    query_tokens = processor.preprocess_text(query)
    if not query_tokens:
        raise ValueError("Query is empty after preprocessing. Try different words.")
    logger.info("Finished phase 3: preprocess query")

    logger.info("Starting phase 4: initialize search engine")
    engine = SearchEngine(documents)
    logger.info("Finished phase 4: initialize search engine")

    build_start = time.perf_counter()
    _build_selected_model(engine, model)
    build_time = time.perf_counter() - build_start

    logger.info("Starting phase 6: execute search")
    search_start = time.perf_counter()
    results = engine.search(model, query_tokens, top_k=top_k)
    search_time = time.perf_counter() - search_start
    logger.info("Finished phase 6: execute search")

    stats = None
    if verbose:
        avg_tokens = sum(len(doc.tokens) for doc in documents) / len(documents) if documents else 0.0
        stats = {
            "model": model,
            "rows_loaded": rows_loaded,
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