import argparse
import logging
import sys
from pathlib import Path

from flask import Flask, render_template, request


# Добавляем корень проекта в path, чтобы переиспользовать общий сервис поиска.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from search_service import ALLOWED_MODELS, run_search


app = Flask(__name__)
logger = logging.getLogger(__name__)
ENABLE_LOGS = False


def configure_logging():
    if ENABLE_LOGS:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    else:
        logging.disable(logging.CRITICAL)


def _to_int(value, default):
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _base_form_data():
    return {
        "query": "",
        "model": "bm25",
        "limit": 1197,
        "top_k": 5,
        "verbose": False,
    }


@app.get("/")
def index():
    return render_template(
        "index.html",
        models=ALLOWED_MODELS,
        form=_base_form_data(),
        results=[],
        search_time=None,
        stats=None,
        error=None,
    )


@app.post("/search")
def search():
    logger.info("Starting phase W1: parse web form")
    form = {
        "query": (request.form.get("query") or "").strip(),
        "model": (request.form.get("model") or "bm25").strip().lower(),
        "limit": request.form.get("limit"),
        "top_k": request.form.get("top_k"),
        "verbose": request.form.get("verbose") == "on",
    }
    logger.info("Finished phase W1: parse web form")

    try:
        logger.info("Starting phase W2: normalize numeric inputs")
        limit = _to_int(form["limit"], 1197)
        top_k = _to_int(form["top_k"], 5)
        logger.info("Finished phase W2: normalize numeric inputs")

        logger.info("Starting phase W3: run search service")
        payload = run_search(
            query=form["query"],
            model=form["model"],
            limit=limit,
            top_k=top_k,
            verbose=form["verbose"],
            data_path=str(ROOT_DIR / "data" / "reddit_ct.csv"),
        )
        logger.info("Finished phase W3: run search service")

        form["limit"] = limit
        form["top_k"] = top_k

        return render_template(
            "index.html",
            models=ALLOWED_MODELS,
            form=form,
            results=payload["results"],
            search_time=payload["search_time"],
            stats=payload["stats"],
            error=None,
        )
    except ValueError as exc:
        logger.exception("Search request failed with validation error")
        error_message = str(exc)
    except FileNotFoundError:
        logger.exception("Search request failed because data file is missing")
        error_message = "Data file not found: data/reddit_ct.csv"
    except Exception as exc:
        logger.exception("Search request failed with unexpected error")
        error_message = f"Search failed: {exc}"

    try:
        form["limit"] = _to_int(form["limit"], 1197)
        form["top_k"] = _to_int(form["top_k"], 5)
    except Exception:
        form["limit"] = 1197
        form["top_k"] = 5

    return render_template(
        "index.html",
        models=ALLOWED_MODELS,
        form=form,
        results=[],
        search_time=None,
        stats=None,
        error=error_message,
    )


def main():
    configure_logging()
    logger.info("Starting phase W0: parse web server arguments")
    parser = argparse.ArgumentParser(description="Run web interface for corpus search")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    logger.info("Finished phase W0: parse web server arguments")

    logger.info("Starting phase W9: start Flask server")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
