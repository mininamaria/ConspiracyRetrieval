import argparse
import logging
import sys

from search_service import ALLOWED_MODELS, run_search


logger = logging.getLogger(__name__)
ENABLE_LOGS = True


def configure_logging():
  if ENABLE_LOGS:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
  else:
    logging.disable(logging.CRITICAL)

# выносим парсер в отдельную функцию
def build_parser():
  parser = argparse.ArgumentParser(
    description="Search in reddit corpus with BM25, Word2Vec, or FastText"
  )
  parser.add_argument("--model", help="Search model: bm25 | word2vec | fasttext")
  parser.add_argument("--query", help="Search query")
  parser.add_argument("--limit", type=int, help="Data limit (rows to load)")
  parser.add_argument(
    "--top-k",
    "--count",
    dest="top_k",
    type=int,
    help="Number of items matching query"
  )
  parser.add_argument("--verbose", action="store_true", help="Show loaded-data and runtime stats")
  return parser

# получение параметров от пользователя
# если в CLI рагументах, то используем их, иначе интерактивно спрашиваем:

# получаем текстовые параметры
def prompt_text(current_value, message, default=None):
  if current_value is not None and str(current_value).strip():
    return str(current_value).strip()

  suffix = f" [{default}]" if default is not None else ""
  value = input(f"{message}{suffix}: ").strip()
  if value:
    return value
  return default

# получаем числовые параметры, проверяем на корерктность
def prompt_int(current_value, message, default, min_value=1, max_value=1197):
  if current_value is not None:
    if current_value < min_value:
      raise ValueError(f"{message} must be >= {min_value}")
    if current_value > max_value:
      raise ValueError(f"{message} must be <= {max_value}")
    return current_value

  while True:
    raw = input(f"{message} [{default}]: ").strip()
    if not raw:
      return default
    try:
      value = int(raw)
      if value < min_value:
        print(f"Value must be >= {min_value}")
        continue
      return value
    except ValueError:
      print("Please enter a valid integer")

# построение конфига для запроса
# сначала пытаемся получить из аргументов, если не указано, то спрашиваем пользователя
def resolve_config(args):
  model = prompt_text(args.model, "Model (bm25/word2vec/fasttext)", default="bm25").lower()
  if model not in ALLOWED_MODELS:
    raise ValueError(f"Unknown model: {model}. Allowed: {', '.join(ALLOWED_MODELS)}")

  query = prompt_text(args.query, "Query")
  if not query:
    raise ValueError("Query must not be empty")

  limit = prompt_int(args.limit, "Data limit", default=1000, min_value=1, max_value=1197)
  top_k = prompt_int(args.top_k, "Number of matching items", default=5, min_value=1)

  return {
    "model": model,
    "query": query,
    "limit": limit,
    "top_k": top_k,
    "verbose": args.verbose,
  }

def print_results(results):
  if not results:
    print("No matches found.")
    return

  for entry in results:
    print(f"{entry['index']}. {entry['title']} | score={entry['score']:.4f}")
    print(f"   {entry['snippet']}")


# подробная статистика по загруженным данным и времени выполнения (выводится только при verbose=True)
def print_verbose_stats(config, stats):
  print("=== Stats ===")
  print(f"Model: {config['model']}")
  print(f"Rows loaded: {stats['rows_loaded']}")
  print(f"Documents created: {stats['documents_created']}")
  print(f"Unique words: {stats['unique_words']}")
  print(f"Average tokens/doc: {stats['avg_tokens_doc']:.2f}")
  print(f"Query tokens: {stats['query_tokens']}")
  print(f"Load time: {stats['load_time']:.4f} s")
  print(f"Model build time: {stats['build_time']:.4f} s")
  print(f"Search time: {stats['search_time']:.4f} s")
  print("============")


def main():
  configure_logging()
  logger.info("Starting phase 1: parse CLI arguments")
  # Создаем парсер, парсим аргуманты и получаем конфиг для запроса
  parser = build_parser()
  args = parser.parse_args()
  logger.info("Finished phase 1: parse CLI arguments")

  try:
    logger.info("Starting phase 2: resolve and validate input config")
    config = resolve_config(args)
    logger.info("Finished phase 2: resolve and validate input config")
  except ValueError as exc:
    print(f"Input error: {exc}", file=sys.stderr)
    return 2

  try:
    logger.info("Starting phase 3: execute search pipeline")
    payload = run_search(
      query=config["query"],
      model=config["model"],
      limit=config["limit"],
      top_k=config["top_k"],
      verbose=config["verbose"],
    )
    logger.info("Finished phase 3: execute search pipeline")
  except ValueError as exc:
    print(f"Input error: {exc}", file=sys.stderr)
    return 2
  except FileNotFoundError:
    print("Data file not found: data/reddit_ct.csv", file=sys.stderr)
    return 1
  except Exception as exc:
    print(f"Search failed: {exc}", file=sys.stderr)
    return 1

  results = payload["results"]
  search_time = payload["search_time"]

  if config["verbose"] and payload["stats"]:
    logger.info("Starting phase 4: print verbose stats")
    print_verbose_stats(config=config, stats=payload["stats"])
    logger.info("Finished phase 4: print verbose stats")

  logger.info("Starting phase 5: print search results")
  print_results(results)
  logger.info("Finished phase 5: print search results")
  # время поиска выводим всегда по требованию.
  print(f"Search time: {search_time:.4f} s")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
