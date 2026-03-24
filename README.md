# CLI Search in the Reddit Conspiracy Corpus

Simple search engine over a Reddit conspiracy corpus ([source](https://www.kaggle.com/datasets/gpreda/reddit-conspiracy-theory)) with three ranking models:
- BM25
- Word2Vec
- FastText

The app supports CLI flags and interactive fallback prompts.

## Features

- Search by query text
- Choose model: `bm25`, `word2vec`, `fasttext`
- Choose data limit (how many rows to load)
- Choose number of matching items (`top-k`)
- `--verbose` mode for full dataset/runtime stats
- Search time is always printed (even without `--verbose`)

## Project Structure

- `main.py` - CLI entry point
- `corpus.py` - data loading and text preprocessing
- `search_engine.py` - model building and search
- `data/reddit_ct.csv` - corpus data file
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.10+
- `pip`

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Run

### Linux/macOS

Show help:

```bash
python3 main.py --help
```

Run with flags:

```bash
python3 main.py --model bm25 --query "climate change" --limit 2000 --top-k 5
```

Run in verbose mode:

```bash
python3 main.py --model fasttext --query "government surveillance" --limit 1500 --top-k 3 --verbose
```

Interactive mode (missing parameters will be asked in terminal):

```bash
python3 main.py
```

### Windows

Show help:

```powershell
py main.py --help
```

Run with flags:

```powershell
py main.py --model bm25 --query "climate change" --limit 2000 --top-k 5
```

Run in verbose mode:

```powershell
py main.py --model fasttext --query "government surveillance" --limit 1500 --top-k 3 --verbose
```

Interactive mode:

```powershell
py main.py
```

## Web Interface

A separate web app is available in `web/` (in progress).

Quick start:

```bash
cd web
python3 -m pip install -r requirements.txt
python3 app.py
```

Custom port:

```bash
python3 app.py --port 8080
```

For full web usage details, see `web/README.md`.

## CLI Options

- `--model` - search model (`bm25`, `word2vec`, `fasttext`)
- `--query` - search query text
- `--limit` - max number of rows to load from dataset
- `--top-k` or `--count` - number of matching results
- `--verbose` - print detailed stats (rows/docs/tokens/timings)
- `-h`, `--help` - print help

If any of `model`, `query`, `limit`, or `top-k` are missing, the app asks interactively.

## Output

### Non-verbose mode

Prints:
- results (rank, title, score, snippet)
- search time

Example format:

```text
1. Some post title | score=0.3456
   Short text snippet from document...
Search time: 0.0421 s
```

### Verbose mode

Prints:
- model name
- rows loaded
- documents created
- unique words
- average tokens per doc
- query tokens
- load/build/search timings
- results list

## Exit Codes

- `0` - success
- `1` - runtime/data/model errors
- `2` - input validation errors

## Notes

- Data file path is currently fixed to `data/reddit_ct.csv`.
- First run may download NLTK stopwords automatically.

