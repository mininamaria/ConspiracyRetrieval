import pandas as pd
import typing
import re
import logging
import pickle
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


logger = logging.getLogger(__name__)

LINK_PATTERN = re.compile(r'https?://\S+|www\.\S+')
TOKEN_PATTERN = re.compile(r"[a-z]+")


def _cache_file_path(data_path: str) -> Path:
    return Path(data_path).resolve().with_name("preprocessed_cache.pkl")


def try_load_preprocessed_cache(data_path: str, limit: typing.Optional[int]):
    cache_path = _cache_file_path(data_path)
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("rb") as f:
            payload = pickle.load(f)

        source_path = Path(data_path).resolve()
        source_mtime = source_path.stat().st_mtime

        if payload.get("source_path") != str(source_path):
            return None
        if payload.get("source_mtime") != source_mtime:
            return None
        if payload.get("limit") != limit:
            return None

        documents = [
            Document(title=item["title"], text=item["text"], tokens=item["tokens"])
            for item in payload.get("documents", [])
        ]
        unique_words = set(payload.get("unique_words", []))
        rows_loaded = int(payload.get("rows_loaded", 0))

        logger.info("Cache hit: loaded preprocessed corpus from %s", cache_path)
        return {
            "documents": documents,
            "unique_words": unique_words,
            "rows_loaded": rows_loaded,
        }
    except Exception as exc:
        logger.warning("Cache load failed, rebuilding preprocessing cache: %s", exc)
        return None


def save_preprocessed_cache(
    data_path: str,
    limit: typing.Optional[int],
    documents: typing.List["Document"],
    unique_words: typing.Set[str],
    rows_loaded: int,
):
    cache_path = _cache_file_path(data_path)
    source_path = Path(data_path).resolve()

    payload = {
        "source_path": str(source_path),
        "source_mtime": source_path.stat().st_mtime,
        "limit": limit,
        "rows_loaded": rows_loaded,
        "documents": [
            {"title": doc.title, "text": doc.text, "tokens": doc.tokens}
            for doc in documents
        ],
        "unique_words": list(unique_words),
    }

    with cache_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Cache saved: preprocessed corpus written to %s", cache_path)

def load_corpus_data(
    filepath: str = "data/reddit_ct.csv",
    limit: typing.Optional[int] = 100,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Загружает и предварительно обрабатывает данные из CSV файла.

    Args:
        filepath (str): путь к csv файлу с данными
        limit (typing.Optional[int]): максимальное количество строк для загрузки, None означает загрузить все строки
        verbose (bool): флаг вывода информационных сообщений в консоль.
                        По умолчанию True

    Returns:
        pd.DataFrame: предварительно обработанный DataFrame с данными
    """
    logger.info("Starting phase 1: load corpus data from %s", filepath)
    # загружаем
    df = pd.read_csv(filepath)
    if verbose:
        print(f"Lines in dataset: {len(df)}")

    # выкидываем ненужные колонки и пустые строки
    df = df.drop(['url', 'comms_num'], axis=1, errors='ignore')
    df = df.dropna()

    # при необходимости ограничиваем количество строк
    if limit:
        df = df[:limit]

    if verbose:
        print(f"Documents processed: {len(df)}")
    logger.info("Finished phase 1: load corpus data, rows=%d", len(df))
    return df

def clean_text(text: str) -> str:
    # убираем ссылки
    text = re.sub(LINK_PATTERN, '', text)

    # убираем абзацы и лишние пробелы
    text = text.replace('\n', ' ')
    text = ' '.join(text.split())

    return text

class Document:
    """
    Класс, представляющий отдельный документ в корпусе.

    Хранит оригинальный текст документа, его заголовок и предобработанные токены.

    Attributes:
        title (str): Заголовок или идентификатор документа
        text (str): Оригинальный текст документа
        tokens (list): Список предобработанных токенов (лемм) документа
    """
    def __init__(self, title, text, tokens):
        self.title = title
        self.text = text
        self.tokens = tokens

    def __repr__(self):
        return f"Document(title='{self.title}', tokens_count={len(self.tokens)})"

    def tostring(self, limit=300):
        return f"-----------------\n{self.title}\n-------\n{self.text[:limit]}..."


class CorpusProcessor:
    def __init__(self):
        logger.info("Starting phase 2: initialize text processor")
        # лемматизатор
        self.lemmatizer = WordNetLemmatizer()

        # список стоп-слов для английского языка
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.stop_words = set(stopwords.words('english'))

        # сет для хранения уникальных слов
        self.unique_words = set()
        logger.info("Finished phase 2: initialize text processor")

    def preprocess_text(self, text: str) -> typing.List[str]:
        """
        Предобработка текста: удаление ссылок, пунктуации, стоп-слов и лишних пробелов
        """
        if not isinstance(text, str) or pd.isna(text):
            return []

        # убираем ссылки
        # убираем абзацы и лишние пробелы
        text = clean_text(text)

        # токенизируем и лемматизируем английский текст
        raw_tokens = TOKEN_PATTERN.findall(text.lower())
        filtered_tokens = []
        for token in raw_tokens:
            if token in self.stop_words:
                continue
            lemma = self.lemmatizer.lemmatize(token)
            if lemma.isalpha() and lemma not in self.stop_words:
                filtered_tokens.append(lemma)

        # добавляем слова в словарь уникальных
        self.unique_words.update(filtered_tokens)

        return filtered_tokens

    def create_documents(self, df: pd.DataFrame, title_column: typing.Optional[str] = None) -> typing.List[Document]:
        total_rows = len(df)
        logger.info("Starting phase 3: create documents from dataframe, total_rows=%d", total_rows)
        documents = []
        skipped_rows = 0

        for processed_rows, (_, row) in enumerate(df.iterrows(), start=1):
            text = clean_text(row['body']) # добавила это позже, мб стоит убрать
            title = str(row[title_column])

            tokens = self.preprocess_text(text)

            if tokens:
                doc = Document(title=title, text=text, tokens=tokens)
                documents.append(doc)
            else:
                skipped_rows += 1

            if processed_rows % 1 == 0 or processed_rows == total_rows:
                logger.info(
                    "Phase 3 progress: processed=%d/%d, created=%d, skipped=%d",
                    processed_rows,
                    total_rows,
                    len(documents),
                    skipped_rows,
                )

        logger.info(
            "Finished phase 3: create documents, processed=%d, created=%d, skipped=%d",
            total_rows,
            len(documents),
            skipped_rows,
        )
        return documents

