import pandas as pd
import string
import typing
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem

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
    return df

def clean_text(text: str) -> str:
    # убираем ссылки
    pattern_link = re.compile(r'https?://\S+|www\.\S+')
    text = re.sub(pattern_link, '', text)

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
        # лемматизатор
        self.mystem = Mystem()

        # список стоп-слов для английского языка
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))

        # сет для хранения уникальных слов
        self.unique_words = set()

    def preprocess_text(self, text: str) -> typing.List[str]:
        # на всякий случай проверим, что с текстом всё в порядке
        if not isinstance(text, str) or pd.isna(text):
            return []

        # убираем ссылки
        # убираем абзацы и лишние пробелы
        text = clean_text(text)

        # убираем пунктуацию, переводим всё в нижний регистр
        text = text.lower()
        text = ''.join([ch for ch in text if ch not in string.punctuation])

        # лемматизируем
        lemmas = self.mystem.lemmatize(text)

        # убираем стоп-слова и неалфавитные единицы
        filtered_tokens = [word for word in lemmas if word not in self.stop_words and word.isalpha()]

        # добавляем слова в словарь уникальных
        self.unique_words.update(filtered_tokens)

        return filtered_tokens

    def create_documents(self, df: pd.DataFrame, title_column: typing.Optional[str] = None) -> typing.List[Document]:
        documents = []

        for idx, row in df.iterrows():
            text = clean_text(row['body']) # добавила это позже, мб стоит убрать
            title = str(row[title_column])

            tokens = self.preprocess_text(text)

            if tokens:
                doc = Document(title=title, text=text, tokens=tokens)
                documents.append(doc)

        return documents

