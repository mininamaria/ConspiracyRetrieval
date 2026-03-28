import numpy as np
import logging
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec, FastText
from collections import Counter
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

class SearchEngine:
    """
    Класс поисковой системы, поддерживающий три метода поиска:
    - BM25 (ранжирование на основе статистики терминов)
    - Word2Vec (векторные представления слов)
    - FastText (векторные представления с учетом подслов)

    Attributes:
        documents (list): список объектов Document
        tokenized_corpus (list): список токенизированных документов
        bm25 (BM25Okapi): модель BM25 для ранжирования
        w2v_model (Word2Vec): обученная модель Word2Vec
        ft_model (FastText): обученная модель FastText
        doc_vectors_w2v (np.ndarray): векторные представления всех документов (Word2Vec)
        doc_vectors_ft (np.ndarray): векторные представления всех документов (FastText)
    """
    def __init__(self, documents):
        """
        Инициализация поисковой системы

        Args:
            documents (list): список объектов Document с полями title, text, tokens
        """
        self.doc_freq = None
        self.word_freq = None
        self.total_docs = None
        logger.info("Starting phase 4: initialize search engine")
        self.documents = documents
        # достаём токены из каждого документа для построения корпуса
        self.tokenized_corpus = [doc.tokens for doc in documents]

        # инициализируем модели
        self.bm25 = None
        self.w2v_model = None
        self.ft_model = None

        # плейсхолдеры для векторного представления документов
        self.doc_vectors_w2v = None
        self.doc_vectors_ft = None
        logger.info("Finished phase 4: initialize search engine")



    ########## BM-25 ##########
    def build_bm25(self):
        """
        Построение индекса BM25 на основе токенизированного корпуса
        BM25 использует TF-IDF для ранжирования документов по релевантности
        """
        logger.info("Starting phase 5.1: build BM25 model")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("Finished phase 5.1: build BM25 model")

    def search_bm25(self, query_tokens, top_k=5):
        """
        Поиск документов при помощи BM25

        Args:
            query_tokens (list): токенизированный запрос
            top_k (int): количество возвращаемых результатов

        Returns:
            list: Список словарей с результатами поиска (title, text, score)
        """
        logger.info("Starting phase 6.1: run BM25 search")
        # получаем оценку релевантности для всех документов
        scores = self.bm25.get_scores(query_tokens)

        # сортируем документы по убыванию оценки и берём top_k
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = self.format_results(top_idx, scores)
        logger.info("Finished phase 6.1: run BM25 search")
        return results

    def search(self, model_name, query_tokens, top_k=5):
        # единая точка поиска по выбранной модели
        model = (model_name or "").lower()

        if model == "bm25":
            if self.bm25 is None:
                raise ValueError("BM25 model is not built")
            return self.search_bm25(query_tokens, top_k)

        if model == "word2vec":
            if self.w2v_model is None:
                raise ValueError("Word2Vec model is not built")
            return self.search_word2vec(query_tokens, top_k)

        if model == "fasttext":
            if self.ft_model is None:
                raise ValueError("FastText model is not built")
            return self.search_fasttext(query_tokens, top_k)

        raise ValueError(f"Unknown model: {model_name}")

    ########## Word2Vec ##########
    def build_word2vec(self, vector_size=100):
        """
        Обучение модели Word2Vec на корпусе документов

        Word2Vec создает векторные представления слов, где близкие по смыслу слова
        представлены каак близкие векторы в пространстве

        Args:
            vector_size (int): размерность векторных представлений слов
        """
        logger.info("Starting phase 5.2: build Word2Vec model")

        # обучаем модель
        self.w2v_model = Word2Vec(
            sentences=self.tokenized_corpus,
            vector_size=vector_size, # размерность векторов
            window=5, # ширина контекстного окна
            min_count=2, # минимальная частота слова
            workers=4 # потоки для обучения
        )

        # создаем векторные представления для всех документов
        self.doc_vectors_w2v = np.array([
            self.calculate_mean_doc_vector(doc.tokens, self.w2v_model)
            for doc in self.documents
        ])
        logger.info("Finished phase 5.2: build Word2Vec model")

    def search_word2vec(self, query_tokens, top_k=5):
        """
        Поиск документов с использованием Word2Vec и косинусной близости

        Args:
            query_tokens (list): токенизированный поисковый запрос
            top_k (int): количество возвращаемых результатов

        Returns:
            list: список словарей с результатами поиска
        """
        logger.info("Starting phase 6.2: run Word2Vec search")
        # получаем вектор запроса (ср. арифм. векторов слов запроса)
        query_vector = self.calculate_mean_doc_vector(query_tokens, self.w2v_model)
        # считаем косинусную близость между вектором запроса и всеми документами
        cos_sims = cosine_similarity([query_vector], self.doc_vectors_w2v)[0]

        # сортируем по убыванию косинусной близости, берём top_k
        top_idx = np.argsort(cos_sims)[::-1][:top_k]

        results = self.format_results(top_idx, cos_sims)
        logger.info("Finished phase 6.2: run Word2Vec search")
        return results

    def weighted_doc_vector(self, tokens):
        vectors = []

        for word in tokens:
            if word in self.ft_model.wv:
                # IDF вес
                df = self.doc_freq.get(word, 1)
                idf = np.log((self.total_docs + 1) / (df + 1))

                vectors.append(self.ft_model.wv[word] * idf)

        if not vectors:
            return np.zeros(self.ft_model.vector_size)

        return np.mean(vectors, axis=0)


    ########## FastText ##########
    def build_fasttext(self, vector_size=100):
        """
        Обучение модели FastText на корпусе

        FastText улучшает Word2Vec, учитывая подслова, что позволяет
        лучше обрабатывать редкие слова

        Args:
            vector_size (int): размерность векторных представлений
        """
        logger.info("Starting phase 5.3: build FastText model")

        # обучаем модель
        self.ft_model = FastText(
            sentences=self.tokenized_corpus,
            vector_size=vector_size,
            window=5,
            min_count=2
        )
        # считаем частоты слов
        all_tokens = [token for doc in self.tokenized_corpus for token in doc]
        self.word_freq = Counter(all_tokens)
        self.total_docs = len(self.documents)

        # считаем DF
        self.doc_freq = Counter()
        for doc in self.tokenized_corpus:
            unique_tokens = set(doc)
            for token in unique_tokens:
                self.doc_freq[token] += 1

        # создаем векторные представления документов с весами
        self.doc_vectors_ft = np.array([
            self.weighted_doc_vector(doc.tokens)
            for doc in self.documents
        ])

        # нормализация
        self.doc_vectors_ft = normalize(self.doc_vectors_ft)
        logger.info("Finished phase 5.3: build FastText model")

    def search_fasttext(self, query_tokens, top_k=5):
        logger.info("Starting phase 6.3: run FastText search")
        # защита от пустого запроса
        if len(query_tokens) == 0:
            logger.info("Query vector is empty")
            return []
        # получаем вектор запроса
        query_vector = self.weighted_doc_vector(query_tokens)

        # считаем косинусную близость
        cos_sims = cosine_similarity([query_vector], self.doc_vectors_ft)[0]

        # сортируем по убыванию, берём top_k
        top_idx = np.argsort(cos_sims)[::-1][:top_k]

        results = self.format_results(top_idx, cos_sims)
        logger.info("Finished phase 6.3: run FastText search")
        return results


    def calculate_mean_doc_vector(self, tokens, model):
        # берём векторы только тех слов, которые есть в модели
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        # ничего нет - возвращаем пустоту (нулевой вектор)
        if not vectors:
            return np.zeros(model.vector_size)

        # возвращаем среднее
        return np.mean(vectors, axis=0)

    def format_results(self, indices, scores):
        results = []

        for rank, i in enumerate(indices, start=1):
            doc = self.documents[i]

            results.append({
                "index": rank,
                "title": doc.title,
                "snippet": doc.text[:300],
                "score": float(scores[i])
            })

        return results