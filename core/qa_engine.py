# -*- coding: utf-8 -*-
"""
QA Engine для инженерной документации.

Возможности:
- Гибридный поиск: semantic + lexical
- SentenceTransformers embeddings
- TF-IDF fallback
- LLM-ответы через Ollama / Gemini
- Mixed-режим: автоматический выбор между Ollama и Gemini
- Сохранение/загрузка индекса
"""

from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests

# Попытка импорта python-dotenv (опционально)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Если python-dotenv не установлен, просто пропускаем
    load_dotenv = None  # type: ignore


# Попытка импорта TfidfVectorizer из sklearn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    # Если scikit-learn не установлен, определяем заглушку
    TfidfVectorizer = None  # type: ignore


@dataclass
class SearchResult:
    """Результат поиска."""
    doc_name: str
    chunk_id: int
    text: str
    score: float
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    filepath: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QAResponse:
    """Ответ на вопрос."""
    question: str
    answer: str
    sources: List[SearchResult] = field(default_factory=list)
    provider: str = "none"
    used_llm: bool = False
    context: str = ""


class QASystem:
    """Система вопросов-ответов по документам."""

    def __init__(
        self,
        use_llm: bool = False,
        llm_provider: str = "ollama",
        model_name: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.15,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.1:8b",
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.0-flash",
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        use_embeddings: bool = True,
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3,
    ):
        self.use_llm = use_llm
        self.llm_provider = (llm_provider or "ollama").strip().lower()
        self.top_k = top_k
        self.min_score = min_score

        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model or "llama3.1:8b"

        self.gemini_api_key = (gemini_api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self.gemini_model = gemini_model or "gemini-2.0-flash"

        if model_name:
            if self.llm_provider in {"ollama", "mixed"}:
                self.ollama_model = model_name
            elif self.llm_provider == "gemini":
                self.gemini_model = model_name

        self.embedding_model_name = embedding_model_name
        self.use_embeddings = use_embeddings
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight

        self.documents: List[Dict[str, Any]] = []
        self.chunks: List[Dict[str, Any]] = []

        self.embedding_model: Any = None
        self.chunk_embeddings: Optional[np.ndarray] = None

        self.vectorizer: Optional[Any] = None
        self.tfidf_matrix: Any = None

        self.is_ready = False
        self.llm_available = False

        if TfidfVectorizer is None:
            print("⚠️ scikit-learn не установлен. Установите: pip install scikit-learn")

        if self.use_embeddings:
            self._try_load_embedding_model()

        if self.use_llm:
            self._validate_llm_config()

    def is_ollama_available(self) -> bool:
        """Проверка доступности Ollama."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=3)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def build_index(self, parsed_docs: List[Dict[str, Any]]) -> bool:
        """Строит индекс по уже обработанным документам."""
        self.documents = parsed_docs or []
        self.chunks = []

        for doc in self.documents:
            doc_name = doc.get("doc_name", "")
            filepath = doc.get("filepath", "")
            filetype = doc.get("filetype", "")
            doc_metadata = doc.get("metadata", {}) or {}

            for chunk in doc.get("chunks", []) or []:
                text = (chunk.get("text") or "").strip()
                if not text:
                    continue

                self.chunks.append(
                    {
                        "doc_name": chunk.get("doc_name", doc_name),
                        "chunk_id": chunk.get("chunk_id", 0),
                        "text": text,
                        "filepath": filepath,
                        "filetype": filetype,
                        "metadata": {**doc_metadata, **chunk.get("metadata", {})},
                    }
                )

        if not self.chunks:
            print("⚠️ Нет фрагментов для индексации")
            self.is_ready = False
            return False

        texts = [c["text"] for c in self.chunks]

        # Создание TF-IDF векторизатора
        if TfidfVectorizer is not None:
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=50000,
                token_pattern=r"(?u)\b[\w\-/\.]+\b",
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(texts) if texts else None
        else:
            self.vectorizer = None
            self.tfidf_matrix = None
            print("⚠️ TF-IDF недоступен: scikit-learn не установлен")

        # Создание эмбеддингов
        if self.embedding_model is not None and texts:
            try:
                embeddings = self.embedding_model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                self.chunk_embeddings = np.asarray(embeddings, dtype=np.float32)
            except Exception as e:
                print(f"⚠️ Ошибка построения эмбеддингов: {e}")
                self.chunk_embeddings = None
        else:
            self.chunk_embeddings = None

        self.is_ready = True
        print(f"✅ Индекс построен: {len(self.chunks)} фрагментов")
        return True

    def index_documents(self, directory: Union[str, Path]) -> bool:
        """Индексирует документы из директории."""
        try:
            from core.parser import DocumentParser

            parser = DocumentParser(chunk_size=1200, chunk_overlap=200)
            docs = parser.parse_directory(directory, recursive=True)
            return self.build_index(docs)
        except ImportError as e:
            print(f"❌ Не удалось импортировать parser: {e}")
            return False
        except Exception as e:
            print(f"❌ Ошибка индексации: {e}")
            return False

    def save_index(self, index_path: Union[str, Path]) -> bool:
        """Сохраняет индекс на диск."""
        try:
            data = {
                "documents": self.documents,
                "chunks": self.chunks,
                "top_k": self.top_k,
                "min_score": self.min_score,
                "embedding_model_name": self.embedding_model_name,
                "use_embeddings": self.use_embeddings,
                "semantic_weight": self.semantic_weight,
                "lexical_weight": self.lexical_weight,
                "chunk_embeddings": self.chunk_embeddings,
                "vectorizer": self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
                "ollama_model": self.ollama_model,
                "gemini_model": self.gemini_model,
                "llm_provider": self.llm_provider,
            }

            index_path = Path(index_path)
            index_path.parent.mkdir(parents=True, exist_ok=True)

            with open(index_path, "wb") as f:
                pickle.dump(data, f)

            print(f"✅ Индекс сохранён: {index_path}")
            return True
        except (OSError, pickle.PickleError) as e:
            print(f"❌ Ошибка сохранения индекса: {e}")
            return False

    def load_index(self, index_path: Union[str, Path]) -> bool:
        """Загружает индекс с диска."""
        try:
            index_path = Path(index_path)
            if not index_path.exists():
                print(f"⚠️ Индекс не найден: {index_path}")
                return False

            with open(index_path, "rb") as f:
                data = pickle.load(f)

            self.documents = data.get("documents", [])
            self.chunks = data.get("chunks", [])
            self.top_k = data.get("top_k", self.top_k)
            self.min_score = data.get("min_score", self.min_score)
            self.embedding_model_name = data.get("embedding_model_name", self.embedding_model_name)
            self.use_embeddings = data.get("use_embeddings", self.use_embeddings)
            self.semantic_weight = data.get("semantic_weight", self.semantic_weight)
            self.lexical_weight = data.get("lexical_weight", self.lexical_weight)
            self.chunk_embeddings = data.get("chunk_embeddings", None)
            self.vectorizer = data.get("vectorizer", None)
            self.tfidf_matrix = data.get("tfidf_matrix", None)
            self.ollama_model = data.get("ollama_model", self.ollama_model)
            self.gemini_model = data.get("gemini_model", self.gemini_model)
            self.llm_provider = data.get("llm_provider", self.llm_provider)

            if self.use_embeddings and self.embedding_model is None:
                self._try_load_embedding_model()

            if self.use_llm:
                self._validate_llm_config()

            self.is_ready = bool(self.chunks)
            print(f"✅ Индекс загружен: {len(self.chunks)} фрагментов")
            return True
        except (OSError, pickle.PickleError, EOFError) as e:
            print(f"❌ Ошибка загрузки индекса: {e}")
            self.is_ready = False
            return False

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        search_type: str = "hybrid",
    ) -> List[SearchResult]:
        """Поиск по индексу."""
        if not self.is_ready or not self.chunks:
            return []

        top_k = top_k or self.top_k

        if search_type == "semantic":
            return self.search_semantic(query, top_k)
        if search_type == "lexical":
            return self.search_lexical(query, top_k)

        return self.search_hybrid(query, top_k)

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        search_type: str = "hybrid",
    ) -> QAResponse:
        """Отвечает на вопрос по документам."""
        results = self.search(question, top_k=top_k, search_type=search_type)
        context = self._build_context(results)

        if not results:
            return QAResponse(
                question=question,
                answer="Не удалось найти релевантную информацию в документах.",
                sources=[],
                provider="none",
                used_llm=False,
                context="",
            )

        if self.use_llm and self.llm_available:
            llm_answer, provider = self._generate_llm_answer(question, context)
            if llm_answer:
                return QAResponse(
                    question=question,
                    answer=llm_answer,
                    sources=results,
                    provider=provider,
                    used_llm=True,
                    context=context,
                )

        fallback_answer = self._generate_extract_answer(results)
        return QAResponse(
            question=question,
            answer=fallback_answer,
            sources=results,
            provider="none",
            used_llm=False,
            context=context,
        )

    def search_semantic(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Семантический поиск по эмбеддингам."""
        if self.chunk_embeddings is None or self.embedding_model is None or not self.chunks:
            return []

        try:
            query_emb = self.embedding_model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            query_emb = np.asarray(query_emb, dtype=np.float32)[0]
            scores = np.dot(self.chunk_embeddings, query_emb)

            # Получаем индексы отсортированные по убыванию
            sorted_indices = np.argsort(scores)[::-1][:top_k]
            results: List[SearchResult] = []

            for idx in sorted_indices:
                score = float(scores[idx])
                if score < self.min_score:
                    continue
                chunk = self.chunks[idx]
                results.append(
                    SearchResult(
                        doc_name=chunk["doc_name"],
                        chunk_id=chunk["chunk_id"],
                        text=chunk["text"],
                        score=score,
                        semantic_score=score,
                        lexical_score=0.0,
                        filepath=chunk.get("filepath", ""),
                        metadata=chunk.get("metadata", {}),
                    )
                )
            return results
        except Exception as e:
            print(f"⚠️ Ошибка semantic search: {e}")
            return []

    def search_lexical(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Лексический поиск по TF-IDF."""
        if self.vectorizer is None or self.tfidf_matrix is None or not self.chunks:
            return []

        try:
            query_vec = self.vectorizer.transform([query])
            scores = (self.tfidf_matrix @ query_vec.T).toarray().ravel()

            sorted_indices = np.argsort(scores)[::-1][:top_k]
            results: List[SearchResult] = []

            for idx in sorted_indices:
                score = float(scores[idx])
                if score < max(self.min_score * 0.5, 0.01):
                    continue
                chunk = self.chunks[idx]
                results.append(
                    SearchResult(
                        doc_name=chunk["doc_name"],
                        chunk_id=chunk["chunk_id"],
                        text=chunk["text"],
                        score=score,
                        semantic_score=0.0,
                        lexical_score=score,
                        filepath=chunk.get("filepath", ""),
                        metadata=chunk.get("metadata", {}),
                    )
                )
            return results
        except Exception as e:
            print(f"⚠️ Ошибка lexical search: {e}")
            return []

    def search_hybrid(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Гибридный поиск semantic + lexical."""
        semantic_scores = np.zeros(len(self.chunks), dtype=np.float32)
        lexical_scores = np.zeros(len(self.chunks), dtype=np.float32)

        if self.chunk_embeddings is not None and self.embedding_model is not None:
            try:
                query_emb = self.embedding_model.encode(
                    [query],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                query_emb = np.asarray(query_emb, dtype=np.float32)[0]
                semantic_scores = np.dot(self.chunk_embeddings, query_emb)
            except Exception as e:
                print(f"⚠️ Ошибка semantic части hybrid search: {e}")

        if self.vectorizer is not None and self.tfidf_matrix is not None:
            try:
                query_vec = self.vectorizer.transform([query])
                lexical_scores = (self.tfidf_matrix @ query_vec.T).toarray().ravel()
            except Exception as e:
                print(f"⚠️ Ошибка lexical части hybrid search: {e}")

        combined = self.semantic_weight * semantic_scores + self.lexical_weight * lexical_scores
        sorted_indices = np.argsort(combined)[::-1][: max(top_k * 2, top_k)]

        results: List[SearchResult] = []
        seen = set()

        for idx in sorted_indices:
            score = float(combined[idx])
            if score < max(self.min_score * 0.5, 0.01):
                continue

            chunk = self.chunks[idx]
            key = (chunk["doc_name"], chunk["chunk_id"])
            if key in seen:
                continue
            seen.add(key)

            results.append(
                SearchResult(
                    doc_name=chunk["doc_name"],
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    score=score,
                    semantic_score=float(semantic_scores[idx]),
                    lexical_score=float(lexical_scores[idx]),
                    filepath=chunk.get("filepath", ""),
                    metadata=chunk.get("metadata", {}),
                )
            )

            if len(results) >= top_k:
                break

        return results

    def _try_load_embedding_model(self) -> None:
        """Пытается загрузить embedding model."""
        try:
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print(f"✅ Эмбеддинг модель загружена: {self.embedding_model_name}")
        except ImportError:
            self.embedding_model = None
            print("⚠️ sentence-transformers не установлен. Установите: pip install sentence-transformers")
        except Exception as e:
            self.embedding_model = None
            print(f"⚠️ Не удалось загрузить эмбеддинг модель: {e}")

    def _validate_llm_config(self) -> None:
        """Проверяет доступность LLM-провайдеров."""
        self.llm_available = False

        if self.llm_provider == "ollama":
            self.llm_available = self.is_ollama_available()

        elif self.llm_provider == "gemini":
            self.llm_available = bool(self.gemini_api_key)

        elif self.llm_provider == "mixed":
            self.llm_available = self.is_ollama_available() or bool(self.gemini_api_key)

        elif self.llm_provider == "none":
            self.llm_available = False

        print(f"ℹ️ LLM provider={self.llm_provider}, available={self.llm_available}")

    @staticmethod
    def _build_context(results: List[SearchResult]) -> str:
        """Собирает контекст из найденных фрагментов."""
        parts = []
        for i, result in enumerate(results, start=1):
            parts.append(
                f"[Источник {i}] {result.doc_name}\n"
                f"Фрагмент #{result.chunk_id}\n"
                f"{result.text}"
            )
        return "\n\n".join(parts)

    def _generate_llm_answer(self, question: str, context: str) -> tuple[Optional[str], str]:
        """Генерация ответа через выбранный провайдер."""
        prompt = self._build_prompt(question, context)

        if self.llm_provider == "ollama":
            answer = self._ask_ollama(prompt)
            return answer, "ollama" if answer else "none"

        if self.llm_provider == "gemini":
            answer = self._ask_gemini(prompt)
            return answer, "gemini" if answer else "none"

        if self.llm_provider == "mixed":
            if self.is_ollama_available():
                answer = self._ask_ollama(prompt)
                if answer:
                    return answer, "ollama"

            if self.gemini_api_key:
                answer = self._ask_gemini(prompt)
                if answer:
                    return answer, "gemini"

        return None, "none"

    def _ask_ollama(self, prompt: str) -> Optional[str]:
        """Запрос к Ollama."""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            text = (data.get("response") or "").strip()
            return text or None
        except Exception as e:
            print(f"⚠️ Ошибка Ollama: {e}")
            return None

    def _ask_gemini(self, prompt: str) -> Optional[str]:
        """Запрос к Gemini."""
        if not self.gemini_api_key:
            return None

        try:
            # Импортируем google.generativeai только при использовании
            import google.generativeai as genai

            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel(self.gemini_model)
            response = model.generate_content(prompt)
            text = getattr(response, "text", None)
            if text:
                text = text.strip()
            return text or None
        except ImportError:
            print("⚠️ google-generativeai не установлен. Установите: pip install google-generativeai")
            return None
        except Exception as e:
            print(f"⚠️ Ошибка Gemini: {e}")
            return None

    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        """Формирует промпт для LLM."""
        return f"""
Ты инженерный помощник по строительной документации.

Используй только информацию из контекста ниже.
Если данных недостаточно, прямо так и скажи.
Если в тексте есть нормы, значения, таблицы, пункты документов — ссылайся на них в ответе.
Отвечай на русском языке, кратко и по делу.

Контекст:
{context}

Вопрос:
{question}
""".strip()

    @staticmethod
    def _generate_extract_answer(results: List[SearchResult]) -> str:
        """Fallback-ответ без LLM."""
        if not results:
            return "Не удалось найти информацию в документах."

        top = results[0]
        fragments = [r.text.strip() for r in results[:3] if r.text.strip()]
        merged = "\n\n".join(fragments)

        merged = re.sub(r"\n{3,}", "\n\n", merged).strip()

        if len(merged) > 1800:
            merged = merged[:1800].rsplit(" ", 1)[0] + "..."

        return (
            f"Найдена релевантная информация в документе «{top.doc_name}».\n\n"
            f"{merged}"
        )