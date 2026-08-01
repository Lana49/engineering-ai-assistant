# -*- coding: utf-8 -*-
"""
QA Engine для инженерной документации.
"""

from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests

from core.table_extractor import extract_tables_from_results, tables_to_dicts

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    load_dotenv = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# НОВЫЙ SDK — google.genai
try:
    import google.genai as genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False


@dataclass(slots=True)
class SearchResult:
    """Результат поиска."""
    doc_name: str
    chunk_id: int
    text: str
    score: float
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    filepath: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QAResponse:
    """Ответ на вопрос."""
    question: str
    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    provider: str = "none"
    used_llm: bool = False
    context: str = ""


class QASystem:
    """Система вопросов-ответов по документам."""

    def __init__(
        self,
        use_llm: bool = False,
        llm_provider: str = "ollama",
        model_name: str | None = None,
        top_k: int = 5,
        min_score: float = 0.15,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.1:8b",
        gemini_api_key: str | None = None,
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

        # Gemini — через google.genai
        self.gemini_api_key = (gemini_api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self.gemini_model = gemini_model or "gemini-2.0-flash"
        self.gemini_available = bool(self.gemini_api_key) and GENAI_AVAILABLE
        self.genai_client: Any = None  # ← ИСПРАВЛЕНО: явная аннотация типа

        if self.gemini_available and genai is not None:
            try:
                self.genai_client = genai.Client(api_key=self.gemini_api_key)
                print(f"✅ Gemini клиент инициализирован (модель: {self.gemini_model})")
            except Exception as e:
                print(f"⚠️ Ошибка инициализации Gemini: {e}")
                self.gemini_available = False

        if model_name:
            if self.llm_provider in {"ollama", "mixed"}:
                self.ollama_model = model_name
            elif self.llm_provider == "gemini":
                self.gemini_model = model_name

        self.embedding_model_name = embedding_model_name
        self.use_embeddings = use_embeddings
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight

        self.documents: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []

        self.embedding_model: SentenceTransformer | None = None
        self.chunk_embeddings: np.ndarray | None = None  # ← ИСПРАВЛЕНО: тип

        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix: Any = None

        self.is_ready = False
        self.llm_available = False

        if TfidfVectorizer is None:
            print("⚠️ scikit-learn не установлен")

        if self.use_embeddings:
            self._try_load_embedding_model()

        if self.use_llm:
            self._validate_llm_config()

    def is_ollama_alive(self) -> bool:
        """Проверяет доступность Ollama."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def is_ollama_available(self) -> bool:
        """Алиас для совместимости."""
        return self.is_ollama_alive()

    def _validate_llm_config(self) -> None:
        """Проверяет доступность LLM-провайдеров."""
        self.llm_available = False
        if self.use_llm:
            if self.llm_provider == "ollama":
                self.llm_available = self.is_ollama_alive()
                if self.llm_available:
                    print(f"✅ Ollama доступен (модель: {self.ollama_model})")
                else:
                    print(f"⚠️ Ollama недоступен ({self.ollama_base_url})")
            elif self.llm_provider == "gemini":
                self.llm_available = self.gemini_available
                if self.llm_available:
                    print(f"✅ Gemini доступен (модель: {self.gemini_model})")
                else:
                    print("⚠️ Gemini недоступен (нет API ключа или не установлен google-genai)")
            elif self.llm_provider == "mixed":
                self.llm_available = self.is_ollama_alive() or self.gemini_available
                if self.llm_available:
                    print("✅ Mixed-режим: хотя бы один LLM доступен")
                else:
                    print("⚠️ Mixed-режим: ни один LLM не доступен")
            elif self.llm_provider == "none":
                self.llm_available = False

        print(f"ℹ️ LLM provider={self.llm_provider}, available={self.llm_available}")

    def _select_provider(self) -> str:
        """Выбирает активного провайдера."""
        if not self.use_llm:
            return "none"

        if self.llm_provider == "ollama":
            return "ollama" if self.is_ollama_alive() else "none"

        if self.llm_provider == "gemini":
            return "gemini" if self.gemini_available else "none"

        if self.llm_provider == "mixed":
            if self.is_ollama_alive():
                return "ollama"
            if self.gemini_available:
                return "gemini"

        return "none"

    def build_index(self, parsed_docs: list[dict[str, Any]]) -> bool:
        """Строит индекс по документам."""
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

                self.chunks.append({
                    "doc_name": chunk.get("doc_name", doc_name),
                    "chunk_id": chunk.get("chunk_id", 0),
                    "text": text,
                    "filepath": filepath,
                    "filetype": filetype,
                    "metadata": {**doc_metadata, **chunk.get("metadata", {})},
                })

        if not self.chunks:
            print("⚠️ Нет фрагментов для индексации")
            self.is_ready = False
            return False

        texts = [c["text"] for c in self.chunks]

        if TfidfVectorizer is not None:
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=50000,
                token_pattern=r"(?u)\b[\w\-/\.]+\b",
            )
            # ← ИСПРАВЛЕНО: проверка что vectorizer не None
            if self.vectorizer is not None:
                self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            else:
                self.tfidf_matrix = None
        else:
            self.vectorizer = None
            self.tfidf_matrix = None
            print("⚠️ TF-IDF недоступен: scikit-learn не установлен")

        if self.embedding_model is not None and texts:
            try:
                embeddings = self.embedding_model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                # ← ИСПРАВЛЕНО: корректное использование np.asarray
                self.chunk_embeddings = np.asarray(embeddings, dtype=np.float32)  # type: ignore
            except Exception as e:
                print(f"⚠️ Ошибка построения эмбеддингов: {e}")
                self.chunk_embeddings = None
        else:
            self.chunk_embeddings = None

        self.is_ready = True
        print(f"✅ Индекс построен: {len(self.chunks)} фрагментов")
        return True

    def index_documents(self, directory: str | Path) -> bool:
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

    def save_index(self, index_path: str | Path) -> bool:
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
        except Exception as e:
            print(f"❌ Ошибка сохранения индекса: {e}")
            return False

    def load_index(self, index_path: str | Path) -> bool:
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
        except Exception as e:
            print(f"❌ Ошибка загрузки индекса: {e}")
            self.is_ready = False
            return False

    def search(
        self,
        query: str,
        top_k: int | None = None,
        search_type: str = "hybrid",
    ) -> list[SearchResult]:
        """Поиск по индексу."""
        if not self.is_ready or not self.chunks:
            return []

        top_k = top_k or self.top_k

        if search_type == "semantic":
            return self._search_semantic(query, top_k)
        if search_type == "lexical":
            return self._search_lexical(query, top_k)

        return self._search_hybrid(query, top_k)

    def _search_semantic(self, query: str, top_k: int) -> list[SearchResult]:
        """Семантический поиск."""
        if self.chunk_embeddings is None or self.embedding_model is None:
            return []

        try:
            query_emb = self.embedding_model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            query_emb = np.asarray(query_emb, dtype=np.float32)[0]  # type: ignore
            scores = np.dot(self.chunk_embeddings, query_emb)  # type: ignore

            sorted_indices = np.argsort(scores)[::-1][:top_k]
            results: list[SearchResult] = []

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

    def _search_lexical(self, query: str, top_k: int) -> list[SearchResult]:
        """Лексический поиск."""
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []

        try:
            query_vec = self.vectorizer.transform([query])
            scores = (self.tfidf_matrix @ query_vec.T).toarray().ravel()

            sorted_indices = np.argsort(scores)[::-1][:top_k]
            results: list[SearchResult] = []

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

    def _search_hybrid(self, query: str, top_k: int) -> list[SearchResult]:
        """Гибридный поиск."""
        semantic_scores = np.zeros(len(self.chunks), dtype=np.float32)
        lexical_scores = np.zeros(len(self.chunks), dtype=np.float32)

        if self.chunk_embeddings is not None and self.embedding_model is not None:
            try:
                query_emb = self.embedding_model.encode(
                    [query],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                query_emb = np.asarray(query_emb, dtype=np.float32)[0]  # type: ignore
                semantic_scores = np.dot(self.chunk_embeddings, query_emb)  # type: ignore
            except Exception as e:
                print(f"⚠️ Ошибка semantic части hybrid search: {e}")

        if self.vectorizer is not None and self.tfidf_matrix is not None:
            try:
                query_vec = self.vectorizer.transform([query])
                lexical_scores = (self.tfidf_matrix @ query_vec.T).toarray().ravel()
            except Exception as e:
                print(f"⚠️ Ошибка lexical части hybrid search: {e}")

        combined = self.semantic_weight * semantic_scores + self.lexical_weight * lexical_scores
        sorted_indices = np.argsort(combined)[::-1][:max(top_k * 2, top_k)]

        results: list[SearchResult] = []
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

    def answer(self, question: str, top_k: int | None = None, search_type: str = "hybrid") -> dict[str, Any]:
        """Отвечает на вопрос."""
        results = self.search(question, top_k=top_k, search_type=search_type)
        context = self._build_context(results)

        if not results:
            return {
                "answer": "Не удалось найти релевантную информацию в документах.",
                "sources": [],
                "tables": [],
                "formulas": [],
                "provider": "none",
                "used_llm": False,
                "context": "",
            }

        provider = self._select_provider()

        if provider != "none":
            prompt = self._build_prompt(question, context)
            llm_answer = None

            if provider == "ollama":
                llm_answer = self._ask_ollama(prompt)
            elif provider == "gemini":
                llm_answer = self._ask_gemini(prompt)

            if llm_answer:
                return {
                    "answer": llm_answer,
                    "sources": self._build_sources(results),
                    "tables": tables_to_dicts(extract_tables_from_results(results)),
                    "formulas": self._extract_formulas_from_results(results),
                    "provider": provider,
                    "used_llm": True,
                    "context": context,
                }

        fallback_answer = self._generate_extract_answer(results)
        return {
            "answer": fallback_answer,
            "sources": self._build_sources(results),
            "tables": tables_to_dicts(extract_tables_from_results(results)),
            "formulas": self._extract_formulas_from_results(results),
            "provider": "none",
            "used_llm": False,
            "context": context,
        }

    def find_definition(self, term: str) -> dict[str, Any]:
        """Ищет определение термина."""
        query = f"определение {term}"
        results = self.search(query, top_k=5)

        for item in results:
            text = item.text
            if term.lower() in text.lower():
                sentence = self._best_sentence_for_term(text, term)
                if sentence:
                    return {
                        "found": True,
                        "definition": sentence,
                        "source": item.doc_name,
                    }

        return {"found": False, "definition": "", "source": ""}

    def _try_load_embedding_model(self) -> None:
        """Загружает модель эмбеддингов."""
        if SentenceTransformer is None:
            self.embedding_model = None
            print("⚠️ sentence-transformers не установлен")
            return

        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print(f"✅ Эмбеддинг модель загружена: {self.embedding_model_name}")
        except Exception as e:
            self.embedding_model = None
            print(f"⚠️ Не удалось загрузить эмбеддинг модель: {e}")

    @staticmethod
    def _build_context(results: list[SearchResult]) -> str:
        """Собирает контекст."""
        parts = []
        for i, result in enumerate(results, start=1):
            parts.append(
                f"[Источник {i}] {result.doc_name}\n"
                f"Фрагмент #{result.chunk_id}\n"
                f"{result.text}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _build_sources(results: list[SearchResult]) -> list[dict[str, Any]]:
        """Собирает источники."""
        seen = set()
        sources = []
        for item in results:
            if item.doc_name not in seen:
                seen.add(item.doc_name)
                sources.append({
                    "doc_name": item.doc_name,
                    "chunk_id": item.chunk_id,
                    "score": item.score,
                    "filepath": item.filepath,
                })
        return sources[:5]

    # ← УДАЛЕНА _extract_tables_from_results — используем из table_extractor

    @staticmethod
    def _extract_formulas_from_results(results: list[SearchResult]) -> list[dict[str, Any]]:
        """Извлекает формулы."""
        formulas = []
        seen = set()
        for result in results[:3]:
            matches = re.findall(r"[A-Za-zА-Яа-я0-9_]+\s*=\s*[^=\n]{3,120}", result.text)
            for match in matches[:3]:
                if match not in seen:
                    seen.add(match)
                    formulas.append({
                        "raw": match,
                        "variables": re.findall(r"[A-Za-zА-Яа-я_]+", match),
                        "source": result.doc_name,
                    })
        return formulas

    @staticmethod
    def _best_sentence_for_term(text: str, term: str) -> str:
        """Находит лучшее предложение с термином."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        term_lower = term.lower()

        best = ""
        best_score = 0
        for sentence in sentences:
            if term_lower in sentence.lower():
                score = 100 - len(sentence)
                if score > best_score:
                    best_score = score
                    best = sentence.strip()

        return best

    def _ask_ollama(self, prompt: str) -> str | None:
        """Запрос к Ollama."""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            return (data.get("response") or "").strip() or None
        except Exception as e:
            print(f"⚠️ Ошибка Ollama: {e}")
            return None

    def _ask_gemini(self, prompt: str) -> str | None:
        """Запрос к Gemini через google.genai SDK."""
        if not self.gemini_available or self.genai_client is None:
            return None

        try:
            response = self.genai_client.models.generate_content(
                model=self.gemini_model,
                contents=[prompt],
            )
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
            return None
        except Exception as e:
            print(f"⚠️ Ошибка Gemini: {e}")
            return None

    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        """Формирует промпт."""
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
    def _generate_extract_answer(results: list[SearchResult]) -> str:
        """Fallback-ответ без LLM."""
        if not results:
            return "Не удалось найти информацию в документах."

        top = results[0]
        fragments = [r.text.strip() for r in results[:3] if r.text.strip()]
        merged = "\n\n".join(fragments)
        merged = re.sub(r"\n{3,}", "\n\n", merged).strip()

        if len(merged) > 1800:
            merged = merged[:1800].rsplit(" ", 1)[0] + "..."

        return f"Найдена релевантная информация в документе «{top.doc_name}».\n\n{merged}"