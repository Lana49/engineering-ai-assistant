# -*- coding: utf-8 -*-
"""
QA Engine для инженерной документации.
Диагностическая версия: считает документы, чанки, батчи эмбеддингов
и подробно показывает причины, почему индекс не сохранился.
"""

from __future__ import annotations

import math
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import logging
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

try:
    import google.genai as genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)
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
        model_name: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.15,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.1:8b",
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        use_embeddings: bool = True,
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3,
        embedding_batch_size: int = 32,
    ):
        self.use_llm = use_llm
        self.llm_provider = (llm_provider or "ollama").strip().lower()
        self.top_k = top_k
        self.min_score = min_score

        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = model_name if model_name and self.llm_provider in {"ollama", "mixed"} else ollama_model

        self.embedding_model_name = embedding_model_name
        self.use_embeddings = use_embeddings
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.embedding_batch_size = max(1, int(embedding_batch_size))

        self.documents: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []

        self.embedding_model: Optional[SentenceTransformer] = None  # Исправлено
        self.chunk_embeddings: Any = None  # Исправлено
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Any = None

        self.is_ready = False
        self.llm_available = False

        self.last_index_diagnostics: dict[str, Any] = {}
        self.last_save_diagnostics: dict[str, Any] = {}
        self.last_load_diagnostics: dict[str, Any] = {}

        if TfidfVectorizer is None:
            print("⚠️ scikit-learn не установлен")

        if self.use_embeddings:
            self._try_load_embedding_model()

        if self.use_llm:
            self._validate_llm_config()

    def _reset_index_diagnostics(self) -> None:
        self.last_index_diagnostics = {
            "documents_total": 0,
            "documents_with_chunks": 0,
            "doc_type_counts": {},
            "chunks_total": 0,
            "chunks_nonempty": 0,
            "chunks_skipped_empty": 0,
            "embeddings_enabled": self.use_embeddings,
            "embedding_model_loaded": self.embedding_model is not None,
            "embedding_batches_started": False,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_batches_total": 0,
            "embedding_texts_total": 0,
            "tfidf_enabled": TfidfVectorizer is not None,
            "tfidf_built": False,
            "success": False,
            "error": "",
        }

    def _try_load_embedding_model(self) -> None:
        if SentenceTransformer is None:
            print("⚠️ sentence-transformers не установлен")
            self.embedding_model = None
            return

        try:
            print(f"📥 Загружаю embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print("✅ Embedding model загружена")
        except Exception as e:
            logger.exception("Не удалось загрузить embedding model %s", self.embedding_model_name)
            self.embedding_model = None

    def is_ollama_alive(self) -> bool:
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def is_ollama_available(self) -> bool:
        return self.is_ollama_alive()

    def _validate_llm_config(self) -> None:
        self.llm_available = False

        if not self.use_llm:
            print("ℹ️ LLM отключён")
            return

        if self.llm_provider == "ollama":
            self.llm_available = self.is_ollama_alive()
            print(
                f"{'✅' if self.llm_available else '⚠️'} "
                f"Ollama {'доступен' if self.llm_available else 'недоступен'} "
                f"(base_url={self.ollama_base_url}, model={self.ollama_model})"
            )

        elif self.llm_provider == "mixed":
            self.llm_available = self.is_ollama_alive()
            print(
                f"{'✅' if self.llm_available else '⚠️'} Mixed LLM provider, "
                f"available={self.llm_available}"
            )
        elif self.llm_provider == "none":
            self.llm_available = False
            print("ℹ️ LLM provider=none")
        else:
            self.llm_available = False
            print(f"⚠️ Неизвестный LLM provider: {self.llm_provider}")

    def _select_provider(self) -> str:
        if not self.use_llm:
            return "none"

        if self.llm_provider == "ollama":
            return "ollama" if self.is_ollama_alive() else "none"

        return "none"

    def build_index(self, parsed_docs: list[dict[str, Any]]) -> bool:
        """Строит индекс по документам с подробной диагностикой батчей."""
        self._reset_index_diagnostics()

        try:
            self.documents = parsed_docs or []
            self.chunks = []

            self.last_index_diagnostics["documents_total"] = len(self.documents)

            doc_type_counts: dict[str, int] = {}
            docs_with_chunks = 0
            skipped_empty = 0

            for doc in self.documents:
                doc_name = doc.get("doc_name", "")
                file_path = doc.get("filepath", doc.get("file_path", ""))
                file_type = doc.get("filetype", doc.get("file_type", "unknown"))
                doc_metadata = doc.get("metadata", {}) or {}
                doc_chunks = doc.get("chunks", []) or []

                doc_type_counts[file_type] = doc_type_counts.get(file_type, 0) + 1
                if doc_chunks:
                    docs_with_chunks += 1

                for chunk in doc_chunks:
                    text = (chunk.get("text") or "").strip()
                    if not text:
                        skipped_empty += 1
                        continue

                    self.chunks.append(
                        {
                            "doc_name": chunk.get("doc_name", doc_name),
                            "chunk_id": chunk.get("chunk_id", 0),
                            "text": text,
                            "filepath": file_path,
                            "filetype": file_type,
                            "metadata": {
                                **doc_metadata,
                                **(chunk.get("metadata", {}) or {}),
                            },
                        }
                    )

            self.last_index_diagnostics["documents_with_chunks"] = docs_with_chunks
            self.last_index_diagnostics["doc_type_counts"] = doc_type_counts
            self.last_index_diagnostics["chunks_total"] = len(self.chunks)
            self.last_index_diagnostics["chunks_nonempty"] = len(self.chunks)
            self.last_index_diagnostics["chunks_skipped_empty"] = skipped_empty


            print("📊 INDEX BUILD DIAGNOSTICS")
            print(f"📄 Документов получено: {len(self.documents)}")
            print(f"📄 Документов с чанками: {docs_with_chunks}")
            print(f"🧩 Всего непустых чанков: {len(self.chunks)}")
            print(f"🗑️ Пустых чанков пропущено: {skipped_empty}")
            print(f"🗂️ Типы документов: {doc_type_counts}")

            if not self.chunks:
                print("❌ После сборки нет ни одного непустого чанка")
                self.is_ready = False
                self.last_index_diagnostics["error"] = "no_nonempty_chunks"
                return False

            texts = [c["text"] for c in self.chunks]

            # TF-IDF
            if TfidfVectorizer is not None:
                self.vectorizer = TfidfVectorizer(max_features=50000,ngram_range=(1, 2),min_df=1,sublinear_tf=True,)
                self.tfidf_matrix = self.vectorizer.fit_transform(texts)
                self.last_index_diagnostics["tfidf_built"] = True
                print(f"✅ TF-IDF построен: shape={self.tfidf_matrix.shape}")
            else:
                self.vectorizer = None
                self.tfidf_matrix = None

            # ЭМБЕДДИНГИ С БАТЧАМИ
            if self.embedding_model is not None and texts:
                total_texts = len(texts)
                batch_size = self.embedding_batch_size
                total_batches = math.ceil(total_texts / batch_size)

                self.last_index_diagnostics["embedding_batches_started"] = True
                self.last_index_diagnostics["embedding_batches_total"] = total_batches
                self.last_index_diagnostics["embedding_texts_total"] = total_texts

                print("🚀 ЗАПУСК БАТЧЕЙ ЭМБЕДДИНГОВ")
                print(f"   model={self.embedding_model_name}")
                print(f"   texts={total_texts}")
                print(f"   batch_size={batch_size}")
                print(f"   total_batches={total_batches}")

                all_embeddings: list[np.ndarray] = []

                for batch_idx in range(total_batches):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, total_texts)
                    batch_texts = texts[start:end]

                    print(
                        f"   📦 Batch {batch_idx + 1}/{total_batches}: "
                        f"items={len(batch_texts)} range=[{start}:{end}]"
                    )

                    batch_embeddings = self.embedding_model.encode(
                        batch_texts,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    all_embeddings.append(np.asarray(batch_embeddings, dtype=np.float32))  # type: ignore

                # Объединяем все батчи
                self.chunk_embeddings = np.vstack(all_embeddings) if all_embeddings else None  # type: ignore

                print(
                    f"✅ ЭМБЕДДИНГИ ПОСТРОЕНЫ: "
                    f"shape={None if self.chunk_embeddings is None else self.chunk_embeddings.shape}"
                )

            else:
                self.chunk_embeddings = None
                if not self.use_embeddings:
                    print("ℹ️ Эмбеддинги отключены настройкой use_embeddings=False")
                elif self.embedding_model is None:
                    print("⚠️ Эмбеддинги не построены: embedding model не загружена")

            self.is_ready = True
            self.last_index_diagnostics["success"] = True
            print("✅ Индекс успешно построен")
            return True

        except Exception as e:
            self.is_ready = False
            self.last_index_diagnostics["error"] = str(e)
            print(f"❌ build_index failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def index_documents(self, directory: str | Path) -> bool:
        try:
            from core.parser import DocumentParser

            directory = Path(directory)
            print(f"📂 index_documents: directory={directory}")

            parser = DocumentParser(chunk_size=1200, chunk_overlap=200)
            docs = parser.parse_directory(directory, recursive=True)

            print(f"📄 parser.parse_directory returned documents={len(docs) if docs else 0}")
            return self.build_index(docs)

        except ImportError as e:
            print(f"❌ Не удалось импортировать parser: {e}")
            return False
        except Exception as e:
            print(f"❌ Ошибка индексации: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_index(self, index_path: str | Path) -> bool:
        """Сохраняет индекс на диск с проверкой."""
        self.last_save_diagnostics = {
            "path": str(index_path),
            "documents_total": len(self.documents),
            "chunks_total": len(self.chunks),
            "has_embeddings": self.chunk_embeddings is not None,
            "has_tfidf": self.tfidf_matrix is not None,
            "success": False,
            "file_exists_after_save": False,
            "file_size_bytes": 0,
            "error": "",
        }

        try:
            index_path = Path(index_path)
            index_path.parent.mkdir(parents=True, exist_ok=True)

            if not self.documents:
                print("⚠️ save_index: documents пустой")
            if not self.chunks:
                print("⚠️ save_index: chunks пустой")

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
                "llm_provider": self.llm_provider,
                "diagnostics": self.last_index_diagnostics,
            }

            print("💾 SAVE INDEX DIAGNOSTICS:")
            print(f"   path={index_path}")
            print(f"   documents={len(self.documents)}")
            print(f"   chunks={len(self.chunks)}")
            print(f"   embeddings_present={self.chunk_embeddings is not None}")
            if self.chunk_embeddings is not None:
                print(f"   embeddings_shape={self.chunk_embeddings.shape}")
            print(f"   tfidf_present={self.tfidf_matrix is not None}")

            with open(index_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            file_exists = index_path.exists()
            file_size = index_path.stat().st_size if file_exists else 0

            self.last_save_diagnostics["file_exists_after_save"] = file_exists
            self.last_save_diagnostics["file_size_bytes"] = file_size
            self.last_save_diagnostics["success"] = file_exists and file_size > 0

            if not file_exists:
                print(f"❌ Индекс не сохранился: файл не создан: {index_path}")
                return False

            if file_size <= 0:
                print(f"❌ Индекс не сохранился корректно: пустой файл: {index_path}")
                return False

            print(f"✅ Индекс сохранён: {index_path} ({file_size} bytes)")
            return True

        except Exception as e:
            self.last_save_diagnostics["error"] = str(e)
            print(f"❌ Ошибка сохранения индекса: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_index(self, index_path: str | Path) -> bool:
        """Загружает индекс с диска."""
        self.last_load_diagnostics = {
            "path": str(index_path),
            "success": False,
            "file_exists_before_load": False,
            "file_size_bytes": 0,
            "error": "",
        }

        try:
            index_path = Path(index_path)
            self.last_load_diagnostics["file_exists_before_load"] = index_path.exists()

            if not index_path.exists():
                print(f"⚠️ load_index: файл не найден: {index_path}")
                self.last_load_diagnostics["error"] = "file_not_found"
                return False

            self.last_load_diagnostics["file_size_bytes"] = index_path.stat().st_size
            print(
                f"📥 Загружаю индекс: {index_path} "
                f"({self.last_load_diagnostics['file_size_bytes']} bytes)"
            )

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
            self.chunk_embeddings = data.get("chunk_embeddings")
            self.vectorizer = data.get("vectorizer")
            self.tfidf_matrix = data.get("tfidf_matrix")
            self.ollama_model = data.get("ollama_model", self.ollama_model)
            self.llm_provider = data.get("llm_provider", self.llm_provider)
            self.last_index_diagnostics = data.get("diagnostics", {})

            self.is_ready = bool(self.chunks)
            self.last_load_diagnostics["success"] = self.is_ready

            print("✅ Индекс загружен")
            print(f"   documents={len(self.documents)}")
            print(f"   chunks={len(self.chunks)}")
            print(f"   embeddings_present={self.chunk_embeddings is not None}")
            if self.chunk_embeddings is not None:
                print(f"   embeddings_shape={self.chunk_embeddings.shape}")
            print(f"   tfidf_present={self.tfidf_matrix is not None}")

            return self.is_ready

        except Exception as e:
            self.last_load_diagnostics["error"] = str(e)
            print(f"❌ Ошибка загрузки индекса: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_diagnostics(self) -> dict[str, Any]:
        """Возвращает полную диагностику состояния."""
        return {
            "is_ready": self.is_ready,
            "documents_total": len(self.documents),
            "chunks_total": len(self.chunks),
            "has_embeddings": self.chunk_embeddings is not None,
            "embeddings_shape": str(self.chunk_embeddings.shape) if self.chunk_embeddings is not None else None,
            "has_tfidf": self.tfidf_matrix is not None,
            "selected_provider": self.get_selected_provider(),
            "last_llm_error": self.last_llm_error,
            "last_index_diagnostics": self.last_index_diagnostics,
            "last_save_diagnostics": self.last_save_diagnostics,
            "last_load_diagnostics": self.last_load_diagnostics,
        }

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        """Поиск по индексу (гибридный)."""
        if not self.is_ready or not self.chunks:
            return []

        query = (query or "").strip()
        if not query:
            return []

        top_k = top_k or self.top_k
        lexical_scores = np.zeros(len(self.chunks), dtype=np.float32)  # type: ignore
        semantic_scores = np.zeros(len(self.chunks), dtype=np.float32)  # type: ignore

        # Лексический поиск (TF-IDF)
        if self.vectorizer is not None and self.tfidf_matrix is not None:
            try:
                query_vec = self.vectorizer.transform([query])
                sim = (self.tfidf_matrix @ query_vec.T).toarray().ravel()
                lexical_scores = sim.astype(np.float32)  # type: ignore
            except Exception as e:
                print(f"⚠️ Ошибка lexical search: {e}")

        # Семантический поиск (эмбеддинги)
        if (
                self.embedding_model is not None
                and self.chunk_embeddings is not None
                and len(self.chunk_embeddings) == len(self.chunks)
        ):
            try:
                q_emb = self.embedding_model.encode(
                    [query],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                q_emb = np.asarray(q_emb, dtype=np.float32)[0]  # type: ignore
                semantic_scores = self.chunk_embeddings @ q_emb  # type: ignore
            except Exception as e:
                print(f"⚠️ Ошибка semantic search: {e}")

        # Гибридная комбинация
        if self.chunk_embeddings is not None:
            scores = self.semantic_weight * semantic_scores + self.lexical_weight * lexical_scores
        else:
            scores = lexical_scores

        ranked_ids = np.argsort(scores)[::-1][:top_k]  # type: ignore
        results: list[SearchResult] = []

        for idx in ranked_ids:
            score = float(scores[idx])
            if score < self.min_score:
                continue

            chunk = self.chunks[idx]
            results.append(
                SearchResult(
                    doc_name=chunk.get("doc_name", ""),
                    chunk_id=int(chunk.get("chunk_id", 0)),
                    text=chunk.get("text", ""),
                    score=score,
                    semantic_score=float(semantic_scores[idx]) if len(semantic_scores) else 0.0,
                    lexical_score=float(lexical_scores[idx]) if len(lexical_scores) else 0.0,
                    filepath=chunk.get("filepath", ""),
                    metadata=chunk.get("metadata", {}) or {},
                )
            )

        return results

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> dict[str, Any]:
        """Отвечает на вопрос."""
        results = self.search(question, top_k=top_k)
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

            if llm_answer and self._has_valid_citations(llm_answer, len(selected_results)):
                return {
                    "answer": self._ensure_structured_answer(llm_answer),
                    "sources": self._build_sources(results),
                    "tables": tables_to_dicts(extract_tables_from_results(results)),
                    "formulas": self._extract_formulas_from_results(results),
                    "provider": provider,
                    "used_llm": True,
                    "context": context,
                }

            if llm_answer:
                logger.warning(
                    "Ответ %s отклонён: в нём нет корректных ссылок на контекст",
                    provider,
                )
        fallback_answer = self._generate_extract_answer(selected_results, question=question)
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
        """Собирает источники с теми же номерами, что и в контексте LLM."""
        sources: list[dict[str, Any]] = []
        for reference_id, item in enumerate(results, start=1):
            metadata = dict(item.metadata or {})
            source = {
                "reference_id": reference_id,
                "doc_name": item.doc_name,
                "chunk_id": item.chunk_id,
                "score": item.score,
                "filepath": item.filepath,
            }
            for key in ("page", "page_numbers", "sheet", "table_title", "location"):
                if metadata.get(key) not in (None, "", []):
                    source[key] = metadata[key]
            sources.append(source)
        return sources

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
        sentences = re.split(r"(?<=[.!?])\s+", text)
        term_lower = term.lower()

        best = ""
        best_score = -1.0
        for sentence in sentences:
            lower = sentence.lower()
            if term_lower not in lower:
                continue

            # чем короче — тем лучше; чем раньше — тем лучше
            length_penalty = 1.0 / (1 + len(sentence) / 200.0)
            position_bonus = 1.0 if lower.find(term_lower) < 100 else 0.7
            score = length_penalty * position_bonus

            if score > best_score:
                best_score = score
                best = sentence.strip()

        return best

    def _ask_ollama(self, prompt: str) -> Optional[str]:
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
            self.last_llm_error = f"Ollama: {type(e).__name__}: {e}"
            logger.exception("Ошибка запроса к Ollama")
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
    def _has_valid_citations(answer: str, source_count: int) -> bool:
        """Проверяет, что LLM ссылается только на переданные фрагменты."""
        references = [
            int(value)
            for value in re.findall(r"\[Источник\s+(\d+)\]", answer, flags=re.IGNORECASE)
        ]
        return bool(references) and all(1 <= value <= source_count for value in references)

    @staticmethod
    def _ensure_structured_answer(answer: str) -> str:
        """Добавляет заголовок, если модель не оформила краткий ответ."""
        if re.search(r"^#{1,3}\s*кратк", answer, flags=re.IGNORECASE | re.MULTILINE):
            return answer.strip()
        return "### Краткий ответ\n\n" + answer.strip()

    @staticmethod
    def _query_terms(question: str) -> set[str]:
        stopwords = {
            "какой", "какая", "какие", "как", "что", "это", "для", "про",
            "нужно", "нужен", "нужна", "пожалуйста", "покажи", "найди",
            "расскажи", "документ", "документа", "базе", "знаний",
        }
        return {
            token.lower()
            for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9_]{3,}", question or "")
            if token.lower() not in stopwords
        }

    @classmethod
    def _select_evidence_excerpt(cls, text: str, question: str) -> str:
        """Выбирает короткое дословное доказательство вместо сырого чанка."""
        clean = re.sub(r"[ \t]+", " ", text or "").strip()
        if not clean:
            return ""
        terms = cls._query_terms(question)
        candidates = re.split(r"(?<=[.!?])\s+|\n+", clean)
        scored: list[tuple[int, int, str]] = []
        for position, candidate in enumerate(candidates):
            candidate = candidate.strip()
            if len(candidate) < 15:
                continue
            lower = candidate.lower()
            hits = sum(1 for term in terms if term in lower)
            scored.append((hits, -position, candidate))

        if scored:
            scored.sort(reverse=True)
            excerpt = scored[0][2]
        else:
            excerpt = clean

        if len(excerpt) > 420:
            excerpt = excerpt[:420].rsplit(" ", 1)[0].rstrip() + "…"
        return excerpt

    @classmethod
    def _generate_extract_answer(
            cls,
            results: list[SearchResult],
            question: str = "",
    ) -> str:
        """Строит структурированный extractive-ответ, когда LLM недоступна."""

        parts: list[str] = [
            "### Краткий ответ","Генеративная модель недоступна; ниже приведены только "
                "дословные релевантные фрагменты из подключённой базы."
            )
        ]
        excerpts: list[str] = []
        for index, result in enumerate(results[:3], start=1):
            excerpt = cls._select_evidence_excerpt(result.text, question)
        if excerpt:
            excerpts.append(f"{excerpt} [Источник {index}]")

        if excerpts:
            parts.extend(excerpts[:2])
        # 1. Сначала таблицы — они самые информативные
        try:
            tables = extract_tables_from_results(results, limit=3)
        except (AttributeError, TypeError, ValueError, KeyError) as exc:
            logger.exception("extract_tables_from_results failed")
            tables = []

        if tables:
            parts.append("📊 **Найденные таблицы:**")
            for table in tables[:3]:
                parts.append(f"\n**{table.title or 'Таблица'}** (источник: {table.source})")

                if table.headers or table.rows:
                    header = " | ".join(str(h) for h in table.headers) if table.headers else ""
                    if header:
                        parts.append(header)
                        parts.append("-" * min(len(header), 80))
                    for row in table.rows[:15]:
                        parts.append(" | ".join(str(c) for c in row))
                else:
                    # fallback: сырой текст таблицы
                    parts.append(table.raw_text[:800])
            parts.append("")

        # 2. Потом текстовые фрагменты — коротко
        text_fragments = []
        for r in results[:3]:
            text = r.text.strip()
            if text and len(text) > 20:
                text_fragments.append(text[:600])

        if text_fragments:
            parts.append("📄 **Текстовые фрагменты:**")
            parts.append("\n\n".join(text_fragments))

        if not parts:
            return "Не удалось найти информацию в документах."

        return f"### Ответ\n\n" + "\n".join(parts)