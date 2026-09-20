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
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional
import logging
import numpy as np
import requests

from core.table_extractor import extract_tables_from_results, tables_to_dicts
from core.prompts import get_system_prompt

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

logger = logging.getLogger(__name__)
DOCUMENT_REFERENCE_RE = re.compile(
    r"(?<!\w)(СП|СНиП|ГОСТ(?:\s+Р)?)\s*(\d+(?:[.-]\d+)*)", re.IGNORECASE,
)


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

    def to_dict(self) -> dict[str, Any]:
        """Словарь для совместимого обмена данными с оркестратором."""
        return {
            "doc_name": self.doc_name, "chunk_id": self.chunk_id,
            "text": self.text, "score": self.score,
            "semantic_score": self.semantic_score, "lexical_score": self.lexical_score,
            "filepath": self.filepath, "metadata": dict(self.metadata),
        }

    @classmethod
    def from_value(cls, value: SearchResult | Mapping[str, Any]) -> SearchResult:
        """Принимает публичный SearchResult или его словарное представление."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("Результат поиска должен быть SearchResult или словарём")
        return cls(
            doc_name=str(value.get("doc_name", "")),
            chunk_id=int(value.get("chunk_id", 0) or 0),
            text=str(value.get("text", "") or ""),
            score=float(value.get("score", 0) or 0),
            semantic_score=float(value.get("semantic_score", 0) or 0),
            lexical_score=float(value.get("lexical_score", 0) or 0),
            filepath=str(value.get("filepath", "")),
            metadata=dict(value.get("metadata", {}) or {}),
        )


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
        self.top_k = max(1, int(top_k))
        self.min_score = max(0.0, min(1.0, float(min_score)))

        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = model_name if model_name and self.llm_provider in {"ollama", "mixed"} else ollama_model

        self.embedding_model_name = embedding_model_name
        self.use_embeddings = use_embeddings
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.embedding_batch_size = max(1, int(embedding_batch_size))

        self.documents: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []

        self.embedding_model: Any = None
        self.chunk_embeddings: Any = None
        self.vectorizer: Any = None
        self.tfidf_matrix: Any = None

        self.is_ready = False
        self.llm_available = False
        self.last_llm_error = ""

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
        except Exception:
            logger.exception("Не удалось загрузить embedding model %s", self.embedding_model_name)
            self.embedding_model = None

    def is_ollama_alive(self) -> bool:
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def is_ollama_available(self) -> bool:
        """Проверяет не только сервер, но и наличие настроенной модели."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
            response.raise_for_status()
            data = response.json()
            names = {
                str(item.get("name") or item.get("model") or "")
                for item in data.get("models", []) if isinstance(item, dict)
            }
            wanted = self.ollama_model if ":" in self.ollama_model else self.ollama_model + ":latest"
            available = self.ollama_model in names or wanted in names
            self.last_llm_error = "" if available else f"В Ollama не установлена модель {self.ollama_model}"
            self.llm_available = available
            return available
        except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
            self.last_llm_error = f"Ollama недоступна: {type(exc).__name__}: {exc}"
            self.llm_available = False
            return False

    def _validate_llm_config(self) -> None:
        self.llm_available = False

        if not self.use_llm:
            print("ℹ️ LLM отключён")
            return

        if self.llm_provider == "ollama":
            self.llm_available = self.is_ollama_available()
            print(
                f"{'✅' if self.llm_available else '⚠️'} "
                f"Ollama {'доступен' if self.llm_available else 'недоступен'} "
                f"(base_url={self.ollama_base_url}, model={self.ollama_model})"
            )

        elif self.llm_provider == "mixed":
            self.llm_available = self.is_ollama_available()
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

        if self.llm_provider in {"ollama", "mixed"}:
            return "ollama" if self.is_ollama_available() else "none"

        return "none"

    def get_selected_provider(self) -> str:
        """Возвращает фактически доступный провайдер (Ollama либо none)."""
        return self._select_provider()

    def build_index(self, parsed_docs: list[dict[str, Any]]) -> bool:
        """Строит индекс по документам с подробной диагностикой батчей."""
        self._reset_index_diagnostics()
        self.is_ready = False
        self.vectorizer = None
        self.tfidf_matrix = None
        self.chunk_embeddings = None

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
                # Символьные n-граммы находят русские формы «вентиляция»/
                # «вентиляции» без дополнительной морфологической библиотеки.
                vectorizer = TfidfVectorizer(
                    max_features=50000, analyzer="char_wb", ngram_range=(3, 5),
                    min_df=1, sublinear_tf=True,
                )
                self.tfidf_matrix = vectorizer.fit_transform(texts)
                self.vectorizer = vectorizer
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

                    try:
                        batch_embeddings = np.asarray(self.embedding_model.encode(
                            batch_texts,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                        ), dtype=np.float32)
                        if batch_embeddings.ndim != 2 or len(batch_embeddings) != len(batch_texts):
                            raise ValueError("Модель вернула неверную размерность эмбеддингов")
                        if not np.isfinite(batch_embeddings).all():
                            raise ValueError("Модель вернула NaN/inf в эмбеддингах")
                        all_embeddings.append(batch_embeddings)
                    except Exception as exc:
                        logger.exception("Ошибка эмбеддингов; сохраняется лексический поиск")
                        self.last_index_diagnostics["embedding_error"] = str(exc)
                        all_embeddings.clear()
                        break

                # Объединяем все батчи
                self.chunk_embeddings = np.vstack(all_embeddings) if all_embeddings else None  # type: ignore

                print(
                    f"{'✅ ЭМБЕДДИНГИ ПОСТРОЕНЫ' if self.chunk_embeddings is not None else '⚠️ ЭМБЕДДИНГИ НЕДОСТУПНЫ'}: "
                    f"shape={None if self.chunk_embeddings is None else self.chunk_embeddings.shape}"
                )

            else:
                self.chunk_embeddings = None
                if not self.use_embeddings:
                    print("ℹ️ Эмбеддинги отключены настройкой use_embeddings=False")
                elif self.embedding_model is None:
                    print("⚠️ Эмбеддинги не построены: embedding model не загружена")

            self.is_ready = self.tfidf_matrix is not None or self.chunk_embeddings is not None
            self.last_index_diagnostics["success"] = self.is_ready
            if not self.is_ready:
                self.last_index_diagnostics["error"] = "no_search_index"
                return False
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

        temporary_path: Path | None = None
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

            # Только полностью записанный файл заменяет рабочий индекс.
            # Временный файл находится на том же диске для атомарного replace.
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=index_path.parent, prefix=f".{index_path.name}.",
                suffix=".tmp", delete=False,
            ) as f:
                temporary_path = Path(f.name)
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            temporary_path.replace(index_path)
            temporary_path = None

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
            logger.exception("Ошибка сохранения индекса %s", index_path)
            return False
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Не удалось удалить временный файл индекса %s", temporary_path, exc_info=True)

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

            if not isinstance(data, dict):
                raise ValueError("Неверный формат файла индекса")
            documents = data.get("documents", [])
            chunks = data.get("chunks", [])
            if not isinstance(documents, list) or not isinstance(chunks, list):
                raise ValueError("documents/chunks должны быть списками")
            if not chunks or any(not isinstance(chunk, dict) for chunk in chunks):
                raise ValueError("В индексе отсутствуют корректные фрагменты")
            vectorizer = data.get("vectorizer")
            tfidf_matrix = data.get("tfidf_matrix")
            if vectorizer is None or not callable(getattr(vectorizer, "transform", None)):
                tfidf_matrix = None
            if tfidf_matrix is not None and tfidf_matrix.shape[0] != len(chunks):
                raise ValueError("Число строк TF-IDF не совпадает с числом фрагментов")
            embeddings = data.get("chunk_embeddings")
            stored_model = data.get("embedding_model_name")
            if embeddings is not None:
                embeddings = np.asarray(embeddings, dtype=np.float32)
                if (
                    not self.use_embeddings or stored_model != self.embedding_model_name
                    or embeddings.ndim != 2 or len(embeddings) != len(chunks)
                    or not np.isfinite(embeddings).all()
                ):
                    logger.warning("Эмбеддинги индекса несовместимы с текущими настройками; используется TF-IDF")
                    embeddings = None
            if tfidf_matrix is None and (embeddings is None or self.embedding_model is None):
                raise ValueError("В индексе нет доступных поисковых данных; перестройте индекс")
            # Настройки текущего запуска и уже загруженную модель нельзя
            # заменять настройками из старого pickle.
            self.documents, self.chunks = documents, chunks
            self.vectorizer, self.tfidf_matrix = vectorizer, tfidf_matrix
            self.chunk_embeddings = embeddings
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
            logger.exception("Ошибка загрузки индекса %s", index_path)
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

        document_refs = self._document_references(query)
        allowed = np.ones(len(self.chunks), dtype=bool)
        if document_refs:
            allowed = np.asarray([
                self._matches_requested_document(chunk.get("doc_name", ""), document_refs)
                for chunk in self.chunks
            ])
            if not allowed.any():
                return []
        # Номер СП обозначает фильтр документа, а не тему ответа. Иначе
        # библиография с этим номером вытесняет содержательный нужный пункт.
        content_query = DOCUMENT_REFERENCE_RE.sub(" ", query)
        content_query = re.sub(r"\b(?:по|согласно)\s*$", " ", content_query, flags=re.IGNORECASE).strip()
        lexical_query = content_query or query

        top_k = self.top_k if top_k is None else max(0, int(top_k))
        if top_k == 0:
            return []
        lexical_scores = np.zeros(len(self.chunks), dtype=np.float32)  # type: ignore
        semantic_scores = np.zeros(len(self.chunks), dtype=np.float32)  # type: ignore
        lexical_available = False
        semantic_available = False

        # Лексический поиск (TF-IDF)
        if self.vectorizer is not None and self.tfidf_matrix is not None:
            try:
                query_vec = self.vectorizer.transform([lexical_query])
                sim = (self.tfidf_matrix @ query_vec.T).toarray().ravel()
                lexical_scores = sim.astype(np.float32)  # type: ignore
                lexical_available = bool(query_vec.nnz)
            except Exception:
                logger.exception("Ошибка лексического поиска")

        # Семантический поиск (эмбеддинги)
        if (
                self.use_embeddings
                and self.embedding_model is not None
                and self.chunk_embeddings is not None
                and len(self.chunk_embeddings) == len(self.chunks)
        ):
            try:
                q_emb = self.embedding_model.encode(
                    [lexical_query],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                q_emb = np.asarray(q_emb, dtype=np.float32)[0]  # type: ignore
                semantic_scores = self.chunk_embeddings @ q_emb  # type: ignore
                if not np.isfinite(semantic_scores).all():
                    raise ValueError("Некорректные оценки семантического поиска")
                semantic_available = True
            except Exception:
                logger.exception("Ошибка семантического поиска; используется TF-IDF")

        # Гибридная комбинация
        if semantic_available and lexical_available:
            semantic_weight = max(0.0, self.semantic_weight)
            lexical_weight = max(0.0, self.lexical_weight)
            total_weight = semantic_weight + lexical_weight
            scores = (
                (semantic_weight * semantic_scores + lexical_weight * lexical_scores) / total_weight
                if total_weight else (semantic_scores + lexical_scores) / 2
            )
        elif semantic_available:
            scores = semantic_scores
        elif lexical_available:
            scores = lexical_scores
        else:
            return []

        clause_refs = self._requested_clauses(query)
        definition_term = self._definition_term(query)
        normative_query = not definition_term and self._requires_exact_evidence(query)
        if clause_refs:
            clause_matches = np.asarray([
                any(self._has_clause_start(chunk.get("text", ""), clause) for clause in clause_refs)
                for chunk in self.chunks
            ])
            allowed &= clause_matches
            if not allowed.any():
                return []
        for index, chunk in enumerate(self.chunks):
            if not allowed[index]:
                scores[index] = -np.inf
                continue
            text = chunk.get("text", "")
            # Точный номер пункта и заголовок определения — сильные
            # детерминированные признаки, независимые от модели эмбеддингов.
            if clause_refs and any(self._has_clause_start(text, clause) for clause in clause_refs):
                scores[index] = max(scores[index], 0.85 + min(max(float(scores[index]), 0), 1) * 0.1)
            if definition_term and self._has_definition(text, definition_term):
                scores[index] = max(scores[index], 0.8 + min(max(float(scores[index]), 0), 1) * 0.1)
            if normative_query and not clause_refs:
                evidence = self._normative_excerpt(text, query)
                if evidence and self._topic_support_score(evidence, query) >= 0.9:
                    scores[index] = max(scores[index], 0.85 + min(max(float(scores[index]), 0), 1) * 0.1)

        ranked_ids = np.argsort(scores)[::-1][:top_k]  # type: ignore
        results: list[SearchResult] = []

        for idx in ranked_ids:
            score = float(scores[idx])
            if not np.isfinite(score) or score <= 0 or score < self.min_score:
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

    @staticmethod
    def _document_references(text: str) -> list[tuple[str, str]]:
        """Извлекает обозначения документов, включая сокращённое «СП 60»."""
        return [(re.sub(r"\s+", "", kind).lower(), number)
                for kind, number in DOCUMENT_REFERENCE_RE.findall(text or "")]

    @classmethod
    def _matches_requested_document(cls, name: str, references: list[tuple[str, str]]) -> bool:
        for kind, number in cls._document_references(name):
            if any(kind == requested_kind and (number == requested_number or number.startswith(requested_number + "."))
                   for requested_kind, requested_number in references):
                return True
        return False

    @staticmethod
    def _requested_clauses(query: str) -> list[str]:
        return re.findall(r"(?:\bпункт(?:а|е|ы|ов)?|\bп\.)\s*(\d+(?:\.\d+)+)(?!\d)", query, flags=re.IGNORECASE)

    @staticmethod
    def _has_clause_start(text: str, clause: str) -> bool:
        return bool(re.search(r"(?:^|\n)\s*" + re.escape(clause) + r"(?:\s|[)]|$)", text))

    @staticmethod
    def _definition_term(question: str) -> str:
        match = re.search(
            r"(?:что\s+(?:такое|означает|значит)|определение(?:\s+термина)?|^термин|^понятие)\s+(.+)",
            question, flags=re.IGNORECASE,
        )
        if not match:
            return ""
        term = DOCUMENT_REFERENCE_RE.sub("", match.group(1))
        # Условия поиска и второе предложение не входят в название термина.
        term = re.split(r"[?!.;\n]|\s+(?:по|согласно|из|в)\s+", term, maxsplit=1, flags=re.IGNORECASE)[0]
        term = re.sub(r"\s+(?:по|согласно|из|в)\s*$", "", term, flags=re.IGNORECASE)
        return term.strip(" \t?!.«»\"'").lower()

    @classmethod
    def _has_definition(cls, text: str, term: str) -> bool:
        return bool(cls._definition_excerpt(text, term))

    @staticmethod
    def _evidence_units(text: str) -> list[str]:
        """Сохраняет условия числовой нормы вместе с её пунктом."""
        # Не разрезаем пункт на отдельные предложения: это теряет исключения.
        numbered = bool(re.search(r"(?:^|\n)\s*\d+(?:\.\d+)+\s+", text or ""))
        boundary = r"\n(?=\s*\d+(?:\.\d+)+\s+)" if numbered else r"\n\s*\n"
        return [part.strip() for part in re.split(boundary, text or "") if part.strip()]

    @classmethod
    def _definition_excerpt(cls, text: str, term: str) -> str:
        """Прямое определение с заголовком; упоминание/ссылка недостаточны."""
        if not term:
            return ""
        pattern = re.compile(
            r"(?:^|\n|(?<=[.!?])\s+)\s*(?:\d+(?:\.\d+)*\s+)?"
            + re.escape(term) + r"\s*(?::|[—–]|\s-\s|\bэто\s)\s*(\S.+)",
            re.IGNORECASE,
        )
        for unit in cls._evidence_units(text):
            match = pattern.search(unit)
            if not match:
                continue
            prefix = unit[:match.start()]
            if re.search(r"библиограф|нормативные\s+ссылки", prefix, re.IGNORECASE):
                continue
            value = match.group(1)
            if re.match(r"(?:см\.\s*|по\s+|согласно\s+)?(?:ГОСТ|СП|СНиП|\[\d+\])", value, re.IGNORECASE):
                continue
            return unit[match.start():].strip()
        return ""

    @classmethod
    def _requires_exact_evidence(cls, question: str) -> bool:
        return bool(cls._definition_term(question) or cls._document_references(question)
                    or cls._requested_clauses(question) or re.search(
                        r"норм[аыуе]|норматив|требован|требует|допуска|допустим|"
                        r"должн|следует|минимальн|максимальн|не\s+(?:менее|более)",
                        question, re.IGNORECASE,
                    ))

    @classmethod
    def _topic_support_score(cls, text: str, question: str) -> float:
        query = DOCUMENT_REFERENCE_RE.sub("", question)
        query = re.sub(r"(?:пункт\w*|п\.)\s*\d+(?:\.\d+)*", "", query, flags=re.IGNORECASE)
        generic = ("како", "каку", "каки", "кака", "определ", "такое", "означ", "значит",
                   "термин", "поняти", "соглас", "треб", "норм", "допус", "долж", "следу", "найти",
                   "минималь", "максималь", "обеспеч", "предусматр", "явля", "составля")
        words = cls._query_terms(query) - {"при", "или", "либо", "более", "менее", "пункт", "такой"}
        stems = {word[:max(4, min(7, len(word) - 2))] for word in words
                 if not word.isdigit() and not word.startswith(generic)}
        if not stems:
            return float(bool(cls._requested_clauses(question)))
        lower = text.lower().replace("ё", "е")
        hits = sum(stem.replace("ё", "е") in lower for stem in stems)
        return hits / len(stems)

    @classmethod
    def _has_topic_support(cls, text: str, question: str) -> bool:
        return cls._topic_support_score(text, question) >= 0.5

    @classmethod
    def _normative_excerpt(cls, text: str, question: str) -> str:
        clauses = cls._requested_clauses(question)
        candidates = []
        for unit in cls._evidence_units(text):
            if re.match(r"(?:\[\d+\]\s*)?(?:ГОСТ|СП|СНиП)\s*\d", unit, re.IGNORECASE):
                continue
            if re.search(r"^(?:\d+\s+)?(?:Библиография|Нормативные ссылки)\b", unit, re.IGNORECASE):
                continue
            if clauses:
                if any(cls._has_clause_start(unit, clause) for clause in clauses):
                    return unit
                continue
            if not cls._has_topic_support(unit, question):
                continue
            if not re.search(r"должн|требует|следует|необходим|допуска|не\s+(?:менее|более)|"
                             r"принима[ею]|предусматр|устанавлива|запрещ", unit, re.IGNORECASE):
                continue
            candidates.append(unit)
        if not candidates:
            return ""
        # Только выбор, без генерации или пересказа числовых ограничений.
        return max(candidates, key=lambda unit: cls._topic_support_score(unit, question))

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> dict[str, Any]:
        """Отвечает на вопрос."""
        results = self.search(question, top_k=top_k)
        return self.answer_from_results(question, results)

    def answer_from_results(
        self,
        question: str,
        results: list[SearchResult | Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Отвечает по уже отобранным фрагментам без повторного поиска.

        Определения и нормы возвращаются дословно по прямому свидетельству.
        Для обычного объяснения проверка ссылок не доказывает все выводы LLM.
        """
        normalized_results = self._normalize_search_results(results)
        # Оркестратор расширяет запрос и может принести фрагменты из других
        # документов. Повторно применяем ограничения исходного вопроса.
        document_refs = self._document_references(question)
        if document_refs:
            normalized_results = [item for item in normalized_results
                                  if self._matches_requested_document(item.doc_name, document_refs)]
        clause_refs = self._requested_clauses(question)
        if clause_refs:
            normalized_results = [item for item in normalized_results
                                  if any(self._has_clause_start(item.text, clause) for clause in clause_refs)]
        definition_term = self._definition_term(question)
        exact_evidence = self._requires_exact_evidence(question)
        if exact_evidence:
            supported = []
            for item in normalized_results:
                excerpt = (self._definition_excerpt(item.text, definition_term) if definition_term
                           else self._normative_excerpt(item.text, question))
                if excerpt:
                    supported.append(replace(item, text=excerpt))
            normalized_results = supported
            if not definition_term:
                # Семантически близкая норма для другого режима (охлаждение /
                # радиационный обогрев) не должна вытеснять более точное условие.
                normalized_results.sort(key=lambda item: self._topic_support_score(item.text, question), reverse=True)
        else:
            normalized_results = [item for item in normalized_results
                                  if self._has_topic_support(item.text, question)]
        if not normalized_results:
            return {
                "answer": (
                    "### Краткий ответ\n\nВ базе знаний не найдены фрагменты, "
                    "достаточные для ответа. "
                    + ("Прямое определение термина в найденном тексте отсутствует. " if definition_term else "")
                    + "Уточните документ, раздел или ключевой термин."
                ),
                "sources": [],
                "tables": [],
                "formulas": [],
                "provider": "none",
                "used_llm": False,
                "context": "",
                "confidence": 0.0,
                "needs_clarification": True,
                "questions": ["Какой документ или раздел нужно проверить?"],
                "grounded": True,
                "evidence_mode": "insufficient",
            }

        # Ограничение контекста удерживает запрос в памяти локальной модели.
        # Копии не изменяют исходные результаты поиска и сам индекс.
        selected_results = [
            replace(item, text=(item.text if exact_evidence else self._context_excerpt(item.text, question)),
                    doc_name=item.doc_name[:200])
            for item in normalized_results[:max(1, min(self.top_k, 6))]
        ]
        context = self._build_context(selected_results)
        sources = self._build_sources(selected_results)
        try:
            tables = tables_to_dicts(extract_tables_from_results(selected_results, min_rows=1))
        except (AttributeError, KeyError, TypeError, ValueError):
            logger.exception("Не удалось извлечь таблицы из найденных фрагментов")
            tables = []
        payload = {
            "sources": sources,
            "tables": tables,
            "formulas": self._extract_formulas_from_results(selected_results),
            "context": context,
            "confidence": round(max(0.0, min(1.0, max(item.score for item in selected_results))), 3),
            "needs_clarification": False,
            "questions": [],
            "grounded": True,
            "evidence_mode": "exact_excerpt" if exact_evidence else "retrieved_context",
        }
        self.last_llm_error = ""
        if exact_evidence:
            # Допустимый номер ссылки не даёт LLM права добавлять норму/число.
            excerpt = selected_results[0].text
            answer = ("### Краткий ответ\n\nТочная выдержка из найденного документа:\n\n> "
                      + excerpt.replace("\n", "\n> ") + "\n\n[Источник 1]\n\n### Ограничения\n\n"
                      "Выдержка приведена без генеративного пересказа. Проверьте область применения "
                      "и условия в документе; она может не покрывать все условия вопроса.")
            return {**payload, "answer": answer, "provider": "none", "used_llm": False}
        provider = self._select_provider()

        if provider != "none":
            prompt = self._build_prompt(question, context)
            llm_answer = self._ask_ollama(prompt)

            if llm_answer and self._has_valid_citations(llm_answer, len(selected_results)):
                return {
                    **payload,
                    "answer": self._ensure_structured_answer(llm_answer),
                    "provider": provider,
                    "used_llm": True,
                }

            if llm_answer:
                logger.warning(
                    "Ответ %s отклонён: в нём нет корректных ссылок на контекст",
                    provider,
                )
                self.last_llm_error = "Ответ модели не содержит корректных ссылок на найденные фрагменты"
        fallback_answer = self._generate_extract_answer(selected_results, question=question)
        return {
            **payload,
            "answer": fallback_answer,
            "provider": "none",
            "used_llm": False,
            "llm_error": self.last_llm_error,
        }

    @staticmethod
    def _normalize_search_results(
        results: list[SearchResult | Mapping[str, Any]] | None,
    ) -> list[SearchResult]:
        normalized: list[SearchResult] = []
        seen: set[tuple[str, int, str]] = set()
        for value in results or []:
            try:
                item = SearchResult.from_value(value)
            except (TypeError, ValueError, OverflowError):
                logger.warning("Пропущен некорректный результат поиска", exc_info=True)
                continue
            key = (item.doc_name, item.chunk_id, item.text)
            if item.text.strip() and np.isfinite(item.score) and key not in seen:
                normalized.append(item)
                seen.add(key)
        return normalized

    @classmethod
    def _context_excerpt(cls, text: str, question: str) -> str:
        """Сохраняет до 1600 символов вокруг предложения по теме вопроса."""
        if len(text) <= 1600:
            return text
        excerpt = cls._select_evidence_excerpt(text, question)
        position = text.find(excerpt.rstrip("…"))
        start = max(0, position - 250) if position >= 0 else 0
        if start:
            line_start = text.rfind("\n", max(0, start - 150), start)
            if line_start >= 0:
                start = line_start + 1
        end = min(len(text), start + 1580)
        return ("…\n" if start else "") + text[start:end] + ("\n…" if end < len(text) else "")

    def find_definition(self, term: str) -> dict[str, Any]:
        """Ищет определение термина."""
        query = f"определение {term}"
        results = self.search(query, top_k=5)

        for item in results:
            sentence = self._definition_excerpt(item.text, term)
            if sentence:
                return {"found": True, "definition": sentence, "source": item.doc_name}

        return {"found": False, "definition": "", "source": ""}

    @staticmethod
    def _build_context(results: list[SearchResult]) -> str:
        """Собирает контекст."""
        parts = []
        for i, result in enumerate(results[:6], start=1):
            metadata = result.metadata or {}
            location = ""
            if metadata.get("page"):
                location = f"; страница {metadata['page']}"
            elif metadata.get("page_numbers"):
                location = f"; страницы {metadata['page_numbers']}"
            if metadata.get("table_title"):
                location += f"; {metadata['table_title']}"
            parts.append(
                f"[Источник {i}] {result.doc_name[:200]}\n"
                f"Фрагмент #{result.chunk_id}{location[:150]}\n"
                f"{result.text[:1600]}"
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
                    "options": {"temperature": 0.1, "num_predict": 700, "num_ctx": 8192},
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            answer = data.get("response")
            if not isinstance(answer, str):
                raise ValueError("Ollama вернула ответ без текстового поля response")
            self.last_llm_error = ""
            return answer.strip() or None
        except (requests.RequestException, ValueError, TypeError, AttributeError) as e:
            self.last_llm_error = f"Ollama: {type(e).__name__}: {e}"
            logger.exception("Ошибка запроса к Ollama")
            return None

    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        """Формирует промпт."""
        return f"""
{get_system_prompt()}

НАЧАЛО КОНТЕКСТА
{context}
КОНЕЦ КОНТЕКСТА

Вопрос:
{question[:3000]}
""".strip()

    @staticmethod
    def _has_valid_citations(answer: str, source_count: int) -> bool:
        """Только синтаксис ссылок, не проверка содержательной обоснованности."""
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
        clauses = cls._requested_clauses(question)
        definition = cls._definition_term(question)
        candidates = re.split(r"(?<=[.!?])\s+|\n+", clean)
        scored: list[tuple[int, int, str]] = []
        for position, candidate in enumerate(candidates):
            candidate = candidate.strip()
            if len(candidate) < 15 or re.match(r"^\[(?:СТРАНИЦА|ТАБЛИЦА|ИСТОЧНИК)\b", candidate, flags=re.IGNORECASE):
                continue
            lower = candidate.lower()
            hits = sum(1 for term in terms if term in lower)
            if any(cls._has_clause_start(candidate, clause) for clause in clauses):
                hits += 100
            if definition and cls._has_definition(candidate, definition):
                hits += 90
            scored.append((hits, -position, candidate))

        if scored:
            scored.sort(reverse=True)
            excerpt = scored[0][2]
            position = -scored[0][1]
            # Не теряем условие/исключение сразу после выбранного предложения.
            if position + 1 < len(candidates):
                following = candidates[position + 1].strip()
                if re.match(r"^(?:При|Если|Кроме|Для этого|В этом случае|В противном случае)\b", following):
                    if len(excerpt) + len(following) < 750:
                        excerpt += " " + following
        else:
            excerpt = clean

        if len(excerpt) > 750:
            excerpt = excerpt[:750].rsplit(" ", 1)[0].rstrip() + "…"
        return excerpt

    @classmethod
    def _generate_extract_answer(
            cls,
            results: list[SearchResult],
            question: str = "",
    ) -> str:
        """Строит структурированный extractive-ответ, когда LLM недоступна."""
        if not results:
            return "### Краткий ответ\n\nВ базе знаний недостаточно данных для точного ответа."
        parts: list[str] = [
            "### Краткий ответ",
            "По найденным документам:",
        ]
        excerpts: list[str] = []
        # Второй по TF-IDF фрагмент может касаться другого условия (например,
        # температуры для другого типа помещения). Без LLM не смешиваем нормы.
        for index, result in enumerate(results[:1], start=1):
            excerpt = cls._select_evidence_excerpt(result.text, question)
            if excerpt:
                excerpts.append(f"> {excerpt}\n\n[Источник {index}]")

        if excerpts:
            parts.append(excerpts[0])
            if len(excerpts) > 1:
                parts.extend(["### Дополнительные фрагменты", *excerpts[1:]])
        parts.extend([
            "### Ограничения",
            "Показаны выдержки без генеративного анализа. Они могут не покрывать "
            "все условия вопроса; проверьте область применения в указанном документе.",
        ])
        return "\n\n".join(parts)


def _run_self_tests() -> None:
    """Офлайн-проверки контракта, поиска, контекста и восстановления индекса."""
    import tempfile

    qa = QASystem(use_llm=False, use_embeddings=False, min_score=0.05)
    docs = [{"doc_name": "Вентиляция.docx", "chunks": [{
        "chunk_id": 1, "text": "Вентиляция обеспечивает организованный воздухообмен в помещениях.",
        "metadata": {"page": 4},
    }]}]
    assert qa.build_index(docs)
    found = qa.search("Как устроена вентиляция?")
    assert found and found[0].doc_name == "Вентиляция.docx"
    assert qa.search("вентиляции")[0].doc_name == "Вентиляция.docx"
    lexical_score = found[0].score
    qa.chunk_embeddings = np.ones((1, 3), dtype=np.float32)
    assert abs(qa.search("Как устроена вентиляция?")[0].score - lexical_score) < 1e-6
    answer = qa.answer_from_results("вентиляция", [found[0].to_dict()])
    assert "[Источник 1]" in answer["answer"] and answer["sources"][0]["page"] == 4
    assert not answer["used_llm"] and answer["provider"] == "none"
    assert qa.answer_from_results("нет данных", [])["needs_clarification"]
    assert not qa._has_valid_citations("Нет источников", 1)
    assert not qa._has_valid_citations("Норма [Источник 2]", 1)
    assert qa._has_valid_citations("Факт [Источник 1]", 1)
    assert qa._definition_term("Что такое рабочая зона по СП 60.13330? Найди определение.") == "рабочая зона"
    assert qa._has_definition("3.1 вентиляция: Организованный воздухообмен.", "вентиляция")
    assert not qa._has_definition("ГОСТ 12.1.005 Воздух рабочей зоны.", "рабочая зона")
    assert qa._has_topic_support("Вентиляция обеспечивает воздухообмен.", "требования к вентиляции")
    assert not qa._has_topic_support("Вентиляция обеспечивает воздухообмен.", "орбита Нептуна")
    context = qa._build_context([replace(found[0], text="а" * 30000)] * 10)
    assert len(context) < 12000 and "страница 4" in context
    assert "НАЧАЛО КОНТЕКСТА" in qa._build_prompt("вопрос", context)
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = Path(temp_dir) / "index.pkl"
        assert qa.save_index(index_path)
        target = QASystem(use_llm=False, use_embeddings=False, top_k=2, min_score=0.02,
                          ollama_model="local-test-model", embedding_model_name="another-model")
        assert target.load_index(index_path)
        assert target.top_k == 2 and target.min_score == 0.02
        assert target.ollama_model == "local-test-model"
        assert target.embedding_model_name == "another-model" and target.chunk_embeddings is None
        assert target.search("вентиляции")


if __name__ == "__main__":
    _run_self_tests()
    print("qa_engine: self-tests passed")

# ИСПРАВЛЕНО: синтаксис; контракт SearchResult/AgentLoop; только Ollama с проверкой модели; устойчивый TF-IDF/semantic поиск; фильтры СП/пунктов и определения; атомарный индекс без замены настроек; ограниченный контекст, ссылки и точные выдержки; офлайн-тесты.
