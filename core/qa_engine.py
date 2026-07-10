# -*- coding: utf-8 -*-
"""
QA Engine для работы с документацией.
Поддерживает:
- Семантический поиск по документам
- Гибридный поиск (FAISS + BM25)
- Извлечение таблиц и формул
- Индексацию документов
- Кэширование индекса
- Поиск определений
"""

import numpy as np
import faiss
from typing import List, Dict, Optional, Any
import pickle
from pathlib import Path
import re
import hashlib
import json
from datetime import datetime


# ========== ИМПОРТЫ С ОБРАБОТКОЙ ОШИБОК ==========

# SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    SentenceTransformer = None
    print("⚠️ sentence-transformers не установлен. pip install sentence-transformers")

# BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None
    print("⚠️ rank-bm25 не установлен. Гибридный поиск недоступен.")

# HuggingFace Hub
try:
    from huggingface_hub import snapshot_download
    SNAPSHOT_AVAILABLE = True
except ImportError:
    SNAPSHOT_AVAILABLE = False
    snapshot_download = None
    print("⚠️ huggingface-hub не установлен. pip install huggingface-hub")


# ========== ОСНОВНОЙ КЛАСС ==========

class QASystem:
    """Система вопрос-ответ на основе документов"""

    def __init__(self, use_llm: bool = False):
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            raise ImportError("sentence-transformers не установлен")

        print("🔄 Загрузка модели эмбеддингов...")
        self.model = SentenceTransformer('intfloat/multilingual-e5-small')
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict] = []
        self.dimension = 384
        self.is_ready = False
        self.use_llm = use_llm
        self.llm_engine = None
        self.bm25_index = None

        # Стоп-фразы для фильтрации
        self.stop_phrases = [
            '(В.', 'δ =', 'q =', 'tпов =', 'R =', 'tв =', 'tн =',
            'поправочный', 'коэффициент, учитывающий'
        ]

        # Кэш определений для быстрого доступа
        self.definitions_cache = {}

        if use_llm:
            try:
                from core.llm_engine import LLMEngine
                self.llm_engine = LLMEngine()
                print("✅ LLM Engine готов")
            except Exception as e:
                print(f"⚠️ LLM Engine не загружен: {e}")
                self.use_llm = False

    # ========== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ==========

    def _is_bad_chunk(self, text: str) -> bool:
        """Проверяет, является ли чанк "плохим" (коротким или содержащим стоп-фразы)"""
        text_stripped = text.strip()
        for phrase in self.stop_phrases:
            if text_stripped.startswith(phrase):
                return True
        if len(text_stripped) < 50:
            return True
        return False

    @staticmethod
    def chunk_text(text: str, doc_name: str, chunk_size=1000, overlap=200):
        """Разбивает текст на чанки с перекрытием"""
        chunks = []
        sentences = re.split(r'[.!?]\s+', text)

        current_chunk = ""
        chunk_id = 0

        for sent in sentences:
            if len(current_chunk) + len(sent) < chunk_size:
                current_chunk += sent + ". "
            else:
                if current_chunk:
                    chunks.append({
                        'id': f"{doc_name}_{chunk_id}",
                        'text': current_chunk.strip(),
                        'doc_name': doc_name,
                        'chunk_id': chunk_id
                    })
                    chunk_id += 1
                    if overlap > 0:
                        words = current_chunk.split()
                        current_chunk = " ".join(words[-overlap // 10:]) + " "
                current_chunk += sent + ". "

        if current_chunk:
            chunks.append({
                'id': f"{doc_name}_{chunk_id}",
                'text': current_chunk.strip(),
                'doc_name': doc_name,
                'chunk_id': chunk_id
            })

        return chunks

    # ========== ЧТЕНИЕ ФАЙЛОВ ==========

    @staticmethod
    def _read_docx(file_path: str) -> str:
        """Чтение DOCX файла с извлечением таблиц"""
        try:
            from docx import Document
            doc = Document(file_path)
            text_parts = []

            # Обычный текст
            for para in doc.paragraphs:
                text_parts.append(para.text)

            # Таблицы
            for table_idx, table in enumerate(doc.tables):
                table_text = []
                table_text.append(f"\n[ТАБЛИЦА {table_idx + 1}]")
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        row_text.append(cell.text.strip())
                    if any(row_text):
                        table_text.append(" | ".join(row_text))
                if len(table_text) > 1:
                    text_parts.append("\n".join(table_text))

            return '\n'.join(text_parts)
        except ImportError:
            return ""
        except Exception:
            return ""

    @staticmethod
    def _read_pdf(file_path: str) -> str:
        """Чтение PDF файла с извлечением таблиц"""
        try:
            import fitz
            doc = fitz.open(file_path)
            text_parts = []

            # Текст из PDF
            for page_num, page in enumerate(doc):
                text_parts.append(f"\n--- Страница {page_num + 1} ---")
                text_parts.append(page.get_text())

            # Пробуем извлечь таблицы через pdfplumber
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        tables = page.extract_tables()
                        for table_idx, table in enumerate(tables):
                            if table and len(table) > 1:
                                table_text = [f"\n[ТАБЛИЦА {table_idx + 1} на стр.{page_num + 1}]"]
                                for row in table:
                                    if row:
                                        row_text = [str(cell or "").strip() for cell in row]
                                        if any(row_text):
                                            table_text.append(" | ".join(row_text))
                                if len(table_text) > 1:
                                    text_parts.append("\n".join(table_text))
            except ImportError:
                pass
            except Exception:
                pass

            return '\n'.join(text_parts)
        except ImportError:
            return ""
        except Exception:
            return ""

    @staticmethod
    def _read_rtf(file_path: str) -> str:
        """Чтение RTF файла"""
        try:
            from striprtf.striprtf import rtf_to_text
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            return rtf_to_text(content).strip()
        except ImportError:
            print("⚠️ striprtf не установлен. pip install striprtf")
            return ""
        except Exception as e:
            print(f"⚠️ Ошибка чтения RTF: {e}")
            return ""

    @staticmethod
    def _read_rtf_pandoc(file_path: str) -> str:
        """Чтение RTF через pandoc"""
        try:
            import pypandoc
            try:
                pypandoc.get_pandoc_version()
            except Exception:
                print("⚠️ Pandoc не установлен")
                return ""
            output = pypandoc.convert_file(
                file_path,
                'plain',
                format='rtf',
                extra_args=['--wrap=none']
            )
            return output.strip()
        except ImportError:
            return ""
        except Exception as e:
            print(f"⚠️ Ошибка pandoc: {e}")
            return ""

    def _read_file(self, file_path: Path) -> str:
        """
        Чтение файла с поддержкой различных форматов

        Поддерживаемые форматы:
        - .docx, .doc - через python-docx
        - .pdf - через PyMuPDF (fitz)
        - .rtf, .RTF - через striprtf или pypandoc
        - .txt, .md, .csv, .json, .xml - как текстовые
        """
        path_str = str(file_path)
        file_ext = Path(file_path).suffix.lower()

        try:
            if file_ext in ['.docx', '.doc']:
                return self._read_docx(path_str)

            elif file_ext == '.pdf':
                return self._read_pdf(path_str)

            elif file_ext in ['.rtf', '.RTF']:
                text = self._read_rtf(path_str)
                if not text or len(text.strip()) < 50:
                    text = self._read_rtf_pandoc(path_str)
                return text

            elif file_ext in ['.txt', '.md', '.csv', '.json', '.xml']:
                with open(path_str, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()

            else:
                print(f"⚠️ Неподдерживаемый формат {file_ext}")
                try:
                    with open(path_str, 'r', encoding='utf-8', errors='ignore') as f:
                        return f.read()
                except Exception:
                    return ""

        except Exception as e:
            print(f"❌ Ошибка чтения файла {file_path.name}: {e}")
            return ""

    # ========== ИНДЕКСАЦИЯ ==========

    @staticmethod
    def _get_documents_hash(documents_dir: Path) -> str:
        """Вычисляет хеш всех документов в директории"""
        hash_md5 = hashlib.md5()
        for file_path in sorted(documents_dir.glob("*")):
            if file_path.suffix.lower() in ['.docx', '.doc', '.pdf', '.rtf']:
                try:
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b''):
                            hash_md5.update(chunk)
                except (IOError, OSError):
                    continue
        return hash_md5.hexdigest()

    @staticmethod
    def _get_hash_file_path(documents_dir: Path) -> Path:
        """Путь к файлу с хешем"""
        return documents_dir.parent / "processed" / "docs_hash.txt"

    def _load_saved_hash(self, documents_dir: Path) -> Optional[str]:
        """Загружает сохранённый хеш документов"""
        hash_file = self._get_hash_file_path(documents_dir)
        if hash_file.exists():
            try:
                with open(hash_file, 'r') as f:
                    return f.read().strip()
            except (IOError, OSError):
                return None
        return None

    def _save_hash(self, documents_dir: Path, hash_value: str):
        """Сохраняет хеш документов"""
        hash_file = self._get_hash_file_path(documents_dir)
        try:
            hash_file.parent.mkdir(parents=True, exist_ok=True)
            with open(hash_file, 'w') as f:
                f.write(hash_value)
        except (IOError, OSError) as e:
            print(f"⚠️ Не удалось сохранить хеш: {e}")

    def _build_bm25_index(self):
        """Создаёт BM25 индекс для гибридного поиска"""
        if not BM25_AVAILABLE or not self.chunks or BM25Okapi is None:
            return
        try:
            tokenized = [chunk['text'].split() for chunk in self.chunks]
            self.bm25_index = BM25Okapi(tokenized)
            print("✅ BM25 индекс создан для гибридного поиска")
        except Exception as e:
            print(f"⚠️ Не удалось создать BM25: {e}")
            self.bm25_index = None

    def index_documents(self, documents_dir: Path):
        """Индексация документов из директории"""
        # Проверяем существование директории
        if not documents_dir.exists():
            print(f"⚠️ Директория {documents_dir} не существует, создаём...")
            documents_dir.mkdir(parents=True, exist_ok=True)

        current_hash = self._get_documents_hash(documents_dir)
        saved_hash = self._load_saved_hash(documents_dir)

        # Проверяем, изменились ли документы
        if saved_hash == current_hash and self.is_ready:
            print("✅ Документы не изменились, индекс актуален")
            return True

        if saved_hash == current_hash:
            index_path = documents_dir.parent / "processed" / "qa_index"
            if index_path.exists():
                if self.load_index(index_path):
                    print("✅ Индекс загружен из кэша")
                    return True

        print("🔄 Документы изменились или индекс отсутствует, пересоздаём...")
        all_chunks = []

        # Скачивание документов из Hugging Face
        if SNAPSHOT_AVAILABLE and snapshot_download is not None:
            try:
                print("📥 Скачивание документов из dataset (snapshot)...")
                snapshot_download(
                    repo_id="Lana49/engineering-docs",
                    repo_type="dataset",
                    local_dir=str(documents_dir),
                    allow_patterns=["*.docx", "*.doc", "*.pdf", "*.rtf", "*.RTF"],
                    local_dir_use_symlinks=False,
                    force_download=False
                )
                print(f"✅ Документы скачаны в {documents_dir}")
            except Exception as e:
                print(f"⚠️ Ошибка скачивания: {e}")
                print("📁 Использую локальные файлы")
        else:
            print("📁 Использую локальные файлы (huggingface-hub не доступен)")

        # Индексация файлов
        files_found = False
        for file_path in documents_dir.glob("*"):
            if file_path.suffix.lower() in ['.docx', '.doc', '.pdf', '.rtf']:
                files_found = True
                try:
                    text = self._read_file(file_path)
                    if text and len(text.strip()) > 100:
                        chunks = self.chunk_text(text, file_path.stem)
                        all_chunks.extend(chunks)
                        print(f"  ✅ {file_path.name}: {len(chunks)} чанков")
                except Exception as e:
                    print(f"  ❌ {file_path.name}: {e}")

        if not files_found:
            print(f"⚠️ В директории {documents_dir} нет поддерживаемых файлов")
            print("Поддерживаемые форматы: .docx, .doc, .pdf, .rtf")
            self.is_ready = False
            return False

        if not all_chunks:
            print("❌ Нет текста для индексации (файлы пусты или повреждены)")
            self.is_ready = False
            return False

        # Генерация эмбеддингов
        print(f"📊 Генерация эмбеддингов для {len(all_chunks)} чанков...")
        texts = [chunk['text'] for chunk in all_chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # Нормализация
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Создание FAISS индекса
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        self.chunks = all_chunks

        # Создание BM25 индекса
        self._build_bm25_index()

        # Сохранение
        self._save_hash(documents_dir, current_hash)

        index_path = documents_dir.parent / "processed" / "qa_index"
        index_path.mkdir(parents=True, exist_ok=True)
        self.save_index(index_path)

        print(f"✅ Индекс создан и сохранён: {self.index.ntotal} векторов")
        self.is_ready = True
        return True

    # ========== ПОИСК ==========

    def _search_bm25(self, query: str, top_k: int = 10) -> List[Dict]:
        """Поиск через BM25"""
        if self.bm25_index is None or not BM25_AVAILABLE:
            return []
        try:
            tokenized_query = query.split()
            scores = self.bm25_index.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[-top_k:][::-1]
            results = []
            for idx in top_indices:
                if scores[idx] > 0 and idx < len(self.chunks):
                    results.append({
                        'text': self.chunks[idx]['text'],
                        'doc_name': self.chunks[idx]['doc_name'],
                        'score': float(scores[idx]),
                        'idx': idx
                    })
            return results
        except Exception as e:
            print(f"⚠️ Ошибка BM25: {e}")
            return []

    def search(self, query: str, top_k: int = 8) -> List[Dict]:
        """Гибридный поиск (FAISS + BM25)"""
        if not self.is_ready or self.index is None:
            return []

        # FAISS поиск
        query_emb = self.model.encode([query])
        query_emb = query_emb / np.linalg.norm(query_emb)

        scores, indices = self.index.search(query_emb.astype(np.float32), top_k * 2)

        faiss_results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunks) and score > 0.3:
                faiss_results.append({
                    'text': self.chunks[idx]['text'],
                    'doc_name': self.chunks[idx]['doc_name'],
                    'score': float(score),
                    'idx': idx
                })

        # BM25 поиск
        bm25_results = self._search_bm25(query, top_k * 2)

        # Объединение результатов
        combined = {}
        for r in faiss_results + bm25_results:
            idx = r['idx']
            if idx not in combined:
                combined[idx] = r
            else:
                combined[idx]['score'] = (combined[idx]['score'] + r['score']) / 2

        sorted_results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        return sorted_results[:top_k]

    def search_with_formulas(self, query: str, top_k: int = 8) -> Dict[str, Any]:
        """Поиск с извлечением таблиц и формул"""
        results = self.search(query, top_k)

        tables = []
        formulas = []
        for chunk in results:
            tables.extend(self._extract_tables(chunk['text'], chunk['doc_name']))
            formulas.extend(self._extract_formulas(chunk['text']))

        # Удаляем дубликаты таблиц
        unique_tables = []
        seen = set()
        for t in tables:
            key = t.get('content', '')[:100]
            if key not in seen:
                seen.add(key)
                unique_tables.append(t)

        # Удаляем дубликаты формул
        unique_formulas = []
        seen = set()
        for f in formulas:
            key = f.get('raw', '')
            if key not in seen:
                seen.add(key)
                unique_formulas.append(f)

        return {
            'results': results,
            'tables': unique_tables[:5],
            'formulas': unique_formulas[:10]
        }

    # ========== ИЗВЛЕЧЕНИЕ ТАБЛИЦ И ФОРМУЛ ==========

    def _extract_tables(self, text: str, doc_name: str) -> List[Dict]:
        """Извлекает таблицы из текста"""
        tables = []
        lines = text.split('\n')
        in_table = False
        table_lines = []
        table_title = ""

        for i, line in enumerate(lines):
            # Ищем начало таблицы
            if '[ТАБЛИЦА]' in line:
                in_table = True
                table_lines = []
                if i > 0 and len(lines[i - 1].strip()) < 100:
                    table_title = lines[i - 1].strip()
                else:
                    table_title = f"Таблица {len(tables) + 1}"
                continue

            # Если внутри таблицы
            if in_table:
                if line.strip() == '':
                    if table_lines:
                        tables.append({
                            'title': table_title,
                            'content': '\n'.join(table_lines),
                            'rows': [l for l in table_lines if l.strip()]
                        })
                        table_lines = []
                        table_title = ""
                    in_table = False
                else:
                    table_lines.append(line.strip())

        if table_lines:
            tables.append({
                'title': table_title,
                'content': '\n'.join(table_lines),
                'rows': [l for l in table_lines if l.strip()]
            })

        return tables

    def _extract_formulas(self, text: str) -> List[Dict]:
        """Извлекает формулы из текста"""
        formulas = []
        pattern = r'([A-Za-zА-Яа-я][_\w]*\s*[=]\s*[^;.\n]+)'

        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) > 3:
                variables = re.findall(r'[A-Za-zА-Яа-я][_\w]*', match)
                formulas.append({
                    'raw': match.strip(),
                    'variables': list(set(variables)),
                    'has_operator': any(c in match for c in ['+', '-', '*', '/', '^', '='])
                })

        bracket_pattern = r'\(([^)]*[=+\-*/^][^)]*)\)'
        bracket_matches = re.findall(bracket_pattern, text)
        for match in bracket_matches:
            if len(match) > 3 and match not in [f['raw'] for f in formulas]:
                formulas.append({
                    'raw': f"({match})",
                    'variables': list(set(re.findall(r'[A-Za-zА-Яа-я][_\w]*', match))),
                    'has_operator': True
                })

        # Удаляем дубликаты
        unique_formulas = []
        seen = set()
        for f in formulas:
            key = f['raw']
            if key not in seen:
                seen.add(key)
                unique_formulas.append(f)

        return unique_formulas

    # ========== ОТВЕТЫ ==========

    def answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Ответ на вопрос с выделением таблиц и формул"""
        search_results = self.search_with_formulas(question, top_k)
        relevant = search_results['results']

        if not relevant:
            return {
                'question': question,
                'answer': "❌ Информация по вашему вопросу не найдена в документации.",
                'sources': [],
                'tables': [],
                'formulas': []
            }

        cleaned_chunks = [c for c in relevant if not self._is_bad_chunk(c['text'])]
        if not cleaned_chunks:
            cleaned_chunks = relevant[:2]

        all_tables = search_results.get('tables', [])
        all_formulas = search_results.get('formulas', [])

        # LLM ответ (если включен)
        if self.use_llm and self.llm_engine:
            context = ""
            for chunk in cleaned_chunks[:3]:
                context += f"\n--- {chunk['doc_name']} ---\n{chunk['text'][:500]}\n"
            try:
                result = self.llm_engine.answer_with_context(question, context, cleaned_chunks[:3])
                result['tables'] = all_tables[:3]
                result['formulas'] = all_formulas[:5]
                return result
            except Exception as e:
                print(f"❌ Ошибка LLM: {e}")

        # Обычный ответ
        first_sentence = cleaned_chunks[0]['text'].split('.')[0] + "."
        answer = f"**📌 Краткий ответ:**\n{first_sentence}\n\n"
        answer += f"**📖 Подробнее из документации:**\n"

        for i, chunk in enumerate(cleaned_chunks[:2], 1):
            text = chunk['text'][:400].replace('\n', ' ')
            answer += f"\n**{i}. {chunk['doc_name']}** (релевантность: {chunk['score']:.2f})\n> {text}...\n"

        # Добавляем таблицы
        if all_tables:
            answer += "\n\n**📊 Найденные таблицы:**\n"
            for table in all_tables[:2]:
                answer += f"\n**{table.get('title', 'Таблица')}**\n"
                content = table.get('content', '')
                if len(content) > 300:
                    content = content[:300] + "..."
                answer += f"```\n{content}\n```\n"

        # Добавляем формулы
        if all_formulas:
            answer += "\n\n**📐 Найденные формулы:**\n"
            for formula in all_formulas[:3]:
                raw = formula.get('raw', '')
                answer += f"\n`{raw}`\n"
                if formula.get('variables'):
                    answer += f"   Переменные: {', '.join(formula['variables'][:5])}\n"

        return {
            'question': question,
            'answer': answer,
            'sources': cleaned_chunks,
            'tables': all_tables[:3],
            'formulas': all_formulas[:5]
        }

    # ========== ОПРЕДЕЛЕНИЯ ==========

    def find_definition(self, term: str) -> Dict[str, Any]:
        """Поиск определения термина в словаре или документации"""
        term_lower = term.lower().strip()

        # Проверяем кэш
        if term_lower in self.definitions_cache:
            return self.definitions_cache[term_lower]

        # Базовый словарь определений
        all_definitions = self._init_definitions()

        # Поиск точного совпадения
        if term_lower in all_definitions:
            result = {
                'definition': all_definitions[term_lower],
                'source': 'Нормативная база',
                'found': True
            }
            self.definitions_cache[term_lower] = result
            return result

        # Поиск частичного совпадения
        for key, value in all_definitions.items():
            if key in term_lower or term_lower in key:
                result = {
                    'definition': value,
                    'source': 'Нормативная база',
                    'found': True
                }
                self.definitions_cache[term_lower] = result
                return result

        # Поиск в документах
        if self.chunks:
            for chunk in self.chunks:
                if term_lower in chunk['text'].lower():
                    text = chunk['text']
                    sentences = text.split('.')
                    for sent in sentences:
                        if term_lower in sent.lower():
                            result = {
                                'definition': sent.strip() + '.',
                                'source': chunk.get('doc_name', 'Документация'),
                                'found': True
                            }
                            self.definitions_cache[term_lower] = result
                            return result

        # Пробуем через answer
        result = self.answer(term)
        if result.get('sources'):
            response = {
                'definition': result['answer'],
                'source': result['sources'][0]['doc_name'],
                'found': True
            }
            self.definitions_cache[term_lower] = response
            return response

        # Не найдено
        response = {
            'definition': f"❌ Определение для термина '{term}' не найдено.",
            'source': 'Нормативная база',
            'found': False
        }
        self.definitions_cache[term_lower] = response
        return response

    def _init_definitions(self) -> Dict[str, str]:
        """Инициализация базового словаря определений"""
        return {
            # Теплотехнические термины
            "гсоп": "**ГСОП (Градусо-сутки отопительного периода)** = (t_в - t_от) × z_от\n\n**Для Москвы:** ГСОП = (20 - (-3,1)) × 214 = 4943 °C·сут",
            "градусо-сутки": "**ГСОП (Градусо-сутки отопительного периода)** = (t_в - t_от) × z_от",
            "dd": "**ГСОП** — Degree Days (Градусо-сутки отопительного периода)",

            "отопление": "**Отопление** — система обогрева помещений. Расчётная температура: 20-22°C.",
            "heating": "**Отопление** — см. определение выше.",

            "вентиляция": "**Вентиляция** — организованный воздухообмен. Норма: 3 м³/ч на 1 м².",
            "ventilation": "**Вентиляция** — см. определение выше.",

            "кондиционирование": "**Кондиционирование** — поддержание температуры, влажности и чистоты воздуха в помещении.",
            "ac": "**Кондиционирование** — Air Conditioning (AC).",

            "овк": "**ОВК** — Отопление, Вентиляция, Кондиционирование (СП 60.13330).",
            "hvac": "**ОВК** — Heating, Ventilation, Air Conditioning (HVAC).",

            # Тепловая изоляция
            "изоляция": "**Тепловая изоляция** — уменьшение теплопередачи. δ = λ × ((t_в - t_н)/q - R_н)",
            "теплоизоляция": "**Тепловая изоляция** — конструкция для уменьшения теплопередачи.",
            "thermal insulation": "**Теплоизоляция** — см. определение выше.",
            "утеплитель": "**Утеплитель** — материал с низкой теплопроводностью для теплоизоляции: минеральная вата, пенополистирол, пенополиуретан.",

            "термическое сопротивление": "**Термическое сопротивление** R = δ/λ, где δ — толщина слоя (м), λ — коэффициент теплопроводности (Вт/(м·К)). Нормируемое значение R_0 ≥ R_тр",
            "сопротивление теплопередаче": "**Сопротивление теплопередаче** R_0 = 1/α_в + R_к + 1/α_н, где α_в и α_н — коэффициенты теплоотдачи внутренней и наружной поверхностей.",

            "теплопроводность": "**Теплопроводность** λ — способность материала проводить тепло. Чем ниже λ, тем лучше теплоизоляция.",
            "thermal conductivity": "**Теплопроводность** — см. определение выше.",
            "коэффициент теплопроводности": "**Коэффициент теплопроводности** λ [Вт/(м·К)] — характеристика материала. Для утеплителей: λ = 0,02-0,05 Вт/(м·К), для кирпича: λ = 0,5-1,0 Вт/(м·К).",

            "теплопотери": "**Теплопотери** — количество тепла, теряемое через ограждающие конструкции. Q = (t_в - t_н) / R_0 × S, где S — площадь поверхности.",
            "тепловой поток": "**Тепловой поток** q [Вт/м²] — плотность теплового потока через единицу площади. q = (t_в - t_н) / R_0",
            "точка росы": "**Точка росы** — температура, при которой водяной пар в воздухе достигает насыщения и начинает конденсироваться.",

            # Энергоресурсы
            "вэр": "**ВЭР** — вторичные энергоресурсы: рекуперация, аккумулирование тепла.",
            "вторичные энергоресурсы": "**ВЭР** — вторичные энергоресурсы: рекуперация, аккумулирование тепла.",
            "рекуперация": "**Рекуперация** — использование тепла вытяжного воздуха для подогрева приточного воздуха. КПД рекуператора до 85%.",
            "рекуператор": "**Рекуператор** — устройство для передачи тепла между вытяжным и приточным воздухом.",
            "теплонасос": "**Тепловой насос** — устройство для передачи тепла от источника низкопотенциального тепла к системе отопления. COP = 3-5.",
            "тепловой насос": "**Тепловой насос** — устройство для отбора тепла от низкопотенциальных источников для отопления и ГВС.",
            "аккумулирование тепла": "**Аккумулирование тепла** — накопление тепловой энергии в тепловых аккумуляторах.",
            "тепловая энергия": "**Тепловая энергия** — энергия теплового движения молекул. Единицы: Дж, кВт·ч, Гкал.",
            "теплоноситель": "**Теплоноситель** — среда, передающая тепловую энергию: вода, пар, воздух, антифриз.",

            # Энергоэффективность
            "энергоэффективность": "**Энергоэффективность** — отношение полезного эффекта к затратам энергии. Классы: A, B, C, D, E.",
            "energy efficiency": "**Энергоэффективность** — см. определение выше.",
            "энергосбережение": "**Энергосбережение** — комплекс мер по снижению потребления энергии.",
            "тепловая защита": "**Тепловая защита здания** — комплекс свойств ограждающих конструкций, обеспечивающих нормируемый тепловой режим помещений (СП 50.13330).",
            "энергопаспорт": "**Энергетический паспорт** — документ с показателями энергоэффективности здания. Классы: A, B, C.",
            "теплотехнический расчёт": "**Теплотехнический расчёт** — определение теплозащитных свойств ограждающих конструкций: R_0 ≥ R_тр, δ = R_тр × λ.",
            "влажностный режим": "**Влажностный режим** — режим эксплуатации ограждающих конструкций по влажности (сухой, нормальный, влажный, мокрый).",

            # Ограждающие конструкции
            "ограждающие конструкции": "**Ограждающие конструкции** — стены, перекрытия, покрытия, окна, двери, разделяющие внутреннюю и наружную среду.",
            "наружная стена": "**Наружная стена** — вертикальная ограждающая конструкция, отделяющая помещение от наружного воздуха.",
            "кровля": "**Кровля** — верхняя ограждающая конструкция здания. Утеплённая (чердачная) или бесчердачная (совмещённая).",
            "перекрытие": "**Перекрытие** — горизонтальная ограждающая конструкция, разделяющая этажи.",
            "светопрозрачные конструкции": "**Светопрозрачные конструкции** — окна, витражи, стеклянные двери. R_0 ≥ 0,6 м²·°С/Вт",
            "стеклопакет": "**Стеклопакет** — герметичная конструкция из двух или трёх стёкол с воздушной прослойкой.",

            # Нормы и стандарты
            "сп 60": "**СП 60.13330** — Свод правил «Отопление, вентиляция и кондиционирование воздуха»",
            "сп 50": "**СП 50.13330** — Свод правил «Тепловая защита зданий»",
            "сп 131": "**СП 131.13330** — Строительная климатология",
            "снип": "**СНиП** — Строительные нормы и правила (заменены на СП).",
            "гост": "**ГОСТ** — Государственный Стандарт.",
            "микроклимат": "**Микроклимат** — климатические условия внутри помещения: температура, влажность, скорость воздуха.",
            "комфортная температура": "**Комфортная температура** — 20-24°C для жилых помещений, 22-24°C для офисов, 18-20°C для спален."
        }

    # ========== СОХРАНЕНИЕ И ЗАГРУЗКА ИНДЕКСА ==========

    def save_index(self, path: Path):
        """Сохраняет индекс в файл"""
        if not self.is_ready or self.index is None:
            return
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"✅ Индекс сохранен в {path}")

    def load_index(self, path: Path) -> bool:
        """Загружает индекс из файла"""
        index_path = path / "index.faiss"
        chunks_path = path / "chunks.pkl"

        if not index_path.exists() or not chunks_path.exists():
            return False

        self.index = faiss.read_index(str(index_path))
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)

        self.is_ready = True
        print(f"✅ Индекс загружен: {self.index.ntotal} векторов")
        return True