import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Any
import pickle
from pathlib import Path
import re
import os

# Для snapshot_download
try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None


class QASystem:
    """Система вопрос-ответ на основе документов"""

    def __init__(self, use_llm: bool = False):
        print("🔄 Загрузка модели эмбеддингов...")
        self.model = SentenceTransformer('intfloat/multilingual-e5-small')
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict] = []
        self.dimension = 384
        self.is_ready = False
        self.use_llm = use_llm
        self.llm_engine = None

        # Стоп-фразы для фильтрации мусорных фрагментов
        self.stop_phrases = [
            '(В.', 'δ =', 'q =', 'tпов =', 'R =', 'tв =', 'tн =',
            'поправочный', 'коэффициент, учитывающий'
        ]

        if use_llm:
            try:
                from core.llm_engine import LLMEngine
                self.llm_engine = LLMEngine()
                print("✅ LLM Engine готов")
            except Exception as e:
                print(f"⚠️ LLM Engine не загружен: {e}")
                self.use_llm = False

    def _is_bad_chunk(self, text: str) -> bool:
        """Проверяет, является ли фрагмент техническим мусором"""
        text_stripped = text.strip()
        for phrase in self.stop_phrases:
            if text_stripped.startswith(phrase):
                return True
        if len(text_stripped) < 50:
            return True
        return False

    @staticmethod
    def chunk_text(text: str, doc_name: str, chunk_size=1000, overlap=200):
        """Разбивает текст на чанки"""
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

    def index_documents(self, documents_dir: Path):
        """Индексирует все документы — скачивает из dataset или использует локальные"""
        all_chunks = []

        # === 1. СКАЧИВАНИЕ ИЗ DATASET ЧЕРЕЗ SNAPSHOT_DOWNLOAD ===
        if snapshot_download is not None:
            try:
                print("📥 Скачивание документов из dataset (snapshot)...")
                snapshot_download(
                    repo_id="Lana49/engineering-docs",
                    repo_type="dataset",
                    local_dir=str(documents_dir),
                    allow_patterns=["*.docx", "*.pdf", "*.rtf"],
                    local_dir_use_symlinks=False,
                    force_download=False
                )
                print(f"✅ Документы скачаны в {documents_dir}")
            except Exception as e:
                print(f"⚠️ Не удалось скачать dataset: {e}")
                print("📁 Использую локальные файлы в data/raw/")
        else:
            print("⚠️ Библиотека huggingface_hub не установлена")
            print("📁 Использую локальные файлы в data/raw/")

        # === 2. ИНДЕКСАЦИЯ ЛОКАЛЬНЫХ ФАЙЛОВ ===
        print(f"📁 Индексация документов из {documents_dir}")

        def read_docx(file_path):
            try:
                from docx import Document
                doc = Document(file_path)
                return '\n'.join([p.text for p in doc.paragraphs])
            except ImportError:
                return ""

        def read_pdf(file_path):
            try:
                import fitz
                doc = fitz.open(file_path)
                return '\n'.join([page.get_text() for page in doc])
            except ImportError:
                return ""

        def read_rtf(file_path):
            try:
                from striprtf.striprtf import rtf_to_text
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    return rtf_to_text(f.read()).strip()
            except ImportError:
                return ""

        def read_file(file_path):
            file_path_str = str(file_path)
            if file_path_str.endswith('.docx'):
                return read_docx(file_path_str)
            elif file_path_str.endswith('.pdf'):
                return read_pdf(file_path_str)
            elif file_path_str.endswith('.rtf'):
                return read_rtf(file_path_str)
            else:
                with open(file_path_str, 'r', encoding='utf-8') as f:
                    return f.read()

        for file_path in documents_dir.glob("*"):
            if file_path.suffix in ['.docx', '.txt', '.pdf', '.rtf']:
                try:
                    text = read_file(file_path)
                    if text:
                        chunks = self.chunk_text(text, file_path.stem)
                        all_chunks.extend(chunks)
                        print(f"  ✅ {file_path.name}: {len(chunks)} чанков")
                except Exception as e:
                    print(f"  ❌ {file_path.name}: {e}")

        # === 3. ПРОВЕРКА И ВЕКТОРИЗАЦИЯ ===
        if not all_chunks:
            print("❌ Нет документов для индексации")
            self.is_ready = False
            return False

        print(f"📊 Генерация эмбеддингов для {len(all_chunks)} чанков...")
        texts = [chunk['text'] for chunk in all_chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        self.chunks = all_chunks

        print(f"✅ Индекс создан: {self.index.ntotal} векторов")
        self.is_ready = True
        return True

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Поиск релевантных чанков по вопросу"""
        if not self.is_ready or self.index is None:
            return []

        query_emb = self.model.encode([query])
        query_emb = query_emb / np.linalg.norm(query_emb)

        scores, indices = self.index.search(query_emb.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score > 0.5 and idx < len(self.chunks):
                results.append({
                    'text': self.chunks[idx]['text'],
                    'doc_name': self.chunks[idx]['doc_name'],
                    'score': float(score)
                })

        return results

    def answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Ответ на вопрос на основе найденных чанков"""
        relevant = self.search(question, top_k)

        if not relevant:
            return {
                'question': question,
                'answer': "❌ Информация по вашему вопросу не найдена в документации.",
                'sources': []
            }

        cleaned_chunks = [c for c in relevant if not self._is_bad_chunk(c['text'])]
        if not cleaned_chunks:
            cleaned_chunks = relevant[:2]

        if self.use_llm and self.llm_engine:
            context = ""
            for chunk in cleaned_chunks[:3]:
                context += f"\n--- {chunk['doc_name']} ---\n{chunk['text'][:500]}\n"
            try:
                return self.llm_engine.answer_with_context(question, context, cleaned_chunks[:3])
            except Exception as e:
                print(f"❌ Ошибка LLM: {e}")

        first_sentence = cleaned_chunks[0]['text'].split('.')[0] + "."
        answer = f"**📌 Краткий ответ:**\n{first_sentence}\n\n"
        answer += f"**📖 Подробнее из документации:**\n"

        for i, chunk in enumerate(cleaned_chunks[:2], 1):
            text = chunk['text'][:400].replace('\n', ' ')
            answer += f"\n**{i}. {chunk['doc_name']}** (релевантность: {chunk['score']:.2f})\n> {text}...\n"

        return {
            'question': question,
            'answer': answer,
            'sources': cleaned_chunks
        }

    def find_definition(self, term: str) -> Dict[str, Any]:
        """Ищет определение термина"""
        term_lower = term.lower()

        # Ручные определения
        definitions = {
            "гсоп": "**ГСОП (Градусо-сутки отопительного периода)** = (t_в - t_от) × z_от\n\n**Для Москвы:** ГСОП = (20 - (-3,1)) × 214 = 4943 °C·сут",
            "градусо-сутки": "**ГСОП (Градусо-сутки отопительного периода)** = (t_в - t_от) × z_от",
            "отопление": "**Отопление** — система обогрева помещений. Расчётная температура: 20-22°C.",
            "вентиляция": "**Вентиляция** — организованный воздухообмен. Норма: 3 м³/ч на 1 м².",
            "изоляция": "**Тепловая изоляция** — уменьшение теплопередачи. δ = λ × ((t_в - t_н)/q - R_н)",
            "теплоизоляция": "**Тепловая изоляция** — конструкция для уменьшения теплопередачи.",
            "вэр": "**ВЭР** — вторичные энергоресурсы: рекуперация, аккумулирование тепла.",
            "вторичные энергоресурсы": "**ВЭР** — вторичные энергоресурсы: рекуперация, аккумулирование тепла.",
            "овк": "**ОВК** — Отопление, Вентиляция, Кондиционирование (СП 60.13330)."
        }

        for key, value in definitions.items():
            if key in term_lower:
                return {'definition': value, 'source': 'Нормативная база', 'found': True}

        # Поиск в документах
        if self.chunks:
            for chunk in self.chunks:
                if term_lower in chunk['text'].lower():
                    return {
                        'definition': chunk['text'].strip(),
                        'source': chunk.get('doc_name', 'Нормативная база'),
                        'found': True
                    }

        result = self.answer(term)
        return {
            'definition': result['answer'],
            'source': result['sources'][0]['doc_name'] if result.get('sources') else 'Нормативная база',
            'found': len(result.get('sources', [])) > 0
        }

    def save_index(self, path: Path):
        """Сохраняет индекс"""
        if not self.is_ready or self.index is None:
            return
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"✅ Индекс сохранен в {path}")

    def load_index(self, path: Path) -> bool:
        """Загружает индекс"""
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