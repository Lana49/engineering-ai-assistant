
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import pickle
from pathlib import Path
import re


class QASystem:
    """Система вопрос-ответ на основе документов"""

    def __init__(self, use_llm: bool = False):
        print("🔄 Загрузка модели эмбеддингов...")
        self.model = SentenceTransformer('intfloat/multilingual-e5-small')
        self.index = None
        self.chunks = []
        self.dimension = 384
        self.is_ready = False
        self.use_llm = use_llm
        self.llm_engine = None

        if use_llm:
            try:
                from core.llm_engine import LLMEngine
                self.llm_engine = LLMEngine()
                print("✅ LLM Engine готов")
            except Exception as e:
                print(f"⚠️ LLM Engine не загружен: {e}")
                self.use_llm = False

    def chunk_text(self, text: str, doc_name: str, chunk_size=500, overlap=100):
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
                        current_chunk = " ".join(words[-overlap//10:]) + " "
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
        """Индексирует все документы в папке"""
        print(f"📁 Индексация документов из {documents_dir}")

        all_chunks = []

        # Читаем все файлы
        try:
            from core.parser import read_docx
        except ImportError:
            from docx import Document
            def read_docx(file_path):
                doc = Document(file_path)
                return '\n'.join([p.text for p in doc.paragraphs])

        for file_path in documents_dir.glob("*"):
            if file_path.suffix in ['.docx', '.txt']:
                try:
                    if file_path.suffix == '.docx':
                        text = read_docx(file_path)
                    else:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()

                    chunks = self.chunk_text(text, file_path.stem)
                    all_chunks.extend(chunks)
                    print(f"  ✅ {file_path.name}: {len(chunks)} чанков")
                except Exception as e:
                    print(f"  ❌ {file_path.name}: {e}")

        if not all_chunks:
            print("❌ Нет документов для индексации")
            return False

        # Генерируем эмбеддинги
        print(f"📊 Генерация эмбеддингов для {len(all_chunks)} чанков...")
        texts = [chunk['text'] for chunk in all_chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # Нормализуем для косинусного сходства
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Создаем FAISS индекс
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        self.chunks = all_chunks

        print(f"✅ Индекс создан: {self.index.ntotal} векторов")
        self.is_ready = True
        return True

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Поиск релевантных чанков по вопросу"""
        if not self.is_ready:
            return []

        # Кодируем вопрос
        query_emb = self.model.encode([query])
        query_emb = query_emb / np.linalg.norm(query_emb)

        # Поиск
        scores, indices = self.index.search(query_emb.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score > 0.5:
                results.append({
                    'text': self.chunks[idx]['text'],
                    'doc_name': self.chunks[idx]['doc_name'],
                    'score': float(score)
                })

        return results

    def answer(self, question: str, top_k: int = 5) -> Dict:
        """Ответ на вопрос на основе найденных чанков"""
        relevant = self.search(question, top_k)

        if not relevant:
            return {
                'question': question,
                'answer': "Извините, не нашел информацию по вашему вопросу в документации.",
                'sources': []
            }

        # Если используем LLM, генерируем умный ответ
        if self.use_llm and self.llm_engine:
            # Формируем контекст
            context = ""
            for chunk in relevant[:3]:
                context += f"\n--- {chunk['doc_name']} ---\n"
                context += chunk['text'][:500] + "\n"

            # Генерируем ответ через LLM
            try:
                result = self.llm_engine.answer_with_context(question, context, relevant[:3])
                return result
            except Exception as e:
                print(f"❌ Ошибка LLM: {e}")
                # Fallback к простому ответу

        # Иначе простой ответ
        answer = f"**По вашему вопросу найдена информация:**\n\n"
        for i, chunk in enumerate(relevant[:3], 1):
            answer += f"**{i}. {chunk['doc_name']}** (релевантность: {chunk['score']:.2f})\n"
            answer += f"{chunk['text'][:400]}\n\n"

        return {
            'question': question,
            'answer': answer,
            'sources': relevant
        }

    def set_top_k(self, top_k: int):
        """Установить количество возвращаемых фрагментов"""
        self.top_k = top_k

    def save_index(self, path: Path):
        """Сохраняет индекс"""
        if not self.is_ready:
            return

        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"✅ Индекс сохранен в {path}")

    def load_index(self, path: Path):
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