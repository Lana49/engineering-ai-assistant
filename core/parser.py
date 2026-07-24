# core/parser.py
"""
Универсальный парсер документов для инженерной базы знаний.

Поддерживает:
- .txt, .md
- .pdf (через PyMuPDF)
- .docx (через python-docx)
- .html / .htm
- .json
- .csv
- .rtf (через striprtf)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ========== ФУНКЦИИ ЧТЕНИЯ ФАЙЛОВ ==========

def read_docx(file_path: str) -> str:
    """Чтение DOCX файла."""
    try:
        from docx import Document
        doc = Document(file_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    except ImportError:
        print("⚠️ Для работы с DOCX установите python-docx: pip install python-docx")
        return ""
    except Exception as e:
        print(f"❌ Ошибка при чтении DOCX {file_path}: {e}")
        return ""


def read_pdf(file_path: str) -> str:
    """Чтение PDF файла через PyMuPDF."""
    try:
        import pymupdf
        doc = pymupdf.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except ImportError:
        print("⚠️ Для работы с PDF установите PyMuPDF: pip install PyMuPDF")
        return ""
    except Exception as e:
        print(f"❌ Ошибка при чтении PDF {file_path}: {e}")
        return ""


def read_rtf(file_path: str) -> str:
    """Чтение RTF файла."""
    try:
        from striprtf.striprtf import rtf_to_text
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            rtf_content = f.read()
        return rtf_to_text(rtf_content).strip()
    except ImportError:
        print("⚠️ Для работы с RTF установите striprtf: pip install striprtf")
        return ""
    except Exception as e:
        print(f"❌ Ошибка при чтении RTF {file_path}: {e}")
        return ""


def read_file(file_path: str) -> str:
    """Универсальное чтение файла."""
    file_path = str(file_path)
    if file_path.endswith('.docx') or file_path.endswith('.doc'):
        return read_docx(file_path)
    elif file_path.endswith('.pdf'):
        return read_pdf(file_path)
    elif file_path.endswith('.rtf') or file_path.endswith('.RTF'):
        return read_rtf(file_path)
    else:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"❌ Ошибка при чтении файла {file_path}: {e}")
            return ""


# ========== DATACLASS ДЛЯ ДОКУМЕНТА ==========

@dataclass
class ParsedDocument:
    """Структура обработанного документа."""
    doc_name: str
    file_path: str
    file_type: str
    text: str
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ========== ОСНОВНОЙ КЛАСС ПАРСЕРА ==========

class DocumentParser:
    """
    Универсальный парсер документов для инженерной базы знаний.
    """

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        min_chunk_size: int = 120,
    ):
        self.chunk_size = max(300, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))
        self.min_chunk_size = max(50, min_chunk_size)

    def parse_file(self, file_path: str | Path) -> Dict[str, Any]:
        """Парсит один файл и возвращает структурированный результат."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        text = read_file(str(path))
        text = self._normalize_text(text)

        chunks = self.split_into_chunks(text, doc_name=path.name)
        metadata = self._build_metadata(path, text, chunks)

        return {
            "doc_name": path.name,
            "file_path": str(path),
            "file_type": path.suffix.lstrip("."),
            "text": text,
            "chunks": chunks,
            "metadata": metadata,
        }

    def parse_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        allowed_extensions: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Парсит все файлы в директории."""
        base = Path(directory)
        if not base.exists():
            raise FileNotFoundError(f"Папка не найдена: {base}")

        allowed = {
            ext.lower()
            for ext in (
                allowed_extensions
                or [".txt", ".md", ".pdf", ".docx", ".html", ".htm", ".json", ".csv", ".rtf"]
            )
        }

        pattern = "**/*" if recursive else "*"
        results: List[Dict[str, Any]] = []

        for path in sorted(base.glob(pattern)):
            if not path.is_file():
                continue
            if path.suffix.lower() not in allowed:
                continue

            try:
                results.append(self.parse_file(path))
            except Exception as exc:
                results.append(
                    {
                        "doc_name": path.name,
                        "file_path": str(path),
                        "file_type": path.suffix.lstrip("."),
                        "text": "",
                        "chunks": [],
                        "metadata": {
                            "error": str(exc),
                            "parsed": False,
                        },
                    }
                )

        return results

    def split_into_chunks(self, text: str, doc_name: str = "") -> List[Dict[str, Any]]:
        """Разбивает текст на смысловые фрагменты с перекрытием."""
        text = self._normalize_text(text)
        if not text:
            return []

        paragraphs = self._split_paragraphs(text)
        chunks: List[Dict[str, Any]] = []

        current = ""
        chunk_id = 0
        start_char = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph

            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(
                    self._make_chunk(
                        chunk_id=chunk_id,
                        doc_name=doc_name,
                        text=current,
                        start_char=start_char,
                    )
                )
                start_char += len(current)
                chunk_id += 1

                overlap_text = current[-self.chunk_overlap:] if self.chunk_overlap else ""
                overlap_text = self._smart_overlap(overlap_text)
                current = (
                    f"{overlap_text}\n\n{paragraph}".strip()
                    if overlap_text
                    else paragraph
                )
            else:
                sentence_parts = self._split_long_text(paragraph)
                for part in sentence_parts:
                    if len(part.strip()) >= self.min_chunk_size:
                        chunks.append(
                            self._make_chunk(
                                chunk_id=chunk_id,
                                doc_name=doc_name,
                                text=part.strip(),
                                start_char=start_char,
                            )
                        )
                        start_char += len(part)
                        chunk_id += 1
                current = ""

        if current and len(current.strip()) >= self.min_chunk_size:
            chunks.append(
                self._make_chunk(
                    chunk_id=chunk_id,
                    doc_name=doc_name,
                    text=current.strip(),
                    start_char=start_char,
                )
            )

        return chunks

    def _make_chunk(
        self, chunk_id: int, doc_name: str, text: str, start_char: int
    ) -> Dict[str, Any]:
        """Создаёт структурированный фрагмент с метаданными."""
        formulas = self.extract_formulas(text)
        table_like = self.detect_table_like_content(text)

        return {
            "chunk_id": chunk_id,
            "doc_name": doc_name,
            "docname": doc_name,  # для обратной совместимости
            "text": text.strip(),
            "start_char": start_char,
            "end_char": start_char + len(text),
            "char_count": len(text),
            "word_count": len(text.split()),
            "has_formula": len(formulas) > 0,
            "has_table_like_content": table_like,
            "formulas": formulas,
        }

    # ========== СТАТИЧЕСКИЕ МЕТОДЫ ==========

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Нормализует текст: удаляет лишние пробелы и символы."""
        if not text:
            return ""

        text = text.replace("\x00", " ")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\t", " ")
        text = re.sub(r"[ ]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_paragraphs(self, text: str) -> List[str]:
        """Разбивает текст на параграфы."""
        parts = re.split(r"\n\s*\n", text)
        result: List[str] = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if len(part) > self.chunk_size * 1.5:
                result.extend(self._split_long_text(part))
            else:
                result.append(part)

        return result

    def _split_long_text(self, text: str) -> List[str]:
        """Разбивает длинный текст на предложения."""
        sentences = re.split(r"(?<=[.!?;:])\s+", text)
        if len(sentences) <= 1:
            return self._hard_split(text)

        result: List[str] = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current.strip())
                    overlap = current[-self.chunk_overlap:] if self.chunk_overlap else ""
                    overlap = self._smart_overlap(overlap)
                    current = f"{overlap} {sentence}".strip() if overlap else sentence
                else:
                    result.extend(self._hard_split(sentence))
                    current = ""

        if current.strip():
            result.append(current.strip())

        return result

    def _hard_split(self, text: str) -> List[str]:
        """Принудительно разбивает текст на части."""
        parts: List[str] = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                split_pos = text.rfind(" ", start, end)
                if split_pos > start + self.min_chunk_size:
                    end = split_pos

            part = text[start:end].strip()
            if part:
                parts.append(part)

            if end >= len(text):
                break

            start = max(end - self.chunk_overlap, start + 1)

        return parts

    @staticmethod
    def _smart_overlap(text: str) -> str:
        """Умное перекрытие для фрагментов."""
        text = text.strip()
        if not text:
            return ""

        split_pos = text.find(" ")
        if 0 < split_pos < len(text) // 2:
            text = text[split_pos + 1:].strip()

        return text

    @staticmethod
    def extract_formulas(text: str) -> List[Dict[str, Any]]:
        """Извлекает формулы из текста по паттернам."""
        formulas: List[Dict[str, Any]] = []

        patterns = [
            r"[A-Za-zА-Яа-яλΔQqRrtTVGLcpnρ]+\s*=\s*[^.\n]{3,120}",
            r"\([^)\n]*[=+\-*/][^)\n]*\)",
        ]

        for pattern in patterns:
            for match in re.findall(pattern, text):
                raw = match.strip()
                if len(raw) < 4:
                    continue

                formulas.append(
                    {
                        "raw": raw,
                        "variables": list(
                            sorted(set(re.findall(r"[A-Za-zА-Яа-яλΔ]+", raw)))
                        ),
                        "has_operator": any(
                            op in raw for op in ["=", "+", "-", "*", "/", "^"]
                        ),
                    }
                )

        unique: List[Dict[str, Any]] = []
        seen = set()

        for item in formulas:
            key = item["raw"]
            if key not in seen:
                seen.add(key)
                unique.append(item)

        return unique[:20]

    @staticmethod
    def detect_table_like_content(text: str) -> bool:
        """Определяет, содержит ли текст табличное содержимое."""
        if "|" in text:
            return True

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return False

        tabular_lines = 0
        for line in lines[:12]:
            if re.search(r"\s{2,}", line):
                tabular_lines += 1
            elif len(re.findall(r"\d+", line)) >= 3:
                tabular_lines += 1

        return tabular_lines >= 2

    @staticmethod
    def _build_metadata(
        path: Path, text: str, chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Собирает метаданные документа."""
        return {
            "parsed": True,
            "file_name": path.name,
            "file_stem": path.stem,
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "char_count": len(text),
            "word_count": len(text.split()),
            "chunk_count": len(chunks),
            "has_formulas": any(chunk.get("has_formula") for chunk in chunks),
            "has_table_like_content": any(
                chunk.get("has_table_like_content") for chunk in chunks
            ),
        }


# ========== УДОБНЫЕ ФУНКЦИИ ДЛЯ ВЫЗОВА ==========

def parse_file(file_path: str | Path, **kwargs) -> Dict[str, Any]:
    """Удобная функция для парсинга одного файла."""
    parser = DocumentParser(**kwargs)
    return parser.parse_file(file_path)


def parse_directory(directory: str | Path, **kwargs) -> List[Dict[str, Any]]:
    """Удобная функция для парсинга директории."""
    parser = DocumentParser(
        chunk_size=kwargs.pop("chunk_size", 1200),
        chunk_overlap=kwargs.pop("chunk_overlap", 200),
        min_chunk_size=kwargs.pop("min_chunk_size", 120),
    )
    return parser.parse_directory(directory, **kwargs)