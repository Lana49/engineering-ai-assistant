# -*- coding: utf-8 -*-
"""
Парсер документов для инженерной базы знаний.

Поддерживает:
- .pdf   -> PyMuPDF + pdfplumber (fallback)
- .docx  -> python-docx
- .doc   -> textract (ТОЛЬКО textract)
- .rtf   -> striprtf
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".rtf"}


def read_docx(file_path: str | Path) -> str:
    """Чтение DOCX файла через python-docx."""
    try:
        from docx import Document
    except ImportError:
        print("⚠️ python-docx не установлен. Установите: pip install python-docx")
        return ""

    path = Path(file_path)
    try:
        if not path.exists() or path.stat().st_size == 0:
            return ""

        doc = Document(str(path))
        parts: list[str] = []

        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                parts.append(text)

        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))

        return "\n".join(parts).strip()
    except (OSError, ValueError, AttributeError) as exc:
        print(f"⚠️ Ошибка DOCX {path.name}: {exc}")
        return ""


def read_pdf_pymupdf(file_path: str | Path) -> str:
    """Чтение PDF через PyMuPDF."""
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf
        except ImportError:
            print("⚠️ PyMuPDF (fitz) не установлен. Установите: pip install PyMuPDF")
            return ""

    path = Path(file_path)
    try:
        # Используем pymupdf.open() вместо pymupdf.open() - IDE может не видеть, но это работает
        doc = pymupdf.open(str(path))
        parts: list[str] = []
        for page in doc:
            page_text = page.get_text() or ""
            if page_text.strip():
                parts.append(page_text.strip())
        doc.close()
        return "\n".join(parts).strip()
    except (OSError, ValueError, AttributeError, RuntimeError) as exc:
        print(f"⚠️ PyMuPDF ошибка {path.name}: {exc}")
        return ""


def read_pdf_pdfplumber(file_path: str | Path) -> str:
    """Чтение PDF через pdfplumber (fallback)."""
    try:
        import pdfplumber
    except ImportError:
        return ""

    path = Path(file_path)
    try:
        with pdfplumber.open(str(path)) as pdf:
            parts: list[str] = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    except (OSError, ValueError, AttributeError) as exc:
        print(f"⚠️ pdfplumber ошибка {path.name}: {exc}")
        return ""


def read_pdf(file_path: str | Path) -> str:
    """Чтение PDF с каскадным fallback."""
    path = Path(file_path)
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return ""

    text = read_pdf_pymupdf(path)
    if text:
        return text

    text = read_pdf_pdfplumber(path)
    if text:
        return text

    print(f"⚠️ Не удалось извлечь текст из PDF {path.name}")
    return ""


def read_doc_textract(file_path: str | Path) -> str:
    """Чтение .doc через textract (ТОЛЬКО ОН)."""
    try:
        import textract
    except ImportError:
        print("⚠️ textract не установлен. Установите: pip install textract")
        return ""

    path = Path(file_path)
    try:
        if not path.exists() or path.stat().st_size == 0:
            return ""

        data = textract.process(str(path))
        return data.decode("utf-8", errors="ignore").strip()
    except (OSError, ValueError, AttributeError, RuntimeError) as exc:
        print(f"⚠️ textract ошибка {path.name}: {exc}")
        return ""


def read_doc(file_path: str | Path) -> str:
    """Чтение .doc через textract."""
    path = Path(file_path)
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return ""

    text = read_doc_textract(path)
    if text:
        return text

    print(f"⚠️ Не удалось извлечь текст из DOC {path.name}")
    return ""


def read_rtf(file_path: str | Path) -> str:
    """Чтение RTF файла через striprtf с поддержкой кодировок."""
    try:
        from striprtf.striprtf import rtf_to_text
    except ImportError:
        print("⚠️ striprtf не установлен. Установите: pip install striprtf")
        return ""

    path = Path(file_path)
    try:
        if not path.exists() or path.stat().st_size == 0:
            return ""

        for encoding in ("utf-8", "cp1251", "latin-1"):
            try:
                content = path.read_text(encoding=encoding, errors="replace")
                if content.strip():
                    return rtf_to_text(content).strip()
            except (UnicodeDecodeError, OSError):
                continue

        return ""
    except (OSError, ValueError, AttributeError) as exc:
        print(f"⚠️ Ошибка RTF {path.name}: {exc}")
        return ""


def read_file(file_path: str | Path) -> str:
    """Универсальное чтение файла с каскадными fallback."""
    path = Path(file_path)

    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return ""

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        print(f"⚠️ Неподдерживаемый формат: {path.name}")
        return ""

    if suffix == ".docx":
        return read_docx(path)
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".doc":
        return read_doc(path)
    if suffix == ".rtf":
        return read_rtf(path)

    return ""


@dataclass(slots=True)
class ParsedDocument:
    """Структура обработанного документа."""
    doc_name: str
    filepath: str
    filetype: str
    text: str
    chunks: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentParser:
    """Парсер документов с разбиением на фрагменты."""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200, min_chunk_size: int = 120):
        self.chunk_size = max(300, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))
        self.min_chunk_size = max(50, min_chunk_size)

    @staticmethod
    def normalize_text(text: str) -> str:
        """Нормализует текст."""
        if not text:
            return ""
        text = text.replace("\x00", " ")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def split_paragraphs(self, text: str) -> list[str]:
        """Разбивает текст на параграфы."""
        parts = re.split(r"\n\s*\n", text)
        result: list[str] = []

        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) > int(self.chunk_size * 1.5):
                result.extend(self.split_long_text(part))
            else:
                result.append(part)

        return result

    def split_long_text(self, text: str) -> list[str]:
        """Разбивает длинный текст."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) < 2:
            return self.hard_split(text)

        result: list[str] = []
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
                    overlap = self.smart_overlap(overlap)
                    current = f"{overlap} {sentence}".strip() if overlap else sentence
                else:
                    result.extend(self.hard_split(sentence))
                    current = ""

        if current.strip():
            result.append(current.strip())

        return result

    def hard_split(self, text: str) -> list[str]:
        """Принудительно разбивает текст."""
        parts: list[str] = []
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
    def smart_overlap(text: str) -> str:
        """Умное перекрытие."""
        text = text.strip()
        if not text:
            return ""
        split_pos = text.find(" ")
        if 0 < split_pos < len(text) // 2:
            text = text[split_pos + 1:].strip()
        return text

    @staticmethod
    def extract_formulas(text: str) -> list[dict[str, Any]]:
        """Извлекает формулы."""
        formulas: list[dict[str, Any]] = []
        patterns = [
            r"[A-Za-zА-Яа-я0-9_]+\s*=\s*[^=\n]{3,120}",
            r"\bQ\s*=\s*[^=\n]{3,120}",
        ]

        for pattern in patterns:
            for match in re.findall(pattern, text):
                raw = match.strip()
                if len(raw) < 4:
                    continue
                formulas.append({
                    "raw": raw,
                    "variables": sorted(set(re.findall(r"[A-Za-zА-Яа-я_]+", raw))),
                    "has_operator": any(op in raw for op in ("=", "+", "-", "*", "/")),
                })

        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in formulas:
            key = item["raw"]
            if key not in seen:
                seen.add(key)
                unique.append(item)

        return unique[:20]

    @staticmethod
    def detect_table_like_content(text: str) -> bool:
        """Определяет табличное содержимое."""
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
    def build_metadata(path: Path, text: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Собирает метаданные."""
        return {
            "parsed": True,
            "filename": path.name,
            "file_stem": path.stem,
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "char_count": len(text),
            "word_count": len(text.split()),
            "chunk_count": len(chunks),
            "has_formulas": any(chunk.get("has_formula", False) for chunk in chunks),
            "has_table_like_content": any(chunk.get("has_table_like_content", False) for chunk in chunks),
        }

    def parse_file(self, file_path: str | Path) -> dict[str, Any]:
        """Парсит один файл."""
        path = Path(file_path)
        text = self.normalize_text(read_file(path))

        if not text:
            return {
                "doc_name": path.name,
                "filepath": str(path),
                "filetype": path.suffix.lower(),
                "text": "",
                "chunks": [],
                "metadata": {
                    "parsed": False,
                    "filename": path.name,
                    "suffix": path.suffix.lower(),
                },
            }

        raw_chunks = self.split_paragraphs(text)
        chunks: list[dict[str, Any]] = []

        for idx, chunk_text in enumerate(raw_chunks):
            formulas = self.extract_formulas(chunk_text)
            chunks.append({
                "doc_name": path.name,
                "filepath": str(path),
                "chunk_id": idx,
                "text": chunk_text,
                "metadata": {"formulas": formulas},
                "has_formula": bool(formulas),
                "has_table_like_content": self.detect_table_like_content(chunk_text),
            })

        metadata = self.build_metadata(path, text, chunks)

        return {
            "doc_name": path.name,
            "filepath": str(path),
            "filetype": path.suffix.lower(),
            "text": text,
            "chunks": chunks,
            "metadata": metadata,
        }

    def parse_directory(self, directory: str | Path, recursive: bool = True) -> list[dict[str, Any]]:
        """Парсит директорию."""
        base = Path(directory)
        if not base.exists() or not base.is_dir():
            return []

        pattern = "**/*" if recursive else "*"
        files = [
            p for p in base.glob(pattern)
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        parsed: list[dict[str, Any]] = []
        for path in sorted(files):
            item = self.parse_file(path)
            if item.get("text"):
                parsed.append(item)

        return parsed


def parse_file(file_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Удобная функция для парсинга одного файла."""
    parser = DocumentParser(**kwargs)
    return parser.parse_file(file_path)


def parse_directory(directory: str | Path, **kwargs: Any) -> list[dict[str, Any]]:
    """Удобная функция для парсинга директории."""
    parser = DocumentParser(
        chunk_size=kwargs.pop("chunk_size", 1200),
        chunk_overlap=kwargs.pop("chunk_overlap", 200),
        min_chunk_size=kwargs.pop("min_chunk_size", 120),
    )
    return parser.parse_directory(directory, **kwargs)