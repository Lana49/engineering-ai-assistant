# -*- coding: utf-8 -*-
"""
Универсальный парсер документов для инженерной базы знаний.

Поддерживает:
- .pdf        -> текст PyMuPDF + таблицы pdfplumber + OCR fallback
- .docx       -> python-docx с сохранением порядка абзацев и таблиц
- .doc        -> textutil (macOS) / textract + antiword (Linux)
- .rtf        -> striprtf с определением кодовой страницы
- .txt/.md    -> обычный текст
- .csv/.tsv   -> таблицы с разделителями
- .xlsx/.xlsm -> листы Excel
"""

from __future__ import annotations
import csv
import json
import logging
import re
import shutil
import subprocess
import sys
from zipfile import BadZipFile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Tuple, Optional

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".rtf",
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".xlsx",
    ".xlsm",
    ".json",
}
logger = logging.getLogger(__name__)
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")


def command_exists(name: str) -> bool:
    """Проверяет, доступна ли системная команда."""
    return shutil.which(name) is not None


def is_supported_file(path: Path) -> bool:
    """Проверяет, поддерживается ли файл по расширению."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def safe_print(message: str) -> None:
    """Безопасный вывод в консоль."""
    try:
        print(message)
    except (OSError, UnicodeEncodeError):
        pass


def _clean_cell(value: Any) -> str:
    """Приводит ячейку к одной строке и не даёт сломать Markdown-таблицу."""
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("|", "\\|")


def _clean_table_rows(rows: Any) -> List[List[str]]:
    """Удаляет пустые строки и выравнивает число столбцов."""
    cleaned: List[List[str]] = []
    for raw_row in rows or []:
        row = [_clean_cell(cell) for cell in (raw_row or [])]
        if any(row):
            cleaned.append(row)

    if not cleaned:
        return []

    width = max(len(row) for row in cleaned)
    return [row + [""] * (width - len(row)) for row in cleaned]


def _table_to_markdown(
    rows: List[List[str]],
    title: str,
    location: str = "",
) -> str:
    """Делает один самодостаточный блок таблицы для поиска и цитат."""
    rows = _clean_table_rows(rows)
    if not rows:
        return ""

    marker = f"[ТАБЛИЦА: {title}"
    if location:
        marker += f" | {location}"
    marker += "]"

    header = rows[0]
    lines = [
        marker,
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


# =========================
# PDF
# =========================

def read_pdf_pymupdf(file_path: str | Path) -> Tuple[str, Optional[str]]:
    """Читает PDF через PyMuPDF."""
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf
        except ImportError:
            return "", "PyMuPDF не установлен"

    path = Path(file_path)
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return "", "файл пустой или не найден"

    try:
        parts: List[str] = []
        with pymupdf.open(str(path)) as doc:
            for page_num, page in enumerate(doc, start=1):
                try:
                    page_text = page.get_text("text", sort=True) or ""
                    if page_text.strip():
                        parts.append(
                            f"[СТРАНИЦА {page_num}]\n{page_text.strip()}"
                        )
                except (AttributeError, ValueError, TypeError, RuntimeError) as exc:
                    logger.warning("PyMuPDF: ошибка страницы %s в %s: %s", page_num, path.name, exc)

        text = "\n\n".join(parts).strip()
        if text:
            return text, "PyMuPDF"
        return "", "текст не извлечён (возможно, сканированный PDF)"
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        return "", f"PyMuPDF ошибка: {exc}"


def read_pdf_tables(file_path: str | Path) -> Tuple[str, Optional[str]]:
    """Извлекает только таблицы PDF и сериализует их в Markdown.

    Отдельный проход нужен потому, что текстовый слой PyMuPDF не
    сохраняет ячейки. Маркеры страницы и таблицы затем попадают в
    метаданные чанка.
    """
    try:
        import pdfplumber
    except ImportError:
        return "", "pdfplumber не установлен"

    path = Path(file_path)
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return "", "файл пустой или не найден"

    blocks: List[str] = []
    table_number = 0
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    for table in page.extract_tables() or []:
                        rows = _clean_table_rows(table)
                        if len(rows) < 2:
                            continue
                        table_number += 1
                        blocks.append(
                            _table_to_markdown(
                                rows,
                                title=f"Таблица {table_number}",
                                location=f"страница {page_num}",
                            )
                        )
                except (AttributeError, ValueError, TypeError) as exc:
                    logger.warning("pdfplumber: ошибка таблиц страницы %s в %s: %s", page_num, path.name, exc)
        return "\n\n".join(blocks).strip(), "pdfplumber tables"
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        return "", f"pdfplumber таблицы: {exc}"

def read_pdf_pdfplumber(file_path: str | Path) -> Tuple[str, Optional[str]]:
    """Читает PDF через pdfplumber как fallback, включая таблицы."""
    try:
        import pdfplumber
    except ImportError:
        return "", "pdfplumber не установлен"

    path = Path(file_path)
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return "", "файл пустой или не найден"

    try:
        parts: List[str] = []
        table_number = 0
        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    page_parts: List[str] = []
                    text = page.extract_text() or ""
                    if text.strip():
                        page_parts.append(
                            f"[СТРАНИЦА {page_num}]\n{text.strip()}"
                        )

                    for table in page.extract_tables() or []:
                        rows = _clean_table_rows(table)
                        if len(rows) < 2:
                            continue
                        table_number += 1
                        page_parts.append(
                            _table_to_markdown(
                                rows,
                                title=f"Таблица {table_number}",
                                location=f"страница {page_num}",
                            )
                        )

                    if page_parts:
                        parts.append("\n\n".join(page_parts))
                except (AttributeError, ValueError, TypeError) as exc:
                    safe_print(
                        f"⚠️ pdfplumber: ошибка страницы {page_num} "
                        f"в {path.name}: {exc}"
                    )

        text = "\n\n".join(parts).strip()
        if text:
            return text, "pdfplumber+tables"
        return "", "текст не извлечён (возможно, сканированный PDF)"
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        return "", f"pdfplumber ошибка: {exc}"


def read_pdf_ocr(
    file_path: str | Path,
    page_numbers: set[int] | None = None,
) -> Tuple[str, Optional[str]]:
    """OCR fallback для сканированных PDF через Tesseract."""
    path = Path(file_path)
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return "", "файл пустой или не найден"

    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf
        except ImportError:
            return "", "PyMuPDF не установлен (нужен для OCR)"

    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return "", "pytesseract не установлен"

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        return "", "Pillow не установлен"

    if not command_exists("tesseract"):
        return "", "Tesseract не установлен в системе"

    try:
        parts: List[str] = []
        with pymupdf.open(str(path)) as doc:  # type: ignore
            total_pages = len(doc)
            safe_print(f"📄 OCR страниц {path.name}: {total_pages}")

            for page_num, page in enumerate(doc, start=1):
                if page_numbers is not None and page_num not in page_numbers:
                    continue
                try:
                    matrix = pymupdf.Matrix(2.5, 2.5)  # type: ignore
                    pix = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csRGB)

                    if pix.alpha:
                        img = Image.frombytes(
                            "RGBA",
                            (pix.width, pix.height),  # ← tuple[int, int]
                            pix.samples
                        ).convert("RGB")
                    else:
                        img = Image.frombytes(
                            "RGB",
                            (pix.width, pix.height),  # ← tuple[int, int]
                            pix.samples
                        )

                    text = pytesseract.image_to_string(
                        img,
                        lang="rus+eng",
                        config="--psm 6 --oem 3",
                    )
                    if text.strip():
                        parts.append(
                            f"[СТРАНИЦА {page_num} | OCR]\n{text.strip()}"
                        )
                        safe_print(f"   ✅ Страница {page_num}/{total_pages} распознана")
                    else:
                        logger.warning("OCR не распознал страницу %s в %s", page_num, path.name)
                except (AttributeError, ValueError, TypeError, OSError, RuntimeError) as exc:
                    logger.warning("OCR: ошибка страницы %s в %s: %s", page_num, path.name, exc)

        text = "\n\n".join(parts).strip()
        if text:
            return text, "OCR (Tesseract)"
        return "", "OCR не распознал текст"
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        return "", f"OCR ошибка: {exc}"


def pdf_pages_needing_ocr(file_path: str | Path, min_chars: int = 8) -> set[int]:
    """Возвращает номера страниц без полезного текстового слоя."""
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf
        except ImportError:
            return set()

    path = Path(file_path)
    try:
        with pymupdf.open(str(path)) as doc:
            return {
                page_num
                for page_num, page in enumerate(doc, start=1)
                if len(re.sub(r"\s+", "", page.get_text("text") or "")) < min_chars
            }
    except (OSError, ValueError, TypeError, RuntimeError):
        logger.exception("Не удалось проверить текстовый слой PDF %s", path)
        return set()


def _merge_pdf_pages(text: str, ocr_text: str, table_text: str) -> str:
    """Объединяет текст, OCR и таблицы в порядке исходных PDF-страниц."""
    pages: dict[int, list[str]] = {}
    unlocated: list[str] = []
    for content in (text, ocr_text):
        for block in re.split(r"(?=^\[СТРАНИЦА\s+\d+)", content, flags=re.M):
            if not block.strip():
                continue
            match = re.match(r"\[СТРАНИЦА\s+(\d+)[^\]]*\]\s*", block)
            if match:
                page = int(match.group(1))
                body = block[match.end():].strip()
                if body:
                    pages.setdefault(page, []).append(body)
            else:
                unlocated.append(block.strip())
    for block in re.split(r"(?=^\[ТАБЛИЦА)", table_text, flags=re.M):
        if not block.strip():
            continue
        page_match = re.search(r"\bстраница\s+(\d+)\b", block.splitlines()[0], flags=re.I)
        if page_match:
            pages.setdefault(int(page_match.group(1)), []).append(block.strip())
        else:
            unlocated.append(block.strip())
    page_blocks = [f"[СТРАНИЦА {page}]\n" + "\n\n".join(pages[page]) for page in sorted(pages)]
    return "\n\n".join(unlocated + page_blocks).strip()


def read_pdf(file_path: str | Path) -> Tuple[str, Optional[str]]:
    """Читает PDF с несколькими fallback-стратегиями."""
    path = Path(file_path)
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return "", "файл не существует или пустой"

    safe_print(f"📄 Чтение PDF: {path.name}")

    text, text_source = read_pdf_pymupdf(path)
    if text:
        ocr_text = ""
        missing_pages = pdf_pages_needing_ocr(path)
        if missing_pages:
            ocr_text, ocr_source = read_pdf_ocr(path, page_numbers=missing_pages)
            recognized_pages = {int(value) for value in re.findall(r"\[СТРАНИЦА\s+(\d+)", ocr_text)}
            unread_pages = missing_pages - recognized_pages
            if unread_pages:
                logger.warning("PDF %s: не прочитаны страницы %s (%s)", path.name, sorted(unread_pages), ocr_source)

        table_text, table_source = read_pdf_tables(path)
        sources = [str(text_source)]
        if ocr_text:
            sources.append("OCR для скан-страниц")
        if table_text:
            sources.append(str(table_source))
        if missing_pages and unread_pages:
            sources.append("не прочитаны страницы: " + ", ".join(map(str, sorted(unread_pages))))
        return _merge_pdf_pages(text, ocr_text, table_text), " + ".join(sources)

    # pdfplumber иногда извлекает текст из PDF, на котором PyMuPDF
    # не вернул результата.
    text, source = read_pdf_pdfplumber(path)
    if text:
        return text, source

    text, source = read_pdf_ocr(path)
    if text:
        return text, source

    return "", f"все методы извлечения PDF не сработали; последняя причина: {source}"


# =========================
# DOCX
# =========================

def read_docx(file_path: str | Path) -> Tuple[str, Optional[str]]:
    """Читает DOCX, сохраняя исходный порядок абзацев и таблиц."""
    try:
        from docx import Document
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError:
        return "", "python-docx не установлен"

    path = Path(file_path)
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return "", "файл пустой или не найден"

    try:
        doc = Document(str(path))
        parts: List[str] = []
        previous_text = ""
        table_number = 0

        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                paragraph = Paragraph(child, doc)
                paragraph_text = paragraph.text.strip()
                if paragraph_text:
                    parts.append(paragraph_text)
                    previous_text = paragraph_text
                continue

            if not isinstance(child, CT_Tbl):
                continue

            table = Table(child, doc)
            rows = _clean_table_rows(
                [
                    [
                        " ".join(
                            paragraph.text.strip()
                            for paragraph in cell.paragraphs
                            if paragraph.text.strip()
                        )
                        for cell in row.cells
                    ]
                    for row in table.rows
                ]
            )
            if not rows:
                continue

            table_number += 1
            title = f"Таблица {table_number}"
            if previous_text and len(previous_text) <= 160:
                if re.search(r"\bтабл(?:ица|ицы|ице|ицу|ицы)?\b", previous_text, re.I):
                    title = previous_text

            parts.append(_table_to_markdown(rows, title=title))

        text = "\n\n".join(parts).strip()
        if text:
            return text, "python-docx"
        return "", "текст не извлечён"
    except (OSError, ValueError, TypeError, ImportError, KeyError, BadZipFile) as exc:
        return "", f"python-docx ошибка: {exc}"


# =========================
# DOC
# =========================

def read_doc_textutil(file_path: str | Path) -> Tuple[str, Optional[str]]:
    """Читает .doc через textutil на macOS."""
    if not IS_MACOS:
        return "", "textutil доступен только на macOS"

    if not command_exists("textutil"):
        return "", "textutil не найден в системе"

    path = Path(file_path)
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return "", "файл пустой или не найден"

    try:
        result = subprocess.run(
            ["textutil", "-stdout", "-convert", "txt", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip(), "textutil"
        stderr = (result.stderr or "").strip()
        return "", f"textutil не вернул текст{': ' + stderr if stderr else ''}"
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as exc:
        return "", f"textutil ошибка: {exc}"


def read_doc_textract(file_path: str | Path) -> Tuple[str, Optional[str]]:
    """Читает .doc через textract."""
    try:
        import textract  # noqa: F401
    except ImportError:
        return "", "textract не установлен"

    path = Path(file_path)
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return "", "файл пустой или не найден"

    try:
        import textract
        data = textract.process(str(path))
        text = data.decode("utf-8", errors="ignore").strip()
        if text:
            return text, "textract"
        return "", "textract не вернул текст"
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        return "", f"textract ошибка: {exc}"


def read_doc_antiword(file_path: str | Path) -> Tuple[str, Optional[str]]:
    """Читает .doc через antiword."""
    if not command_exists("antiword"):
        return "", "antiword не установлен"

    path = Path(file_path)
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return "", "файл пустой или не найден"

    try:
        result = subprocess.run(
            ["antiword", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip(), "antiword"
        stderr = (result.stderr or "").strip()
        return "", f"antiword не вернул текст{': ' + stderr if stderr else ''}"
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as exc:
        return "", f"antiword ошибка: {exc}"


def read_doc(file_path: str | Path) -> Tuple[str, Optional[str]]:
    """Читает .doc с платформенным fallback."""
    path = Path(file_path)
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return "", "файл не существует или пустой"

    safe_print(f"📄 Чтение DOC: {path.name}")

    try:
        with path.open("rb") as handle:
            if handle.read(32).lstrip().startswith(b"{\\rtf"):
                return read_rtf(path)
    except OSError as exc:
        return "", f"ошибка чтения DOC: {exc}"

    readers = [read_doc_textutil, read_doc_antiword, read_doc_textract] if IS_MACOS else [read_doc_antiword, read_doc_textract]
    failures: list[str] = []
    for reader in readers:
        try:
            text, source = reader(path)
        except Exception as exc:
            # Изолированная граница стороннего конвертера: textract использует
            # собственные типы ошибок. Другие документы продолжают читаться.
            logger.exception("Ошибка конвертера %s для %s", reader.__name__, path.name)
            failures.append(f"{reader.__name__}: {exc}")
            continue
        if text:
            return text, source
        failures.append(str(source))
    return "", "все методы извлечения DOC не сработали: " + "; ".join(failures)


# =========================
# RTF
# =========================

def _strip_rtf_objects(raw: bytes) -> bytes:
    """Пропускает вложенные рисунки/объекты, сохраняя текст и RTF-структуру.

    В нормативных RTF сотни мегабайт могут занимать шестнадцатеричные
    изображения. Они не участвуют в текстовом поиске и не должны четырежды
    проходить через декодер. Скобки считаются с учётом escape и bin-блоков.
    """
    object_start = re.compile(rb"\{\\(?:\*\\)?(?:pict|shppict|nonshppict|object|objdata)\b")
    token_pattern = re.compile(rb"\\(?:[{}\\]|bin(\d+) ?)|[{}]")
    parts: list[bytes] = []
    cursor = 0
    while match := object_start.search(raw, cursor):
        depth = 1
        position = match.end()
        while depth and (token := token_pattern.search(raw, position)):
            position = token.end()
            if token.group(1):
                position += int(token.group(1))
            elif token.group(0) == b"{":
                depth += 1
            elif token.group(0) == b"}":
                depth -= 1
        if depth:
            logger.warning("RTF содержит незакрытую группу изображения/объекта")
            break
        parts.append(raw[cursor:match.start()])
        cursor = position
    if not parts:
        return raw
    parts.append(raw[cursor:])
    return b"".join(parts)


def read_rtf(file_path: str | Path) -> Tuple[str, Optional[str]]:
    """Читает RTF через striprtf."""
    try:
        from striprtf.striprtf import rtf_to_text
    except ImportError:
        return "", "striprtf не установлен"

    path = Path(file_path)
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return "", "файл пустой или не найден"

    try:
        raw = _strip_rtf_objects(path.read_bytes())
    except OSError as exc:
        return "", f"ошибка чтения RTF: {exc}"

    header = raw[:4096].decode("latin-1", errors="ignore")
    codepage_match = re.search(r"\\ansicpg(\d+)", header)
    declared_encoding = (
        f"cp{codepage_match.group(1)}" if codepage_match else None
    )
    encodings = [declared_encoding, "utf-8", "cp1251", "latin-1"]

    candidates: List[tuple[int, str, str]] = []
    for encoding in dict.fromkeys(item for item in encodings if item):
        try:
            content = raw.decode(encoding, errors="strict")
            if content.strip():
                text = rtf_to_text(content, encoding=encoding, errors="replace").strip()
                if text:
                    # Выбираем вариант без mojibake и символов замены.
                    penalty = text.count("�") * 100
                    penalty += sum(text.count(token) for token in ("Ð", "Ñ", "Ã", "Â"))
                    # Строгое декодирование + явно заявленная кодировка
                    # дают хороший первый вариант без повторных проходов.
                    if penalty == 0:
                        return text, f"striprtf ({encoding})"
                    candidates.append((penalty, text, encoding))
        except (LookupError, UnicodeDecodeError, ValueError):
            continue

    if candidates:
        _, text, encoding = min(candidates, key=lambda item: item[0])
        return text, f"striprtf ({encoding})"

    return "", "не удалось прочитать RTF"


# =========================
# Текст и табличные форматы
# =========================

def _decode_text_bytes(raw: bytes) -> Tuple[str, str]:
    """Декодирует обычный текст без молчаливой потери кириллицы."""
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16"), "utf-16"

    candidates: List[tuple[int, str, str]] = []
    for encoding in ("utf-8", "cp1251", "latin-1"):
        try:
            value = raw.decode(encoding, errors="strict")
        except UnicodeDecodeError:
            continue
        penalty = value.count("�") * 100
        penalty += sum(value.count(token) for token in ("Ð", "Ñ", "Ã", "Â"))
        candidates.append((penalty, value, encoding))

    if not candidates:
        return raw.decode("utf-8", errors="replace"), "utf-8/replace"
    _, value, encoding = min(candidates, key=lambda item: item[0])
    return value, encoding


def read_plain_text(file_path: str | Path) -> Tuple[str, Optional[str]]:
    """Читает TXT и Markdown."""
    path = Path(file_path)
    try:
        text, encoding = _decode_text_bytes(path.read_bytes())
    except OSError as exc:
        return "", f"ошибка чтения текста: {exc}"
    return text.strip(), f"plain text ({encoding})"


def read_delimited_table(file_path: str | Path) -> Tuple[str, Optional[str]]:
    """Читает CSV/TSV и превращает строки в один табличный блок."""
    path = Path(file_path)
    try:
        text, encoding = _decode_text_bytes(path.read_bytes())
    except OSError as exc:
        return "", f"ошибка чтения таблицы: {exc}"

    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    if path.suffix.lower() == ".csv":
        try:
            delimiter = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ";" if text.count(";") > text.count(",") else ","

    rows = _clean_table_rows(csv.reader(text.splitlines(), delimiter=delimiter))
    if not rows:
        return "", "таблица пуста"
    return (
        _table_to_markdown(rows, title=path.stem),
        f"delimited table ({encoding}, delimiter={delimiter!r})",
    )


def read_excel(file_path: str | Path) -> Tuple[str, Optional[str]]:
    """Читает видимые и скрытые листы XLSX/XLSM через openpyxl."""
    try:
        from openpyxl import load_workbook
    except ImportError:
        return "", "openpyxl не установлен"

    path = Path(file_path)
    try:
        workbook = load_workbook(
            filename=str(path),
            read_only=True,
            data_only=False,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return "", f"openpyxl ошибка: {exc}"

    blocks: List[str] = []
    try:
        for sheet in workbook.worksheets:
            rows = _clean_table_rows(sheet.iter_rows(values_only=True))
            if not rows:
                continue
            blocks.append(
                _table_to_markdown(
                    rows,
                    title=f"Лист {sheet.title}",
                    location=f"лист {sheet.title}",
                )
            )
    finally:
        workbook.close()

    if not blocks:
        return "", "в книге Excel нет непустых листов"
    return "\n\n".join(blocks), "openpyxl"


def read_json_document(file_path: str | Path) -> Tuple[str, Optional[str]]:
    """Читает JSON-выгрузку; поле full_text использует как основной корпус."""
    path = Path(file_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return "", f"JSON ошибка: {exc}"

    if isinstance(payload, dict) and isinstance(payload.get("full_text"), str):
        source_name = str(payload.get("file") or path.name)
        return (
            f"[ИСТОЧНИК: {source_name}]\n{payload['full_text'].strip()}",
            "json full_text",
        )

    if isinstance(payload, list) and payload and all(isinstance(row, dict) for row in payload):
        headers = list(dict.fromkeys(key for row in payload for key in row.keys()))
        rows = [headers] + [[row.get(key, "") for key in headers] for row in payload]
        return _table_to_markdown(rows, title=path.stem), "json records"

    return json.dumps(payload, ensure_ascii=False, indent=2), "json"


# =========================
# Универсальное чтение
# =========================

def read_file(file_path: str | Path) -> Tuple[str, Optional[str]]:
    """Универсальное чтение файла по расширению."""
    path = Path(file_path)

    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return "", "файл не существует или пустой"

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return "", f"неподдерживаемый формат: {path.suffix}"

    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    if suffix == ".doc":
        return read_doc(path)
    if suffix == ".rtf":
        return read_rtf(path)
    if suffix in {".txt", ".md"}:
        return read_plain_text(path)
    if suffix in {".csv", ".tsv"}:
        return read_delimited_table(path)
    if suffix in {".xlsx", ".xlsm"}:
        return read_excel(path)
    if suffix == ".json":
        return read_json_document(path)

    return "", "неизвестный формат"


# =========================
# Структура результата
# =========================

@dataclass(slots=True)
class ParsedDocument:
    """Структура обработанного документа."""
    doc_name: str
    filepath: str
    filetype: str
    text: str
    chunks: List[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentParser:
    """Парсер документов с разбиением текста на фрагменты."""

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        min_chunk_size: int = 120,
    ) -> None:
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

        normalized_lines: List[str] = []
        for line in text.split("\n"):
            if "\t" in line:
                cells = [_clean_cell(cell) for cell in line.split("\t")]
                if len(cells) >= 2 and sum(bool(cell) for cell in cells) >= 2:
                    normalized_lines.append("| " + " | ".join(cells) + " |")
                    continue
            normalized_lines.append(re.sub(r" {2,}", " ", line).strip())

        text = "\n".join(normalized_lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def split_paragraphs(self, text: str) -> List[str]:
        """Группирует абзацы и сохраняет адрес источника в каждом фрагменте."""
        parts = re.split(r"\n\s*\n|(?=^\[(?:СТРАНИЦА|ИСТОЧНИК|ТАБЛИЦА))", text, flags=re.M)
        result: List[str] = []
        location: dict[str, str] = {}
        pending: list[str] = []

        def add_prefix(fragment: str, markers: dict[str, str]) -> str:
            return "\n".join([*markers.values(), fragment]).strip()

        def flush() -> None:
            if not pending:
                return
            paragraph = "\n\n".join(pending)
            fragments = self.split_long_text(paragraph) if len(paragraph) > self.chunk_size else [paragraph]
            result.extend(add_prefix(fragment, location) for fragment in fragments)
            pending.clear()

        for part in parts:
            part = part.strip()
            if not part:
                continue
            marker = re.match(r"\[(СТРАНИЦА|ИСТОЧНИК)[^\]]*\]", part)
            if marker:
                flush()
                marker_kind = marker.group(1)
                if marker_kind == "ИСТОЧНИК":
                    location.clear()
                location[marker_kind] = marker.group(0)
                part = part[marker.end():].strip()
                if not part:
                    continue
            if self._looks_like_table_block(part):
                flush()
                table_location = dict(location)
                table_page = self.extract_chunk_metadata(part).get("page")
                if table_page:
                    table_location["СТРАНИЦА"] = f"[СТРАНИЦА {table_page}]"
                result.extend(add_prefix(fragment, table_location) for fragment in self.split_table_block(part))
            else:
                if pending and len("\n\n".join(pending + [part])) > self.chunk_size:
                    flush()
                pending.append(part)
        flush()
        return result

    @staticmethod
    def _looks_like_table_block(text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False
        if lines[0].startswith("[ТАБЛИЦА"):
            return True
        pipe_rows = sum(1 for line in lines if line.startswith("|") and line.endswith("|"))
        return pipe_rows >= 2

    def split_table_block(self, text: str) -> List[str]:
        """
        Делит длинную таблицу по строкам и повторяет маркер и шапку.

        Это позволяет найти строку в большом Excel/PDF и понять её
        столбцы даже после чанкинга.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return []

        prefix: List[str] = []
        cursor = 0
        if lines[0].startswith("[ТАБЛИЦА"):
            prefix.append(lines[0])
            cursor = 1

        if cursor < len(lines):
            prefix.append(lines[cursor])
            cursor += 1
        if cursor < len(lines) and re.fullmatch(r"\|?[\s:|\-]+\|?", lines[cursor]):
            prefix.append(lines[cursor])
            cursor += 1

        rows = lines[cursor:]
        if not rows:
            return ["\n".join(prefix)]

        chunks: List[str] = []
        current = list(prefix)
        for row in rows:
            candidate = "\n".join(current + [row])
            if len(candidate) > self.chunk_size and len(current) > len(prefix):
                chunks.append("\n".join(current))
                current = list(prefix)
            current.append(row)

        if len(current) > len(prefix):
            chunks.append("\n".join(current))
        return chunks or [text]

    def split_long_text(self, text: str) -> List[str]:
        """Разбивает длинный текст по предложениям."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) < 2:
            return self.hard_split(text)

        result: List[str] = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > self.chunk_size:
                if current:
                    result.append(current.strip())
                    current = ""
                result.extend(self.hard_split(sentence))
                continue

            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current.strip())
                    overlap = current[-self.chunk_overlap:] if self.chunk_overlap else ""
                    overlap = self.smart_overlap(overlap)
                    with_overlap = f"{overlap} {sentence}".strip() if overlap else sentence
                    current = with_overlap if len(with_overlap) <= self.chunk_size else sentence
                else:
                    result.extend(self.hard_split(sentence))
                    current = ""

        if current.strip():
            result.append(current.strip())

        return result

    def hard_split(self, text: str) -> List[str]:
        """Принудительно разбивает длинный текст."""
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
    def smart_overlap(text: str) -> str:
        """Подчищает overlap, чтобы не начинать с обрывка слова."""
        text = text.strip()
        if not text:
            return ""
        split_pos = text.find(" ")
        if 0 < split_pos < len(text) // 2:
            text = text[split_pos + 1:].strip()
        return text

    @staticmethod
    def extract_formulas(text: str) -> List[dict[str, Any]]:
        """Пытается извлечь простые формулы."""
        formulas: List[dict[str, Any]] = []
        patterns = [
            r"[A-Za-zА-Яа-я0-9_]+\s*=\s*[^=\n]{3,120}",
            r"\bQ\s*=\s*[^=\n]{3,120}",
            r"\bR\s*=\s*[^=\n]{3,120}",
        ]

        for pattern in patterns:
            for match in re.findall(pattern, text):
                raw = match.strip()
                if len(raw) < 4:
                    continue
                formulas.append(
                    {
                        "raw": raw,
                        "variables": sorted(set(re.findall(r"[A-Za-zА-Яа-я_]+", raw))),
                        "has_operator": any(op in raw for op in ("=", "+", "-", "*", "/")),
                    }
                )

        unique: List[dict[str, Any]] = []
        seen: set[str] = set()
        for item in formulas:
            key = item["raw"]
            if key not in seen:
                seen.add(key)
                unique.append(item)

        return unique[:20]

    @staticmethod
    def detect_table_like_content(text: str) -> bool:
        """Определяет, похож ли фрагмент на таблицу."""
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
    def extract_chunk_metadata(text: str) -> dict[str, Any]:
        """Извлекает адрес таблицы/страницы из служебных маркеров."""
        page_numbers = [
            int(value)
            for value in re.findall(r"\[СТРАНИЦА\s+(\d+)", text, flags=re.I)
        ]
        if not page_numbers:
            page_numbers = [
                int(value)
                for value in re.findall(r"\bстраница\s+(\d+)\b", text, flags=re.I)
            ]

        table_match = re.search(r"\[ТАБЛИЦА:\s*([^\]|]+)", text, flags=re.I)
        sheet_match = re.search(r"\bлист\s+([^\]|]+)", text, flags=re.I)
        source_match = re.search(r"\[ИСТОЧНИК:\s*([^\]]+)", text, flags=re.I)

        result: dict[str, Any] = {}
        if page_numbers:
            result["page_numbers"] = sorted(set(page_numbers))
            result["page"] = min(page_numbers)
        if table_match:
            result["table_title"] = table_match.group(1).strip()
        if sheet_match:
            result["sheet"] = sheet_match.group(1).strip()
        if source_match:
            result["logical_source"] = source_match.group(1).strip()
        return result

    @staticmethod
    def build_metadata(
        path: Path,
        text: str,
        chunks: List[dict[str, Any]],
        source: str | None = None,
    ) -> dict[str, Any]:
        """Собирает метаданные документа."""
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
            "has_table_like_content": any(
                chunk.get("has_table_like_content", False) for chunk in chunks
            ),
            "extraction_source": source or "unknown",
            "platform": sys.platform,
        }

    def parse_file(self, file_path: str | Path) -> dict[str, Any]:
        """Парсит один файл."""
        path = Path(file_path)
        safe_print(f"📄 Парсинг: {path.name}")

        try:
            text, source = read_file(path)
            text = self.normalize_text(text)
        except Exception as exc:
            # Один повреждённый файл не должен отменять индексацию корпуса.
            logger.exception("Ошибка парсинга файла %s", path)
            safe_print(f"⚠️ Ошибка парсинга {path.name}: {exc}")
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
                    "error": str(exc),
                },
            }

        if not text:
            reason = source or "неизвестная причина"
            logger.warning("Текст не извлечён из %s: %s", path, reason)
            safe_print(f"⚠️ Не удалось извлечь текст из {path.name}: {reason}")
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
                    "reason": reason,
                },
            }

        safe_print(f"✅ Текст извлечён из {path.name}: {len(text)} символов через {source}")

        raw_chunks = self.split_paragraphs(text)
        chunks: List[dict[str, Any]] = []

        for idx, chunk_text in enumerate(raw_chunks):
            formulas = self.extract_formulas(chunk_text)
            chunk_metadata = self.extract_chunk_metadata(chunk_text)
            chunks.append(
                {
                    "doc_name": path.name,
                    "filepath": str(path),
                    "chunk_id": idx,
                    "text": chunk_text,
                    "metadata": {
                        "formulas": formulas,
                        **chunk_metadata,
                    },
                    "has_formula": bool(formulas),
                    "has_table_like_content": self.detect_table_like_content(chunk_text),
                }
            )

        metadata = self.build_metadata(path, text, chunks, source)

        return {
            "doc_name": path.name,
            "filepath": str(path),
            "filetype": path.suffix.lower(),
            "text": text,
            "chunks": chunks,
            "metadata": metadata,
        }

    def parse_directory(self, directory: str | Path, recursive: bool = True) -> List[dict[str, Any]]:
        """Парсит директорию с документами."""
        base = Path(directory)
        if not base.exists() or not base.is_dir():
            safe_print(f"⚠️ Директория не существует: {base}")
            return []

        pattern = "**/*" if recursive else "*"
        all_files = [p for p in base.glob(pattern) if p.is_file()]
        supported_files = [p for p in all_files if is_supported_file(p)]
        unsupported_files = [p for p in all_files if not is_supported_file(p)]

        safe_print(f"📦 Всего файлов в {base}: {len(all_files)}")
        safe_print(f"✅ Поддерживаемых: {len(supported_files)}")
        safe_print(f"⛔ Неподдерживаемых: {len(unsupported_files)}")

        if unsupported_files:
            safe_print("Неподдерживаемые файлы (первые 10):")
            for p in unsupported_files[:10]:
                safe_print(f"   - {p.name} [{p.suffix}]")

        parsed: List[dict[str, Any]] = []
        failed: List[str] = []

        for path in sorted(supported_files):
            try:
                item = self.parse_file(path)
                if item.get("text"):
                    parsed.append(item)
                else:
                    failed.append(path.name)
            except Exception as exc:
                # Граница обработки независимых пользовательских файлов.
                logger.exception("Критическая ошибка парсинга %s", path)
                safe_print(f"⚠️ Критическая ошибка на файле {path.name}: {exc}")
                failed.append(path.name)

        safe_print(f"✅ Успешно распарсено: {len(parsed)}")
        if failed:
            safe_print(f"⚠️ Не удалось распарсить: {len(failed)}")
            for name in failed[:10]:
                safe_print(f"   - {name}")

        return parsed


def parse_file(file_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Удобная функция парсинга одного файла."""
    parser = DocumentParser(**kwargs)
    return parser.parse_file(file_path)


def parse_directory(directory: str | Path, **kwargs: Any) -> List[dict[str, Any]]:
    """Удобная функция парсинга директории."""
    parser = DocumentParser(
        chunk_size=kwargs.pop("chunk_size", 1200),
        chunk_overlap=kwargs.pop("chunk_overlap", 200),
        min_chunk_size=kwargs.pop("min_chunk_size", 120),
    )
    recursive = kwargs.pop("recursive", True)
    return parser.parse_directory(directory, recursive=recursive)


def _run_self_tests() -> None:
    """Проверяет адреса фрагментов, целостность таблиц и RTF-объектов."""
    table = _table_to_markdown([["Город", "Дни"], ["Томск", "225"]], "Климат")
    parser = DocumentParser(chunk_size=300, min_chunk_size=50)
    chunks = parser.split_paragraphs(table)
    assert chunks and "Томск" in chunks[0]
    assert ".rtf" in SUPPORTED_EXTENSIONS
    page_chunks = parser.split_paragraphs("[СТРАНИЦА 7]\n" + "Описание системы. " * 100)
    assert len(page_chunks) > 2
    assert all(parser.extract_chunk_metadata(chunk)["page"] == 7 for chunk in page_chunks)
    assert _strip_rtf_objects(rb"{\rtf1 Before {\pict abc{xyz}123} After}") == rb"{\rtf1 Before  After}"
    assert "| Город |  | Дни |" in parser.normalize_text("Город\t\tДни")
    ordered = _merge_pdf_pages("[СТРАНИЦА 2]\nSecond", "[СТРАНИЦА 1 | OCR]\nFirst", "")
    assert ordered.index("First") < ordered.index("Second")


if __name__ == "__main__":
    _run_self_tests()

# ИСПРАВЛЕНО: PDF-текст/таблицы/OCR объединяются по страницам; адрес страницы
# сохраняется во всех чанках; DOCX сохраняет порядок; RTF учитывает кодировку
# и быстро пропускает изображения; DOC использует платформенные конвертеры;
# таблицы сохраняют пустые ячейки/заголовки; ошибки логируются; добавлены самотесты.
