# core/table_extractor.py
"""
Модуль для интеллектуального извлечения таблиц из текста.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False


@dataclass(slots=True)
class ExtractedTable:
    """Класс для хранения извлечённой таблицы."""
    title: str
    headers: list[str]
    rows: list[list[str]]
    raw_text: str
    source: str
    confidence: float = 0.0

    def to_markdown(self) -> str:
        """Преобразует таблицу в Markdown."""
        if not self.headers or not self.rows:
            return f"**{self.title}**\n\n(Таблица пуста или не распознана)"

        lines = [
            f"**{self.title}**",
            "",
            "| " + " | ".join(self.headers) + " |",
            "| " + " | ".join(["---"] * len(self.headers)) + " |",
        ]

        for row in self.rows[:20]:
            padded = row + [""] * (len(self.headers) - len(row))
            visible = padded[:len(self.headers)]
            lines.append("| " + " | ".join(str(cell).strip()[:80] for cell in visible) + " |")

        if len(self.rows) > 20:
            lines.append(f"*... и ещё {len(self.rows) - 20} строк*")

        return "\n".join(lines)

    def to_dataframe(self):
        """Преобразует в pandas DataFrame."""
        if not PANDAS_AVAILABLE or pd is None:
            return None

        if not self.headers:
            return pd.DataFrame(self.rows)

        rows_padded: list[list[str]] = []
        for row in self.rows:
            padded = row + [""] * (len(self.headers) - len(row))
            rows_padded.append(padded[:len(self.headers)])

        return pd.DataFrame(rows_padded, columns=self.headers)

    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь для JSON."""
        return {
            "title": self.title,
            "headers": self.headers,
            "rows": self.rows[:50],
            "row_count": len(self.rows),
            "source": self.source,
            "confidence": self.confidence,
        }


class TableExtractor:
    """Извлекает таблицы из текста по структурным признакам."""

    def __init__(self) -> None:
        self._cache: dict[str, list[ExtractedTable]] = {}

    def extract(self, text: str, source: str = "", min_rows: int = 2) -> list[ExtractedTable]:
        """Извлекает таблицы из текста."""
        if not text or not text.strip():
            return []

        cache_key = f"{source}_{hash(text[:1000])}_{min_rows}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        tables: list[ExtractedTable] = []
        tables.extend(self._extract_pipe_tables(text, source))
        tables.extend(self._extract_marker_tables(text, source))
        tables.extend(self._extract_aligned_tables(text, source))
        tables.extend(self._extract_keyword_tables(text, source))

        unique_tables = self._deduplicate_tables(tables)
        unique_tables.sort(key=lambda item: item.confidence, reverse=True)

        filtered = [table for table in unique_tables if len(table.rows) >= min_rows]
        self._cache[cache_key] = filtered
        return filtered

    def _extract_pipe_tables(self, text: str, source: str) -> list[ExtractedTable]:
        """Извлекает таблицы с разделителями |."""
        tables: list[ExtractedTable] = []
        lines = text.splitlines()
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            if "|" in line and len([p for p in line.split("|") if p.strip()]) >= 2:
                title = self._find_title(lines, i)
                table_lines: list[str] = []
                j = i

                while j < len(lines) and "|" in lines[j]:
                    current = lines[j].strip()
                    if current:
                        table_lines.append(current)
                    j += 1

                if len(table_lines) >= 2:
                    headers, rows = self._parse_pipe_table(table_lines)
                    if headers or rows:
                        tables.append(
                            ExtractedTable(
                                title=title,
                                headers=headers,
                                rows=rows,
                                raw_text="\n".join(table_lines),
                                source=source,
                                confidence=0.90,
                            )
                        )
                i = j
            else:
                i += 1

        return tables

    def _extract_marker_tables(self, text: str, source: str) -> list[ExtractedTable]:
        """Извлекает таблицы по маркеру [ТАБЛИЦА]."""
        tables: list[ExtractedTable] = []
        lines = text.splitlines()
        in_table = False
        table_lines: list[str] = []
        title = "Таблица"

        for i, line in enumerate(lines):
            stripped = line.strip()

            if "[ТАБЛИЦА]" in stripped:
                in_table = True
                table_lines = []
                if i > 0 and len(lines[i - 1].strip()) < 100:
                    title = lines[i - 1].strip() or "Таблица"
                continue

            if in_table:
                if not stripped:
                    if table_lines:
                        headers, rows = self._parse_plain_table(table_lines)
                        if rows:
                            tables.append(
                                ExtractedTable(
                                    title=title,
                                    headers=headers,
                                    rows=rows,
                                    raw_text="\n".join(table_lines),
                                    source=source,
                                    confidence=0.85,
                                )
                            )
                    in_table = False
                    title = "Таблица"
                    table_lines = []
                else:
                    table_lines.append(stripped)

        if in_table and table_lines:
            headers, rows = self._parse_plain_table(table_lines)
            if rows:
                tables.append(
                    ExtractedTable(
                        title=title,
                        headers=headers,
                        rows=rows,
                        raw_text="\n".join(table_lines),
                        source=source,
                        confidence=0.80,
                    )
                )

        return tables

    def _extract_aligned_tables(self, text: str, source: str) -> list[ExtractedTable]:
        """Извлекает таблицы по выравниванию текста в колонки."""
        tables: list[ExtractedTable] = []
        lines = text.splitlines()
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            parts = re.split(r"\s{3,}", line)
            if len(parts) >= 3:
                title = self._find_title(lines, i)
                table_lines = [line]
                j = i + 1

                while j < len(lines):
                    next_line = lines[j].strip()
                    next_parts = re.split(r"\s{3,}", next_line)
                    if next_line and len(next_parts) >= 3:
                        table_lines.append(next_line)
                        j += 1
                    else:
                        break

                if len(table_lines) >= 2:
                    parsed_rows = [re.split(r"\s{3,}", item.strip()) for item in table_lines]
                    parsed_rows = [row for row in parsed_rows if any(cell.strip() for cell in row)]

                    if len(parsed_rows) >= 2:
                        headers = [cell.strip() for cell in parsed_rows[0]]
                        rows = [[cell.strip() for cell in row] for row in parsed_rows[1:]]
                        tables.append(
                            ExtractedTable(
                                title=title,
                                headers=headers,
                                rows=rows,
                                raw_text="\n".join(table_lines),
                                source=source,
                                confidence=0.60,
                            )
                        )
                i = j
            else:
                i += 1

        return tables

    def _extract_keyword_tables(self, text: str, source: str) -> list[ExtractedTable]:
        """Извлекает таблицы по ключевым словам."""
        tables: list[ExtractedTable] = []
        keywords = [
            "таблица", "табл", "table",
            "показатели", "значения", "параметры",
            "данные", "список", "перечень",
            "температура", "давление", "расход",
            "город", "регион", "климат",
        ]

        lines = text.splitlines()

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            has_keyword = any(keyword in line_lower for keyword in keywords)
            has_numbers = bool(re.search(r"\d+[.,]?\d*", line))

            if has_keyword and has_numbers and len(line.strip()) > 30:
                table_lines = [line.strip()]
                j = i + 1
                collected = 0

                while j < len(lines) and collected < 10:
                    next_line = lines[j].strip()

                    if next_line and re.search(r"\d+[.,]?\d*", next_line):
                        table_lines.append(next_line)
                        collected += 1
                    elif next_line and len(next_line) < 80 and table_lines:
                        table_lines[-1] += " " + next_line
                    else:
                        break

                    j += 1

                if len(table_lines) >= 3:
                    headers, rows = self._parse_plain_table(table_lines)
                    if rows:
                        prev_title = lines[i - 1].strip() if i > 0 else ""
                        title = prev_title if prev_title and len(prev_title) < 80 else "Таблица"
                        tables.append(
                            ExtractedTable(
                                title=title,
                                headers=headers,
                                rows=rows,
                                raw_text="\n".join(table_lines),
                                source=source,
                                confidence=0.50,
                            )
                        )

        return tables

    @staticmethod
    def _find_title(lines: list[str], index: int) -> str:
        """Находит заголовок таблицы."""
        for j in range(max(0, index - 3), index):
            line = lines[j].strip()
            if line and len(line) < 100 and not line.startswith("|") and "|" not in line:
                return line
        return "Таблица"

    @staticmethod
    def _parse_pipe_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
        """Парсит таблицу с разделителями |."""
        if not lines:
            return [], []

        cleaned_lines = [line for line in lines if line.strip()]
        if len(cleaned_lines) < 2:
            return [], []

        if re.match(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$", cleaned_lines[1]):
            header_line = cleaned_lines[0]
            data_lines = cleaned_lines[2:]
        else:
            header_line = cleaned_lines[0]
            data_lines = cleaned_lines[1:]

        headers = TableExtractor._split_pipe_line(header_line)
        rows: list[list[str]] = []

        for line in data_lines:
            row = TableExtractor._split_pipe_line(line)
            if row and any(cell.strip() for cell in row):
                rows.append(row)

        return headers, rows

    @staticmethod
    def _split_pipe_line(line: str) -> list[str]:
        """Разбивает строку по |."""
        if not line:
            return []

        cleaned = line.strip()
        if cleaned.startswith("|"):
            cleaned = cleaned[1:]
        if cleaned.endswith("|"):
            cleaned = cleaned[:-1]

        return [cell.strip() for cell in cleaned.split("|")]

    @staticmethod
    def _parse_plain_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
        """Парсит обычную таблицу без |."""
        if not lines:
            return [], []

        cleaned_lines = [line for line in lines if line.strip()]
        if len(cleaned_lines) < 2:
            return [], []

        first_line = cleaned_lines[0]
        second_line = cleaned_lines[1]

        if re.match(r"^[\d\s.,;:%()\-–—/]+$", second_line):
            headers = TableExtractor._split_by_spaces(first_line)
            data_lines = cleaned_lines[1:]
        else:
            headers = []
            data_lines = cleaned_lines

        rows: list[list[str]] = []
        for line in data_lines:
            row = TableExtractor._split_by_spaces(line)
            if row and any(cell.strip() for cell in row):
                rows.append(row)

        return headers, rows

    @staticmethod
    def _split_by_spaces(line: str) -> list[str]:
        """Разбивает строку по группам пробелов."""
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) >= 2:
            return [part.strip() for part in parts if part.strip()]
        return [part.strip() for part in line.strip().split() if part.strip()]

    @staticmethod
    def _deduplicate_tables(tables: list[ExtractedTable]) -> list[ExtractedTable]:
        """Удаляет дубликаты таблиц."""
        seen: set[tuple[str, str, str]] = set()
        unique_tables: list[ExtractedTable] = []

        for table in tables:
            key = (
                table.source.strip().lower(),
                table.title.strip().lower(),
                table.raw_text[:200].strip(),
            )
            if key not in seen:
                seen.add(key)
                unique_tables.append(table)

        return unique_tables

    def deduplicate_tables(self, tables: list[ExtractedTable]) -> list[ExtractedTable]:
        """Публичный метод для дедупликации таблиц."""
        return self._deduplicate_tables(tables)

    def extract_from_chunks(self, chunks: list[dict[str, Any]], source: str = "") -> list[ExtractedTable]:
        """Извлекает таблицы из списка фрагментов."""
        all_tables: list[ExtractedTable] = []

        for chunk in chunks:
            text = str(chunk.get("text", "") or "")
            doc_name = str(chunk.get("doc_name") or chunk.get("docname") or source)
            if not text.strip():
                continue
            tables = self.extract(text, doc_name)
            all_tables.extend(tables)

        return self._deduplicate_tables(all_tables)

    def search_table(self, query: str, chunks: list[dict[str, Any]]) -> ExtractedTable | None:
        """Ищет таблицу по запросу."""
        all_tables = self.extract_from_chunks(chunks)
        if not all_tables:
            return None

        query_lower = query.lower().strip()
        scored_tables: list[tuple[ExtractedTable, float]] = []

        for table in all_tables:
            score = 0.0

            if query_lower in table.title.lower():
                score += 1.0

            for header in table.headers:
                if query_lower in header.lower():
                    score += 0.3

            for row in table.rows:
                row_text = " ".join(row).lower()
                if query_lower in row_text:
                    score += 0.5

            score += table.confidence * 0.2
            scored_tables.append((table, score))

        scored_tables.sort(key=lambda item: item[1], reverse=True)
        return scored_tables[0][0] if scored_tables and scored_tables[0][1] > 0 else None

    @staticmethod
    def format_table_for_response(table: ExtractedTable | None) -> str:
        """Форматирует таблицу для ответа пользователю."""
        if not table:
            return "Таблица не найдена"

        lines = [
            f"**📊 {table.title}**",
            f"*Источник: {table.source}*",
            "",
            table.to_markdown(),
            "",
            f"*Всего строк: {len(table.rows)}*",
        ]
        return "\n".join(lines)


def patch_qa_system_with_table_extractor() -> None:
    """Добавляет улучшенное извлечение таблиц в QASystem."""
    try:
        from core.qa_engine import QASystem
    except ImportError:
        try:
            from core.qaengine import QASystem
        except ImportError:
            return

    def enhanced_answer(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Улучшенный ответ с распознаванием таблиц."""
        extractor = TableExtractor()

        if hasattr(self, "search_with_formulas"):
            result = self.search_with_formulas(question, top_k)
            relevant = result.get("results", []) if isinstance(result, dict) else []
            all_formulas = result.get("formulas", []) if isinstance(result, dict) else []
        else:
            relevant = self.search(question, top_k=top_k) if hasattr(self, "search") else []
            all_formulas = []

        if not relevant:
            return {
                "question": question,
                "answer": "❌ Информация по вашему вопросу не найдена в документации.",
                "sources": [],
                "tables": [],
                "formulas": [],
            }

        all_tables: list[ExtractedTable] = []

        for chunk in relevant:
            if not isinstance(chunk, dict):
                continue
            chunk_text = str(chunk.get("text", "") or "")
            chunk_doc = str(chunk.get("doc_name") or chunk.get("docname") or "")
            tables = extractor.extract(chunk_text, chunk_doc)
            all_tables.extend(tables)

        unique_tables = extractor.deduplicate_tables(all_tables)

        if hasattr(self, "is_bad_chunk"):
            cleaned_chunks = [
                chunk for chunk in relevant
                if isinstance(chunk, dict) and not self.is_bad_chunk(str(chunk.get("text", "")))
            ]
        else:
            cleaned_chunks = [chunk for chunk in relevant if isinstance(chunk, dict)]

        if not cleaned_chunks:
            cleaned_chunks = [chunk for chunk in relevant[:2] if isinstance(chunk, dict)]

        answer_lines = ["**📌 Краткий ответ:**"]

        first_text = str(cleaned_chunks[0].get("text", "")).strip() if cleaned_chunks else ""
        if first_text:
            first_sentence = re.split(r"(?<=[.!?])\s+", first_text)[0].strip()
            answer_lines.append(first_sentence if first_sentence else "Найдена релевантная информация в документации.")
            answer_lines.append("")
        else:
            answer_lines.append("Найдена релевантная информация в документации.")
            answer_lines.append("")

        if unique_tables:
            answer_lines.append("**📊 Найденные таблицы:**")
            answer_lines.append("")

            for i, table in enumerate(unique_tables[:2], start=1):
                answer_lines.append(f"**Таблица {i}: {table.title}**")
                answer_lines.append(f"*Источник: {table.source}*")
                answer_lines.append("")

                if table.headers:
                    visible_headers = table.headers[:6]
                    answer_lines.append("| " + " | ".join(visible_headers) + " |")
                    answer_lines.append("| " + " | ".join(["---"] * len(visible_headers)) + " |")

                    for row in table.rows[:5]:
                        padded = row + [""] * (len(visible_headers) - len(row))
                        answer_lines.append(
                            "| " + " | ".join(str(cell).strip()[:30] for cell in padded[:len(visible_headers)]) + " |"
                        )
                else:
                    for row in table.rows[:5]:
                        answer_lines.append("- " + " | ".join(str(cell).strip() for cell in row[:6]))

                if len(table.rows) > 5:
                    answer_lines.append(f"*... и ещё {len(table.rows) - 5} строк*")

                answer_lines.append("")

        if all_formulas:
            answer_lines.append("**📐 Найденные формулы:**")
            for formula in all_formulas[:3]:
                if isinstance(formula, dict):
                    raw = str(formula.get("raw", "") or "")
                    if raw:
                        answer_lines.append(f"`{raw}`")

        answer_lines.append("")
        answer_lines.append("**📚 Источники:**")
        for src in cleaned_chunks[:2]:
            src_name = src.get("doc_name") or src.get("docname") or "Документ"
            score = float(src.get("score", 0) or 0)
            answer_lines.append(f"• {src_name} (релевантность: {score:.2f})")

        return {
            "question": question,
            "answer": "\n".join(answer_lines),
            "sources": cleaned_chunks,
            "tables": [table.to_dict() for table in unique_tables[:5]],
            "formulas": all_formulas[:5],
            "_raw_tables": unique_tables,
        }

    QASystem.answer = enhanced_answer