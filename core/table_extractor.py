# core/table_extractor.py
"""
Извлечение таблиц из текстов документов.
Единый источник для всех операций с таблицами.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedTable:
    """Структура извлечённой таблицы."""
    title: str = ""
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    raw_text: str = ""
    source: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь для JSON-сериализации."""
        return {
            "title": self.title,
            "headers": self.headers,
            "rows": self.rows,
            "content": self.raw_text,
            "raw_text": self.raw_text,
            "doc_name": self.source,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "row_count": len(self.rows),
        }

    def is_valid(self, min_rows: int = 2) -> bool:
        """Проверяет, что таблица содержит достаточно данных."""
        return len(self.rows) >= min_rows and bool(self.raw_text.strip())


class TableExtractor:
    """
    Извлекает таблицы из текста различными методами.
    """

    def __init__(self):
        self._seen: set[tuple[str, str]] = set()

    def extract(
        self,
        text: str,
        source: str = "",
        min_rows: int = 2,
        max_tables: int = 10,
    ) -> list[ExtractedTable]:
        """Извлекает таблицы из текста всеми доступными методами."""
        if not text or not text.strip():
            return []

        all_tables: list[ExtractedTable] = []

        # 1. Таблицы с маркером [ТАБЛИЦА ...]
        marker_tables = self._extract_marker_tables(text, source)
        all_tables.extend(marker_tables)

        # 2. Таблицы в Markdown-формате (с |)
        pipe_tables = self._extract_pipe_tables(text, source)
        all_tables.extend(pipe_tables)

        # 3. Таблицы по выравниванию (пробелы)
        aligned_tables = self._extract_aligned_tables(text, source)
        all_tables.extend(aligned_tables)

        # 4. Таблицы по ключевым словам
        keyword_tables = self._extract_keyword_tables(text, source)
        all_tables.extend(keyword_tables)

        # Дедупликация и фильтрация
        unique_tables = self._deduplicate_tables(all_tables)
        valid_tables = [t for t in unique_tables if t.is_valid(min_rows)]
        valid_tables.sort(key=lambda t: t.confidence, reverse=True)

        return valid_tables[:max_tables]

    def _extract_marker_tables(self, text: str, source: str) -> list[ExtractedTable]:
        """Извлекает таблицы по маркеру [ТАБЛИЦА ...]."""
        tables = []
        lines = text.split("\n")
        in_table = False
        table_lines = []
        table_title = ""

        for i, line in enumerate(lines):
            if "[ТАБЛИЦА" in line:
                in_table = True
                table_lines = []

                if i > 0 and len(lines[i - 1].strip()) < 100:
                    table_title = lines[i - 1].strip()
                else:
                    table_title = f"Таблица {len(tables) + 1}"
                continue

            if in_table:
                if line.strip() == "":
                    if table_lines:
                        table = self._build_table_from_lines(
                            table_lines, table_title, source, confidence=0.95
                        )
                        if table.is_valid():
                            tables.append(table)
                        table_lines = []
                        table_title = ""
                    in_table = False
                else:
                    table_lines.append(line.strip())

        if in_table and table_lines:
            table = self._build_table_from_lines(
                table_lines, table_title, source, confidence=0.95
            )
            if table.is_valid():
                tables.append(table)

        return tables

    def _extract_pipe_tables(self, text: str, source: str) -> list[ExtractedTable]:
        """Извлекает Markdown-таблицы с разделителями |."""
        tables = []
        lines = text.split("\n")
        table_lines = []
        in_table = False

        for line in lines:
            if "|" in line:
                stripped = line.strip()
                if stripped.count("|") >= 2:
                    if not in_table:
                        in_table = True
                        table_lines = []
                    table_lines.append(stripped)
                else:
                    if in_table and table_lines:
                        table = self._build_table_from_pipe_lines(
                            table_lines, source, confidence=0.85
                        )
                        if table.is_valid():
                            tables.append(table)
                        table_lines = []
                    in_table = False
            else:
                if in_table and table_lines:
                    table = self._build_table_from_pipe_lines(
                        table_lines, source, confidence=0.85
                    )
                    if table.is_valid():
                        tables.append(table)
                    table_lines = []
                in_table = False

        if in_table and table_lines:
            table = self._build_table_from_pipe_lines(
                table_lines, source, confidence=0.85
            )
            if table.is_valid():
                tables.append(table)

        return tables

    def _extract_aligned_tables(self, text: str, source: str) -> list[ExtractedTable]:
        """Извлекает таблицы по выравниванию текста."""
        tables = []
        lines = text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            if re.search(r"\d+\s*[мМ]?[м²³]?[°С]?", line):
                parts = re.split(r"\s{2,}", line)
                if len(parts) >= 3:
                    table_lines = [line]
                    j = i + 1

                    while j < len(lines):
                        next_line = lines[j].strip()
                        if not next_line:
                            break
                        next_parts = re.split(r"\s{2,}", next_line)
                        if len(next_parts) >= 3 and any(
                            re.search(r"\d", p) for p in next_parts
                        ):
                            table_lines.append(next_line)
                            j += 1
                        else:
                            break

                    if len(table_lines) >= 3:
                        table = self._build_table_from_aligned_lines(
                            table_lines, source, confidence=0.7
                        )
                        if table.is_valid():
                            tables.append(table)
                        i = j
                        continue
            i += 1

        return tables

    def _extract_keyword_tables(self, text: str, source: str) -> list[ExtractedTable]:
        """Извлекает таблицы по ключевым словам."""
        tables = []
        keywords = [
            "таблица", "табл", "таблиц", "показатели", "значения",
            "параметры", "характеристики", "нормативы", "требования",
        ]

        lines = text.split("\n")
        i = 0

        while i < len(lines):
            line_lower = lines[i].strip().lower()
            if any(kw in line_lower for kw in keywords) and ":" in lines[i]:
                title = lines[i].strip()
                table_lines = []
                j = i + 1

                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line:
                        break
                    if any(kw in next_line.lower() for kw in keywords):
                        break
                    if re.search(r"\d+\s*[мМ]?[м²³]?[°С]?", next_line):
                        table_lines.append(next_line)
                    elif len(next_line) > 50 and j < i + 5:
                        table_lines.append(next_line)
                    else:
                        break
                    j += 1

                if len(table_lines) >= 2:
                    table = self._build_table_from_keyword_lines(
                        table_lines, title, source, confidence=0.6
                    )
                    if table.is_valid():
                        tables.append(table)
                    i = j
                    continue
            i += 1

        return tables

    @staticmethod
    def _build_table_from_lines(
        lines: list[str],
        title: str,
        source: str,
        confidence: float = 0.8,
    ) -> ExtractedTable:
        """Строит таблицу из строк."""
        rows = []
        for line in lines:
            line = line.strip()
            if line:
                if "\t" in line:
                    parts = line.split("\t")
                else:
                    parts = re.split(r"\s{2,}", line)
                if len(parts) > 1:
                    rows.append([p.strip() for p in parts])
                else:
                    rows.append([line])

        headers = rows[0] if rows and len(rows) > 0 else []
        data_rows = rows[1:] if len(rows) > 1 else []

        return ExtractedTable(
            title=title,
            headers=headers,
            rows=data_rows,
            raw_text="\n".join(lines),
            source=source,
            confidence=confidence,
        )

    @staticmethod
    def _build_table_from_pipe_lines(
        lines: list[str],
        source: str,
        confidence: float = 0.85,
    ) -> ExtractedTable:
        """Строит таблицу из строк с разделителями |."""
        rows = []
        title = f"Таблица из {source}" if source else "Таблица"

        for line in lines:
            line = line.strip()
            if line:
                if line.startswith("|"):
                    line = line[1:]
                if line.endswith("|"):
                    line = line[:-1]
                parts = [p.strip() for p in line.split("|")]
                if any(parts):
                    rows.append(parts)

        filtered_rows = []
        for row in rows:
            if not all(p.replace("-", "").strip() == "" for p in row):
                filtered_rows.append(row)

        headers = filtered_rows[0] if filtered_rows else []
        data_rows = filtered_rows[1:] if len(filtered_rows) > 1 else []

        return ExtractedTable(
            title=title,
            headers=headers,
            rows=data_rows,
            raw_text="\n".join(lines),
            source=source,
            confidence=confidence,
        )

    @staticmethod
    def _build_table_from_aligned_lines(
        lines: list[str],
        source: str,
        confidence: float = 0.7,
    ) -> ExtractedTable:
        """Строит таблицу из выровненных строк."""
        rows = []
        title = f"Таблица из {source}" if source else "Таблица"

        for line in lines:
            line = line.strip()
            if line:
                parts = re.split(r"\s{2,}", line)
                if len(parts) > 1:
                    rows.append(parts)
                else:
                    rows.append([line])

        headers = rows[0] if rows and len(rows) > 0 else []
        data_rows = rows[1:] if len(rows) > 1 else []

        return ExtractedTable(
            title=title,
            headers=headers,
            rows=data_rows,
            raw_text="\n".join(lines),
            source=source,
            confidence=confidence,
        )

    @staticmethod
    def _build_table_from_keyword_lines(
        lines: list[str],
        title: str,
        source: str,
        confidence: float = 0.6,
    ) -> ExtractedTable:
        """Строит таблицу из строк по ключевым словам."""
        rows = []
        for line in lines:
            line = line.strip()
            if line:
                if "\t" in line:
                    parts = line.split("\t")
                else:
                    parts = re.split(r"\s{2,}", line)
                if len(parts) > 1:
                    rows.append([p.strip() for p in parts])
                else:
                    rows.append([line])

        headers = rows[0] if rows and len(rows) > 0 else []
        data_rows = rows[1:] if len(rows) > 1 else []

        return ExtractedTable(
            title=title,
            headers=headers,
            rows=data_rows,
            raw_text="\n".join(lines),
            source=source,
            confidence=confidence,
        )

    @staticmethod
    def _deduplicate_tables(
        tables: list[ExtractedTable],
    ) -> list[ExtractedTable]:
        """Удаляет дубликаты таблиц."""
        unique = []
        seen: set[tuple[str, str]] = set()

        for table in tables:
            key = (table.source, table.raw_text[:200])
            if key not in seen:
                seen.add(key)
                unique.append(table)

        return unique


# ========= УДОБНЫЕ ФУНКЦИИ ДЛЯ ИМПОРТА =========

def extract_tables(text: str, doc_name: str = "", min_rows: int = 2) -> list[ExtractedTable]:
    """Быстрый импорт: извлекает таблицы из текста."""
    extractor = TableExtractor()
    return extractor.extract(text, source=doc_name, min_rows=min_rows)


def extract_tables_from_results(results: list[Any], min_rows: int = 2, limit: int = 3) -> list[ExtractedTable]:
    """Извлекает таблицы из результатов поиска."""
    extractor = TableExtractor()
    all_tables: list[ExtractedTable] = []
    seen: set[tuple[str, str]] = set()

    for result in results[:limit]:
        if hasattr(result, "text"):
            text = getattr(result, "text", "")
            doc_name = getattr(result, "doc_name", "Документ")
        elif isinstance(result, dict):
            text = result.get("text", "")
            doc_name = result.get("doc_name", "Документ")
        else:
            continue

        if not text.strip():
            continue

        tables = extractor.extract(text, source=doc_name, min_rows=min_rows)
        for table in tables:
            key = (table.source, table.raw_text[:200])
            if key not in seen:
                seen.add(key)
                all_tables.append(table)

    return all_tables


def find_city_in_tables(tables: list[ExtractedTable], city: str) -> dict[str, Any] | None:
    """Ищет город в таблицах и возвращает информацию о найденной строке."""
    city_lower = city.lower().strip()

    for table in tables:
        for idx, row in enumerate(table.rows):
            if isinstance(row, list):
                for cell in row:
                    if city_lower in str(cell).lower():
                        return {
                            "table": table,
                            "row": row,
                            "index": idx,
                            "row_text": " ".join(str(c) for c in row)
                        }
            elif isinstance(row, str):
                if city_lower in row.lower():
                    return {
                        "table": table,
                        "row": row,
                        "index": idx,
                        "row_text": row
                    }

    return None


def tables_to_dicts(tables: list[ExtractedTable]) -> list[dict[str, Any]]:
    """Преобразует список ExtractedTable в список словарей."""
    return [t.to_dict() for t in tables]