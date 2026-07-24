# core/table_extractor.py
"""
Модуль для интеллектуального извлечения таблиц из текста.
Распознаёт таблицы по структурным признакам, а не по маркерам.
"""

import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

# Импортируем pandas только когда он нужен
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


@dataclass
class ExtractedTable:
    """Класс для хранения извлечённой таблицы"""
    title: str
    headers: List[str]
    rows: List[List[str]]
    raw_text: str
    source: str
    confidence: float = 0.0

    def to_markdown(self) -> str:
        """Преобразует таблицу в Markdown"""
        if not self.headers or not self.rows:
            return f"**{self.title}**\n\n(Таблица пуста или не распознана)"

        lines = [
            f"**{self.title}**",
            "",
            "| " + " | ".join(self.headers) + " |",
            "| " + " | ".join(["---"] * len(self.headers)) + " |"
        ]

        for row in self.rows[:20]:
            padded = row + [""] * (len(self.headers) - len(row))
            lines.append("| " + " | ".join(str(cell).strip()[:50] for cell in padded) + " |")

        if len(self.rows) > 20:
            lines.append(f"*... и ещё {len(self.rows) - 20} строк*")

        return "\n".join(lines)

    def to_dataframe(self):
        """Преобразует в pandas DataFrame (если pandas доступен)"""
        if not PANDAS_AVAILABLE:
            return None

        if not self.headers:
            return pd.DataFrame(self.rows)

        rows_padded = []
        for row in self.rows:
            padded = row + [""] * (len(self.headers) - len(row))
            rows_padded.append(padded)

        return pd.DataFrame(rows_padded, columns=self.headers)

    def to_dict(self) -> Dict:
        """Преобразует в словарь для JSON-сериализации"""
        return {
            "title": self.title,
            "headers": self.headers,
            "rows": self.rows[:50],
            "row_count": len(self.rows),
            "source": self.source,
            "confidence": self.confidence
        }


class TableExtractor:
    """Извлекает таблицы из текста по структурным признакам"""

    def __init__(self):
        self._cache: Dict[str, List[ExtractedTable]] = {}

    def extract(self, text: str, source: str = "", min_rows: int = 2) -> List[ExtractedTable]:
        """
        Извлекает таблицы из текста

        Args:
            text: Текст для анализа
            source: Источник (имя документа)
            min_rows: Минимальное количество строк для таблицы

        Returns:
            Список извлечённых таблиц
        """
        cache_key = f"{source}_{hash(text[:500])}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        tables = []

        # 1. Поиск таблиц с разделителями (|)
        pipe_tables = self._extract_pipe_tables(text, source)
        tables.extend(pipe_tables)

        # 2. Поиск таблиц по маркеру [ТАБЛИЦА]
        marker_tables = self._extract_marker_tables(text, source)
        tables.extend(marker_tables)

        # 3. Поиск таблиц по выравниванию (колонки)
        align_tables = self._extract_aligned_tables(text, source)
        tables.extend(align_tables)

        # 4. Поиск таблиц по ключевым словам
        keyword_tables = self._extract_keyword_tables(text, source)
        tables.extend(keyword_tables)

        # Удаляем дубликаты и сортируем по уверенности
        unique_tables = self._deduplicate_tables(tables)
        unique_tables.sort(key=lambda t: t.confidence, reverse=True)

        # Фильтруем по минимальному количеству строк
        filtered = [t for t in unique_tables if len(t.rows) >= min_rows]

        self._cache[cache_key] = filtered
        return filtered

    def _extract_pipe_tables(self, text: str, source: str) -> List[ExtractedTable]:
        """Извлекает таблицы с разделителями |"""
        tables = []
        lines = text.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            if '|' in line and len(line.split('|')) >= 3:
                title = self._find_title(lines, i)
                table_lines = []
                j = i

                while j < len(lines) and '|' in lines[j]:
                    table_lines.append(lines[j].strip())
                    j += 1

                if len(table_lines) >= 2:
                    headers, rows = self._parse_pipe_table(table_lines)
                    if headers and len(rows) >= 1:
                        tables.append(ExtractedTable(
                            title=title,
                            headers=headers,
                            rows=rows,
                            raw_text='\n'.join(table_lines),
                            source=source,
                            confidence=0.9
                        ))
                i = j
            else:
                i += 1

        return tables

    def _extract_marker_tables(self, text: str, source: str) -> List[ExtractedTable]:
        """Извлекает таблицы по маркеру [ТАБЛИЦА]"""
        tables = []
        lines = text.split('\n')
        in_table = False
        table_lines = []
        title = "Таблица"

        for i, line in enumerate(lines):
            if '[ТАБЛИЦА]' in line:
                in_table = True
                table_lines = []
                if i > 0 and len(lines[i - 1].strip()) < 100:
                    title = lines[i - 1].strip()
                continue

            if in_table:
                if line.strip() == '':
                    if table_lines:
                        headers, rows = self._parse_plain_table(table_lines)
                        if rows:
                            tables.append(ExtractedTable(
                                title=title,
                                headers=headers,
                                rows=rows,
                                raw_text='\n'.join(table_lines),
                                source=source,
                                confidence=0.85
                            ))
                    in_table = False
                    title = "Таблица"
                else:
                    table_lines.append(line.strip())

        if in_table and table_lines:
            headers, rows = self._parse_plain_table(table_lines)
            if rows:
                tables.append(ExtractedTable(
                    title=title,
                    headers=headers,
                    rows=rows,
                    raw_text='\n'.join(table_lines),
                    source=source,
                    confidence=0.8
                ))

        return tables

    def _extract_aligned_tables(self, text: str, source: str) -> List[ExtractedTable]:
        """Извлекает таблицы по выравниванию текста в колонки"""
        tables = []
        lines = text.split('\n')

        for i, line in enumerate(lines):
            if not line.strip():
                continue

            parts = re.split(r'\s{3,}', line.strip())
            if len(parts) >= 3:
                title = self._find_title(lines, i)
                table_lines = [line.strip()]
                j = i + 1

                while j < len(lines):
                    next_line = lines[j].strip()
                    next_parts = re.split(r'\s{3,}', next_line)
                    if next_line and len(next_parts) >= 3:
                        table_lines.append(next_line)
                        j += 1
                    else:
                        break

                if len(table_lines) >= 2:
                    rows = [re.split(r'\s{3,}', line) for line in table_lines]
                    rows = [r for r in rows if r and any(c.strip() for c in r)]

                    if len(rows) >= 2:
                        headers = rows[0]
                        data_rows = rows[1:]
                        tables.append(ExtractedTable(
                            title=title,
                            headers=headers,
                            rows=data_rows,
                            raw_text='\n'.join(table_lines),
                            source=source,
                            confidence=0.6
                        ))

        return tables

    def _extract_keyword_tables(self, text: str, source: str) -> List[ExtractedTable]:
        """Извлекает таблицы по ключевым словам"""
        tables = []
        keywords = [
            'таблица', 'табл', 'table',
            'показатели', 'значения', 'параметры',
            'данные', 'список', 'перечень',
            'температура', 'давление', 'расход',
            'город', 'регион', 'климат'
        ]

        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            has_keyword = any(kw in line_lower for kw in keywords)
            has_numbers = bool(re.search(r'\d+[.,]?\d*', line))

            if has_keyword and has_numbers and len(line) > 30:
                table_lines = [line.strip()]
                j = i + 1
                consecutive = 0

                while j < len(lines) and consecutive < 10:
                    next_line = lines[j].strip()
                    if next_line and bool(re.search(r'\d+[.,]?\d*', next_line)):
                        table_lines.append(next_line)
                        consecutive += 1
                    elif next_line and len(next_line) < 80:
                        table_lines[-1] += " " + next_line
                    else:
                        break
                    j += 1

                if len(table_lines) >= 3:
                    headers, rows = self._parse_plain_table(table_lines)
                    if rows:
                        title = lines[i - 1].strip() if i > 0 and len(lines[i - 1].strip()) < 80 else "Таблица"
                        tables.append(ExtractedTable(
                            title=title,
                            headers=headers,
                            rows=rows,
                            raw_text='\n'.join(table_lines),
                            source=source,
                            confidence=0.5
                        ))

        return tables

    @staticmethod
    def _find_title(lines: List[str], index: int) -> str:
        """Находит заголовок для таблицы"""
        for j in range(max(0, index - 3), index):
            line = lines[j].strip()
            if line and len(line) < 100 and not line.startswith('|'):
                if '|' not in line:
                    return line
        return "Таблица"

    @staticmethod
    def _parse_pipe_table(lines: List[str]) -> Tuple[List[str], List[List[str]]]:
        """Парсит таблицу с разделителями |"""
        if not lines:
            return [], []

        cleaned_lines = [l for l in lines if l.strip()]
        if len(cleaned_lines) < 2:
            return [], []

        # Проверяем на разделительную строку (|---|)
        if re.match(r'^\s*\|?\s*:?-+:?\s*\|', cleaned_lines[1]):
            header_line = cleaned_lines[0]
            data_lines = cleaned_lines[2:]
        else:
            header_line = cleaned_lines[0]
            data_lines = cleaned_lines[1:]

        headers = TableExtractor._split_pipe_line(header_line)
        rows = []

        for line in data_lines:
            row = TableExtractor._split_pipe_line(line)
            if row:
                rows.append(row)

        return headers, rows

    @staticmethod
    def _split_pipe_line(line: str) -> List[str]:
        """Разбивает строку с разделителями |"""
        if not line:
            return []

        cleaned = line.strip()
        if cleaned.startswith('|'):
            cleaned = cleaned[1:]
        if cleaned.endswith('|'):
            cleaned = cleaned[:-1]

        return [cell.strip() for cell in cleaned.split('|')]

    @staticmethod
    def _parse_plain_table(lines: List[str]) -> Tuple[List[str], List[List[str]]]:
        """Парсит обычную таблицу (без разделителей)"""
        if not lines:
            return [], []

        cleaned_lines = [l for l in lines if l.strip()]
        if len(cleaned_lines) < 2:
            return [], []

        first_line = cleaned_lines[0]
        second_line = cleaned_lines[1] if len(cleaned_lines) > 1 else ""

        # Если вторая строка содержит только числа и знаки — это данные
        if re.match(r'^[\d\s.,-]+$', second_line):
            headers = TableExtractor._split_by_spaces(first_line)
            data_lines = cleaned_lines[1:]
        else:
            headers = []
            data_lines = cleaned_lines

        rows = []
        for line in data_lines:
            if line.strip():
                row = TableExtractor._split_by_spaces(line)
                if row:
                    rows.append(row)

        return headers, rows

    @staticmethod
    def _split_by_spaces(line: str) -> List[str]:
        """Разбивает строку по пробелам, сохраняя числа с точкой"""
        parts = re.split(r'\s{2,}', line.strip())
        if len(parts) >= 2:
            return parts
        return line.strip().split()

    @staticmethod
    def _deduplicate_tables(tables: List[ExtractedTable]) -> List[ExtractedTable]:
        """Удаляет дубликаты таблиц"""
        seen = set()
        unique = []

        for table in tables:
            key = table.raw_text[:100]
            if key not in seen:
                seen.add(key)
                unique.append(table)

        return unique

    def extract_from_chunks(self, chunks: List[Dict], source: str = "") -> List[ExtractedTable]:
        """Извлекает таблицы из списка фрагментов"""
        all_tables = []

        for chunk in chunks:
            text = chunk.get('text', '')
            doc_name = chunk.get('doc_name', source)
            tables = self.extract(text, doc_name)
            all_tables.extend(tables)

        return self._deduplicate_tables(all_tables)

    def search_table(self, query: str, chunks: List[Dict]) -> Optional[ExtractedTable]:
        """Ищет таблицу по запросу"""
        all_tables = self.extract_from_chunks(chunks)

        if not all_tables:
            return None

        query_lower = query.lower()
        scored = []

        for table in all_tables:
            score = 0.0

            if query_lower in table.title.lower():
                score += 1.0

            for row in table.rows:
                row_text = ' '.join(row).lower()
                if query_lower in row_text:
                    score += 0.5

            for header in table.headers:
                if query_lower in header.lower():
                    score += 0.3

            scored.append((table, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        if scored and scored[0][1] > 0:
            return scored[0][0]

        return None

    @staticmethod
    def format_table_for_response(table: ExtractedTable) -> str:
        """Форматирует таблицу для ответа пользователю"""
        if not table:
            return "Таблица не найдена"

        lines = [
            f"**📊 {table.title}**",
            f"*Источник: {table.source}*",
            "",
            table.to_markdown(),
            "",
            f"*Всего строк: {len(table.rows)}*"
        ]

        return "\n".join(lines)


# ========== ИНТЕГРАЦИЯ В QASystem ==========

def patch_qa_system_with_table_extractor():
    """
    Добавляет улучшенное извлечение таблиц в QASystem
    """
    try:
        from core.qa_engine import QASystem

        def enhanced_answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
            """Улучшенный ответ с распознаванием таблиц"""
            from core.table_extractor import TableExtractor

            # Стандартный поиск
            result = self.search_with_formulas(question, top_k)
            relevant = result['results']

            if not relevant:
                return {
                    'question': question,
                    'answer': "❌ Информация по вашему вопросу не найдена в документации.",
                    'sources': [],
                    'tables': [],
                    'formulas': []
                }

            # Улучшенное извлечение таблиц
            extractor = TableExtractor()
            all_tables = []

            for chunk in relevant:
                tables = extractor.extract(chunk['text'], chunk.get('doc_name', ''))
                all_tables.extend(tables)

            # Дедупликация
            seen = set()
            unique_tables = []

            for t in all_tables:
                key = t.raw_text[:100]
                if key not in seen:
                    seen.add(key)
                    unique_tables.append(t)

            # Формируем ответ
            cleaned_chunks = [c for c in relevant if not self._is_bad_chunk(c['text'])]
            if not cleaned_chunks:
                cleaned_chunks = relevant[:2]

            answer_lines = ["**📌 Краткий ответ:**"]

            # Первый фрагмент
            first_text = cleaned_chunks[0]['text']
            first_sentence = first_text.split('.')[0] + "."
            answer_lines.append(first_sentence)
            answer_lines.append("")

            # Таблицы
            if unique_tables:
                answer_lines.append("**📊 Найденные таблицы:**")
                answer_lines.append("")

                for i, table in enumerate(unique_tables[:2], 1):
                    answer_lines.append(f"**Таблица {i}: {table.title}**")
                    answer_lines.append(f"*Источник: {table.source}*")
                    answer_lines.append("")

                    if table.headers:
                        answer_lines.append("| " + " | ".join(table.headers[:6]) + " |")
                        answer_lines.append("| " + " | ".join(["---"] * len(table.headers[:6])) + " |")

                        for row in table.rows[:5]:
                            padded = row + [""] * (len(table.headers[:6]) - len(row))
                            answer_lines.append("| " + " | ".join(str(cell).strip()[:30] for cell in padded[:6]) + " |")
                    else:
                        for row in table.rows[:5]:
                            answer_lines.append("- " + " | ".join(row))

                    if len(table.rows) > 5:
                        answer_lines.append(f"*... и ещё {len(table.rows) - 5} строк*")

                    answer_lines.append("")

            # Формулы
            all_formulas = result.get('formulas', [])
            if all_formulas:
                answer_lines.append("**📐 Найденные формулы:**")
                for formula in all_formulas[:3]:
                    raw = formula.get('raw', '')
                    answer_lines.append(f"\n`{raw}`")

            # Источники
            answer_lines.append("")
            answer_lines.append("**📚 Источники:**")
            for src in cleaned_chunks[:2]:
                answer_lines.append(f"• {src.get('doc_name', 'Документ')} (релевантность: {src.get('score', 0):.2f})")

            return {
                'question': question,
                'answer': '\n'.join(answer_lines),
                'sources': cleaned_chunks,
                'tables': [t.to_dict() for t in unique_tables[:5]],
                'formulas': all_formulas[:5],
                '_raw_tables': unique_tables
            }

        # Подменяем метод
        QASystem.answer = enhanced_answer
        print("✅ QASystem расширен улучшенным извлечением таблиц")

    except ImportError as e:
        print(f"⚠️ Не удалось применить патч: {e}")
    except Exception as e:
        print(f"⚠️ Ошибка при патче: {e}")