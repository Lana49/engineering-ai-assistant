# core/table_calculator.py
"""
Модуль для поиска таблиц, извлечения климатических данных и выполнения расчётов.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.table_extractor import ExtractedTable, TableExtractor


@dataclass(slots=True)
class ClimateData:
    """Климатические данные для города."""
    city: str
    t_ot: float | None = None
    z_ot: int | None = None
    t_n: float | None = None
    t_avg: float | None = None
    source: str = ""
    confidence: float = 0.0


class TableCalculator:
    """Поиск таблиц и выполнение расчётов на основе данных из таблиц."""

    _KNOWN_CITIES = {
        "москва", "санкт-петербург", "новосибирск", "екатеринбург",
        "нижний новгород", "казань", "челябинск", "омск", "самара",
        "ростов-на-дону", "уфа", "красноярск", "пермь", "воронеж",
        "волгоград", "краснодар", "сочи", "владивосток", "иркутск",
        "тюмень", "барнаул", "хабаровск", "новокузнецк", "магнитогорск",
        "томск", "кемерово", "астрахань", "архангельск", "мурманск",
        "якутск", "чита", "брянск", "курск", "тверь", "рязань",
        "ярославль", "иваново", "смоленск", "липецк", "орел", "орёл",
        "белгород", "ставрополь", "грозный", "махачкала", "нальчик",
        "владикавказ", "киров", "сургут", "нижневартовск", "тобольск",
        "минск", "алматы", "ташкент", "баку", "ереван", "тбилиси",
        "бишкек", "душанбе",
    }

    def __init__(self, qa_system: Any = None) -> None:
        self.qa_system = qa_system
        self.extractor = TableExtractor()
        self._climate_cache: dict[str, ClimateData] = {}

    @staticmethod
    def _normalize_city_name(text: str) -> str:
        """Нормализует название города."""
        return re.sub(r"\s+", " ", text.lower().strip().replace("ё", "е"))

    @staticmethod
    def _chunk_doc_name(chunk: dict[str, Any], default: str = "") -> str:
        """Извлекает имя документа из чанка."""
        value = chunk.get("doc_name") or chunk.get("docname") or default
        return str(value) if value is not None else default

    def _qa_ready(self) -> bool:
        """Проверяет готовность QA-системы."""
        if not self.qa_system:
            return False

        return bool(
            getattr(self.qa_system, "is_ready", False)
            or getattr(self.qa_system, "isready", False)
        )

    def find_climate_table(self, query: str, city_name: str | None = None) -> ExtractedTable | None:
        """Находит климатическую таблицу по запросу."""
        if not self._qa_ready():
            return None

        search_queries = [
            query,
            f"{query} СП 131.13330 климат",
            f"{query} температура отопительный период",
            "СП 131.13330 таблица климатические параметры",
            "климатические параметры холодного периода года",
        ]

        if city_name:
            search_queries.insert(0, f"{city_name} климат СП 131.13330")

        all_tables: list[ExtractedTable] = []

        for search_query in search_queries:
            try:
                chunks = self.qa_system.search(search_query, top_k=5)
            except (AttributeError, TypeError, ValueError):
                continue

            if not chunks:
                continue

            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue

                text = str(chunk.get("text", "") or "")
                if not text.strip():
                    continue

                doc_name = self._chunk_doc_name(chunk)
                tables = self.extractor.extract(text, doc_name)

                for table in tables:
                    if self._is_climate_table(table):
                        all_tables.append(table)

        seen: set[tuple[str, str, str]] = set()
        unique_tables: list[ExtractedTable] = []

        for table in all_tables:
            key = (
                str(table.source).strip().lower(),
                str(table.title).strip().lower(),
                str(table.raw_text[:200]).strip(),
            )
            if key not in seen:
                seen.add(key)
                unique_tables.append(table)

        unique_tables.sort(key=lambda item: item.confidence, reverse=True)
        return unique_tables[0] if unique_tables else None

    def _is_climate_table(self, table: ExtractedTable) -> bool:
        """Проверяет, является ли таблица климатической."""
        climate_keywords = [
            "климат", "температура", "отопительный период", "холодный период",
            "средняя температура", "продолжительность", "градусо-сутки",
            "наружный воздух", "расчетные параметры", "расчётные параметры",
            "t_от", "z_от", "t_н", "tв", "tн", "t_в",
        ]

        title_lower = table.title.lower()
        if any(keyword in title_lower for keyword in climate_keywords):
            return True

        headers_text = " ".join(table.headers).lower()
        if any(keyword in headers_text for keyword in climate_keywords):
            return True

        for row in table.rows[:10]:
            row_text = " ".join(row).lower()
            if any(keyword in row_text for keyword in climate_keywords):
                return True

        has_city = any(self._is_city_name(cell) for row in table.rows for cell in row)
        has_numbers = any(re.search(r"-?\d+[.,]?\d*", cell) for row in table.rows for cell in row)

        return has_city and has_numbers

    @classmethod
    def _is_city_name(cls, text: str) -> bool:
        """Проверяет, является ли текст названием города."""
        normalized = cls._normalize_city_name(text)
        if len(normalized) < 2:
            return False
        return normalized in {city.replace("ё", "е") for city in cls._KNOWN_CITIES}

    @classmethod
    def _row_contains_city(cls, row: list[str], city_name: str) -> bool:
        """Проверяет, содержит ли строка нужный город."""
        target = cls._normalize_city_name(city_name)

        for cell in row:
            cell_normalized = cls._normalize_city_name(cell)
            if cell_normalized == target:
                return True

        row_joined = " | ".join(cls._normalize_city_name(cell) for cell in row)
        return bool(re.search(rf"(^|\W){re.escape(target)}($|\W)", row_joined))

    def extract_climate_data(self, table: ExtractedTable, city_name: str) -> ClimateData | None:
        """Извлекает климатические данные для указанного города."""
        if not table or not city_name:
            return None

        cache_key = f"{self._normalize_city_name(city_name)}::{hash(table.raw_text[:300])}"
        if cache_key in self._climate_cache:
            return self._climate_cache[cache_key]

        data = ClimateData(city=city_name, source=table.source)

        found_row: list[str] | None = None
        for row in table.rows:
            if self._row_contains_city(row, city_name):
                found_row = row
                break

        if not found_row:
            return None

        headers = [self._normalize_city_name(h) for h in table.headers] if table.headers else []

        if headers and len(headers) == len(found_row):
            self._extract_by_headers(headers, found_row, data)

        if data.t_ot is None or data.z_ot is None:
            self._extract_by_markers(found_row, data)

        if data.t_ot is None or data.z_ot is None:
            self._extract_by_plain_scan(found_row, data)

        confidence = 0.0
        if data.t_ot is not None:
            confidence += 0.4
        if data.z_ot is not None:
            confidence += 0.4
        if data.t_n is not None:
            confidence += 0.2

        data.confidence = confidence
        self._climate_cache[cache_key] = data
        return data

    def _extract_by_headers(self, headers: list[str], row: list[str], data: ClimateData) -> None:
        """Извлекает значения по заголовкам таблицы."""
        for header, cell in zip(headers, row):
            value = self._extract_first_number(cell)
            if value is None:
                continue

            if (
                data.t_ot is None
                and ("t_от" in header or "отоп" in header or "средн" in header)
                and -50 <= value <= 30
            ):
                data.t_ot = float(value)
                continue

            if (
                data.z_ot is None
                and ("z_от" in header or "продолж" in header or "сут" in header or "дней" in header)
                and 1 <= value <= 400
            ):
                data.z_ot = int(round(value))
                continue

            if (
                data.t_n is None
                and ("t_н" in header or "пятиднев" in header or "холодн" in header or "наруж" in header)
                and -70 <= value <= 20
            ):
                data.t_n = float(value)
                continue

            if data.t_avg is None and "среднегод" in header and -50 <= value <= 30:
                data.t_avg = float(value)

    def _extract_by_markers(self, row: list[str], data: ClimateData) -> None:
        """Извлекает значения по маркерам."""
        for cell in row:
            cell_lower = self._normalize_city_name(cell)

            if data.t_ot is None and re.search(r"t_от|средняя.*температура|отопительный период", cell_lower):
                value = self._extract_first_number(cell)
                if value is not None and -50 <= value <= 30:
                    data.t_ot = float(value)
                    continue

            if data.z_ot is None and re.search(r"z_от|продолжительность|сут|дней", cell_lower):
                value = self._extract_first_number(cell)
                if value is not None and 1 <= value <= 400:
                    data.z_ot = int(round(value))
                    continue

            if data.t_n is None and re.search(r"t_н|холодн.*пятидневк|наружн", cell_lower):
                value = self._extract_first_number(cell)
                if value is not None and -70 <= value <= 20:
                    data.t_n = float(value)
                    continue

    def _extract_by_plain_scan(self, row: list[str], data: ClimateData) -> None:
        """Запасное извлечение без заголовков."""
        numbers: list[float] = []

        for cell in row:
            value = self._extract_first_number(cell)
            if value is not None:
                numbers.append(float(value))

        if len(numbers) >= 2:
            for num in numbers:
                if data.t_ot is None and -30 <= num <= 25:
                    data.t_ot = float(num)
                    break

            for num in numbers:
                if data.z_ot is None and 1 <= num <= 400:
                    data.z_ot = int(round(num))
                    break

            for num in numbers:
                if data.t_n is None and -70 <= num <= -5:
                    data.t_n = float(num)
                    break

        if len(numbers) >= 3 and (data.t_ot is None or data.z_ot is None):
            for i, num in enumerate(numbers):
                if i == 0 and data.t_ot is None and -30 <= num <= 25:
                    data.t_ot = float(num)
                elif i == 1 and data.z_ot is None and 1 <= num <= 400:
                    data.z_ot = int(round(num))
                elif i == 2 and data.t_n is None and -70 <= num <= 15:
                    data.t_n = float(num)

    @staticmethod
    def _extract_first_number(text: str) -> float | None:
        """Извлекает первое число из строки."""
        if not text:
            return None

        match = re.search(r"-?\d+(?:[.,]\d+)?", text)
        if not match:
            return None

        try:
            return float(match.group(0).replace(",", "."))
        except ValueError:
            return None

    def calculate_degree_days_from_table(self, city_name: str, t_v: float = 20.0) -> dict[str, Any]:
        """Рассчитывает ГСОП по данным климатической таблицы."""
        result: dict[str, Any] = {
            "city": city_name,
            "t_v": t_v,
            "success": False,
            "data": None,
            "degree_days": None,
            "answer": "",
            "table": None,
            "source": "",
        }

        try:
            table = self.find_climate_table(f"климатические данные {city_name}", city_name)

            if not table:
                result["answer"] = f"❌ Не найдена климатическая таблица для города {city_name}"
                return result

            result["table"] = table
            result["source"] = table.source

            data = self.extract_climate_data(table, city_name)
            if not data or data.confidence < 0.3:
                result["answer"] = (
                    f"⚠️ Не удалось извлечь климатические данные для {city_name}"
                    if not data
                    else f"⚠️ Не удалось надёжно извлечь климатические данные для {city_name}. Уверенность: {data.confidence:.0%}"
                )
                return result

            result["data"] = data

            if data.t_ot is None or data.z_ot is None:
                result["answer"] = (
                    f"⚠️ Недостаточно данных для расчёта ГСОП для {city_name}:\n"
                    f"- t_от: {data.t_ot if data.t_ot is not None else 'не найдено'}\n"
                    f"- z_от: {data.z_ot if data.z_ot is not None else 'не найдено'}"
                )
                return result

            degree_days = (t_v - data.t_ot) * data.z_ot

            result["degree_days"] = degree_days
            result["success"] = True

            answer_lines = [
                f"🌍 **ГСОП для {city_name} = {degree_days:.0f} °C·сут**",
                "",
                "📊 **Исходные данные из таблицы:**",
                f"- Город: {city_name}",
                f"- t_в = {t_v} °C",
                f"- t_от = {data.t_ot:.1f} °C",
                f"- z_от = {data.z_ot} сут",
                "",
                "📐 **Формула:** ГСОП = (t_в - t_от) × z_от",
                f"🔢 **Подстановка:** ({t_v} - ({data.t_ot:.1f})) × {data.z_ot} = {degree_days:.0f}",
                "",
                f"📚 **Источник:** {data.source}",
                f"✅ Уверенность извлечения данных: {data.confidence:.0%}",
            ]

            if data.t_n is not None:
                answer_lines.extend([
                    "",
                    f"📌 Также найдено: t_н = {data.t_n:.1f} °C (температура холодной пятидневки)"
                ])

            result["answer"] = "\n".join(answer_lines)

        except (ValueError, TypeError, AttributeError) as exc:
            result["answer"] = f"❌ Ошибка при расчёте: {exc}"

        return result

    def calculate_gsop_from_table(self, city_name: str, t_v: float = 20.0) -> dict[str, Any]:
        """Алиас для calculate_degree_days_from_table."""
        return self.calculate_degree_days_from_table(city_name, t_v)

    def calculate_ventilation_from_table(
        self,
        city_name: str,
        air_flow: float,
        t_v: float = 20.0,
    ) -> dict[str, Any]:
        """Рассчитывает расход теплоты на вентиляцию по данным таблицы."""
        result: dict[str, Any] = {
            "city": city_name,
            "air_flow": air_flow,
            "t_v": t_v,
            "success": False,
            "degree_days_result": None,
            "ventilation": None,
            "answer": "",
            "source": "",
            "table": None,
        }

        degree_days_result = self.calculate_degree_days_from_table(city_name, t_v)

        if not degree_days_result["success"]:
            result["answer"] = str(degree_days_result["answer"])
            result["source"] = str(degree_days_result.get("source", ""))
            result["table"] = degree_days_result.get("table")
            return result

        data = degree_days_result["data"]
        if not isinstance(data, ClimateData):
            result["answer"] = f"⚠️ Не удалось получить климатические данные для {city_name}"
            return result

        if data.t_n is None:
            result["answer"] = (
                f"⚠️ Не найдена температура наружного воздуха t_н для {city_name}. "
                "Она нужна для расчёта вентиляции."
            )
            result["source"] = str(degree_days_result.get("source", ""))
            result["table"] = degree_days_result.get("table")
            return result

        q_vent = 0.335 * air_flow * (t_v - data.t_n)

        result["degree_days_result"] = degree_days_result
        result["ventilation"] = q_vent
        result["success"] = True
        result["source"] = str(degree_days_result.get("source", ""))
        result["table"] = degree_days_result.get("table")

        answer_lines = [
            f"💨 **Расход теплоты на вентиляцию для {city_name} = {q_vent:.0f} Вт**",
            "",
            "📊 **Исходные данные из таблицы:**",
            f"- Город: {city_name}",
            f"- L = {air_flow} м³/ч",
            f"- t_в = {t_v} °C",
            f"- t_н = {data.t_n:.1f} °C",
            "",
            "📐 **Формула:** Q_в = 0.335 × L × (t_в - t_н)",
            f"🔢 **Подстановка:** 0.335 × {air_flow} × ({t_v} - ({data.t_n:.1f})) = {q_vent:.0f}",
            "",
            f"📚 **Источник:** {data.source}",
            "",
            "🌍 **ГСОП для справки:**",
            f"- ГСОП = {degree_days_result['degree_days']:.0f} °C·сут",
        ]

        result["answer"] = "\n".join(answer_lines)
        return result

    def calculate_heat_loss_from_table(
        self,
        city_name: str,
        area: float,
        resistance: float,
        t_v: float = 20.0,
    ) -> dict[str, Any]:
        """Рассчитывает теплопотери через ограждение по данным таблицы."""
        result: dict[str, Any] = {
            "city": city_name,
            "area": area,
            "resistance": resistance,
            "t_v": t_v,
            "success": False,
            "data": None,
            "heat_loss": None,
            "answer": "",
            "source": "",
            "table": None,
        }

        try:
            table = self.find_climate_table(f"климатические данные {city_name}", city_name)

            if not table:
                result["answer"] = f"❌ Не найдена климатическая таблица для города {city_name}"
                return result

            data = self.extract_climate_data(table, city_name)

            if not data or data.t_n is None:
                result["answer"] = f"⚠️ Не найдена температура наружного воздуха t_н для {city_name}"
                result["source"] = table.source
                result["table"] = table
                return result

            delta_t = t_v - data.t_n
            q_loss = (area * delta_t) / resistance

            result["data"] = data
            result["heat_loss"] = q_loss
            result["success"] = True
            result["source"] = table.source
            result["table"] = table

            answer_lines = [
                f"🔥 **Теплопотери через ограждение для {city_name} = {q_loss:.0f} Вт**",
                "",
                "📊 **Исходные данные:**",
                f"- Город: {city_name}",
                f"- A = {area} м²",
                f"- R = {resistance} м²·°C/Вт",
                f"- t_в = {t_v} °C",
                f"- t_н = {data.t_n:.1f} °C (из таблицы)",
                f"- Δt = {delta_t:.1f} °C",
                "",
                "📐 **Формула:** Q = (A × Δt) / R",
                f"🔢 **Подстановка:** ({area} × {delta_t:.1f}) / {resistance} = {q_loss:.0f}",
                "",
                f"📚 **Источник:** {data.source}",
            ]

            result["answer"] = "\n".join(answer_lines)

        except (ValueError, TypeError, AttributeError, ZeroDivisionError) as exc:
            result["answer"] = f"❌ Ошибка при расчёте: {exc}"

        return result

    def get_cities_from_table(self, query: str = "климатические параметры городов") -> list[str]:
        """Извлекает список городов из найденной климатической таблицы."""
        table = self.find_climate_table(query)
        if not table:
            return []

        cities: list[str] = []
        for row in table.rows:
            for cell in row:
                if self._is_city_name(cell):
                    cities.append(cell.strip())

        return sorted(set(cities))


def patch_app_with_table_calculator() -> None:
    """Заглушка для совместимости."""
    print("✅ TableCalculator готов к использованию")