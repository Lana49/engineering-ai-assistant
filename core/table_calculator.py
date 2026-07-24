# core/table_calculator.py
"""
Модуль для поиска таблиц, извлечения климатических данных и выполнения расчётов.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.table_extractor import TableExtractor, ExtractedTable


@dataclass
class ClimateData:
    """Климатические данные для города."""
    city: str
    t_ot: Optional[float] = None
    z_ot: Optional[int] = None
    t_n: Optional[float] = None
    t_avg: Optional[float] = None
    source: str = ""
    confidence: float = 0.0


class TableCalculator:
    """Поиск таблиц и выполнение расчётов на основе данных из таблиц."""

    def __init__(self, qa_system=None):
        self.qa_system = qa_system
        self.extractor = TableExtractor()
        self._climate_cache: Dict[str, ClimateData] = {}

    @staticmethod
    def _chunk_doc_name(chunk: Dict[str, Any], default: str = "") -> str:
        """Извлекает имя документа из чанка."""
        return chunk.get("doc_name") or chunk.get("docname", default)

    def _qa_ready(self) -> bool:
        """Проверяет готовность QA-системы."""
        if not self.qa_system:
            return False
        return bool(
            getattr(self.qa_system, "is_ready", False)
            or getattr(self.qa_system, "isready", False)
        )

    def find_climate_table(self, query: str, city_name: Optional[str] = None) -> Optional[ExtractedTable]:
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

        all_tables: List[ExtractedTable] = []

        for search_query in search_queries:
            try:
                chunks = self.qa_system.search(search_query, top_k=5)
            except Exception:
                continue

            for chunk in chunks:
                text = chunk.get("text", "")
                doc_name = self._chunk_doc_name(chunk)
                tables = self.extractor.extract(text, doc_name)

                for table in tables:
                    if self._is_climate_table(table):
                        all_tables.append(table)

        # Удаляем дубликаты
        seen = set()
        unique_tables: List[ExtractedTable] = []

        for table in all_tables:
            key = table.raw_text[:200]
            if key not in seen:
                seen.add(key)
                unique_tables.append(table)

        unique_tables.sort(key=lambda t: t.confidence, reverse=True)
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

    @staticmethod
    def _is_city_name(text: str) -> bool:
        """Проверяет, является ли текст названием города."""
        if not text or len(text.strip()) < 2:
            return False

        cities = {
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

        normalized = text.lower().strip().replace("ё", "е")
        cities_norm = {city.replace("ё", "е") for city in cities}

        return normalized in cities_norm or any(city in normalized for city in cities_norm)

    def extract_climate_data(self, table: ExtractedTable, city_name: str) -> Optional[ClimateData]:
        """Извлекает климатические данные для указанного города."""
        if not table or not city_name:
            return None

        cache_key = f"{city_name.lower().strip()}::{hash(table.raw_text[:300])}"
        if cache_key in self._climate_cache:
            return self._climate_cache[cache_key]

        city_lower = city_name.lower().strip().replace("ё", "е")
        data = ClimateData(city=city_name, source=table.source)

        found_row = None
        for row in table.rows:
            row_text = " ".join(row).lower().replace("ё", "е")
            if city_lower in row_text:
                found_row = row
                break

        if not found_row:
            return None

        headers = [h.lower().strip().replace("ё", "е") for h in table.headers] if table.headers else []

        if headers and len(headers) == len(found_row):
            self._extract_by_headers(headers, found_row, data)

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

    def _extract_by_headers(self, headers: List[str], row: List[str], data: ClimateData) -> None:
        """Извлекает значения по заголовкам таблицы."""
        for header, cell in zip(headers, row):
            value = self._extract_first_number(cell)
            if value is None:
                continue

            header_text = header.lower()

            if (
                data.t_ot is None
                and ("t_от" in header_text or "отоп" in header_text or "средн" in header_text)
                and -50 <= value <= 30
            ):
                data.t_ot = float(value)
                continue

            if (
                data.z_ot is None
                and ("z_от" in header_text or "продолж" in header_text or "сут" in header_text or "дней" in header_text)
                and 1 <= value <= 400
            ):
                data.z_ot = int(round(value))
                continue

            if (
                data.t_n is None
                and ("t_н" in header_text or "пятиднев" in header_text or "холодн" in header_text or "наруж" in header_text)
                and -70 <= value <= 20
            ):
                data.t_n = float(value)
                continue

            if data.t_avg is None and "среднегод" in header_text and -50 <= value <= 30:
                data.t_avg = float(value)

    def _extract_by_plain_scan(self, row: List[str], data: ClimateData) -> None:
        """Запасной вариант извлечения без заголовков."""
        for cell in row:
            cell_norm = cell.lower().replace("ё", "е").replace(",", ".").strip()

            if data.z_ot is None:
                days_match = re.search(r"(\d+)\s*сут", cell_norm)
                if days_match:
                    data.z_ot = int(days_match.group(1))
                    continue

            value = self._extract_first_number(cell_norm)
            if value is None:
                continue

            if data.t_ot is None and -30 <= value <= 25:
                data.t_ot = float(value)
                continue

            if data.z_ot is None and 1 <= value <= 400:
                data.z_ot = int(round(value))
                continue

            if data.t_n is None and -70 <= value <= 15:
                data.t_n = float(value)

    @staticmethod
    def _extract_first_number(text: str) -> Optional[float]:
        """Извлекает первое число из строки."""
        if not text:
            return None

        match = re.search(r"-?\d+[.,]?\d*", text)
        if not match:
            return None

        try:
            return float(match.group(0).replace(",", "."))
        except ValueError:
            return None

    def calculate_degree_days_from_table(self, city_name: str, t_v: float = 20.0) -> Dict[str, Any]:
        """Рассчитывает ГСОП по данным климатической таблицы."""
        result: Dict[str, Any] = {
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

        except Exception as exc:
            result["answer"] = f"❌ Ошибка при расчёте: {exc}"

        return result

    def calculate_gsop_from_table(self, city_name: str, t_v: float = 20.0) -> Dict[str, Any]:
        """Алиас для calculate_degree_days_from_table."""
        return self.calculate_degree_days_from_table(city_name, t_v)

    def calculate_ventilation_from_table(
        self,
        city_name: str,
        air_flow: float,
        t_v: float = 20.0,
    ) -> Dict[str, Any]:
        """Рассчитывает расход теплоты на вентиляцию по данным таблицы."""
        result: Dict[str, Any] = {
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
            result["answer"] = degree_days_result["answer"]
            result["source"] = degree_days_result.get("source", "")
            result["table"] = degree_days_result.get("table")
            return result

        data = degree_days_result["data"]

        if data.t_n is None:
            result["answer"] = (
                f"⚠️ Не найдена температура наружного воздуха t_н для {city_name}. "
                "Она нужна для расчёта вентиляции."
            )
            result["source"] = degree_days_result.get("source", "")
            result["table"] = degree_days_result.get("table")
            return result

        q_vent = 0.335 * air_flow * (t_v - data.t_n)

        result["degree_days_result"] = degree_days_result
        result["ventilation"] = q_vent
        result["success"] = True
        result["source"] = degree_days_result.get("source", "")
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
    ) -> Dict[str, Any]:
        """Рассчитывает теплопотери через ограждение по данным таблицы."""
        result: Dict[str, Any] = {
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

        except Exception as exc:
            result["answer"] = f"❌ Ошибка при расчёте: {exc}"

        return result

    def get_cities_from_table(self, query: str = "климатические параметры городов") -> List[str]:
        """Извлекает список городов из найденной климатической таблицы."""
        table = self.find_climate_table(query)
        if not table:
            return []

        cities: List[str] = []
        for row in table.rows:
            for cell in row:
                if self._is_city_name(cell):
                    cities.append(cell.strip())

        return sorted(set(cities))


def patch_app_with_table_calculator() -> None:
    """Заглушка для совместимости."""
    print("✅ TableCalculator готов к использованию")