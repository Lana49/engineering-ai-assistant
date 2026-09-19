# core/table_calculator.py
"""
Калькулятор для расчётов на основе таблиц и встроенных данных.
"""

from __future__ import annotations
import logging
import re
import os
from typing import Any
logger = logging.getLogger(__name__)
# ВСТРОЕННЫЕ КЛИМАТИЧЕСКИЕ ДАННЫЕ (FALLBACK)
CLIMATE_DATA = {
    "москва": {"t_ot": -3.1, "z_ot": 214, "t_n": -25},
    "санкт-петербург": {"t_ot": -1.8, "z_ot": 220, "t_n": -24},
    "новосибирск": {"t_ot": -8.7, "z_ot": 230, "t_n": -39},
    "екатеринбург": {"t_ot": -6.9, "z_ot": 227, "t_n": -35},
    "казань": {"t_ot": -5.8, "z_ot": 218, "t_n": -32},
    "нижний новгород": {"t_ot": -5.6, "z_ot": 220, "t_n": -30},
    "челябинск": {"t_ot": -7.5, "z_ot": 230, "t_n": -34},
    "омск": {"t_ot": -8.4, "z_ot": 235, "t_n": -37},
    "самара": {"t_ot": -6.0, "z_ot": 215, "t_n": -30},
    "ростов-на-дону": {"t_ot": -1.2, "z_ot": 210, "t_n": -22},
    "уфа": {"t_ot": -6.4, "z_ot": 222, "t_n": -33},
    "красноярск": {"t_ot": -9.2, "z_ot": 238, "t_n": -40},
    "пермь": {"t_ot": -6.3, "z_ot": 225, "t_n": -35},
    "воронеж": {"t_ot": -3.4, "z_ot": 205, "t_n": -26},
    "волгоград": {"t_ot": -2.0, "z_ot": 202, "t_n": -23},
    "краснодар": {"t_ot": 0.2, "z_ot": 190, "t_n": -19},
    "тюмень": {"t_ot": -8.2, "z_ot": 232, "t_n": -37},
}


class TableCalculator:
    def __init__(self, qa_system=None):
        self.qa_system = qa_system

    def _search_tables(self, query: str, top_k: int = 5) -> list:
        """Ищет таблицы через QA-систему."""
        if self.qa_system is None:
            return []

        try:
            from core.table_extractor import extract_tables_from_results
            results = self.qa_system.search(query, top_k=top_k)
            return extract_tables_from_results(results)
        except (AttributeError, KeyError, TypeError, ValueError):
            logger.exception("TableCalculator: ошибка поиска таблиц")
            return []

    CLIMATE_HEADER_ALIASES: dict[str, tuple[str, ...]] = {
        "t_ot": (
            "tот", "t_от", "средняя температура отопительного периода",
            "температура отопительного периода", "средняя температура",
        ),
        "z_ot": (
            "zот", "z_от", "продолжительность отопительного периода",
            "продолжительность периода", "продолжительность",
        ),
        "t_n": (
            "tн", "t_н", "расчетная температура наружного воздуха",
            "температура наружного воздуха", "температура наиболее холодной пятидневки",
        ),
    }

    def _normalized(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").lower().replace("ё", "е")).strip()

    def _parse_number(value: Any) -> float | None:
        """Извлекает одно числовое значение из ячейки без угадывания столбца."""
        match = re.search(r"[-+]?\d+(?:[.,]\d+)?", str(value or ""))
        if not match:
            return None
        try:
            return float(match.group(0).replace(",", "."))
        except ValueError:
            return None

    def _matches_city(candidate: str, known_city: str) -> bool:
        """Сравнивает названия городов по словам, не путая Томск с Омском."""
        candidate = _normalized(candidate).replace("-", " ")
        known_city = _normalized(known_city).replace("-", " ")
        return bool(re.search(rf"(?<![а-яa-z]){re.escape(known_city)}(?![а-яa-z])", candidate))
    def calculate_gsop_from_table(self, city: str, t_v: float = 20.0) -> dict[str, Any]:
        """Рассчитывает ГСОП по явно подписанным данным климатической таблицы."""
        table_values = self._climate_values_from_tables(city, ("t_ot", "z_ot"))
        source: dict[str, Any] | None = None
        if table_values:
            data, source = table_values
        elif self._allow_builtin_fallback():
            data = self._builtin_climate(city)
            if data:
                source = {"doc_name": "Контрольный встроенный набор (требуется сверка со СП 131.13330)"}
        else:
            data = None

        if not data:
            return self._missing_climate_response(city, ("t_ot", "z_ot"))
        t_ot = data["t_ot"]
        z_ot = int(data["z_ot"])
        gsop = (t_v - t_ot) * z_ot
        source_name = str((source or {}).get("doc_name") or "Таблица климатических данных")
        source_line = f"📚 Источник: {source_name}"
        return {
            "answer": (
                f"### ГСОП для {city.capitalize()}\n\n"
                f"📊 Исходные данные:\n"
                f"- t_в = {t_v:.1f} °C\n"
                f"- t_от = {t_ot:.1f} °C\n"
                f"- z_от = {z_ot} сут\n\n"
                f"🔢 Расчёт: ({t_v:.1f} - {t_ot:.1f}) × {z_ot} = **{gsop:.0f} °C·сут**\n\n"
                f"{source_line}"
            ),
            "result": gsop,
            "sources": [source or {"doc_name": source_name}],
            "tables": [],
            "formulas": [{"raw": "(t_в - t_от) × z_от", "name": "ГСОП", "source": "СП 131.13330"}],
            "confidence": 0.9 if table_values else 0.55,
            "needs_clarification": False,
            "questions": [],
            "query_type": "calculation",
        }


    def calculate_ventilation_from_table(self, city: str, air_flow: float, t_v: float = 20.0) -> dict[str, Any]:
        """Рассчитывает расход теплоты на вентиляцию."""
        table_values = self._climate_values_from_tables(city, ("t_n",))
        source: dict[str, Any] | None = None
        if table_values:
            data, source = table_values
        elif self._allow_builtin_fallback():
            data = self._builtin_climate(city)
            if data:
                source = {"doc_name": "Контрольный встроенный набор (требуется сверка со СП 131.13330)"}
        else:
            data = None

        if not data:
            return self._missing_climate_response(city, ("t_n",))

        t_n = data["t_n"]
        q_vent = 0.335 * air_flow * (t_v - t_n)
        source_name = str((source or {}).get("doc_name") or "Таблица климатических данных")

        return {
            "answer": (
                f"### Расход теплоты на вентиляцию для {city.capitalize()}\n\n"
                f"📊 Исходные данные:\n"
                f"- L = {air_flow:.0f} м³/ч\n"
                f"- t_в = {t_v:.1f} °C\n"
                f"- t_н = {t_n:.1f} °C (из климатических данных)\n\n"
                f"🔢 Расчёт: 0.335 × {air_flow:.0f} × ({t_v:.1f} - {t_n:.1f}) = **{q_vent:.0f} Вт**\n\n"
                f"📚 Источники: СП 60.13330; {source_name}"
            ),
            "result": q_vent,
            "sources": [{"doc_name": "СП 60.13330"}, source or {"doc_name": source_name}],
            "tables": [],
            "formulas": [
                {"raw": "Q_в = 0.335 × L × (t_в - t_н)", "name": "Расход теплоты на вентиляцию",
                 "source": "СП 60.13330"}
            ],
            "confidence": 0.9 if table_values else 0.55,
            "needs_clarification": False,
            "questions": [],
            "query_type": "calculation",
        }

    def calculate_heat_loss_from_table(self, city: str, area: float, resistance: float, t_v: float = 20.0) -> dict[str, Any]:
        """Рассчитывает теплопотери через ограждение."""
        table_values = self._climate_values_from_tables(city, ("t_n",))
        source: dict[str, Any] | None = None
        if table_values:
            data, source = table_values
        elif self._allow_builtin_fallback():
            data = self._builtin_climate(city)
            if data:
                source = {"doc_name": "Контрольный встроенный набор (требуется сверка со СП 131.13330)"}
        else:
            data = None
        if not data:
            return self._missing_climate_response(city, ("t_n",))

        t_n = data["t_n"]
        delta_t = t_v - t_n
        q_loss = (area * delta_t) / resistance
        source_name = str((source or {}).get("doc_name") or "Таблица климатических данных")

        return {
            "answer": (
                f"### Теплопотери для {city.capitalize()}\n\n"
                f"📊 Исходные данные:\n"
                f"- A = {area:.1f} м²\n"
                f"- R = {resistance:.3f} м²·°C/Вт\n"
                f"- t_в = {t_v:.1f} °C\n"
                f"- t_н = {t_n:.1f} °C (из климатических данных)\n"
                f"- Δt = {delta_t:.1f} °C\n\n"
                f"🔢 Расчёт: ({area:.1f} × {delta_t:.1f}) / {resistance:.3f} = **{q_loss:.0f} Вт**\n\n"
                f"📚 Источники: СП 50.13330; {source_name}"
            ),
            "result": q_loss,
            "sources": [{"doc_name": "СП 50.13330"}, source or {"doc_name": source_name}],
            "tables": [],
            "formulas": [
                {"raw": "Q = (A × Δt) / R", "name": "Теплопотери через ограждение", "source": "СП 50.13330"}
            ],
            "confidence": 0.9 if table_values else 0.55,
            "needs_clarification": False,
            "questions": [],
            "query_type": "calculation",
        }


def patch_app_with_table_calculator():
    """Заглушка для совместимости."""
    pass
  @staticmethod
    def _column_index(headers: list[Any], parameter: str) -> int | None:
        """Находит нужный климатический столбец по заголовку таблицы."""
        aliases = CLIMATE_HEADER_ALIASES.get(parameter, ())
        for index, header in enumerate(headers or []):
            normalized_header = _normalized(header).replace("_", "")
            if any(alias.replace("_", "") in normalized_header for alias in aliases):
                return index
        return None

    def _climate_values_from_tables(
        self,
        city: str,
        required: tuple[str, ...],
    ) -> tuple[dict[str, float], dict[str, Any]] | None:
        """Берёт климатические параметры строго из подписанных столбцов.

        Прежняя версия брала первые два числа из строки. В нормативных
        таблицах это мог быть номер строки или координата, поэтому расчёт
        становился неверным. Здесь значение принимается только если найден
        и город, и заголовок соответствующего столбца.
        """
        tables = self._search_tables(
            f"{city} климатические параметры температура отопительный период",
            top_k=10,
        )
        if not tables:
            return None

        try:
            from core.table_extractor import find_city_in_tables
        except ImportError:
            return None

        for table in tables:
            found = find_city_in_tables([table], city)
            if not found:
                continue

            row = list(found.get("row") or [])
            values: dict[str, float] = {}
            for parameter in required:
                column = self._column_index(list(table.headers or []), parameter)
                if column is None or column >= len(row):
                    break
                value = _parse_number(row[column])
                if value is None:
                    break
                values[parameter] = value

            if len(values) == len(required):
                source = {
                    "doc_name": table.source or "Таблица климатических данных",
                    "table_title": table.title,
                    "metadata": dict(table.metadata or {}),
                }
                return values, source

        return None

    @staticmethod
    def _builtin_climate(city: str) -> dict[str, float] | None:
        """Возвращает fallback только при точном совпадении города."""
        city_normalized = _normalized(city).replace("-", " ")
        for known_city, data in CLIMATE_DATA.items():
            if _matches_city(city_normalized, known_city):
                return dict(data)
        return None

    @staticmethod
    def _allow_builtin_fallback() -> bool:
        """Fallback выключен по умолчанию: ответы должны опираться на базу."""
        return os.getenv("ALLOW_BUILTIN_CLIMATE_FALLBACK", "false").lower() == "true"

    @staticmethod
    def _missing_climate_response(city: str, required: tuple[str, ...]) -> dict[str, Any]:
        labels = {
            "t_ot": "t_от — среднюю температуру отопительного периода",
            "z_ot": "z_от — продолжительность отопительного периода",
            "t_n": "t_н — расчётную температуру наружного воздуха",
        }
        missing = ", ".join(labels.get(value, value) for value in required)
        return {
            "answer": (
                f"⚠️ В подключённой базе не найдены климатические данные для города «{city}».\n\n"
                f"Для расчёта нужны: {missing}. Загрузите таблицу СП 131.13330 "
                "или передайте эти параметры явно."
            ),
            "sources": [],
            "tables": [],
            "formulas": [],
            "confidence": 0.0,
            "needs_clarification": True,
            "questions": ["Укажите недостающие климатические параметры или загрузите источник."],
            "query_type": "calculation",
        }