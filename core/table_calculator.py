# core/table_calculator.py
"""
Калькулятор для расчётов на основе таблиц и встроенных данных.
"""

from __future__ import annotations
import logging
import math
import os
import re
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


CLIMATE_HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    "t_ot": (
        "tот", "tot", "средняя температура отопительного периода",
        "температура отопительного периода", "heating temperature",
    ),
    "z_ot": (
        "zот", "zot", "продолжительность отопительного периода", "heating days",
    ),
    "t_n": (
        "tн", "tn", "расчетная температура наружного воздуха",
        "температура наружного воздуха", "температура наиболее холодной пятидневки",
    ),
}


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower().replace("ё", "е")).strip()


def _parse_number(value: Any) -> float | None:
    """Читает одну числовую ячейку; диапазоны и несколько чисел отклоняет."""
    if isinstance(value, bool) or value is None:
        return None
    normalized = str(value).strip().replace("−", "-").replace("–", "-")
    match = re.fullmatch(
        r"([-+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:[eE][-+]?\d+)?)"
        r"\s*(?:°\s*[cс]|сут(?:ки|ок)?\.?|дн(?:ей|я)?\.?)?",
        normalized, flags=re.I,
    )
    if not match:
        return None
    try:
        number = float(match.group(1).replace(",", "."))
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def _matches_city(candidate: str, known_city: str) -> bool:
    """Сравнивает названия городов по словам, не путая Томск с Омском."""
    candidate_normalized = _normalized(candidate).replace("-", " ")
    city_normalized = _normalized(known_city).replace("-", " ")
    return bool(city_normalized and re.search(rf"(?<![а-яa-z]){re.escape(city_normalized)}(?![а-яa-z])", candidate_normalized))


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
            # Чанк большой таблицы может содержать ровно одну нужную строку.
            return extract_tables_from_results(results, min_rows=1, limit=len(results))
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, OSError):
            logger.exception("TableCalculator: ошибка поиска таблиц")
            return []

    @staticmethod
    def _column_index(headers: list[Any], parameter: str) -> int | None:
        """Находит нужный климатический столбец по заголовку таблицы."""
        aliases = CLIMATE_HEADER_ALIASES.get(parameter, ())
        matches: list[int] = []
        for index, header in enumerate(headers or []):
            normalized_header = _normalized(header).replace("_", "")
            if any(re.search(rf"(?<![а-яa-z]){re.escape(alias)}(?![а-яa-z])", normalized_header) for alias in aliases):
                matches.append(index)
        # Две колонки пятидневки (0,92 и 0,98) требуют выбора условия.
        return matches[0] if len(matches) == 1 else None

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

        candidates: list[tuple[dict[str, float], dict[str, Any]]] = []
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
                if value is None or not self._valid_climate_value(parameter, value):
                    break
                values[parameter] = value

            if len(values) == len(required):
                source = {
                    "doc_name": table.source or "Таблица климатических данных",
                    "table_title": table.title,
                    "metadata": dict(table.metadata or {}),
                }
                candidates.append((values, source))

        if not candidates:
            return None
        distinct_values = {tuple(candidate[0][key] for key in required) for candidate in candidates}
        if len(distinct_values) > 1:
            logger.warning("Противоречивые климатические таблицы для %s; требуется уточнить источник", city)
            return None
        return candidates[0]

    @staticmethod
    def _valid_climate_value(parameter: str, value: float) -> bool:
        if not math.isfinite(value):
            return False
        if parameter == "z_ot":
            return 0 < value <= 366 and value.is_integer()
        return -100 <= value <= 60

    @staticmethod
    def _invalid_input_response(message: str) -> dict[str, Any]:
        """Возвращает уточнение вместо физически некорректного числа."""
        return {
            "answer": f"Для расчёта нужно уточнить исходные данные: {message}",
            "sources": [], "tables": [], "formulas": [], "confidence": 0.0,
            "needs_clarification": True, "questions": [message], "query_type": "calculation",
        }

    @staticmethod
    def _valid_input(value: Any, *, positive: bool = False, temperature: bool = False) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return False
        return (not positive or value > 0) and (not temperature or value > -273.15)

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
                f"В подключённой базе не найдены однозначные климатические данные для города «{city}».\n\n"
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

    def calculate_gsop_from_table(self, city: str, t_v: float = 20.0) -> dict[str, Any]:
        """Рассчитывает ГСОП по явно подписанным данным климатической таблицы."""
        if not self._valid_input(t_v, temperature=True):
            return self._invalid_input_response("t_в должна быть конечным числом выше −273,15 °C.")
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
        if t_v <= t_ot:
            return self._invalid_input_response("Для ГСОП t_в должна быть выше t_от. Проверьте температуры.")
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
            "formulas": [{"raw": "(t_в - t_от) × z_от", "name": "ГСОП", "source": "Встроенная расчётная формула"}],
            "confidence": 0.9 if table_values else 0.55,
            "needs_clarification": False,
            "questions": [],
            "query_type": "calculation",
        }


    def calculate_ventilation_from_table(self, city: str, air_flow: float, t_v: float = 20.0) -> dict[str, Any]:
        """Рассчитывает расход теплоты на вентиляцию."""
        if not self._valid_input(air_flow, positive=True) or not self._valid_input(t_v, temperature=True):
            return self._invalid_input_response("L должен быть конечным положительным числом в м³/ч; t_в — температурой выше −273,15 °C.")
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
        if t_v <= t_n:
            return self._invalid_input_response("Для нагрева воздуха t_в должна быть выше t_н.")
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
                f"Коэффициент 0,335 — приближённая объёмная теплоёмкость воздуха в Вт·ч/(м³·°C).\n\n"
                f"📚 Источник климатических данных: {source_name}"
            ),
            "result": q_vent,
            "sources": [source or {"doc_name": source_name}],
            "tables": [],
            "formulas": [
                {"raw": "Q_в = 0.335 × L × (t_в - t_н)", "name": "Расход теплоты на вентиляцию",
                 "source": "Встроенная приближённая расчётная формула"}
            ],
            "confidence": 0.9 if table_values else 0.55,
            "needs_clarification": False,
            "questions": [],
            "query_type": "calculation",
        }

    def calculate_heat_loss_from_table(self, city: str, area: float, resistance: float, t_v: float = 20.0) -> dict[str, Any]:
        """Рассчитывает теплопотери через ограждение."""
        if not all(self._valid_input(value, positive=True) for value in (area, resistance)):
            return self._invalid_input_response("A и R должны быть конечными положительными числами; R не может быть равным нулю.")
        if not self._valid_input(t_v, temperature=True):
            return self._invalid_input_response("t_в должна быть конечным числом выше −273,15 °C.")
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
        if delta_t <= 0:
            return self._invalid_input_response("Для расчёта теплопотерь t_в должна быть выше t_н.")
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
                f"📚 Источник климатических данных: {source_name}"
            ),
            "result": q_loss,
            "sources": [source or {"doc_name": source_name}],
            "tables": [],
            "formulas": [
                {"raw": "Q = (A × Δt) / R", "name": "Теплопотери через ограждение", "source": "Встроенная расчётная формула"}
            ],
            "confidence": 0.9 if table_values else 0.55,
            "needs_clarification": False,
            "questions": [],
            "query_type": "calculation",
        }


def patch_app_with_table_calculator():
    """Заглушка для совместимости."""
    pass


def _run_self_tests() -> None:
    """Проверяет города, числовые ячейки и валидацию входных данных."""
    assert not _matches_city("томск", "омск")
    assert TableCalculator().calculate_gsop_from_table("Томск")["needs_clarification"]
    assert _parse_number("−8,4 °C") == -8.4
    assert _parse_number("от -8 до -12") is None
    assert _parse_number("NaN") is None
    assert TableCalculator().calculate_heat_loss_from_table("Томск", 20, 0)["needs_clarification"]
    assert TableCalculator._column_index(["t_н 0,92", "t_н 0,98"], "t_n") is None


if __name__ == "__main__":
    _run_self_tests()

# ИСПРАВЛЕНО: методы возвращены в класс; определены вспомогательные функции;
# поиск всех результатов и однострочных таблиц; однозначные столбцы/числа;
# проверка диапазонов и деления на ноль; Томск не подменяется Омском;
# источники содержат только найденные документы; добавлены самотесты.
