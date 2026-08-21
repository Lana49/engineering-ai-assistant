# core/table_calculator.py
"""
Калькулятор для расчётов на основе таблиц и встроенных данных.
"""

from __future__ import annotations

import re
from typing import Any

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
            return []

    def calculate_gsop_from_table(self, city: str, t_v: float = 20.0) -> dict[str, Any]:
        """Рассчитывает ГСОП по встроенным данным (fallback)."""
        city_lower = city.lower().strip()

        # Сначала пробуем найти в таблицах
        try:
            tables = self._search_tables(
                f"{city} климатические параметры температура отопительный период",
                top_k=10
            )
            if tables:
                from core.table_extractor import find_city_in_tables
                found = find_city_in_tables(tables, city_lower)
                if found:
                    row_text = found.get("row_text", "")
                    numbers = re.findall(r"-?\d+[.,]?\d*", row_text)
                    if len(numbers) >= 2:
                        try:
                            t_ot = float(numbers[0].replace(",", "."))
                            z_ot = int(float(numbers[1].replace(",", ".")))
                            gsop = (t_v - t_ot) * z_ot
                            return {
                                "answer": (
                                    f"🌍 **ГСОП для {city.capitalize()} = {gsop:.0f} °C·сут**\n\n"
                                    f"📊 Из таблицы: t_от={t_ot:.1f}°C, z_от={z_ot} сут\n"
                                    f"({t_v:.1f} - {t_ot:.1f}) × {z_ot} = {gsop:.0f}"
                                ),
                                "result": gsop,
                                "sources": [{"doc_name": found.get("table", {}).source}],
                                "confidence": 0.9,
                            }
                        except (ValueError, IndexError):
                            pass
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

        # Fallback на встроенные данные
        data = CLIMATE_DATA.get(city_lower)
        if not data:
            for known_city in CLIMATE_DATA:
                if known_city in city_lower:
                    data = CLIMATE_DATA[known_city]
                    break

        if not data:
            return {
                "answer": f"❌ Климатические данные для города **{city}** не найдены.\n\nДоступные города: {', '.join(list(CLIMATE_DATA.keys())[:5])}...",
                "sources": [],
                "tables": [],
                "confidence": 0.0,
                "needs_clarification": True,
                "questions": [f"Укажите город из списка."],
                "query_type": "calculation",
            }

        t_ot = data["t_ot"]
        z_ot = data["z_ot"]
        gsop = (t_v - t_ot) * z_ot

        return {
            "answer": (
                f"🌍 **ГСОП для {city.capitalize()} = {gsop:.0f} °C·сут**\n\n"
                f"📊 Исходные данные:\n"
                f"- t_в = {t_v:.1f} °C\n"
                f"- t_от = {t_ot:.1f} °C\n"
                f"- z_от = {z_ot} сут\n\n"
                f"🔢 Расчёт: ({t_v:.1f} - {t_ot:.1f}) × {z_ot} = {gsop:.0f}\n\n"
                f"📚 Источник: СП 131.13330 (встроенные данные)"
            ),
            "result": gsop,
            "sources": [{"doc_name": "СП 131.13330"}],
            "tables": [],
            "formulas": [{"raw": "(t_в - t_от) × z_от", "name": "ГСОП", "source": "СП 131.13330"}],
            "confidence": 0.95,
            "needs_clarification": False,
            "questions": [],
            "query_type": "calculation",
        }

    @staticmethod
    def calculate_ventilation_from_table(self, city: str, air_flow: float) -> dict[str, Any]:
        """Рассчитывает расход теплоты на вентиляцию."""
        city_lower = city.lower().strip()
        data = CLIMATE_DATA.get(city_lower)

        if not data:
            for known_city in CLIMATE_DATA:
                if known_city in city_lower:
                    data = CLIMATE_DATA[known_city]
                    break

        if not data:
            return {
                "answer": f"❌ Климатические данные для {city} не найдены.",
                "sources": [],
                "tables": [],
                "confidence": 0.0,
                "needs_clarification": True,
                "questions": ["Уточните город."],
                "query_type": "calculation",
            }

        t_n = data["t_n"]
        t_v = 20.0
        q_vent = 0.335 * air_flow * (t_v - t_n)

        return {
            "answer": (
                f"💨 **Расход теплоты на вентиляцию для {city.capitalize()} = {q_vent:.0f} Вт**\n\n"
                f"📊 Исходные данные:\n"
                f"- L = {air_flow:.0f} м³/ч\n"
                f"- t_в = {t_v:.1f} °C\n"
                f"- t_н = {t_n:.1f} °C (из климатических данных)\n\n"
                f"🔢 Расчёт: 0.335 × {air_flow:.0f} × ({t_v:.1f} - {t_n:.1f}) = {q_vent:.0f} Вт\n\n"
                f"📚 Источник: СП 60.13330 + СП 131.13330"
            ),
            "result": q_vent,
            "sources": [{"doc_name": "СП 60.13330 / СП 131.13330"}],
            "tables": [],
            "formulas": [
                {"raw": "Q_в = 0.335 × L × (t_в - t_н)", "name": "Расход теплоты на вентиляцию",
                 "source": "СП 60.13330"}
            ],
            "confidence": 0.9,
            "needs_clarification": False,
            "questions": [],
            "query_type": "calculation",
        }

    @staticmethod
    def calculate_heat_loss_from_table(self, city: str, area: float, resistance: float) -> dict[str, Any]:
        """Рассчитывает теплопотери через ограждение."""
        city_lower = city.lower().strip()
        data = CLIMATE_DATA.get(city_lower)

        if not data:
            for known_city in CLIMATE_DATA:
                if known_city in city_lower:
                    data = CLIMATE_DATA[known_city]
                    break

        if not data:
            return {
                "answer": f"❌ Климатические данные для {city} не найдены.",
                "sources": [],
                "tables": [],
                "confidence": 0.0,
                "needs_clarification": True,
                "questions": ["Уточните город."],
                "query_type": "calculation",
            }

        t_n = data["t_n"]
        t_v = 20.0
        delta_t = t_v - t_n
        q_loss = (area * delta_t) / resistance

        return {
            "answer": (
                f"🔥 **Теплопотери для {city.capitalize()} = {q_loss:.0f} Вт**\n\n"
                f"📊 Исходные данные:\n"
                f"- A = {area:.1f} м²\n"
                f"- R = {resistance:.3f} м²·°C/Вт\n"
                f"- t_в = {t_v:.1f} °C\n"
                f"- t_н = {t_n:.1f} °C (из климатических данных)\n"
                f"- Δt = {delta_t:.1f} °C\n\n"
                f"🔢 Расчёт: ({area:.1f} × {delta_t:.1f}) / {resistance:.3f} = {q_loss:.0f} Вт\n\n"
                f"📚 Источник: СП 50.13330 + СП 131.13330"
            ),
            "result": q_loss,
            "sources": [{"doc_name": "СП 50.13330 / СП 131.13330"}],
            "tables": [],
            "formulas": [
                {"raw": "Q = (A × Δt) / R", "name": "Теплопотери через ограждение", "source": "СП 50.13330"}
            ],
            "confidence": 0.9,
            "needs_clarification": False,
            "questions": [],
            "query_type": "calculation",
        }


def patch_app_with_table_calculator():
    """Заглушка для совместимости."""
    pass