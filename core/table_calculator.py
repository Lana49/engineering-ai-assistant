# core/table_calculator.py
"""
Калькулятор для расчётов на основе табличных данных.
Использует TableExtractor для поиска таблиц.
"""

from __future__ import annotations

import re
from typing import Any

from core.table_extractor import TableExtractor, find_city_in_tables, tables_to_dicts


class TableCalculator:
    """
    Выполняет расчёты, используя данные из таблиц документов.
    """

    def __init__(self, qa_system=None):
        self.qa_system = qa_system
        self.extractor = TableExtractor()
        self.city_data_cache: dict[str, dict[str, Any]] = {}

    def _search_tables(self, query: str, top_k: int = 5) -> list:
        """Ищет таблицы по запросу."""
        if self.qa_system is None:
            return []

        from core.table_extractor import extract_tables_from_results
        results = self.qa_system.search(query, top_k=top_k)
        return extract_tables_from_results(results)

    def calculate_gsop_from_table(self, city: str) -> dict[str, Any]:
        """Рассчитывает ГСОП по данным из таблицы."""
        city_lower = city.lower().strip()

        tables = self._search_tables(
            "климатические параметры температура отопительный период",
            top_k=10
        )

        if not tables:
            tables = self._search_tables(
                "температура наружного воздуха отопительный период",
                top_k=10
            )

        if not tables:
            return {
                "answer": f"❌ Не найдены климатические таблицы для города {city}.",
                "sources": [],
                "tables": [],
                "confidence": 0.0,
                "needs_clarification": False,
                "questions": [],
                "query_type": "calculation",
            }

        found = find_city_in_tables(tables, city_lower)

        if not found:
            for table in tables:
                for row in table.rows:
                    if isinstance(row, list):
                        row_text = " ".join(str(c) for c in row)
                    else:
                        row_text = str(row)

                    if city_lower in row_text.lower():
                        found = {
                            "table": table,
                            "row": row,
                            "row_text": row_text,
                            "index": table.rows.index(row)
                        }
                        break
                if found:
                    break

        if not found:
            table_names = [t.source for t in tables[:3]]
            return {
                "answer": f"❌ Город {city} не найден в климатических таблицах.\n\n"
                           f"🔍 Найдены таблицы: {', '.join(table_names)}",
                "sources": [{"doc_name": t.source} for t in tables[:3]],
                "tables": tables_to_dicts(tables[:3]),
                "confidence": 0.0,
                "needs_clarification": True,
                "questions": [f"Уточните название города или проверьте таблицы."],
                "query_type": "calculation",
            }

        row_text = found.get("row_text", "")
        numbers = re.findall(r"-?\d+[.,]?\d*", row_text)

        if len(numbers) < 2:
            return {
                "answer": f"⚠️ Не удалось извлечь климатические данные для города {city} из таблицы.",
                "sources": [{"doc_name": found["table"].source}],
                "tables": tables_to_dicts([found["table"]]),
                "confidence": 0.2,
                "needs_clarification": True,
                "questions": ["Проверьте формат данных в таблице."],
                "query_type": "calculation",
            }

        try:
            t_ot = float(numbers[0].replace(",", "."))
            z_ot = int(float(numbers[1].replace(",", ".")))
            t_v = 20.0

            gsop = (t_v - t_ot) * z_ot

            answer = (
                f"🌍 **ГСОП для {city} = {gsop:.2f} °C·сут**\n\n"
                f"📊 **Данные из таблицы:**\n"
                f"- t_от = {t_ot:.1f} °C\n"
                f"- z_от = {z_ot} сут\n"
                f"- t_в = {t_v:.1f} °C (принято по умолчанию)\n\n"
                f"🔢 **Расчёт:** ({t_v:.1f} - {t_ot:.1f}) × {z_ot} = {gsop:.2f}\n\n"
                f"📚 **Источник:** {found['table'].source}"
            )

            return {
                "answer": answer,
                "sources": [{"doc_name": found["table"].source}],
                "tables": tables_to_dicts([found["table"]]),
                "confidence": 0.9,
                "needs_clarification": False,
                "questions": [],
                "query_type": "calculation",
                "formulas": [{
                    "raw": "(t_в - t_от) × z_от",
                    "name": "ГСОП",
                    "source": found["table"].source,
                }],
            }

        except (ValueError, IndexError) as e:
            return {
                "answer": f"⚠️ Ошибка парсинга климатических данных для города {city}: {e}",
                "sources": [{"doc_name": found["table"].source}],
                "tables": tables_to_dicts([found["table"]]),
                "confidence": 0.2,
                "needs_clarification": True,
                "questions": ["Проверьте формат данных в таблице."],
                "query_type": "calculation",
            }

    def calculate_ventilation_from_table(self, city: str, air_flow: float) -> dict[str, Any]:
        """Рассчитывает расход теплоты на вентиляцию."""
        climate_data = self.calculate_gsop_from_table(city)

        if climate_data.get("confidence", 0) < 0.5:
            return {
                "answer": f"❌ Не удалось получить климатические данные для {city}.",
                "sources": climate_data.get("sources", []),
                "tables": climate_data.get("tables", []),
                "confidence": 0.0,
                "needs_clarification": True,
                "questions": ["Уточните город или проверьте климатические таблицы."],
                "query_type": "calculation",
            }

        source = climate_data.get("answer", "")
        t_ot_match = re.search(r"t_от\s*=\s*([-+]?\d+[.,]?\d*)", source)
        t_ot = float(t_ot_match.group(1).replace(",", ".")) if t_ot_match else -8.0

        t_n = t_ot
        t_v = 20.0
        q_vent = 0.335 * air_flow * (t_v - t_n)

        answer = (
            f"💨 **Расход теплоты на вентиляцию для {city} = {q_vent:.2f} Вт**\n\n"
            f"📊 **Исходные данные:**\n"
            f"- Расход воздуха L = {air_flow:.1f} м³/ч\n"
            f"- t_в = {t_v:.1f} °C\n"
            f"- t_н = {t_n:.1f} °C (из климатических данных)\n\n"
            f"🔢 **Расчёт:** 0.335 × {air_flow:.1f} × ({t_v:.1f} - {t_n:.1f}) = {q_vent:.2f} Вт\n\n"
            f"📚 **Источник:** {climate_data.get('sources', [{}])[0].get('doc_name', 'Климатическая таблица')}"
        )

        return {
            "answer": answer,
            "sources": climate_data.get("sources", []),
            "tables": climate_data.get("tables", []),
            "confidence": 0.85,
            "needs_clarification": False,
            "questions": [],
            "query_type": "calculation",
            "formulas": [{
                "raw": "Q_в = 0.335 × L × (t_в - t_н)",
                "name": "Расход теплоты на вентиляцию",
                "source": "СП 60.13330",
            }],
        }

    def calculate_heat_loss_from_table(self, city: str, area: float, resistance: float) -> dict[str, Any]:
        """Рассчитывает теплопотери через ограждение."""
        climate_data = self.calculate_gsop_from_table(city)

        if climate_data.get("confidence", 0) < 0.5:
            return {
                "answer": f"❌ Не удалось получить климатические данные для {city}.",
                "sources": climate_data.get("sources", []),
                "tables": climate_data.get("tables", []),
                "confidence": 0.0,
                "needs_clarification": True,
                "questions": ["Уточните город или проверьте климатические таблицы."],
                "query_type": "calculation",
            }

        source = climate_data.get("answer", "")
        t_ot_match = re.search(r"t_от\s*=\s*([-+]?\d+[.,]?\d*)", source)
        t_ot = float(t_ot_match.group(1).replace(",", ".")) if t_ot_match else -8.0

        t_n = t_ot
        t_v = 20.0
        delta_t = t_v - t_n
        q_loss = (area * delta_t) / resistance

        answer = (
            f"🔥 **Теплопотери через ограждение для {city} = {q_loss:.2f} Вт**\n\n"
            f"📊 **Исходные данные:**\n"
            f"- Площадь A = {area:.2f} м²\n"
            f"- Сопротивление R = {resistance:.3f} м²·°C/Вт\n"
            f"- t_в = {t_v:.1f} °C\n"
            f"- t_н = {t_n:.1f} °C (из климатических данных)\n"
            f"- Δt = {delta_t:.1f} °C\n\n"
            f"🔢 **Расчёт:** ({area:.2f} × {delta_t:.1f}) / {resistance:.3f} = {q_loss:.2f} Вт\n\n"
            f"📚 **Источник:** {climate_data.get('sources', [{}])[0].get('doc_name', 'Климатическая таблица')}"
        )

        return {
            "answer": answer,
            "sources": climate_data.get("sources", []),
            "tables": climate_data.get("tables", []),
            "confidence": 0.85,
            "needs_clarification": False,
            "questions": [],
            "query_type": "calculation",
            "formulas": [{
                "raw": "Q = (A × Δt) / R",
                "name": "Теплопотери через ограждение",
                "source": "СП 50.13330",
            }],
        }


def patch_app_with_table_calculator():
    """Заглушка для совместимости с app.py."""
    pass