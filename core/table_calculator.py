# core/table_calculator.py
"""
Модуль для поиска таблиц, извлечения данных и выполнения расчётов.
Поддерживает:
- Поиск климатических таблиц по городу
- Извлечение параметров (t_от, z_от, t_н)
- Расчёт ГСОП на основе данных из таблицы
- Расчёт теплопотерь и вентиляции на основе климатических данных
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from core.table_extractor import TableExtractor, ExtractedTable


@dataclass
class ClimateData:
    """Климатические данные для города"""
    city: str
    t_ot: Optional[float] = None  # Средняя температура отопительного периода
    z_ot: Optional[int] = None    # Продолжительность отопительного периода
    t_n: Optional[float] = None   # Температура наиболее холодной пятидневки
    t_avg: Optional[float] = None # Среднегодовая температура
    source: str = ""
    confidence: float = 0.0


class TableCalculator:
    """
    Поиск таблиц и выполнение расчётов на основе данных из них
    """

    def __init__(self, qa_system=None):
        self.qa_system = qa_system
        self.extractor = TableExtractor()
        self._climate_cache: Dict[str, ClimateData] = {}

    def find_climate_table(self, query: str, city_name: str = None) -> Optional[ExtractedTable]:
        """
        Находит климатическую таблицу по запросу

        Args:
            query: Поисковый запрос
            city_name: Название города (опционально)

        Returns:
            Извлечённая таблица или None
        """
        if not self.qa_system or not self.qa_system.is_ready:
            return None

        # Расширяем запрос для поиска климатических таблиц
        search_queries = [
            query,
            f"{query} СП 131.13330 климат",
            f"{query} температура отопительный период",
            "СП 131.13330 таблица климатические параметры",
            "климатические параметры холодного периода года"
        ]

        if city_name:
            search_queries.insert(0, f"{city_name} климат СП 131.13330")

        all_tables = []

        for search_query in search_queries:
            chunks = self.qa_system.search(search_query, top_k=5)

            for chunk in chunks:
                text = chunk.get('text', '')
                doc_name = chunk.get('doc_name', '')
                tables = self.extractor.extract(text, doc_name)

                for table in tables:
                    if self._is_climate_table(table):
                        all_tables.append(table)

        # Удаляем дубликаты
        seen = set()
        unique_tables = []
        for table in all_tables:
            key = table.raw_text[:100]
            if key not in seen:
                seen.add(key)
                unique_tables.append(table)

        # Сортируем по уверенности
        unique_tables.sort(key=lambda t: t.confidence, reverse=True)

        return unique_tables[0] if unique_tables else None

    def _is_climate_table(self, table: ExtractedTable) -> bool:
        """Проверяет, является ли таблица климатической"""
        climate_keywords = [
            'климат', 'температура', 'отопительный период', 'холодный период',
            'средняя температура', 'продолжительность', 'градусо-сутки',
            'зимний период', 'наружный воздух', 'расчетные параметры',
            't_от', 'z_от', 't_н', 't_в'
        ]

        title_lower = table.title.lower()
        if any(kw in title_lower for kw in climate_keywords):
            return True

        for row in table.rows[:5]:
            row_text = ' '.join(row).lower()
            if any(kw in row_text for kw in climate_keywords):
                return True

        has_city = any(self._is_city_name(cell) for row in table.rows for cell in row)
        has_numbers = any(re.search(r'-?\d+[.,]?\d*', cell) for row in table.rows for cell in row)

        return has_city and has_numbers

    @staticmethod
    def _is_city_name(text: str) -> bool:
        """Проверяет, является ли текст названием города"""
        if not text or len(text) < 2:
            return False

        cities = {
            'москва', 'санкт-петербург', 'новосибирск', 'екатеринбург',
            'нижний новгород', 'казань', 'челябинск', 'омск', 'самара',
            'ростов-на-дону', 'уфа', 'красноярск', 'пермь', 'воронеж',
            'волгоград', 'краснодар', 'сочи', 'владивосток', 'иркутск',
            'тюмень', 'барнаул', 'хабаровск', 'новокузнецк', 'магнитогорск',
            'киев', 'минск', 'алматы', 'ташкент', 'баку', 'ереван', 'тбилиси'
        }

        text_lower = text.lower().strip()
        return text_lower in cities or any(city in text_lower for city in cities)

    def extract_climate_data(self, table: ExtractedTable, city_name: str) -> Optional[ClimateData]:
        """
        Извлекает климатические данные для города из таблицы

        Args:
            table: Таблица с климатическими данными
            city_name: Название города

        Returns:
            ClimateData или None
        """
        if not table or not city_name:
            return None

        cache_key = f"{city_name}_{hash(table.raw_text[:100])}"
        if cache_key in self._climate_cache:
            return self._climate_cache[cache_key]

        city_lower = city_name.lower().strip()
        data = ClimateData(city=city_name, source=table.source)

        # Ищем строку с городом
        found_row = None
        for row in table.rows:
            row_text = ' '.join(row).lower()
            if city_lower in row_text:
                found_row = row
                break

        if not found_row:
            return None

        # Парсим числовые значения из строки
        for cell in found_row:
            cell_clean = cell.replace(',', '.')

            # Поиск температуры
            temp_match = re.search(r'(-?\d+[.,]?\d*)\s*[°С]', cell_clean)
            if temp_match:
                temp = float(temp_match.group(1).replace(',', '.'))
                if data.t_ot is None:
                    data.t_ot = temp
                elif data.t_n is None:
                    data.t_n = temp

            # Поиск продолжительности (сутки)
            days_match = re.search(r'(\d+)\s*сут', cell_clean)
            if days_match:
                data.z_ot = int(days_match.group(1))

            # Поиск обычных чисел
            num_match = re.search(r'(-?\d+[.,]?\d*)', cell_clean)
            if num_match and not temp_match and not days_match:
                num = float(num_match.group(1).replace(',', '.'))
                if data.t_ot is None and -50 < num < 30:
                    data.t_ot = num
                elif data.z_ot is None and 0 < num < 400:
                    data.z_ot = int(num)
                elif data.t_n is None and -60 < num < 10:
                    data.t_n = num

        # Если не нашли t_ot и t_n, пробуем другие строки
        if data.t_ot is None or data.z_ot is None:
            for row in table.rows:
                row_text = ' '.join(row).lower()
                if city_lower in row_text:
                    continue

                for cell in row:
                    cell_lower = cell.lower()
                    if 'средняя' in cell_lower and 'температура' in cell_lower:
                        num = re.search(r'(-?\d+[.,]?\d*)', cell)
                        if num and data.t_ot is None:
                            data.t_ot = float(num.group(1).replace(',', '.'))
                    if 'продолжительность' in cell_lower or 'сут' in cell_lower:
                        num = re.search(r'(\d+)', cell)
                        if num and data.z_ot is None:
                            data.z_ot = int(num.group(1))
                    if 'холодной' in cell_lower and 'пятидневки' in cell_lower:
                        num = re.search(r'(-?\d+[.,]?\d*)', cell)
                        if num and data.t_n is None:
                            data.t_n = float(num.group(1).replace(',', '.'))

        # Вычисляем уверенность
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

    def calculate_gsop_from_table(self, city_name: str, t_v: float = 20.0) -> Dict[str, Any]:
        """
        Находит таблицу, извлекает климатические данные и рассчитывает ГСОП

        Args:
            city_name: Название города
            t_v: Внутренняя температура (по умолчанию 20°C)

        Returns:
            Dict с результатами расчёта
        """
        result = {
            'city': city_name,
            't_v': t_v,
            'success': False,
            'data': None,
            'gsop': None,
            'answer': '',
            'table': None,
            'source': ''
        }

        try:
            table = self.find_climate_table(
                f"климатические данные {city_name}",
                city_name
            )

            if not table:
                result['answer'] = f"❌ Не найдена климатическая таблица для города {city_name}"
                return result

            result['table'] = table
            result['source'] = table.source

            data = self.extract_climate_data(table, city_name)

            if not data or data.confidence < 0.3:
                result['answer'] = (
                    f"⚠️ Не удалось извлечь климатические данные для {city_name}. "
                    f"Уверенность: {data.confidence:.0%}" if data else "Данные не найдены"
                )
                return result

            result['data'] = data

            if data.t_ot is None or data.z_ot is None:
                result['answer'] = (
                    f"⚠️ Недостаточно данных для расчёта ГСОП для {city_name}:\n"
                    f"- t_от: {data.t_ot if data.t_ot is not None else 'не найдено'}\n"
                    f"- z_от: {data.z_ot if data.z_ot is not None else 'не найдено'}"
                )
                return result

            gsop = (t_v - data.t_ot) * data.z_ot

            result['gsop'] = gsop
            result['success'] = True

            answer_lines = [
                f"🌍 **ГСОП для {city_name} = {gsop:.0f} °C·сут**",
                "",
                "📊 **Исходные данные из таблицы:**",
                f"- Город: {city_name}",
                f"- t_в = {t_v} °C",
                f"- t_от = {data.t_ot:.1f} °C",
                f"- z_от = {data.z_ot} сут",
                "",
                "📐 **Формула:** ГСОП = (t_в - t_от) × z_от",
                f"🔢 **Подстановка:** ({t_v} - ({data.t_ot:.1f})) × {data.z_ot} = {gsop:.0f}",
                "",
                f"📚 **Источник:** {data.source}",
                f"✅ Уверенность извлечения данных: {data.confidence:.0%}"
            ]

            if data.t_n is not None:
                answer_lines.append("")
                answer_lines.append(f"📌 Также найдено: t_н = {data.t_n:.1f} °C (температура холодной пятидневки)")

            result['answer'] = '\n'.join(answer_lines)

        except Exception as e:
            result['answer'] = f"❌ Ошибка при расчёте: {e}"

        return result

    def calculate_ventilation_from_table(
        self,
        city_name: str,
        air_flow: float,
        t_v: float = 20.0
    ) -> Dict[str, Any]:
        """
        Находит таблицу и рассчитывает расход теплоты на вентиляцию

        Args:
            city_name: Название города
            air_flow: Расход воздуха (м³/ч)
            t_v: Внутренняя температура

        Returns:
            Dict с результатами расчёта
        """
        result = {
            'city': city_name,
            'air_flow': air_flow,
            't_v': t_v,
            'success': False,
            'gsop_result': None,
            'ventilation': None,
            'answer': ''
        }

        gsop_result = self.calculate_gsop_from_table(city_name, t_v)

        if not gsop_result['success']:
            result['answer'] = gsop_result['answer']
            return result

        data = gsop_result['data']

        if data.t_n is None:
            result['answer'] = (
                f"⚠️ Не найдена температура наружного воздуха t_н для {city_name}. "
                "Необходимо для расчёта вентиляции."
            )
            return result

        q_vent = 0.335 * air_flow * (t_v - data.t_n)

        result['gsop_result'] = gsop_result
        result['ventilation'] = q_vent
        result['success'] = True

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
            f"- ГСОП = {gsop_result['gsop']:.0f} °C·сут"
        ]

        result['answer'] = '\n'.join(answer_lines)

        return result

    def calculate_heat_loss_from_table(
        self,
        city_name: str,
        area: float,
        resistance: float,
        t_v: float = 20.0
    ) -> Dict[str, Any]:
        """
        Находит таблицу и рассчитывает теплопотери через ограждение

        Args:
            city_name: Название города
            area: Площадь конструкции (м²)
            resistance: Сопротивление теплопередаче (м²·°C/Вт)
            t_v: Внутренняя температура

        Returns:
            Dict с результатами расчёта
        """
        result = {
            'city': city_name,
            'area': area,
            'resistance': resistance,
            't_v': t_v,
            'success': False,
            'data': None,
            'heat_loss': None,
            'answer': ''
        }

        try:
            table = self.find_climate_table(
                f"климатические данные {city_name}",
                city_name
            )

            if not table:
                result['answer'] = f"❌ Не найдена климатическая таблица для города {city_name}"
                return result

            data = self.extract_climate_data(table, city_name)

            if not data or data.t_n is None:
                result['answer'] = (
                    f"⚠️ Не найдена температура наружного воздуха t_н для {city_name}"
                )
                return result

            delta_t = t_v - data.t_n
            q_loss = (area * delta_t) / resistance

            result['data'] = data
            result['heat_loss'] = q_loss
            result['success'] = True

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
                f"📚 **Источник:** {data.source}"
            ]

            result['answer'] = '\n'.join(answer_lines)

        except Exception as e:
            result['answer'] = f"❌ Ошибка при расчёте: {e}"

        return result

    def get_cities_from_table(self, query: str = "климатические параметры городов") -> List[str]:
        """
        Извлекает список городов из климатической таблицы

        Returns:
            Список названий городов
        """
        table = self.find_climate_table(query)

        if not table:
            return []

        cities = []
        for row in table.rows:
            for cell in row:
                if self._is_city_name(cell):
                    cities.append(cell.strip())

        return list(set(cities))


# ========== ИНТЕГРАЦИЯ В APP.PY ==========

def patch_app_with_table_calculator():
    """
    Добавляет возможность расчёта на основе таблиц в приложение
    """
    print("✅ TableCalculator готов к использованию")