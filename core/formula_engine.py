# core/formula_engine.py
"""
Движок инженерных расчётов.

Поддерживает:
- ГСОП (градусо-сутки отопительного периода)
- Расход теплоты на вентиляцию
- Теплопотери через ограждение
- Термическое сопротивление слоя
- Требуемая толщина изоляции
- Удельный тепловой поток
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CalculationResult:
    """Структура результата расчёта."""
    answer: str
    sources: List[Dict[str, Any]]
    formulas: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    confidence: float = 0.0
    needs_clarification: bool = False
    questions: Optional[List[str]] = None
    query_type: str = "calculation"


@dataclass
class ReasoningStep:
    """Шаг цепочки рассуждений."""
    step_id: int
    description: str
    result: Any = None
    confidence: float = 0.0


@dataclass
class Material:
    """Материал с теплофизическими свойствами."""
    name: str
    lambda_value: float
    density: float = 0.0
    specific_heat: Optional[float] = None
    source: Optional[str] = None


@dataclass
class CityClimate:
    """Климатические данные города."""
    name: str
    t_ot: float
    z_ot: int
    t_n: float
    source: str = "Извлечено из документов"


class FormulaEngine:
    """
    Безопасный движок инженерных расчётов без eval.
    Совместим с async/await интерфейсом.
    """

    def __init__(self, qa_system=None):
        self.qa_system = qa_system

        # Инициализация всех атрибутов в __init__
        self.reasoning_steps: List[str] = []
        self.materials: Dict[str, Material] = {}
        self.cities: Dict[str, CityClimate] = {}
        self._material_cache: Dict[str, Material] = {}
        self._city_cache: Dict[str, CityClimate] = {}
        self._table_calculator = None
        self._on_city_not_found = None
        self._on_material_not_found = None

        # Загружаем кэш
        self._load_cache()

        # База формул с хендлерами
        self.formulas = self._init_formulas()

        # Список городов для поиска
        self.city_list = [
            "москва", "санкт-петербург", "новосибирск", "екатеринбург",
            "казань", "нижний новгород", "челябинск", "омск", "самара",
            "ростов-на-дону", "уфа", "красноярск", "пермь", "воронеж",
            "волгоград", "краснодар", "тюмень", "иркутск", "барнаул",
            "владивосток", "хабаровск", "томск", "ярославль", "ижевск",
            "сочи", "астрахань", "тверь", "тула"
        ]

    # ========== ИНИЦИАЛИЗАЦИЯ ФОРМУЛ ==========

    def _init_formulas(self) -> Dict[str, Dict]:
        """База формул с безопасными хендлерами."""
        return {
            "gsop": {
                "id": "gsop",
                "name": "ГСОП",
                "expression": "ГСОП = (t_в - t_от) × z_от",
                "description": "Расчёт градусо-суток отопительного периода.",
                "legend": (
                    "- t_v / t_в — внутренняя температура воздуха, °C\n"
                    "- t_ot / t_от — средняя температура наружного воздуха за отопительный период, °C\n"
                    "- z_ot / z_от — продолжительность отопительного периода, сут\n"
                    "- Результат: °C·сут"
                ),
                "source": "СП 131.13330",
                "unit": "°C·сут",
                "required_params": ["t_v", "t_ot", "z_ot"],
                "aliases": ["гсоп", "градусо-сутки", "градусосутки", "dd"],
                "handler": self._calc_gsop
            },
            "ventilation_heat": {
                "id": "ventilation_heat",
                "name": "Расход теплоты на вентиляцию",
                "expression": "Q_в = 0.335 × L × (t_в - t_н)",
                "description": "Расчёт расхода теплоты на нагрев приточного воздуха.",
                "legend": (
                    "- Q_v / Q_в — расход теплоты, Вт\n"
                    "- L — расход воздуха, м³/ч\n"
                    "- t_v / t_в — температура внутреннего воздуха, °C\n"
                    "- t_n / t_н — температура наружного воздуха, °C\n"
                    "- 0.335 — коэффициент пересчёта"
                ),
                "source": "СП 60.13330",
                "unit": "Вт",
                "required_params": ["L", "t_v", "t_n"],
                "aliases": ["вентиляция", "расход теплоты", "нагрев воздуха", "приточный воздух"],
                "handler": self._calc_ventilation_heat
            },
            "heat_loss": {
                "id": "heat_loss",
                "name": "Теплопотери через ограждение",
                "expression": "Q = (A × Δt) / R",
                "description": "Расчёт теплопотерь через конструкцию по площади и сопротивлению теплопередаче.",
                "legend": (
                    "- Q — теплопотери, Вт\n"
                    "- A — площадь конструкции, м²\n"
                    "- delta_t / Δt — разность температур внутри и снаружи, °C\n"
                    "- R — сопротивление теплопередаче, м²·°C/Вт"
                ),
                "source": "СП 50.13330",
                "unit": "Вт",
                "required_params": ["A", "delta_t", "R"],
                "aliases": ["теплопотери", "потери тепла", "ограждение", "через стену"],
                "handler": self._calc_heat_loss
            },
            "thermal_resistance_layer": {
                "id": "thermal_resistance_layer",
                "name": "Сопротивление слоя",
                "expression": "R = δ / λ",
                "description": "Расчёт сопротивления теплопередаче отдельного слоя материала.",
                "legend": (
                    "- R — сопротивление теплопередаче, м²·°C/Вт\n"
                    "- delta / δ — толщина слоя, м\n"
                    "- lambda_value / λ — коэффициент теплопроводности, Вт/(м·°C)"
                ),
                "source": "СП 50.13330",
                "unit": "м²·°C/Вт",
                "required_params": ["delta", "lambda_value"],
                "aliases": ["сопротивление слоя", "термическое сопротивление", "r слоя"],
                "handler": self._calc_thermal_resistance_layer
            },
            "required_insulation_thickness": {
                "id": "required_insulation_thickness",
                "name": "Требуемая толщина изоляции",
                "expression": "δ = R_тр × λ",
                "description": "Приближённый расчёт требуемой толщины теплоизоляции.",
                "legend": (
                    "- delta / δ — требуемая толщина изоляции, м\n"
                    "- R_tr / R_тр — требуемое сопротивление теплопередаче, м²·°C/Вт\n"
                    "- lambda_value / λ — коэффициент теплопроводности материала, Вт/(м·°C)"
                ),
                "source": "СП 50.13330 / СП 61.13330",
                "unit": "м",
                "required_params": ["R_tr", "lambda_value"],
                "aliases": ["толщина изоляции", "требуемая толщина", "утеплитель", "изоляция"],
                "handler": self._calc_required_insulation_thickness
            },
            "pipe_surface_heat_flux": {
                "id": "pipe_surface_heat_flux",
                "name": "Удельный тепловой поток",
                "expression": "q = Q / L",
                "description": "Расчёт удельного теплового потока на единицу длины.",
                "legend": (
                    "- q — удельный тепловой поток, Вт/м\n"
                    "- Q — общий тепловой поток, Вт\n"
                    "- L — длина участка, м"
                ),
                "source": "СП 61.13330",
                "unit": "Вт/м",
                "required_params": ["Q", "L"],
                "aliases": ["удельный поток", "тепловой поток", "поток на метр"],
                "handler": self._calc_pipe_surface_heat_flux
            }
        }

    # ========== КЭШИРОВАНИЕ ==========

    @staticmethod
    def _get_cache_path() -> Path:
        """Возвращает путь к файлу кэша."""
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / "formula_engine_cache.json"

    def _load_cache(self) -> None:
        """Загружает кэш из файла."""
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for key, mat_data in data.get('materials', {}).items():
                self.materials[key] = Material(
                    name=mat_data['name'],
                    lambda_value=mat_data['lambda'],
                    density=mat_data.get('density', 0),
                    specific_heat=mat_data.get('specific_heat'),
                    source=mat_data.get('source', 'Из кэша')
                )
                self._material_cache[key] = self.materials[key]

            for key, city_data in data.get('cities', {}).items():
                self.cities[key] = CityClimate(
                    name=city_data['name'],
                    t_ot=city_data['t_ot'],
                    z_ot=city_data['z_ot'],
                    t_n=city_data.get('t_n', city_data['t_ot'] - 20),
                    source=city_data.get('source', 'Из кэша')
                )
                self._city_cache[key] = self.cities[key]

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"⚠️ Ошибка загрузки кэша: {e}")

    def _save_cache(self) -> None:
        """Сохраняет кэш в файл."""
        try:
            cache_path = self._get_cache_path()

            data = {'materials': {}, 'cities': {}}

            for key, mat in self.materials.items():
                if mat.source and ("Извлечено" in mat.source or "Пользователь" in mat.source):
                    data['materials'][key] = {
                        'name': mat.name,
                        'lambda': mat.lambda_value,
                        'density': mat.density,
                        'specific_heat': mat.specific_heat,
                        'source': mat.source
                    }

            for key, city in self.cities.items():
                data['cities'][key] = {
                    'name': city.name,
                    't_ot': city.t_ot,
                    'z_ot': city.z_ot,
                    't_n': city.t_n,
                    'source': city.source
                }

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️ Ошибка сохранения кэша: {e}")

    # ========== ИЗВЛЕЧЕНИЕ ДАННЫХ ==========

    @staticmethod
    def _extract_value(text: str, patterns: List[str], convert=float):
        """Извлекает значение из текста по паттернам."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = match.group(1).replace(',', '.')
                    return convert(value)
                except (ValueError, TypeError):
                    continue
        return None

    def _extract_city_from_text(self, text: str) -> Optional[str]:
        """Извлекает название города из текста."""
        text_lower = text.lower()
        for city in sorted(self.city_list, key=len, reverse=True):
            if city in text_lower:
                return city
        return None

    @staticmethod
    def _extract_parameters_from_text(text: str) -> Dict[str, float]:
        """Извлекает числовые параметры из текста."""
        t = text.lower().replace(",", ".")
        params: Dict[str, float] = {}

        patterns = {
            "area": [
                r"площад[ьяи]\s*(?:=|:)?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*м2",
                r"(\d+(?:\.\d+)?)\s*м²",
            ],
            "resistance": [
                r"сопротивлен[^\d]{0,20}(\d+(?:\.\d+)?)",
                r"\br\s*=\s*(\d+(?:\.\d+)?)",
            ],
            "temp_inside": [
                r"внутр[^\d-]{0,20}(-?\d+(?:\.\d+)?)",
                r"tв\s*=\s*(-?\d+(?:\.\d+)?)",
                r"tin\s*=\s*(-?\d+(?:\.\d+)?)",
            ],
            "temp_outside": [
                r"наруж[^\d-]{0,20}(-?\d+(?:\.\d+)?)",
                r"tн\s*=\s*(-?\d+(?:\.\d+)?)",
                r"tout\s*=\s*(-?\d+(?:\.\d+)?)",
            ],
            "volume": [
                r"об[ъь]ем[^\d]{0,20}(\d+(?:\.\d+)?)",
                r"\bv\s*=\s*(\d+(?:\.\d+)?)",
            ],
            "airflow": [
                r"расход[^\d]{0,20}(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*м3/ч",
                r"(\d+(?:\.\d+)?)\s*м³/ч",
                r"\bl\s*=\s*(\d+(?:\.\d+)?)",
            ],
            "air_changes": [
                r"кратност[^\d]{0,20}(\d+(?:\.\d+)?)",
                r"\bn\s*=\s*(\d+(?:\.\d+)?)",
            ],
            "lambda": [
                r"лямбд[^\d]{0,20}(\d+(?:\.\d+)?)",
                r"lambda\s*=\s*(\d+(?:\.\d+)?)",
                r"λ\s*=\s*(\d+(?:\.\d+)?)",
            ],
            "r_required": [
                r"требуем[^\d]{0,20}(\d+(?:\.\d+)?)",
                r"rтр\s*=\s*(\d+(?:\.\d+)?)",
            ],
            "r_existing": [
                r"существующ[^\d]{0,20}(\d+(?:\.\d+)?)",
                r"rсущ\s*=\s*(\d+(?:\.\d+)?)",
            ],
            "temp_ot": [
                r"t_от\s*=\s*(-?\d+(?:\.\d+)?)",
                r"средняя температура[^\d-]{0,20}(-?\d+(?:\.\d+)?)",
            ],
            "z_ot": [
                r"z_от\s*=\s*(\d+)",
                r"продолжительн[^\d]{0,20}(\d+)\s*сут",
            ]
        }

        for key, pats in patterns.items():
            for pat in pats:
                match = re.search(pat, t, re.IGNORECASE)
                if match:
                    try:
                        params[key] = float(match.group(1))
                        break
                    except (ValueError, TypeError):
                        pass

        return params

    # ========== ЦЕПОЧКА РАССУЖДЕНИЙ ==========

    def _reset_reasoning(self) -> None:
        """Сбрасывает цепочку рассуждений."""
        self.reasoning_steps = []

    def _add_reasoning(self, text: str) -> None:
        """Добавляет шаг в цепочку рассуждений."""
        self.reasoning_steps.append(text)

    def get_reasoning_chain(self) -> str:
        """Возвращает цепочку рассуждений."""
        if not self.reasoning_steps:
            return "🔍 Цепочка расчёта пуста."
        return "\n".join([f"{i + 1}. {step}" for i, step in enumerate(self.reasoning_steps)])

    # ========== ОПРЕДЕЛЕНИЕ ТИПА РАСЧЁТА ==========

    def _detect_formula_key(self, query: str) -> Optional[str]:
        """Определяет тип расчёта по запросу."""
        query_lower = query.lower()

        for formula_key, meta in self.formulas.items():
            aliases = meta.get("aliases", [])
            if any(alias in query_lower for alias in aliases):
                return formula_key

        if "гсоп" in query_lower or "градусо" in query_lower:
            return "gsop"

        if "вентиляц" in query_lower or "расход теплоты" in query_lower or "приточ" in query_lower:
            return "ventilation_heat"

        if "теплопотер" in query_lower or "ограждени" in query_lower:
            return "heat_loss"

        if "сопротивление" in query_lower and "слой" in query_lower:
            return "thermal_resistance_layer"

        if "изоляция" in query_lower or "утеплитель" in query_lower:
            return "required_insulation_thickness"

        return None

    # ========== ОСНОВНОЙ МЕТОД РАСЧЁТА ==========

    async def answer_calculation(
        self,
        query: str,
        parameters: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Главный async-метод для выполнения расчёта.
        """
        self._reset_reasoning()
        self._add_reasoning(f"Получен запрос: {query}")

        try:
            parameters = parameters or {}

            # Извлекаем параметры из текста
            extracted_params = self._extract_parameters_from_text(query)
            merged_params = {**extracted_params, **parameters}

            # Определяем тип расчёта
            formula_key = self._detect_formula_key(query)

            if not formula_key:
                self._add_reasoning("Формула не распознана")
                return self._build_error_response(
                    "⚠️ Не удалось определить тип расчёта.\n\n"
                    "Поддерживаемые расчёты:\n"
                    "- ГСОП (градусо-сутки)\n"
                    "- Расход теплоты на вентиляцию\n"
                    "- Теплопотери через ограждение\n"
                    "- Термическое сопротивление слоя\n"
                    "- Требуемая толщина изоляции\n"
                    "- Удельный тепловой поток"
                )

            formula_meta = self.formulas[formula_key]
            self._add_reasoning(f"Определена формула: {formula_meta['name']}")

            # Проверяем наличие обязательных параметров
            required = formula_meta.get("required_params", [])
            missing = [p for p in required if p not in merged_params]

            if missing:
                self._add_reasoning(f"Не хватает параметров: {', '.join(missing)}")
                return self._build_missing_params_response(formula_meta, missing)

            # Выполняем расчёт
            handler = formula_meta.get("handler")
            if handler is None:
                raise ValueError(f"Для формулы '{formula_key}' не задан handler")

            # Пробуем табличный расчёт для ГСОП
            if formula_key == "gsop":
                table_result = await self._try_table_calculation(query)
                if table_result:
                    return table_result

            result = handler(merged_params, formula_meta)

            if not result.get("reasoning"):
                result["reasoning"] = self.get_reasoning_chain()

            return self._format_result(result, formula_meta)

        except (ValueError, TypeError, ZeroDivisionError) as e:
            self._add_reasoning(f"Ошибка расчёта: {e}")
            return {
                "answer": f"❌ Ошибка расчёта: {e}",
                "sources": [],
                "tables": [],
                "formulas": [],
                "confidence": 0.0,
                "query_type": "calculation",
                "needs_clarification": False,
                "questions": []
            }

    async def _try_table_calculation(
        self,
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """Пробует выполнить расчёт через табличный калькулятор."""
        city = self._extract_city_from_text(query)

        if not city:
            return None

        try:
            from core.table_calculator import TableCalculator

            if self._table_calculator is None:
                self._table_calculator = TableCalculator(self.qa_system)

            table_result = self._table_calculator.calculate_gsop_from_table(city)

            if isinstance(table_result, dict) and table_result.get("answer"):
                return table_result

        except (ImportError, AttributeError, ValueError) as e:
            self._add_reasoning(f"Табличный расчёт не удался: {e}")

        return None

    @staticmethod
    def _build_error_response(message: str) -> Dict[str, Any]:
        """Создаёт ответ с ошибкой."""
        return {
            "answer": message,
            "sources": [],
            "tables": [],
            "formulas": [],
            "confidence": 0.0,
            "query_type": "calculation",
            "needs_clarification": True,
            "questions": ["Уточните, какой именно расчёт требуется."]
        }

    @staticmethod
    def _build_missing_params_response(
        formula_meta: Dict[str, Any],
        missing: List[str]
    ) -> Dict[str, Any]:
        """Создаёт ответ с сообщением о недостающих параметрах."""
        required = formula_meta.get("required_params", [])

        answer = (
            f"⚠️ Недостаточно данных для расчёта «{formula_meta['name']}».\n\n"
            f"Нужно указать параметры: {', '.join(required)}.\n\n"
            f"📐 Формула: {formula_meta['expression']}\n"
            f"📚 Источник: {formula_meta['source']}"
        )

        return {
            "answer": answer,
            "sources": [{"doc_name": formula_meta['source']}],
            "tables": [],
            "formulas": [{
                "raw": formula_meta['expression'],
                "name": formula_meta['name'],
                "source": formula_meta['source']
            }],
            "confidence": 0.3,
            "query_type": "calculation",
            "needs_clarification": True,
            "questions": [f"Укажите: {', '.join(missing)}"]
        }

    @staticmethod
    def _format_result(result: Dict, formula_meta: Dict) -> Dict[str, Any]:
        """Форматирует результат расчёта."""
        if "sources" not in result:
            source = result.get("source", formula_meta.get("source", ""))
            result["sources"] = [{"doc_name": source}] if source else []

        if "tables" not in result:
            result["tables"] = []

        if "formulas" not in result:
            result["formulas"] = [{
                "raw": formula_meta["expression"],
                "name": formula_meta["name"],
                "source": formula_meta["source"]
            }]

        result["query_type"] = "calculation"
        result["needs_clarification"] = False
        result["questions"] = []

        return result

    # ========== ХЕНДЛЕРЫ РАСЧЁТОВ ==========

    def _calc_gsop(self, params: dict, formula_meta: dict) -> dict:
        """Расчёт ГСОП."""
        t_v = float(params["t_v"])
        t_ot = float(params.get("t_ot", params.get("temp_ot", 0)))
        z_ot = float(params.get("z_ot", params.get("zot", 0)))

        self._add_reasoning(f"Определены параметры: t_v={t_v}, t_ot={t_ot}, z_ot={z_ot}")
        self._add_reasoning("Применена формула ГСОП = (t_в - t_от) × z_от")

        result_value = (t_v - t_ot) * z_ot

        self._add_reasoning(f"Подстановка: ({t_v} - ({t_ot})) × {z_ot} = {result_value}")

        answer = (
            f"🌍 **ГСОП = {result_value:.2f} {formula_meta['unit']}**\n\n"
            f"📐 **Формула:** {formula_meta['expression']}\n"
            f"📊 **Исходные данные:**\n"
            f"- t_в = {t_v} °C\n"
            f"- t_от = {t_ot} °C\n"
            f"- z_от = {z_ot} сут\n\n"
            f"🔢 **Подстановка:** ({t_v} - ({t_ot})) × {z_ot} = {result_value:.2f}\n\n"
            f"📚 **Источник:** {formula_meta['source']}"
        )

        return {
            "answer": answer,
            "params": {"t_v": t_v, "t_ot": t_ot, "z_ot": z_ot},
            "result": result_value,
            "source": formula_meta["source"]
        }

    def _calc_ventilation_heat(self, params: dict, formula_meta: dict) -> dict:
        """Расчёт расхода теплоты на вентиляцию."""
        air_flow = float(params["L"])
        t_v = float(params["t_v"])
        t_n = float(params["t_n"])

        self._add_reasoning(f"Определены параметры: air_flow={air_flow}, t_v={t_v}, t_n={t_n}")
        self._add_reasoning("Применена формула Q_в = 0.335 × L × (t_в - t_н)")

        result_value = 0.335 * air_flow * (t_v - t_n)

        self._add_reasoning(f"Подстановка: 0.335 × {air_flow} × ({t_v} - ({t_n})) = {result_value}")

        answer = (
            f"💨 **Расход теплоты на вентиляцию = {result_value:.2f} {formula_meta['unit']}**\n\n"
            f"📐 **Формула:** {formula_meta['expression']}\n"
            f"📊 **Исходные данные:**\n"
            f"- L = {air_flow} м³/ч\n"
            f"- t_в = {t_v} °C\n"
            f"- t_н = {t_n} °C\n\n"
            f"🔢 **Подстановка:** 0.335 × {air_flow} × ({t_v} - ({t_n})) = {result_value:.2f}\n\n"
            f"📚 **Источник:** {formula_meta['source']}"
        )

        return {
            "answer": answer,
            "params": {"L": air_flow, "t_v": t_v, "t_n": t_n},
            "result": result_value,
            "source": formula_meta["source"]
        }

    def _calc_heat_loss(self, params: dict, formula_meta: dict) -> dict:
        """Расчёт теплопотерь через ограждение."""
        area = float(params["A"])
        delta_t = float(params["delta_t"])
        resistance = float(params["R"])

        if resistance == 0:
            raise ValueError("Сопротивление R не может быть равно 0")

        self._add_reasoning(f"Определены параметры: area={area}, delta_t={delta_t}, resistance={resistance}")
        self._add_reasoning("Применена формула Q = (A × Δt) / R")

        result_value = (area * delta_t) / resistance

        self._add_reasoning(f"Подстановка: ({area} × {delta_t}) / {resistance} = {result_value}")

        answer = (
            f"🔥 **Теплопотери через ограждение = {result_value:.2f} {formula_meta['unit']}**\n\n"
            f"📐 **Формула:** {formula_meta['expression']}\n"
            f"📊 **Исходные данные:**\n"
            f"- A = {area} м²\n"
            f"- Δt = {delta_t} °C\n"
            f"- R = {resistance} м²·°C/Вт\n\n"
            f"🔢 **Подстановка:** ({area} × {delta_t}) / {resistance} = {result_value:.2f}\n\n"
            f"📚 **Источник:** {formula_meta['source']}"
        )

        return {
            "answer": answer,
            "params": {"A": area, "delta_t": delta_t, "R": resistance},
            "result": result_value,
            "source": formula_meta["source"]
        }

    def _calc_thermal_resistance_layer(self, params: dict, formula_meta: dict) -> dict:
        """Расчёт сопротивления слоя."""
        delta = float(params["delta"])
        lambda_value = float(params["lambda_value"])

        if lambda_value == 0:
            raise ValueError("Теплопроводность не может быть равна 0")

        self._add_reasoning(f"Определены параметры: delta={delta}, lambda={lambda_value}")
        self._add_reasoning("Применена формула R = δ / λ")

        result_value = delta / lambda_value

        self._add_reasoning(f"Подстановка: {delta} / {lambda_value} = {result_value}")

        answer = (
            f"🧱 **Сопротивление слоя = {result_value:.3f} {formula_meta['unit']}**\n\n"
            f"📐 **Формула:** {formula_meta['expression']}\n"
            f"📊 **Исходные данные:**\n"
            f"- δ = {delta} м\n"
            f"- λ = {lambda_value} Вт/(м·°C)\n\n"
            f"🔢 **Подстановка:** {delta} / {lambda_value} = {result_value:.3f}\n\n"
            f"📚 **Источник:** {formula_meta['source']}"
        )

        return {
            "answer": answer,
            "params": {"delta": delta, "lambda_value": lambda_value},
            "result": result_value,
            "source": formula_meta["source"]
        }

    def _calc_required_insulation_thickness(self, params: dict, formula_meta: dict) -> dict:
        """Расчёт требуемой толщины изоляции."""
        r_required = float(params["R_tr"])
        lambda_value = float(params["lambda_value"])

        self._add_reasoning(f"Определены параметры: r_required={r_required}, lambda={lambda_value}")
        self._add_reasoning("Применена формула δ = R_тр × λ")

        result_value = r_required * lambda_value
        result_mm = result_value * 1000

        self._add_reasoning(f"Подстановка: {r_required} × {lambda_value} = {result_value}")

        answer = (
            f"📏 **Требуемая толщина изоляции = {result_value:.3f} м ({result_mm:.0f} мм)**\n\n"
            f"📐 **Формула:** {formula_meta['expression']}\n"
            f"📊 **Исходные данные:**\n"
            f"- R_тр = {r_required} м²·°C/Вт\n"
            f"- λ = {lambda_value} Вт/(м·°C)\n\n"
            f"🔢 **Подстановка:** {r_required} × {lambda_value} = {result_value:.3f} м\n\n"
            f"📚 **Источник:** {formula_meta['source']}"
        )

        return {
            "answer": answer,
            "params": {"R_tr": r_required, "lambda_value": lambda_value},
            "result": result_value,
            "source": formula_meta["source"]
        }

    def _calc_pipe_surface_heat_flux(self, params: dict, formula_meta: dict) -> dict:
        """Расчёт удельного теплового потока."""
        heat_flow = float(params["Q"])
        length = float(params["L"])

        if length == 0:
            raise ValueError("Длина не может быть равна 0")

        self._add_reasoning(f"Определены параметры: heat_flow={heat_flow}, length={length}")
        self._add_reasoning("Применена формула q = Q / L")

        result_value = heat_flow / length

        self._add_reasoning(f"Подстановка: {heat_flow} / {length} = {result_value}")

        answer = (
            f"🔥 **Удельный тепловой поток = {result_value:.2f} {formula_meta['unit']}**\n\n"
            f"📐 **Формула:** {formula_meta['expression']}\n"
            f"📊 **Исходные данные:**\n"
            f"- Q = {heat_flow} Вт\n"
            f"- L = {length} м\n\n"
            f"🔢 **Подстановка:** {heat_flow} / {length} = {result_value:.2f}\n\n"
            f"📚 **Источник:** {formula_meta['source']}"
        )

        return {
            "answer": answer,
            "params": {"Q": heat_flow, "L": length},
            "result": result_value,
            "source": formula_meta["source"]
        }

    # ========== ПУБЛИЧНЫЕ МЕТОДЫ ДЛЯ APP.PY ==========

    def get_available_formulas(self) -> list:
        """Возвращает список формул для отображения в sidebar."""
        return [
            {
                "id": item["id"],
                "name": item["name"],
                "expression": item["expression"],
                "description": item["description"],
                "legend": item.get("legend", ""),
                "source": item.get("source", "")
            }
            for item in self.formulas.values()
        ]

    def get_materials(self) -> List[Dict]:
        """Возвращает список материалов."""
        return [{'key': k, 'name': v.name, 'lambda': v.lambda_value} for k, v in self.materials.items()]

    def get_cities(self) -> List[Dict]:
        """Возвращает список городов с климатическими данными."""
        return [{'key': k, 'name': v.name, 't_ot': v.t_ot, 'z_ot': v.z_ot} for k, v in self.cities.items()]

    def set_city_callback(self, callback):
        """Устанавливает callback для случая, когда город не найден."""
        self._on_city_not_found = callback

    def set_material_callback(self, callback):
        """Устанавливает callback для случая, когда материал не найден."""
        self._on_material_not_found = callback