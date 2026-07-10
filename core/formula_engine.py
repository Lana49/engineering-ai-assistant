# -*- coding: utf-8 -*-
"""
Интеллектуальный движок для инженерных расчетов.
Без eval - безопасные методы расчёта.
Данные извлекаются из документов через QASystem.
Поддерживает:
- ГСОП (градусо-сутки отопительного периода)
- Расход теплоты на вентиляцию
- Теплопотери через ограждение
- Термическое сопротивление слоя
- Требуемая толщина изоляции
- Удельный тепловой поток
"""

import re
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReasoningStep:
    step_id: int
    description: str
    result: Any = None
    confidence: float = 0.0


@dataclass
class Material:
    name: str
    lambda_value: float
    density: float = 0.0
    specific_heat: Optional[float] = None
    source: Optional[str] = None


@dataclass
class CityClimate:
    name: str
    t_ot: float
    z_ot: int
    t_n: float
    source: str = "Извлечено из документов"


@dataclass
class ExtractedParameter:
    name: str
    value: float
    unit: str
    source: str
    confidence: float = 0.9


class FormulaEngine:
    """
    Безопасный движок инженерных расчётов без eval.
    """

    def __init__(self, qa_system=None):
        self.qa_system = qa_system
        self.reasoning_steps: List[str] = []
        self.extracted_params: Dict[str, ExtractedParameter] = {}
        self.materials: Dict[str, Material] = {}
        self.cities: Dict[str, CityClimate] = {}
        self._material_cache: Dict[str, Material] = {}
        self._city_cache: Dict[str, CityClimate] = {}
        self._on_city_not_found = None
        self._on_material_not_found = None

        # Загружаем кэш
        self._load_cache()

        # База формул с хендлерами
        self.formulas = self._init_formulas()

    # ========== ИНИЦИАЛИЗАЦИЯ ФОРМУЛ ==========

    def _init_formulas(self) -> Dict[str, Dict]:
        """База формул с безопасными хендлерами"""
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

    def _get_cache_path(self) -> Path:
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / "formula_engine_cache.json"

    def _load_cache(self):
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

        except Exception as e:
            print(f"⚠️ Ошибка загрузки кэша: {e}")

    def _save_cache(self):
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

        except Exception as e:
            print(f"⚠️ Ошибка сохранения кэша: {e}")

    # ========== ИЗВЛЕЧЕНИЕ ДАННЫХ ИЗ ДОКУМЕНТОВ ==========

    def _extract_value(self, text: str, patterns: List[str], convert=float):
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = match.group(1).replace(',', '.')
                    return convert(value)
                except (ValueError, TypeError):
                    continue
        return None

    async def _get_city_data(self, city_name: str) -> Optional[CityClimate]:
        if not city_name:
            return None

        if city_name in self._city_cache:
            return self._city_cache[city_name]

        if self.qa_system and self.qa_system.is_ready:
            queries = [
                f"климат {city_name} СП 131.13330",
                f"{city_name} температура отопительный период",
            ]

            for query in queries:
                results = self.qa_system.search(query, top_k=3)
                for chunk in results:
                    text = chunk.get('text', '')
                    doc_name = chunk.get('doc_name', '')

                    t_ot = self._extract_value(text, [
                        r'средняя температура.*?(-?\d+[.,]?\d*)\s*°С',
                        r't_от\s*=\s*(-?\d+[.,]?\d*)'
                    ])

                    z_ot = self._extract_value(text, [
                        r'продолжительность.*?(\d+)\s*сут',
                        r'z_от\s*=\s*(\d+)'
                    ], int)

                    t_n = self._extract_value(text, [
                        r'температура наружного.*?(-?\d+[.,]?\d*)\s*°С',
                        r't_н\s*=\s*(-?\d+[.,]?\d*)'
                    ])

                    if t_ot is not None and z_ot is not None:
                        city_data = CityClimate(
                            name=city_name,
                            t_ot=t_ot,
                            z_ot=int(z_ot),
                            t_n=t_n if t_n is not None else t_ot - 20,
                            source=f"Извлечено из {doc_name}"
                        )
                        self._city_cache[city_name] = city_data
                        self.cities[city_name] = city_data
                        self._save_cache()
                        return city_data

        if self._on_city_not_found:
            return await self._on_city_not_found(city_name)

        return None

    async def _get_material_data(self, material_name: str) -> Optional[Material]:
        if not material_name:
            return None

        if material_name in self._material_cache:
            return self._material_cache[material_name]

        if self.qa_system and self.qa_system.is_ready:
            queries = [
                f"{material_name} теплопроводность",
                f"{material_name} λ Вт/(м·°С)"
            ]

            for query in queries:
                results = self.qa_system.search(query, top_k=3)
                for chunk in results:
                    text = chunk.get('text', '')
                    doc_name = chunk.get('doc_name', '')

                    lambda_val = self._extract_value(text, [
                        rf'{material_name}.*?(\d+[.,]?\d*)\s*Вт/\(м·°С\)',
                        rf'λ\s*=\s*(\d+[.,]?\d*)\s*[Вв]т/\(м·°С\)'
                    ])

                    density = self._extract_value(text, [
                        rf'{material_name}.*?(\d+[.,]?\d*)\s*кг/м³'
                    ])

                    if lambda_val is not None:
                        material = Material(
                            name=material_name,
                            lambda_value=lambda_val,
                            density=density or 0,
                            source=f"Извлечено из {doc_name}"
                        )
                        self._material_cache[material_name] = material
                        self.materials[material_name] = material
                        self._save_cache()
                        return material

        if self._on_material_not_found:
            return await self._on_material_not_found(material_name)

        return None

    # ========== БЕЗОПАСНЫЕ МЕТОДЫ РАСЧЁТА ==========

    def _reset_reasoning(self):
        self.reasoning_steps = []

    def _add_reasoning(self, text: str):
        if not hasattr(self, "reasoning_steps") or self.reasoning_steps is None:
            self.reasoning_steps = []
        self.reasoning_steps.append(text)

    def get_reasoning_chain(self) -> str:
        if not getattr(self, "reasoning_steps", None):
            return "🔍 Цепочка расчёта пуста."
        return "\n".join([f"{i + 1}. {step}" for i, step in enumerate(self.reasoning_steps)])

    def _extract_number_after_keywords(self, query: str, keywords: list):
        query_lower = query.lower()
        for keyword in keywords:
            pattern = rf"{keyword}\s*[:=]?\s*(-?\d+[.,]?\d*)"
            match = re.search(pattern, query_lower)
            if match:
                return float(match.group(1).replace(",", "."))
        return None

    def _extract_all_numbers(self, query: str) -> list:
        raw_numbers = re.findall(r"-?\d+[.,]?\d*", query)
        values = []
        for item in raw_numbers:
            try:
                values.append(float(item.replace(",", ".")))
            except ValueError:
                continue
        return values

    def _normalize_params(self, params: dict) -> dict:
        if not params:
            return {}

        normalized = dict(params)

        alias_map = {
            "t_в": "t_v", "tв": "t_v", "tv": "t_v", "t_v": "t_v",
            "t_от": "t_ot", "tот": "t_ot", "tot": "t_ot", "t_ot": "t_ot",
            "z_от": "z_ot", "zот": "z_ot", "zot": "z_ot", "z_ot": "z_ot",
            "t_н": "t_n", "tн": "t_n", "tn": "t_n", "t_n": "t_n",
            "l": "L", "L": "L",
            "a": "A", "A": "A",
            "r": "R", "R": "R",
            "δ": "delta", "delta": "delta",
            "λ": "lambda_value", "lambda": "lambda_value", "lambda_value": "lambda_value",
            "r_тр": "R_tr", "rтр": "R_tr", "r_tr": "R_tr", "R_tr": "R_tr",
            "q": "Q", "Q": "Q"
        }

        result = {}
        for key, value in normalized.items():
            norm_key = alias_map.get(str(key), str(key))
            result[norm_key] = value

        return result

    def _parse_params_from_query(self, query: str) -> dict:
        params = {}

        extracted_pairs = [
            ("t_v", self._extract_number_after_keywords(query, [r"t_в", r"tв", r"t_v", r"tv", r"внутренняя температура", r"внутри"])),
            ("t_ot", self._extract_number_after_keywords(query, [r"t_от", r"tот", r"t_ot", r"tot", r"средняя температура", r"отопительного периода"])),
            ("z_ot", self._extract_number_after_keywords(query, [r"z_от", r"zот", r"z_ot", r"zot", r"продолжительность", r"сут"])),
            ("L", self._extract_number_after_keywords(query, [r"\bl\b", r"расход воздуха", r"воздух"])),
            ("t_n", self._extract_number_after_keywords(query, [r"t_н", r"tн", r"t_n", r"tn", r"наружная температура", r"снаружи"])),
            ("A", self._extract_number_after_keywords(query, [r"\ba\b", r"площадь"])),
            ("delta_t", self._extract_number_after_keywords(query, [r"delta_t", r"Δt", r"дельта t", r"разность температур"])),
            ("R", self._extract_number_after_keywords(query, [r"\br\b", r"сопротивление"])),
            ("delta", self._extract_number_after_keywords(query, [r"δ", r"delta", r"толщина"])),
            ("lambda_value", self._extract_number_after_keywords(query, [r"λ", r"lambda", r"теплопроводность"])),
            ("R_tr", self._extract_number_after_keywords(query, [r"r_тр", r"r_tr", r"требуемое сопротивление"]))
        ]

        for key, value in extracted_pairs:
            if value is not None:
                params[key] = value

        all_numbers = self._extract_all_numbers(query)
        if all_numbers:
            params["_all_numbers"] = all_numbers

        return params

    def _autofill_gsop_params(self, params: dict) -> dict:
        filled = dict(params)
        nums = filled.get("_all_numbers", [])

        ordered_keys = ["t_v", "t_ot", "z_ot"]
        if len(nums) >= 3:
            for idx, key in enumerate(ordered_keys):
                if key not in filled and idx < len(nums):
                    filled[key] = nums[idx]

        return filled

    def _autofill_ventilation_params(self, params: dict) -> dict:
        filled = dict(params)
        nums = filled.get("_all_numbers", [])

        ordered_keys = ["L", "t_v", "t_n"]
        if len(nums) >= 3:
            for idx, key in enumerate(ordered_keys):
                if key not in filled:
                    filled[key] = nums[idx]

        return filled

    def _autofill_heat_loss_params(self, params: dict) -> dict:
        filled = dict(params)
        nums = filled.get("_all_numbers", [])

        ordered_keys = ["A", "delta_t", "R"]
        if len(nums) >= 3:
            for idx, key in enumerate(ordered_keys):
                if key not in filled:
                    filled[key] = nums[idx]

        return filled

    def _build_missing_params_response(self, formula_meta: dict, params: dict) -> dict:
        required = formula_meta.get("required_params", [])
        missing = [p for p in required if p not in params]

        answer = (
            f"⚠️ Недостаточно данных для расчёта «{formula_meta['name']}».\n\n"
            f"Нужно указать параметры: {', '.join(required)}.\n\n"
            f"📐 Формула: {formula_meta['expression']}\n"
            f"📚 Источник: {formula_meta['source']}"
        )

        if formula_meta.get("legend"):
            answer += f"\n\n**Обозначения:**\n{formula_meta['legend']}"

        self._add_reasoning(f"Не хватает параметров: {', '.join(missing)}")

        return {
            "answer": answer,
            "formula": {
                "raw": formula_meta["expression"],
                "name": formula_meta["name"],
                "source": formula_meta["source"]
            },
            "params": params,
            "result": None,
            "reasoning": self.get_reasoning_chain(),
            "source": formula_meta["source"],
            "sources": [{"doc_name": formula_meta["source"]}],
            "tables": [],
            "formulas": [{
                "raw": formula_meta["expression"],
                "name": formula_meta["name"],
                "source": formula_meta["source"]
            }]
        }

    def _calc_gsop(self, params: dict, formula_meta: dict) -> dict:
        params = self._autofill_gsop_params(params)

        required = formula_meta["required_params"]
        if any(p not in params for p in required):
            return self._build_missing_params_response(formula_meta, params)

        t_v = float(params["t_v"])
        t_ot = float(params["t_ot"])
        z_ot = float(params["z_ot"])

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
            "formula": {
                "raw": formula_meta["expression"],
                "name": formula_meta["name"],
                "source": formula_meta["source"]
            },
            "params": {"t_v": t_v, "t_ot": t_ot, "z_ot": z_ot},
            "result": result_value,
            "reasoning": self.get_reasoning_chain(),
            "source": formula_meta["source"],
            "sources": [{"doc_name": formula_meta["source"]}],
            "tables": [],
            "formulas": [{
                "raw": formula_meta["expression"],
                "name": formula_meta["name"],
                "source": formula_meta["source"]
            }]
        }

    def _calc_ventilation_heat(self, params: dict, formula_meta: dict) -> dict:
        params = self._autofill_ventilation_params(params)

        required = formula_meta["required_params"]
        if any(p not in params for p in required):
            return self._build_missing_params_response(formula_meta, params)

        L = float(params["L"])
        t_v = float(params["t_v"])
        t_n = float(params["t_n"])

        self._add_reasoning(f"Определены параметры: L={L}, t_v={t_v}, t_n={t_n}")
        self._add_reasoning("Применена формула Q_в = 0.335 × L × (t_в - t_н)")

        result_value = 0.335 * L * (t_v - t_n)

        self._add_reasoning(f"Подстановка: 0.335 × {L} × ({t_v} - ({t_n})) = {result_value}")

        answer = (
            f"💨 **Расход теплоты на вентиляцию = {result_value:.2f} {formula_meta['unit']}**\n\n"
            f"📐 **Формула:** {formula_meta['expression']}\n"
            f"📊 **Исходные данные:**\n"
            f"- L = {L} м³/ч\n"
            f"- t_в = {t_v} °C\n"
            f"- t_н = {t_n} °C\n\n"
            f"🔢 **Подстановка:** 0.335 × {L} × ({t_v} - ({t_n})) = {result_value:.2f}\n\n"
            f"📚 **Источник:** {formula_meta['source']}"
        )

        return {
            "answer": answer,
            "formula": {
                "raw": formula_meta["expression"],
                "name": formula_meta["name"],
                "source": formula_meta["source"]
            },
            "params": {"L": L, "t_v": t_v, "t_n": t_n},
            "result": result_value,
            "reasoning": self.get_reasoning_chain(),
            "source": formula_meta["source"],
            "sources": [{"doc_name": formula_meta["source"]}],
            "tables": [],
            "formulas": [{
                "raw": formula_meta["expression"],
                "name": formula_meta["name"],
                "source": formula_meta["source"]
            }]
        }

    def _calc_heat_loss(self, params: dict, formula_meta: dict) -> dict:
        params = self._autofill_heat_loss_params(params)

        required = formula_meta["required_params"]
        if any(p not in params for p in required):
            return self._build_missing_params_response(formula_meta, params)

        A = float(params["A"])
        delta_t = float(params["delta_t"])
        R = float(params["R"])

        if R == 0:
            raise ValueError("Сопротивление R не может быть равно 0")

        self._add_reasoning(f"Определены параметры: A={A}, delta_t={delta_t}, R={R}")
        self._add_reasoning("Применена формула Q = (A × Δt) / R")

        result_value = (A * delta_t) / R

        self._add_reasoning(f"Подстановка: ({A} × {delta_t}) / {R} = {result_value}")

        answer = (
            f"🔥 **Теплопотери через ограждение = {result_value:.2f} {formula_meta['unit']}**\n\n"
            f"📐 **Формула:** {formula_meta['expression']}\n"
            f"📊 **Исходные данные:**\n"
            f"- A = {A} м²\n"
            f"- Δt = {delta_t} °C\n"
            f"- R = {R} м²·°C/Вт\n\n"
            f"🔢 **Подстановка:** ({A} × {delta_t}) / {R} = {result_value:.2f}\n\n"
            f"📚 **Источник:** {formula_meta['source']}"
        )

        return {
            "answer": answer,
            "formula": {
                "raw": formula_meta["expression"],
                "name": formula_meta["name"],
                "source": formula_meta["source"]
            },
            "params": {"A": A, "delta_t": delta_t, "R": R},
            "result": result_value,
            "reasoning": self.get_reasoning_chain(),
            "source": formula_meta["source"],
            "sources": [{"doc_name": formula_meta["source"]}],
            "tables": [],
            "formulas": [{
                "raw": formula_meta["expression"],
                "name": formula_meta["name"],
                "source": formula_meta["source"]
            }]
        }

    def _calc_thermal_resistance_layer(self, params: dict, formula_meta: dict) -> dict:
        required = formula_meta["required_params"]
        if any(p not in params for p in required):
            return self._build_missing_params_response(formula_meta, params)

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
            "formula": {
                "raw": formula_meta["expression"],
                "name": formula_meta["name"],
                "source": formula_meta["source"]
            },
            "params": {"delta": delta, "lambda_value": lambda_value},
            "result": result_value,
            "reasoning": self.get_reasoning_chain(),
            "source": formula_meta["source"],
            "sources": [{"doc_name": formula_meta["source"]}],
            "tables": [],
            "formulas": [{
                "raw": formula_meta["expression"],
                "name": formula_meta["name"],
                "source": formula_meta["source"]
            }]
        }

    def _calc_required_insulation_thickness(self, params: dict, formula_meta: dict) -> dict:
        required = formula_meta["required_params"]
        if any(p not in params for p in required):
            return self._build_missing_params_response(formula_meta, params)

        R_tr = float(params["R_tr"])
        lambda_value = float(params["lambda_value"])

        self._add_reasoning(f"Определены параметры: R_tr={R_tr}, lambda={lambda_value}")
        self._add_reasoning("Применена формула δ = R_тр × λ")

        result_value = R_tr * lambda_value
        result_mm = result_value * 1000

        self._add_reasoning(f"Подстановка: {R_tr} × {lambda_value} = {result_value}")

        answer = (
            f"📏 **Требуемая толщина изоляции = {result_value:.3f} м ({result_mm:.0f} мм)**\n\n"
            f"📐 **Формула:** {formula_meta['expression']}\n"
            f"📊 **Исходные данные:**\n"
            f"- R_тр = {R_tr} м²·°C/Вт\n"
            f"- λ = {lambda_value} Вт/(м·°C)\n\n"
            f"🔢 **Подстановка:** {R_tr} × {lambda_value} = {result_value:.3f} м\n\n"
            f"📚 **Источник:** {formula_meta['source']}"
        )

        return {
            "answer": answer,
            "formula": {
                "raw": formula_meta["expression"],
                "name": formula_meta["name"],
                "source": formula_meta["source"]
            },
            "params": {"R_tr": R_tr, "lambda_value": lambda_value},
            "result": result_value,
            "reasoning": self.get_reasoning_chain(),
            "source": formula_meta["source"],
            "sources": [{"doc_name": formula_meta["source"]}],
            "tables": [],
            "formulas": [{
                "raw": formula_meta["expression"],
                "name": formula_meta["name"],
                "source": formula_meta["source"]
            }]
        }

    def _calc_pipe_surface_heat_flux(self, params: dict, formula_meta: dict) -> dict:
        required = formula_meta["required_params"]
        if any(p not in params for p in required):
            return self._build_missing_params_response(formula_meta, params)

        Q = float(params["Q"])
        L = float(params["L"])

        if L == 0:
            raise ValueError("Длина не может быть равна 0")

        self._add_reasoning(f"Определены параметры: Q={Q}, L={L}")
        self._add_reasoning("Применена формула q = Q / L")

        result_value = Q / L

        self._add_reasoning(f"Подстановка: {Q} / {L} = {result_value}")

        answer = (
            f"🔥 **Удельный тепловой поток = {result_value:.2f} {formula_meta['unit']}**\n\n"
            f"📐 **Формула:** {formula_meta['expression']}\n"
            f"📊 **Исходные данные:**\n"
            f"- Q = {Q} Вт\n"
            f"- L = {L} м\n\n"
            f"🔢 **Подстановка:** {Q} / {L} = {result_value:.2f}\n\n"
            f"📚 **Источник:** {formula_meta['source']}"
        )

        return {
            "answer": answer,
            "formula": {
                "raw": formula_meta["expression"],
                "name": formula_meta["name"],
                "source": formula_meta["source"]
            },
            "params": {"Q": Q, "L": L},
            "result": result_value,
            "reasoning": self.get_reasoning_chain(),
            "source": formula_meta["source"],
            "sources": [{"doc_name": formula_meta["source"]}],
            "tables": [],
            "formulas": [{
                "raw": formula_meta["expression"],
                "name": formula_meta["name"],
                "source": formula_meta["source"]
            }]
        }

    # ========== ОСНОВНОЙ МЕТОД ==========

    def _detect_formula_key(self, query: str):
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

    async def answer_calculation(self, query: str) -> dict:
        self._reset_reasoning()
        self._add_reasoning(f"Получен запрос: {query}")

        try:
            formula_key = self._detect_formula_key(query)
            if not formula_key:
                self._add_reasoning("Формула не распознана")
                return {
                    "answer": (
                        "⚠️ Не удалось определить тип расчёта.\n\n"
                        "Поддерживаемые расчёты:\n"
                        "- ГСОП (градусо-сутки)\n"
                        "- Расход теплоты на вентиляцию\n"
                        "- Теплопотери через ограждение\n"
                        "- Термическое сопротивление слоя\n"
                        "- Требуемая толщина изоляции\n"
                        "- Удельный тепловой поток"
                    ),
                    "formula": None,
                    "params": {},
                    "result": None,
                    "reasoning": self.get_reasoning_chain(),
                    "source": "",
                    "sources": [],
                    "tables": [],
                    "formulas": []
                }

            formula_meta = self.formulas[formula_key]
            self._add_reasoning(f"Определена формула: {formula_meta['name']}")

            params = self._parse_params_from_query(query)
            params = self._normalize_params(params)
            self._add_reasoning(f"Извлечены параметры: {params}")

            handler = formula_meta.get("handler")
            if handler is None:
                raise ValueError(f"Для формулы '{formula_key}' не задан handler")

            result = handler(params, formula_meta)

            if not result.get("reasoning"):
                result["reasoning"] = self.get_reasoning_chain()

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

            return result

        except Exception as e:
            self._add_reasoning(f"Ошибка расчёта: {e}")
            return {
                "answer": f"❌ Ошибка расчёта: {e}",
                "formula": None,
                "params": {},
                "result": None,
                "reasoning": self.get_reasoning_chain(),
                "source": "",
                "sources": [],
                "tables": [],
                "formulas": []
            }

    # ========== ПУБЛИЧНЫЕ МЕТОДЫ ДЛЯ APP.PY ==========

    def get_available_formulas(self) -> list:
        """Возвращает список формул для отображения в sidebar"""
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

    def get_reasoning_chain(self) -> str:
        """Возвращает цепочку рассуждений"""
        if not getattr(self, "reasoning_steps", None):
            return "🔍 Цепочка расчёта пуста."
        return "\n".join([f"{i + 1}. {step}" for i, step in enumerate(self.reasoning_steps)])

    def set_city_callback(self, callback):
        self._on_city_not_found = callback

    def set_material_callback(self, callback):
        self._on_material_not_found = callback

    def get_materials(self) -> List[Dict]:
        return [{'key': k, 'name': v.name, 'lambda': v.lambda_value} for k, v in self.materials.items()]

    def get_cities(self) -> List[Dict]:
        return [{'key': k, 'name': v.name, 't_ot': v.t_ot, 'z_ot': v.z_ot} for k, v in self.cities.items()]