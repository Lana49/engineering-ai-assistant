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
from typing import Any

from core.table_calculator import TableCalculator


@dataclass(slots=True)
class Material:
    """Материал с теплофизическими свойствами."""
    name: str
    lambda_value: float
    density: float = 0.0
    specific_heat: float | None = None
    source: str | None = None


@dataclass(slots=True)
class CityClimate:
    """Климатические данные города."""
    name: str
    t_ot: float
    z_ot: int
    t_n: float
    source: str = ""


class FormulaEngine:
    """
    Безопасный движок инженерных расчётов без eval.
    """

    def __init__(self, qa_system=None):
        self.qa_system = qa_system
        self.reasoning_steps: list[str] = []
        self.materials: dict[str, Material] = {}
        self.cities: dict[str, CityClimate] = {}
        self._material_cache: dict[str, Material] = {}
        self._city_cache: dict[str, CityClimate] = {}
        self._table_calculator = None
        self._on_city_not_found = None
        self._on_material_not_found = None

        self._load_cache()
        self.formulas = self._init_formulas()
        self.city_list = sorted([
            "москва", "санкт-петербург", "новосибирск", "екатеринбург",
            "казань", "нижний новгород", "челябинск", "омск", "самара",
            "ростов-на-дону", "уфа", "красноярск", "пермь", "воронеж",
            "волгоград", "краснодар", "тюмень"
        ], key=len, reverse=True)

    def _init_formulas(self) -> dict[str, dict[str, Any]]:
        """База формул с безопасными хендлерами."""
        return {
            "gsop": {
                "id": "gsop",
                "name": "ГСОП",
                "expression": "(t_в - t_от) × z_от",
                "description": "Расчёт градусо-суток отопительного периода.",
                "legend": [
                    "t_в — внутренняя температура, °C",
                    "t_от — средняя температура отопительного периода, °C",
                    "z_от — продолжительность отопительного периода, сут",
                ],
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
                "legend": [
                    "L — расход воздуха, м³/ч",
                    "t_в — внутренняя температура, °C",
                    "t_н — наружная температура, °C",
                ],
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
                "legend": [
                    "A — площадь конструкции, м²",
                    "Δt — разность температур внутри и снаружи, °C",
                    "R — сопротивление теплопередаче, м²·°C/Вт",
                ],
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
                "legend": [
                    "δ — толщина слоя, м",
                    "λ — коэффициент теплопроводности, Вт/(м·°C)",
                ],
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
                "legend": [
                    "R_тр — требуемое сопротивление теплопередаче, м²·°C/Вт",
                    "λ — коэффициент теплопроводности материала, Вт/(м·°C)",
                ],
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
                "legend": [
                    "Q — общий тепловой поток, Вт",
                    "L — длина участка, м",
                ],
                "source": "СП 61.13330",
                "unit": "Вт/м",
                "required_params": ["Q", "L"],
                "aliases": ["удельный поток", "тепловой поток", "поток на метр"],
                "handler": self._calc_pipe_surface_heat_flux
            }
        }

    @staticmethod
    def _get_cache_path() -> Path:
        cache_dir = Path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "formula_engine_cache.json"

    def _load_cache(self) -> None:
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return

        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))

            for key, mat_data in data.get("materials", {}).items():
                material = Material(
                    name=mat_data["name"],
                    lambda_value=float(mat_data["lambda_value"]),
                    density=float(mat_data.get("density", 0.0)),
                    specific_heat=mat_data.get("specific_heat"),
                    source=mat_data.get("source"),
                )
                self.materials[key] = material
                self._material_cache[key] = material

            for key, city_data in data.get("cities", {}).items():
                city = CityClimate(
                    name=city_data["name"],
                    t_ot=float(city_data["t_ot"]),
                    z_ot=int(city_data["z_ot"]),
                    t_n=float(city_data.get("t_n", city_data["t_ot"] - 20)),
                    source=city_data.get("source", ""),
                )
                self.cities[key] = city
                self._city_cache[key] = city

        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            self.materials = {}
            self.cities = {}
            self._material_cache = {}
            self._city_cache = {}

    def _save_cache(self) -> None:
        try:
            payload = {
                "materials": {
                    key: {
                        "name": mat.name,
                        "lambda_value": mat.lambda_value,
                        "density": mat.density,
                        "specific_heat": mat.specific_heat,
                        "source": mat.source,
                    }
                    for key, mat in self.materials.items()
                },
                "cities": {
                    key: {
                        "name": city.name,
                        "t_ot": city.t_ot,
                        "z_ot": city.z_ot,
                        "t_n": city.t_n,
                        "source": city.source,
                    }
                    for key, city in self.cities.items()
                },
            }
            self._get_cache_path().write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except OSError:
            pass

    @staticmethod
    def _extract_number(text: str, patterns: list[str]) -> float | None:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1).replace(",", "."))
                except ValueError:
                    continue
        return None

    def _extract_city_from_text(self, text: str) -> str | None:
        text_lower = text.lower()
        for city in self.city_list:
            if city in text_lower:
                return city
        return None

    def _extract_parameters_from_text(self, text: str) -> dict[str, float]:
        t = text.lower().replace(",", ".")
        params: dict[str, float] = {}

        patterns = {
            "A": [r"(?:a|площадь)\s*[:=]?\s*(-?\d+(?:\.\d+)?)"],
            "R": [r"(?:\br\b|сопротивление)\s*[:=]?\s*(-?\d+(?:\.\d+)?)"],
            "t_v": [r"(?:tv|tв|внутренняя(?: температура)?)\s*[:=]?\s*(-?\d+(?:\.\d+)?)"],
            "t_n": [r"(?:tn|tн|наружная(?: температура)?)\s*[:=]?\s*(-?\d+(?:\.\d+)?)"],
            "t_ot": [r"(?:tot|tот|средняя температура отопительного периода)\s*[:=]?\s*(-?\d+(?:\.\d+)?)"],
            "z_ot": [r"(?:zot|zот|продолжительность отопительного периода)\s*[:=]?\s*(-?\d+(?:\.\d+)?)"],
            "L": [r"(?:\bl\b|расход воздуха|длина)\s*[:=]?\s*(-?\d+(?:\.\d+)?)"],
            "delta": [r"(?:delta|толщина|δ)\s*[:=]?\s*(-?\d+(?:\.\d+)?)"],
            "lambda_value": [r"(?:lambda|λ|теплопроводность)\s*[:=]?\s*(-?\d+(?:\.\d+)?)"],
            "R_tr": [r"(?:rtr|rтр|требуемое сопротивление)\s*[:=]?\s*(-?\d+(?:\.\d+)?)"],
            "Q": [r"(?:\bq\b|тепловой поток)\s*[:=]?\s*(-?\d+(?:\.\d+)?)"],
            "delta_t": [r"(?:deltat|Δt|разность температур)\s*[:=]?\s*(-?\d+(?:\.\d+)?)"],
        }

        for key, pats in patterns.items():
            value = self._extract_number(t, pats)
            if value is not None:
                params[key] = value

        return params

    def _detect_formula_key(self, query: str) -> str | None:
        query_lower = query.lower()

        for formula_key, meta in self.formulas.items():
            aliases = meta.get("aliases", [])
            if any(alias in query_lower for alias in aliases):
                return formula_key

        if "гсоп" in query_lower or "градусо" in query_lower:
            return "gsop"
        if "вентиляц" in query_lower or "приточ" in query_lower:
            return "ventilation_heat"
        if "теплопотер" in query_lower or "ограждени" in query_lower:
            return "heat_loss"
        if "сопротивление" in query_lower and "слой" in query_lower:
            return "thermal_resistance_layer"
        if "изоляция" in query_lower or "утеплитель" in query_lower:
            return "required_insulation_thickness"

        return None

    async def answer_calculation(
        self,
        query: str,
        parameters: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Главный async-метод для выполнения расчёта.
        """
        self.reasoning_steps = []
        self.reasoning_steps.append(f"Получен запрос: {query}")

        try:
            extracted = self._extract_parameters_from_text(query)
            merged = {**extracted, **(parameters or {})}

            formula_key = self._detect_formula_key(query)
            if not formula_key:
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
            self.reasoning_steps.append(f"Определена формула: {formula_meta['name']}")

            # Для ГСОП, вентиляции и теплопотерь пытаемся найти данные в таблице
            if formula_key in ["gsop", "ventilation_heat", "heat_loss"]:
                table_result = await self._try_table_calculation(query)
                if table_result:
                    return table_result

            required = formula_meta.get("required_params", [])
            missing = [p for p in required if p not in merged]

            if missing:
                self.reasoning_steps.append(f"Не хватает параметров: {', '.join(missing)}")
                return self._build_missing_params_response(formula_meta, missing)

            handler = formula_meta.get("handler")
            if handler is None:
                raise ValueError(f"Для формулы '{formula_key}' не задан handler")

            result = handler(merged, formula_meta)
            if not result.get("reasoning"):
                result["reasoning"] = self._get_reasoning_chain()

            return self._format_result(result, formula_meta)

        except (ValueError, TypeError, ZeroDivisionError) as e:
            self.reasoning_steps.append(f"Ошибка расчёта: {e}")
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

    async def _try_table_calculation(self, query: str) -> dict[str, Any] | None:
        """
        Пытается выполнить расчёт на основе данных из таблицы.
        Поддерживает: ГСОП, вентиляцию, теплопотери.
        """
        import re

        if self._table_calculator is None:
            self._table_calculator = TableCalculator(self.qa_system)

        city = self._extract_city_from_text(query)
        if not city:
            return None

        query_lower = query.lower()

        # Определяем тип расчёта
        is_ventilation = "вентиляц" in query_lower or "расход теплоты" in query_lower or "нагрев воздуха" in query_lower
        is_heat_loss = "теплопотер" in query_lower or "потери тепла" in query_lower or "ограждени" in query_lower

        try:
            if is_ventilation:
                # Ищем расход воздуха
                flow_match = re.search(r'(\d+[.,]?\d*)\s*м³/ч', query_lower)
                if not flow_match:
                    flow_match = re.search(r'расход\s*(\d+[.,]?\d*)', query_lower)
                    if not flow_match:
                        flow_match = re.search(r'L\s*=\s*(\d+[.,]?\d*)', query_lower)

                if flow_match:
                    air_flow = float(flow_match.group(1).replace(',', '.'))
                    result = self._table_calculator.calculate_ventilation_from_table(city, air_flow)
                    if result and result.get("answer"):
                        return result
                return None

            elif is_heat_loss:
                # Ищем площадь и сопротивление
                area_match = re.search(r'площадь\s*(\d+[.,]?\d*)', query_lower)
                res_match = re.search(r'сопротивление\s*(\d+[.,]?\d*)', query_lower)

                if not area_match:
                    area_match = re.search(r'A\s*=\s*(\d+[.,]?\d*)', query_lower)
                if not res_match:
                    res_match = re.search(r'R\s*=\s*(\d+[.,]?\d*)', query_lower)

                if area_match and res_match:
                    area = float(area_match.group(1).replace(',', '.'))
                    resistance = float(res_match.group(1).replace(',', '.'))
                    result = self._table_calculator.calculate_heat_loss_from_table(city, area, resistance)
                    if result and result.get("answer"):
                        return result
                return None

            else:
                # По умолчанию — ГСОП
                result = self._table_calculator.calculate_gsop_from_table(city)
                if result and result.get("answer"):
                    return result
                return None

        except (ImportError, AttributeError, TypeError, ValueError) as e:
            # Логируем ошибку, но не прерываем выполнение
            print(f"⚠️ Ошибка при расчёте из таблицы: {e}")
            return None

    def _get_reasoning_chain(self) -> str:
        if not self.reasoning_steps:
            return "🔍 Цепочка расчёта пуста."
        return "\n".join([f"{i + 1}. {step}" for i, step in enumerate(self.reasoning_steps)])

    def _calc_gsop(self, params: dict[str, float], formula_meta: dict[str, Any]) -> dict[str, Any]:
        t_v = float(params["t_v"])
        t_ot = float(params["t_ot"])
        z_ot = float(params["z_ot"])

        self.reasoning_steps.append(f"Подставлены параметры: t_v={t_v}, t_ot={t_ot}, z_ot={z_ot}")
        result_value = (t_v - t_ot) * z_ot
        self.reasoning_steps.append(f"Вычисление: ({t_v} - {t_ot}) * {z_ot} = {result_value}")

        answer = (
            f"🌍 **ГСОП = {result_value:.2f} {formula_meta['unit']}**\n\n"
            f"📐 **Формула:** {formula_meta['expression']}\n"
            f"📊 **Исходные данные:**\n"
            f"- t_в = {t_v} °C\n"
            f"- t_от = {t_ot} °C\n"
            f"- z_от = {z_ot} сут\n\n"
            f"🔢 **Подстановка:** ({t_v} - {t_ot}) × {z_ot} = {result_value:.2f}\n\n"
            f"📚 **Источник:** {formula_meta['source']}"
        )

        return {
            "answer": answer,
            "result": result_value,
            "source": formula_meta["source"]
        }

    def _calc_ventilation_heat(self, params: dict[str, float], formula_meta: dict[str, Any]) -> dict[str, Any]:
        air_flow = float(params["L"])
        t_v = float(params["t_v"])
        t_n = float(params["t_n"])

        self.reasoning_steps.append(f"Подставлены параметры: L={air_flow}, t_v={t_v}, t_n={t_n}")
        result_value = 0.335 * air_flow * (t_v - t_n)
        self.reasoning_steps.append(f"Вычисление: 0.335 * {air_flow} * ({t_v} - {t_n}) = {result_value}")

        answer = (
            f"💨 **Расход теплоты на вентиляцию = {result_value:.2f} {formula_meta['unit']}**\n\n"
            f"📐 **Формула:** {formula_meta['expression']}\n"
            f"📊 **Исходные данные:**\n"
            f"- L = {air_flow} м³/ч\n"
            f"- t_в = {t_v} °C\n"
            f"- t_н = {t_n} °C\n\n"
            f"🔢 **Подстановка:** 0.335 × {air_flow} × ({t_v} - {t_n}) = {result_value:.2f}\n\n"
            f"📚 **Источник:** {formula_meta['source']}"
        )

        return {
            "answer": answer,
            "result": result_value,
            "source": formula_meta["source"]
        }

    def _calc_heat_loss(self, params: dict[str, float], formula_meta: dict[str, Any]) -> dict[str, Any]:
        area = float(params["A"])
        delta_t = float(params["delta_t"])
        resistance = float(params["R"])

        if resistance == 0:
            raise ValueError("Сопротивление R не может быть равно 0")

        self.reasoning_steps.append(f"Подставлены параметры: A={area}, delta_t={delta_t}, R={resistance}")
        result_value = (area * delta_t) / resistance
        self.reasoning_steps.append(f"Вычисление: {area} * {delta_t} / {resistance} = {result_value}")

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
            "result": result_value,
            "source": formula_meta["source"]
        }

    def _calc_thermal_resistance_layer(self, params: dict[str, float], formula_meta: dict[str, Any]) -> dict[str, Any]:
        delta = float(params["delta"])
        lambda_value = float(params["lambda_value"])

        if lambda_value == 0:
            raise ValueError("Теплопроводность не может быть равна 0")

        self.reasoning_steps.append(f"Подставлены параметры: delta={delta}, lambda={lambda_value}")
        result_value = delta / lambda_value
        self.reasoning_steps.append(f"Вычисление: {delta} / {lambda_value} = {result_value}")

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
            "result": result_value,
            "source": formula_meta["source"]
        }

    def _calc_required_insulation_thickness(self, params: dict[str, float], formula_meta: dict[str, Any]) -> dict[str, Any]:
        r_required = float(params["R_tr"])
        lambda_value = float(params["lambda_value"])

        self.reasoning_steps.append(f"Подставлены параметры: R_tr={r_required}, lambda={lambda_value}")
        result_value = r_required * lambda_value
        result_mm = result_value * 1000
        self.reasoning_steps.append(f"Вычисление: {r_required} * {lambda_value} = {result_value}")

        answer = (
            f"📏 **Требуемая толщина изоляции = {result_value:.3f} м ({result_mm:.1f} мм)**\n\n"
            f"📐 **Формула:** {formula_meta['expression']}\n"
            f"📊 **Исходные данные:**\n"
            f"- R_тр = {r_required} м²·°C/Вт\n"
            f"- λ = {lambda_value} Вт/(м·°C)\n\n"
            f"🔢 **Подстановка:** {r_required} × {lambda_value} = {result_value:.3f} м\n\n"
            f"📚 **Источник:** {formula_meta['source']}"
        )

        return {
            "answer": answer,
            "result": result_value,
            "source": formula_meta["source"]
        }

    def _calc_pipe_surface_heat_flux(self, params: dict[str, float], formula_meta: dict[str, Any]) -> dict[str, Any]:
        heat_flow = float(params["Q"])
        length = float(params["L"])

        if length == 0:
            raise ValueError("Длина не может быть равна 0")

        self.reasoning_steps.append(f"Подставлены параметры: Q={heat_flow}, L={length}")
        result_value = heat_flow / length
        self.reasoning_steps.append(f"Вычисление: {heat_flow} / {length} = {result_value}")

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
            "result": result_value,
            "source": formula_meta["source"]
        }

    @staticmethod
    def _build_error_response(message: str) -> dict[str, Any]:
        return {
            "answer": message,
            "sources": [],
            "tables": [],
            "formulas": [],
            "confidence": 0.0,
            "query_type": "calculation",
            "needs_clarification": False,
            "questions": []
        }

    @staticmethod
    def _build_missing_params_response(
        formula_meta: dict[str, Any],
        missing: list[str]
    ) -> dict[str, Any]:
        required = ", ".join(formula_meta.get("required_params", []))

        answer = (
            f"⚠️ Недостаточно данных для расчёта «{formula_meta['name']}».\n\n"
            f"Нужно указать параметры: {required}.\n\n"
            f"📐 Формула: {formula_meta['expression']}\n"
            f"📚 Источник: {formula_meta['source']}"
        )

        return {
            "answer": answer,
            "sources": [{"doc_name": formula_meta.get("source", "")}],
            "tables": [],
            "formulas": [{
                "raw": formula_meta["expression"],
                "name": formula_meta["name"],
                "source": formula_meta.get("source", ""),
            }],
            "confidence": 0.3,
            "query_type": "calculation",
            "needs_clarification": True,
            "questions": [f"Укажите значения: {', '.join(missing)}"]
        }

    @staticmethod
    def _format_result(result: dict[str, Any], formula_meta: dict[str, Any]) -> dict[str, Any]:
        if "sources" not in result:
            source = result.get("source", formula_meta.get("source", ""))
            result["sources"] = [{"doc_name": source}] if source else []

        if "tables" not in result:
            result["tables"] = []

        if "formulas" not in result:
            result["formulas"] = [{
                "raw": formula_meta["expression"],
                "name": formula_meta["name"],
                "source": formula_meta.get("source", ""),
            }]

        result["confidence"] = result.get("confidence", 0.9)
        result["query_type"] = "calculation"
        result["needs_clarification"] = result.get("needs_clarification", False)
        result["questions"] = result.get("questions", [])

        return result

    def get_available_formulas(self) -> list[dict[str, Any]]:
        """Возвращает список формул для отображения в sidebar."""
        return [
            {
                "id": key,
                "name": meta["name"],
                "expression": meta["expression"],
                "description": meta.get("description", ""),
                "required_params": meta.get("required_params", []),
                "source": meta.get("source", ""),
            }
            for key, meta in self.formulas.items()
        ]

    def get_reasoning_chain(self) -> str:
        """Возвращает цепочку рассуждений."""
        return self._get_reasoning_chain()