"""
core/formula_engine.py

Детерминированный движок инженерных расчётов.

Ключевые принципы:
1. Числовые расчёты выполняются детерминированно, без LLM.
2. QueryParser / extract_variables / extract_city — основной источник параметров.
3. Если все обязательные параметры уже есть, расчёт выполняется напрямую.
4. Таблицы используются только для добора климатических данных или справочных значений.
5. Движок совместим с более старой архитектурой проекта:
   - callbacks on_city_not_found / on_material_not_found
   - cache materials / cities
   - helper-методы build_*_response
   - reasoning chain
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


try:
    from core.query_parser import extract_city, extract_variables
except ImportError:
    def extract_city(text: str) -> str | None:
        text = (text or "").strip()
        m = re.search(r"\bдля\s+([А-ЯA-ZЁ][А-ЯA-ZЁа-яa-zё\- ]{1,50})", text)
        return m.group(1).strip() if m else None

    def extract_variables(text: str) -> dict[str, float]:
        aliases = {
            "l": "L",
            "tv": "t_v",
            "t_v": "t_v",
            "tв": "t_v",
            "tn": "t_n",
            "t_n": "t_n",
            "tн": "t_n",
            "tot": "t_ot",
            "t_ot": "t_ot",
            "tот": "t_ot",
            "zot": "z_ot",
            "z_ot": "z_ot",
            "zот": "z_ot",
            "a": "A",
            "r": "R",
            "delta_t": "delta_t",
            "dt": "delta_t",
            "delta": "delta",
            "λ": "lambda_value",
            "lambda": "lambda_value",
            "lambda_value": "lambda_value",
            "r_tr": "R_tr",
            "rtr": "R_tr",
            "q": "Q",
        }
        normalized = (
            (text or "")
            .lower()
            .replace(",", ".")
            .replace("°с", "")
            .replace("°c", "")
            .replace("м³", "м3")
        )
        values: dict[str, float] = {}
        for alias, canonical in aliases.items():
            pattern = rf"(?:\b{re.escape(alias)}\b)\s*=\s*(-?\d+(?:\.\d+)?)"
            m = re.search(pattern, normalized, flags=re.IGNORECASE)
            if m:
                values[canonical] = float(m.group(1))
        return values


try:
    from core.table_calculator import TableCalculator
except ImportError:
    TableCalculator = None  # type: ignore


@dataclass(slots=True)
class Material:
    name: str
    lambda_value: float
    density: float = 0.0
    specific_heat: float | None = None
    source: str = ""


@dataclass(slots=True)
class CityClimate:
    name: str
    t_ot: float
    z_ot: int
    t_n: float
    source: str = ""


@dataclass(slots=True)
class FormulaMeta:
    id: str
    name: str
    expression: str
    description: str
    source: str
    unit: str
    required_params: list[str]
    climate_params: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    handler_name: str = ""


class FormulaEngine:
    """
    Выполняет инженерные расчёты и добирает климатические данные при необходимости.
    """

    CLIMATE_DEPENDENT = frozenset({
        "gsop",
        "ventilation_heat",
        "heat_loss",
    })

    PARAM_LABELS = {
        "L": "L — расход воздуха, м³/ч",
        "t_v": "t_v — температура внутреннего воздуха, °C",
        "t_n": "t_n — температура наружного воздуха, °C",
        "t_ot": "t_ot — средняя температура отопительного периода, °C",
        "z_ot": "z_ot — продолжительность отопительного периода, сут",
        "A": "A — площадь ограждения, м²",
        "delta_t": "delta_t — разность температур, °C",
        "R": "R — сопротивление теплопередаче, м²·°C/Вт",
        "delta": "delta — толщина слоя, м",
        "lambda_value": "lambda_value — теплопроводность λ, Вт/(м·°C)",
        "R_tr": "R_tr — требуемое сопротивление, м²·°C/Вт",
        "Q": "Q — тепловая мощность, Вт",
    }

    CITY_ALIASES_COMMON = {
        "мск": "Москва",
        "москва": "Москва",
        "спб": "Санкт-Петербург",
        "питер": "Санкт-Петербург",
        "санкт петербург": "Санкт-Петербург",
        "екб": "Екатеринбург",
        "екатеринбург": "Екатеринбург",
        "нн": "Нижний Новгород",
    }

    MATERIAL_ALIASES_COMMON = {
        "минвата": "минеральная вата",
        "минеральная вата": "минеральная вата",
        "каменная вата": "минеральная вата",
        "базальтовая вата": "минеральная вата",
        "пеноплекс": "экструдированный пенополистирол",
        "эппс": "экструдированный пенополистирол",
        "xps": "экструдированный пенополистирол",
        "пенополистирол": "пенополистирол",
        "кирпич": "кирпич",
        "бетон": "бетон",
        "газобетон": "газобетон",
    }

    def __init__(self, qa_system: Any = None):
        self.qa_system = qa_system
        self.reasoning_steps: list[str] = []

        self.materials: dict[str, Material] = {}
        self.cities: dict[str, CityClimate] = {}

        self._material_cache: dict[str, Material] = {}
        self._city_cache: dict[str, CityClimate] = {}
        self._table_calculator: Any = None

        self.on_city_not_found: Callable[[str], Any] | None = None
        self.on_material_not_found: Callable[[str], Any] | None = None

        self.formulas: dict[str, dict[str, Any]] = self._init_formulas()

        self.city_list: list[str] = []
        self.city_aliases: dict[str, str] = {}
        self.material_aliases: dict[str, str] = dict(self.MATERIAL_ALIASES_COMMON)

        self._load_cache()
        self.city_list = sorted(self.cities.keys(), key=len, reverse=True)
        self.city_aliases = self._build_city_aliases()

    def _init_formulas(self) -> dict[str, dict[str, Any]]:
        return {
            "gsop": {
                "id": "gsop",
                "name": "ГСОП",
                "expression": "ГСОП = (t_в - t_от) × z_от",
                "description": "Градусо-сутки отопительного периода.",
                "legend": {
                    "t_v": "температура внутреннего воздуха, °C",
                    "t_ot": "средняя температура отопительного периода, °C",
                    "z_ot": "продолжительность отопительного периода, сут",
                },
                "source": "СП 131.13330",
                "unit": "°C·сут",
                "required_params": ["t_v", "t_ot", "z_ot"],
                "climate_params": ["t_ot", "z_ot"],
                "aliases": [
                    "гсоп",
                    "градусо-сутки",
                    "градусосутки",
                    "градусо суток",
                    "dd",
                ],
                "handler": self._calc_gsop,
            },
            "ventilation_heat": {
                "id": "ventilation_heat",
                "name": "Расход теплоты на вентиляцию",
                "expression": "Q_в = 0.335 × L × (t_в - t_н)",
                "description": "Расчёт расхода теплоты на нагрев наружного воздуха.",
                "legend": {
                    "L": "расход воздуха, м³/ч",
                    "t_v": "температура внутреннего воздуха, °C",
                    "t_n": "температура наружного воздуха, °C",
                },
                "source": "СП 60.13330",
                "unit": "Вт",
                "required_params": ["L", "t_v", "t_n"],
                "climate_params": ["t_n"],
                "aliases": [
                    "вентиляция",
                    "вентиляционный",
                    "расход теплоты на вентиляцию",
                    "теплота на вентиляцию",
                    "теплопотери на вентиляцию",
                    "приточный воздух",
                    "приточная вентиляция",
                    "нагрев воздуха",
                ],
                "handler": self._calc_ventilation_heat,
            },
            "heat_loss": {
                "id": "heat_loss",
                "name": "Теплопотери через ограждение",
                "expression": "Q = (A × Δt) / R",
                "description": "Расчёт теплопотерь через ограждающую конструкцию.",
                "legend": {
                    "A": "площадь ограждения, м²",
                    "delta_t": "разность температур, °C",
                    "R": "сопротивление теплопередаче, м²·°C/Вт",
                },
                "source": "СП 50.13330",
                "unit": "Вт",
                "required_params": ["A", "delta_t", "R"],
                "climate_params": ["delta_t"],
                "aliases": [
                    "теплопотери",
                    "потери тепла",
                    "потери теплоты",
                    "тепловые потери",
                    "ограждение",
                    "наружная стена",
                    "потери через стену",
                ],
                "handler": self._calc_heat_loss,
            },
            "thermal_resistance_layer": {
                "id": "thermal_resistance_layer",
                "name": "Сопротивление теплопередаче слоя",
                "expression": "R = δ / λ",
                "description": "Термическое сопротивление однородного слоя материала.",
                "legend": {
                    "delta": "толщина слоя, м",
                    "lambda_value": "теплопроводность λ, Вт/(м·°C)",
                },
                "source": "СП 50.13330",
                "unit": "м²·°C/Вт",
                "required_params": ["delta", "lambda_value"],
                "climate_params": [],
                "aliases": [
                    "сопротивление слоя",
                    "термическое сопротивление",
                    "термическое сопротивление слоя",
                    "r слоя",
                ],
                "handler": self._calc_thermal_resistance,
            },
            "required_insulation_thickness": {
                "id": "required_insulation_thickness",
                "name": "Требуемая толщина изоляции",
                "expression": "δ = R_тр × λ",
                "description": "Оценка требуемой толщины утеплителя по требуемому сопротивлению.",
                "legend": {
                    "R_tr": "требуемое сопротивление, м²·°C/Вт",
                    "lambda_value": "теплопроводность λ, Вт/(м·°C)",
                },
                "source": "СП 50.13330",
                "unit": "м",
                "required_params": ["R_tr", "lambda_value"],
                "climate_params": [],
                "aliases": [
                    "толщина утеплителя",
                    "толщина изоляции",
                    "требуемая толщина",
                    "толщина теплоизоляции",
                ],
                "handler": self._calc_insulation_thickness,
            },
            "pipe_surface_heat_flux": {
                "id": "pipe_surface_heat_flux",
                "name": "Удельный тепловой поток",
                "expression": "q = Q / L",
                "description": "Удельный тепловой поток на единицу длины.",
                "legend": {
                    "Q": "тепловая мощность, Вт",
                    "L": "длина, м",
                },
                "source": "СП 61.13330",
                "unit": "Вт/м",
                "required_params": ["Q", "L"],
                "climate_params": [],
                "aliases": [
                    "удельный поток",
                    "удельный тепловой поток",
                    "тепловой поток трубы",
                    "теплопоток",
                ],
                "handler": self._calc_pipe_heat_flux,
            },
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
                    name=str(mat_data["name"]),
                    lambda_value=float(mat_data["lambda_value"]),
                    density=float(mat_data.get("density", 0.0)),
                    specific_heat=mat_data.get("specific_heat"),
                    source=str(mat_data.get("source", "")),
                )
                self.materials[key] = material
                self._material_cache[key] = material

            for key, city_data in data.get("cities", {}).items():
                city = CityClimate(
                    name=str(city_data["name"]),
                    t_ot=float(city_data["t_ot"]),
                    z_ot=int(city_data["z_ot"]),
                    t_n=float(city_data.get("t_n", city_data["t_ot"])),
                    source=str(city_data.get("source", "")),
                )
                self.cities[key] = city
                self._city_cache[key] = city

        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            self.materials = {}
            self.cities = {}
            self._material_cache = {}
            self._city_cache = {}

    def _save_cache(self) -> None:
        try:
            payload = {
                "materials": {
                    key: asdict(mat)
                    for key, mat in self.materials.items()
                },
                "cities": {
                    key: asdict(city)
                    for key, city in self.cities.items()
                },
            }
            self._get_cache_path().write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _build_city_aliases(self) -> dict[str, str]:
        aliases = dict(self.CITY_ALIASES_COMMON)

        for city_name, city in self.cities.items():
            canonical = city.name or city_name
            lowered = city_name.lower().strip()
            aliases[lowered] = canonical
            aliases[canonical.lower().strip()] = canonical

            normalized = lowered.replace("-", " ").replace("ё", "е")
            aliases[normalized] = canonical

        return aliases

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text or ""
        text = text.lower()
        text = text.replace("ё", "е")
        text = text.replace(",", ".")
        text = text.replace("°с", "")
        text = text.replace("°c", "")
        text = text.replace("м³", "м3")
        text = text.replace("m³", "m3")
        text = text.replace("–", "-").replace("—", "-")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _merge_parameters(
        extracted: dict[str, float],
        provided: dict[str, Any],
    ) -> dict[str, float]:
        result: dict[str, float] = {}

        for key, value in extracted.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[key] = float(value)

        for key, value in provided.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[key] = float(value)

        return result

    def _extract_parameters_from_text(self, text: str) -> dict[str, float]:
        values = dict(extract_variables(text) or {})

        normalized = self._normalize_text(text)

        extra_patterns: list[tuple[str, str]] = [
            (r"\bрасход воздуха\s*=?\s*(-?\d+(?:\.\d+)?)", "L"),
            (r"\bтемпература внутри(?:\s*воздуха)?\s*=?\s*(-?\d+(?:\.\d+)?)", "t_v"),
            (r"\bтемпература внутреннего воздуха\s*=?\s*(-?\d+(?:\.\d+)?)", "t_v"),
            (r"\bтемпература наружного воздуха\s*=?\s*(-?\d+(?:\.\d+)?)", "t_n"),
            (r"\bнаружная температура\s*=?\s*(-?\d+(?:\.\d+)?)", "t_n"),
            (r"\bплощадь\s*=?\s*(-?\d+(?:\.\d+)?)", "A"),
            (r"\bсопротивление\s*=?\s*(-?\d+(?:\.\d+)?)", "R"),
            (r"\bтолщина\s*=?\s*(-?\d+(?:\.\d+)?)", "delta"),
            (r"\bтеплопроводность\s*=?\s*(-?\d+(?:\.\d+)?)", "lambda_value"),
            (r"\bдельта t\s*=?\s*(-?\d+(?:\.\d+)?)", "delta_t"),
            (r"\bразность температур\s*=?\s*(-?\d+(?:\.\d+)?)", "delta_t"),
            (r"\bтребуемое сопротивление\s*=?\s*(-?\d+(?:\.\d+)?)", "R_tr"),
            (r"\bмощность\s*=?\s*(-?\d+(?:\.\d+)?)", "Q"),
        ]

        for pattern, name in extra_patterns:
            if name not in values:
                m = re.search(pattern, normalized, re.IGNORECASE)
                if m:
                    values[name] = float(m.group(1))

        return values

    def _extract_city_from_text(self, text: str) -> str | None:
        extracted = extract_city(text)
        if extracted:
            return self._normalize_city_name(extracted)

        normalized = self._normalize_text(text)

        for alias, canonical in sorted(
            self.city_aliases.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if alias and re.search(rf"\b{re.escape(alias)}\b", normalized):
                return canonical

        m = re.search(
            r"\b(?:для|по|в|город|города)\s+([а-яa-z\- ]{2,50})\b",
            normalized,
        )
        if m:
            return self._normalize_city_name(m.group(1))

        return None

    def _normalize_city_name(self, city: str | None) -> str | None:
        if not city:
            return None

        key = self._normalize_text(city)
        if key in self.city_aliases:
            return self.city_aliases[key]

        for known_city in self.cities.values():
            if self._normalize_text(known_city.name) == key:
                return known_city.name

        words = [w.capitalize() for w in key.split() if w]
        return " ".join(words) if words else None

    def _normalize_material_name(self, name: str | None) -> str | None:
        if not name:
            return None

        key = self._normalize_text(name)
        if key in self.material_aliases:
            return self.material_aliases[key]

        for material in self.materials.values():
            if self._normalize_text(material.name) == key:
                return material.name

        return key

    def _detect_formula_key(
        self,
        query: str,
        params: dict[str, float] | None = None,
    ) -> str | None:
        params = params or {}
        q = self._normalize_text(query)

        candidates: list[tuple[float, str]] = []

        for formula_key, meta in self.formulas.items():
            required = set(meta["required_params"])
            aliases = meta.get("aliases", [])

            parameter_hits = len(required.intersection(params))
            complete = parameter_hits == len(required)

            alias_hits = sum(
                1 for alias in aliases
                if alias and self._normalize_text(alias) in q
            )

            score = parameter_hits * 2.0
            score += alias_hits * 3.0

            if complete:
                score += 10.0

            if formula_key == "ventilation_heat":
                if any(token in q for token in ("вентиляц", "приточ", "воздухообмен")):
                    score += 4.0

            if formula_key == "gsop":
                if "гсоп" in q or "градусо" in q:
                    score += 4.0

            if formula_key == "heat_loss":
                if "теплопотер" in q or "ограждени" in q:
                    score += 4.0

            if formula_key == "thermal_resistance_layer":
                if "сопротивлен" in q and ("слоя" in q or "теплопередаче" in q):
                    score += 4.0

            if formula_key == "required_insulation_thickness":
                if "толщин" in q and ("утепл" in q or "изоляц" in q):
                    score += 4.0

            if formula_key == "pipe_surface_heat_flux":
                if "удельн" in q or "теплопоток" in q:
                    score += 4.0

            if score > 0:
                candidates.append((score, formula_key))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        best_score, best_key = candidates[0]
        return best_key if best_score >= 2.0 else None

    def _get_missing_params(
        self,
        formula_key: str,
        params: dict[str, float],
    ) -> list[str]:
        required = self.formulas[formula_key]["required_params"]
        return [name for name in required if name not in params]

    def can_calculate_directly(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> bool:
        extracted = self._extract_parameters_from_text(query)
        params = self._merge_parameters(extracted, parameters or {})
        formula_key = self._detect_formula_key(query, params)

        if formula_key is None:
            return False

        return not self._get_missing_params(formula_key, params)

    async def answer_calculation(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        entities: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.reasoning_steps = []

        provided_params = dict(parameters or {})
        entities = dict(entities or {})

        extracted_params = self._extract_parameters_from_text(query)
        params = self._merge_parameters(extracted_params, provided_params)

        self.reasoning_steps.append("Параметры извлечены и нормализованы")
        self.reasoning_steps.append(f"Параметры: {params}")

        formula_key = self._detect_formula_key(query, params)
        if formula_key is None:
            return self._error(
                "Не удалось определить тип расчёта.\n\n"
                "Поддерживаются: ГСОП, вентиляция, теплопотери, "
                "сопротивление слоя, толщина утеплителя и удельный тепловой поток."
            )

        meta = self.formulas[formula_key]
        self.reasoning_steps.append(f"Выбрана формула: {meta['name']}")

        missing = self._get_missing_params(formula_key, params)

        if not missing:
            self.reasoning_steps.append("Все обязательные параметры переданы")
            return self._execute_formula(formula_key, params)

        city = entities.get("city") or self._extract_city_from_text(query)
        if city:
            self.reasoning_steps.append(f"Определён город: {city}")

        material_name = self._extract_material_name(query)
        if material_name:
            self.reasoning_steps.append(f"Определён материал: {material_name}")

        params = self._try_enrich_from_material(formula_key, params, material_name)

        missing = self._get_missing_params(formula_key, params)
        if not missing:
            self.reasoning_steps.append("Недостающие параметры добраны из материала")
            return self._execute_formula(formula_key, params)

        climate_params = set(meta.get("climate_params", []))
        missing_set = set(missing)

        can_use_climate_table = (
            city is not None
            and formula_key in self.CLIMATE_DEPENDENT
            and missing_set
            and missing_set.issubset(climate_params)
        )

        if can_use_climate_table:
            self.reasoning_steps.append(
                f"Не хватает климатических параметров {missing}; "
                f"используется таблица для города «{city}»"
            )

            table_result = await self._try_table_calculation(
                formula_key=formula_key,
                city=city,
                params=params,
            )

            if table_result is not None:
                table_result.setdefault("reasoning", "\n".join(self.reasoning_steps))
                return table_result

        self.reasoning_steps.append(f"Недостающие параметры: {missing}")
        return self._missing_params(meta, missing)

    def _execute_formula(
        self,
        formula_key: str,
        params: dict[str, float],
    ) -> dict[str, Any]:
        meta = self.formulas[formula_key]
        handler = meta["handler"]

        try:
            result = handler(params, meta)
            result["reasoning"] = "\n".join(self.reasoning_steps)
            return self._format_result(result, meta)
        except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
            return self._error(f"Ошибка расчёта: {exc}")

    async def _try_table_calculation(
        self,
        formula_key: str,
        city: str,
        params: dict[str, float],
    ) -> dict[str, Any] | None:
        if self.qa_system is None or TableCalculator is None:
            return None

        if self._table_calculator is None:
            self._table_calculator = TableCalculator(self.qa_system)

        try:
            if formula_key == "gsop":
                result = self._table_calculator.calculate_gsop_from_table(
                    city=city,
                    t_v=params.get("t_v", 20.0),
                )

            elif formula_key == "ventilation_heat":
                air_flow = params.get("L")
                if air_flow is None:
                    return None

                result = self._table_calculator.calculate_ventilation_from_table(
                    city=city,
                    air_flow=air_flow,
                    t_v=params.get("t_v", 20.0),
                )

            elif formula_key == "heat_loss":
                area = params.get("A")
                resistance = params.get("R")

                if area is None or resistance is None:
                    return None

                result = self._table_calculator.calculate_heat_loss_from_table(
                    city=city,
                    area=area,
                    resistance=resistance,
                    t_v=params.get("t_v", 20.0),
                )
            else:
                return None

            if not result or result.get("confidence", 0.0) < 0.5:
                return None

            self.reasoning_steps.append("Табличные климатические данные успешно получены")
            result["reasoning"] = "\n".join(self.reasoning_steps)
            result.setdefault("query_type", "calculation")
            result.setdefault("needs_clarification", False)
            result.setdefault("questions", [])
            return result

        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self.reasoning_steps.append(f"Табличный метод недоступен: {exc}")
            return None

    def _extract_material_name(self, text: str) -> str | None:
        normalized = self._normalize_text(text)

        for alias, canonical in sorted(
            self.material_aliases.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if alias and re.search(rf"\b{re.escape(alias)}\b", normalized):
                return canonical

        return None

    def _try_enrich_from_material(
        self,
        formula_key: str,
        params: dict[str, float],
        material_name: str | None,
    ) -> dict[str, float]:
        if not material_name:
            return params

        material = self.get_material(material_name)
        if material is None:
            return params

        updated = dict(params)

        if formula_key in {"thermal_resistance_layer", "required_insulation_thickness"}:
            if "lambda_value" not in updated:
                updated["lambda_value"] = material.lambda_value
                self.reasoning_steps.append(
                    f"Теплопроводность λ взята из данных материала: {material.lambda_value}"
                )

        return updated

    def add_material(
        self,
        name: str,
        lambda_value: float,
        density: float = 0.0,
        specific_heat: float | None = None,
        source: str = "",
    ) -> None:
        key = self._normalize_text(name)
        material = Material(
            name=name,
            lambda_value=float(lambda_value),
            density=float(density),
            specific_heat=specific_heat,
            source=source,
        )
        self.materials[key] = material
        self._material_cache[key] = material
        self.material_aliases[key] = name
        self._save_cache()

    def add_city(
        self,
        name: str,
        t_ot: float,
        z_ot: int,
        t_n: float | None = None,
        source: str = "",
    ) -> None:
        key = self._normalize_text(name)
        city = CityClimate(
            name=name,
            t_ot=float(t_ot),
            z_ot=int(z_ot),
            t_n=float(t_n if t_n is not None else t_ot),
            source=source,
        )
        self.cities[key] = city
        self._city_cache[key] = city
        self.city_aliases[key] = name
        self.city_list = sorted(self.cities.keys(), key=len, reverse=True)
        self._save_cache()

    def get_material(self, name: str | None) -> Material | None:
        canonical = self._normalize_material_name(name)
        if not canonical:
            return None

        key = self._normalize_text(canonical)

        if key in self.materials:
            return self.materials[key]

        if self.on_material_not_found is not None:
            try:
                self.on_material_not_found(canonical)
            except (KeyError, TypeError, ValueError, AttributeError, OSError, json.JSONDecodeError):
                pass

        return None

    def get_city_climate(self, name: str | None) -> CityClimate | None:
        canonical = self._normalize_city_name(name)
        if not canonical:
            return None

        key = self._normalize_text(canonical)

        if key in self.cities:
            return self.cities[key]

        if self.on_city_not_found is not None:
            try:
                self.on_city_not_found(canonical)
            except Exception:
                pass

        return None

    def find_closest_city(self, query_city: str | None) -> str | None:
        if not query_city:
            return None

        normalized = self._normalize_text(query_city)

        if normalized in self.city_aliases:
            return self.city_aliases[normalized]

        for alias, canonical in self.city_aliases.items():
            if normalized in alias or alias in normalized:
                return canonical

        return None

    @staticmethod
    def _calc_gsop(
        p: dict[str, float],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        t_v = p["t_v"]
        t_ot = p["t_ot"]
        z_ot = p["z_ot"]

        if z_ot <= 0:
            raise ValueError("z_ot должно быть больше 0")

        value = (t_v - t_ot) * z_ot

        return {
            "answer": (
                "### ГСОП\n\n"
                f"Подстановка: ({t_v:g} - {t_ot:g}) × {z_ot:g}\n\n"
                f"**Результат: {value:.0f} {meta['unit']}**"
            ),
            "result": value,
            "params": p,
            "source": meta["source"],
        }

    @staticmethod
    def _calc_ventilation_heat(
        p: dict[str, float],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        air_flow = p["L"]
        t_v = p["t_v"]
        t_n = p["t_n"]

        if air_flow < 0:
            raise ValueError("Расход воздуха L не может быть отрицательным")

        value = 0.335 * air_flow * (t_v - t_n)

        return {
            "answer": (
                "### Расход теплоты на вентиляцию\n\n"
                f"Подстановка: Q_в = 0.335 × {air_flow:g} × ({t_v:g} - {t_n:g})\n\n"
                f"**Результат: {value:.1f} Вт**, или **{value / 1000:.3f} кВт**"
            ),
            "result": value,
            "params": p,
            "source": meta["source"],
        }

    @staticmethod
    def _calc_heat_loss(
        p: dict[str, float],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        area = p["A"]
        delta_t = p["delta_t"]
        resistance = p["R"]

        if area <= 0:
            raise ValueError("Площадь A должна быть больше 0")
        if resistance <= 0:
            raise ValueError("Сопротивление R должно быть больше 0")

        value = (area * delta_t) / resistance

        return {
            "answer": (
                "### Теплопотери через ограждение\n\n"
                f"Подстановка: Q = ({area:g} × {delta_t:g}) / {resistance:g}\n\n"
                f"**Результат: {value:.1f} {meta['unit']}**, "
                f"или **{value / 1000:.3f} кВт**"
            ),
            "result": value,
            "params": p,
            "source": meta["source"],
        }

    @staticmethod
    def _calc_thermal_resistance(
        p: dict[str, float],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        delta = p["delta"]
        lambda_value = p["lambda_value"]

        if delta <= 0:
            raise ValueError("Толщина δ должна быть больше 0")
        if lambda_value <= 0:
            raise ValueError("Теплопроводность λ должна быть больше 0")

        value = delta / lambda_value

        return {
            "answer": (
                "### Сопротивление слоя\n\n"
                f"Подстановка: R = {delta:g} / {lambda_value:g}\n\n"
                f"**Результат: {value:.4f} {meta['unit']}**"
            ),
            "result": value,
            "params": p,
            "source": meta["source"],
        }

    @staticmethod
    def _calc_insulation_thickness(
        p: dict[str, float],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        r_required = p["R_tr"]
        lambda_value = p["lambda_value"]

        if r_required <= 0:
            raise ValueError("R_tr должно быть больше 0")
        if lambda_value <= 0:
            raise ValueError("Теплопроводность λ должна быть больше 0")

        value = r_required * lambda_value

        return {
            "answer": (
                "### Требуемая толщина изоляции\n\n"
                f"Подстановка: δ = {r_required:g} × {lambda_value:g}\n\n"
                f"**Результат: {value:.4f} м**, или **{value * 1000:.0f} мм**"
            ),
            "result": value,
            "params": p,
            "source": meta["source"],
        }

    @staticmethod
    def _calc_pipe_heat_flux(
        p: dict[str, float],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        heat_power = p["Q"]
        length = p["L"]

        if length <= 0:
            raise ValueError("Длина L должна быть больше 0")

        value = heat_power / length

        return {
            "answer": (
                "### Удельный тепловой поток\n\n"
                f"Подстановка: q = {heat_power:g} / {length:g}\n\n"
                f"**Результат: {value:.2f} {meta['unit']}**"
            ),
            "result": value,
            "params": p,
            "source": meta["source"],
        }

    def _missing_params(
        self,
        meta: dict[str, Any],
        missing: list[str],
    ) -> dict[str, Any]:
        human_missing = [self.PARAM_LABELS.get(name, name) for name in missing]

        return {
            "answer": (
                f"⚠️ Недостаточно данных для расчёта «{meta['name']}».\n\n"
                "Укажите:\n- "
                + "\n- ".join(human_missing)
                + f"\n\nФормула: {meta['expression']}"
            ),
            "sources": [],
            "tables": [],
            "formulas": [{
                "raw": meta["expression"],
                "name": meta["name"],
                "source": meta["source"],
            }],
            "formula": {
                "raw": meta["expression"],
                "name": meta["name"],
                "source": meta["source"],
            },
            "confidence": 0.2,
            "needs_clarification": True,
            "questions": [f"Укажите: {', '.join(human_missing)}"],
            "query_type": "calculation",
        }

    @staticmethod
    def _format_result(
        result: dict[str, Any],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        formula = {
            "raw": meta["expression"],
            "name": meta["name"],
            "source": meta["source"],
        }

        return {
            "answer": result.get("answer", ""),
            "sources": [{"doc_name": result.get("source", meta["source"])}],
            "tables": result.get("tables", []),
            "formulas": [formula],
            "formula": formula,
            "params": result.get("params", {}),
            "result": result.get("result"),
            "reasoning": result.get("reasoning", ""),
            "confidence": result.get("confidence", 0.95),
            "needs_clarification": False,
            "questions": [],
            "query_type": "calculation",
        }

    @staticmethod
    def _error(message: str) -> dict[str, Any]:
        return {
            "answer": f"❌ {message}",
            "sources": [],
            "tables": [],
            "formulas": [],
            "confidence": 0.0,
            "needs_clarification": True,
            "questions": ["Уточните запрос и исходные данные."],
            "query_type": "calculation",
        }

    def _build_error_response(self, message: str) -> dict[str, Any]:
        return self._error(message)

    def _build_missing_params_response(
        self,
        formula_meta: dict[str, Any],
        missing: list[str],
    ) -> dict[str, Any]:
        return self._missing_params(formula_meta, missing)

    def get_available_formulas(self) -> list[dict[str, Any]]:
        return [
            {
                "id": key,
                "name": meta["name"],
                "expression": meta["expression"],
                "description": meta.get("description", ""),
                "required_params": meta["required_params"],
                "climate_params": meta.get("climate_params", []),
                "source": meta["source"],
                "unit": meta["unit"],
                "aliases": meta.get("aliases", []),
                "legend": meta.get("legend", {}),
            }
            for key, meta in self.formulas.items()
        ]

    def get_formula(self, formula_key: str) -> dict[str, Any] | None:
        return self.formulas.get(formula_key)

    def get_reasoning_chain(self) -> str:
        return "\n".join(self.reasoning_steps)

    def _get_reasoning_chain(self) -> str:
        return self.get_reasoning_chain()

    def explain_formula(self, formula_key: str) -> dict[str, Any]:
        meta = self.formulas.get(formula_key)
        if not meta:
            return self._error(f"Формула «{formula_key}» не найдена.")

        legend_lines = []
        for key, value in meta.get("legend", {}).items():
            legend_lines.append(f"- {key}: {value}")

        answer = (
            f"### {meta['name']}\n\n"
            f"Формула: {meta['expression']}\n\n"
            f"{meta.get('description', '')}\n\n"
            f"{chr(10).join(legend_lines)}\n\n"
            f"Источник: {meta['source']}"
        )

        return {
            "answer": answer,
            "sources": [{"doc_name": meta["source"]}],
            "tables": [],
            "formulas": [{
                "raw": meta["expression"],
                "name": meta["name"],
                "source": meta["source"],
            }],
            "formula": {
                "raw": meta["expression"],
                "name": meta["name"],
                "source": meta["source"],
            },
            "confidence": 0.95,
            "needs_clarification": False,
            "questions": [],
            "query_type": "formula_info",
        }

    def try_calculate(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        entities: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Синхронная совместимость со старым кодом.
        Если проект вызывает FormulaEngine без await, можно использовать этот метод.
        """
        extracted = self._extract_parameters_from_text(query)
        params = self._merge_parameters(extracted, parameters or {})
        formula_key = self._detect_formula_key(query, params)

        if formula_key is None:
            return self._error("Не удалось определить тип расчёта.")

        meta = self.formulas[formula_key]
        missing = self._get_missing_params(formula_key, params)

        if missing:
            return self._missing_params(meta, missing)

        return self._execute_formula(formula_key, params)

    def try_table_calculation(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        entities: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Совместимость со старым API.
        Только определяет возможность табличного пути, но не async.
        Полезно для старых веток AgentLoop.
        """
        provided_params = dict(parameters or {})
        entities = dict(entities or {})

        extracted_params = self._extract_parameters_from_text(query)
        params = self._merge_parameters(extracted_params, provided_params)

        formula_key = self._detect_formula_key(query, params)
        if formula_key is None:
            return None

        meta = self.formulas[formula_key]
        missing = self._get_missing_params(formula_key, params)
        city = entities.get("city") or self._extract_city_from_text(query)

        climate_params = set(meta.get("climate_params", []))
        missing_set = set(missing)

        if (
            city is None
            or formula_key not in self.CLIMATE_DEPENDENT
            or not missing_set
            or not missing_set.issubset(climate_params)
        ):
            return None

        return {
            "formula_key": formula_key,
            "city": city,
            "params": params,
            "missing": missing,
            "query_type": "calculation",
            "can_use_table": True,
        }

    def preload_default_reference_data(self) -> None:
        """
        Можно вызвать при старте приложения, если хочешь заранее
        добавить базовые материалы/города в кэш.
        """
        defaults_materials = [
            ("минеральная вата", 0.040, 50, None, "default"),
            ("экструдированный пенополистирол", 0.032, 35, None, "default"),
            ("пенополистирол", 0.038, 25, None, "default"),
            ("кирпич", 0.700, 1800, None, "default"),
            ("бетон", 1.690, 2400, None, "default"),
            ("газобетон", 0.120, 500, None, "default"),
        ]
        for name, lam, density, cp, source in defaults_materials:
            key = self._normalize_text(name)
            if key not in self.materials:
                self.add_material(name, lam, density, cp, source)

        defaults_cities = [
            ("Москва", -3.1, 214, -3.1, "default"),
            ("Санкт-Петербург", -1.3, 219, -1.3, "default"),
            ("Екатеринбург", -6.0, 230, -6.0, "default"),
            ("Новосибирск", -8.7, 230, -8.7, "default"),
        ]
        for name, t_ot, z_ot, t_n, source in defaults_cities:
            key = self._normalize_text(name)
            if key not in self.cities:
                self.add_city(name, t_ot, z_ot, t_n, source)

    def debug_snapshot(self) -> dict[str, Any]:
        return {
            "materials_count": len(self.materials),
            "cities_count": len(self.cities),
            "formula_count": len(self.formulas),
            "city_aliases_count": len(self.city_aliases),
            "material_aliases_count": len(self.material_aliases),
            "qa_system_attached": self.qa_system is not None,
            "table_calculator_ready": self._table_calculator is not None,
            "reasoning_steps": list(self.reasoning_steps),
        }

    def validate_formula_contracts(self) -> list[str]:
        errors: list[str] = []

        for key, meta in self.formulas.items():
            for field_name in (
                "id",
                "name",
                "expression",
                "source",
                "unit",
                "required_params",
                "aliases",
            ):
                if field_name not in meta:
                    errors.append(f"{key}: missing field {field_name}")

            if "handler" not in meta or not callable(meta["handler"]):
                errors.append(f"{key}: invalid handler")

            if not isinstance(meta.get("required_params", []), list):
                errors.append(f"{key}: required_params must be list")

        return errors

    def search_materials(self, query: str) -> list[dict[str, Any]]:
        normalized = self._normalize_text(query)
        result: list[dict[str, Any]] = []

        for material in self.materials.values():
            name_norm = self._normalize_text(material.name)
            if normalized in name_norm or name_norm in normalized:
                result.append({
                    "name": material.name,
                    "lambda_value": material.lambda_value,
                    "density": material.density,
                    "specific_heat": material.specific_heat,
                    "source": material.source,
                })

        return result[:10]

    def search_cities(self, query: str) -> list[dict[str, Any]]:
        normalized = self._normalize_text(query)
        result: list[dict[str, Any]] = []

        for city in self.cities.values():
            name_norm = self._normalize_text(city.name)
            if normalized in name_norm or name_norm in normalized:
                result.append({
                    "name": city.name,
                    "t_ot": city.t_ot,
                    "z_ot": city.z_ot,
                    "t_n": city.t_n,
                    "source": city.source,
                })

        return result[:10]
