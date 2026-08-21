# core/query_parser.py
"""
Единый парсер инженерных запросов.
Извлекает параметры, город, коды документов.
ЕДИНСТВЕННЫЙ ИСТОЧНИК для всех слоёв.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


NUMBER = r"[-+]?\d+(?:[.,]\d+)?"


@dataclass(slots=True)
class ParsedQuery:
    """Результат парсинга запроса."""
    text: str
    normalized_text: str
    parameters: dict[str, float] = field(default_factory=dict)
    raw_numbers: list[float] = field(default_factory=list)
    city: str | None = None
    document_codes: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    intent: str = "search"


# Алиасы параметров (каноническое имя → варианты написания)
PARAM_ALIASES: dict[str, tuple[str, ...]] = {
    "L": (
        "l",
        "расход воздуха",
        "расход воздушный",
        "воздухообмен",
        "подача воздуха",
    ),
    "t_v": (
        "tv",
        "tв",
        "t_v",
        "t в",
        "температура внутри",
        "внутренняя температура",
        "температура внутреннего воздуха",
    ),
    "t_n": (
        "tn",
        "tн",
        "t_n",
        "t н",
        "температура наружного воздуха",
        "наружная температура",
        "температура снаружи",
    ),
    "t_ot": (
        "tot",
        "tот",
        "t_ot",
        "t от",
        "средняя температура отопительного периода",
        "температура отопительного периода",
    ),
    "z_ot": (
        "zot",
        "zот",
        "z_ot",
        "z от",
        "продолжительность отопительного периода",
        "длительность отопительного периода",
    ),
    "A": (
        "a",
        "площадь",
        "площадь ограждения",
        "площадь стены",
    ),
    "R": (
        "r",
        "сопротивление теплопередаче",
        "термическое сопротивление",
        "сопротивление",
    ),
    "delta_t": (
        "deltat",
        "delta t",
        "дельта t",
        "разность температур",
        "перепад температур",
    ),
    "delta": (
        "delta",
        "дельта",
        "толщина слоя",
        "толщина утеплителя",
        "толщина",
    ),
    "lambda_value": (
        "lambda",
        "λ",
        "лямбда",
        "теплопроводность",
        "коэффициент теплопроводности",
    ),
    "R_tr": (
        "rtr",
        "r_tr",
        "r тр",
        "требуемое сопротивление",
        "нормируемое сопротивление",
    ),
    "Q": (
        "q",
        "тепловая мощность",
        "тепловая нагрузка",
        "тепловой поток",
        "теплопотери",
    ),
}

# Города для климатических данных
KNOWN_CITIES = [
    "москва", "санкт-петербург", "новосибирск", "екатеринбург",
    "казань", "нижний новгород", "челябинск", "омск", "самара",
    "ростов-на-дону", "уфа", "красноярск", "пермь", "воронеж",
    "волгоград", "краснодар", "тюмень"
]


def normalize_engineering_text(text: str) -> str:
    """Нормализует текст для поиска параметров."""
    if not text:
        return ""

    value = text.lower().strip()

    replacements = {
        "³": "3",
        "м³": "м3",
        "°с": "",
        "°c": "",
        "град.с": "",
        "град c": "",
        "×": "x",
        "–": "-",
        "—": "-",
        "＝": "=",
        ",": ".",
        "tв": "t_v",
        "tн": "t_n",
        "tот": "t_ot",
        "zот": "z_ot",
        "rтр": "r_tr",
        "ё": "е",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _alias_pattern(alias: str) -> str:
    """Готовит alias для безопасного поиска."""
    escaped = re.escape(alias)
    return rf"(?<![a-zа-я0-9_]){escaped}(?![a-zа-я0-9_])"


def _value_after_alias(text: str, alias: str) -> float | None:
    """Ищет число после алиаса: L = 500"""
    pattern = rf"{_alias_pattern(alias)}\s*(?:=|:|равно|составляет|-->|-)\s*({NUMBER})"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return None


def _value_before_alias(text: str, alias: str) -> float | None:
    """Ищет число перед алиасом: L 500 или расход 500"""
    pattern = rf"({NUMBER})\s*(?:м3/ч|м2|м|вт|квт|сут)?\s*{_alias_pattern(alias)}"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return None


def extract_variables(text: str) -> dict[str, float]:
    """
    Извлекает именованные инженерные параметры.

    Приоритет: явная запись `имя = число`, затем вариант `число имя`.
    Не назначает безымянные числа параметрам.
    """
    if not text:
        return {}

    normalized = normalize_engineering_text(text)
    values: dict[str, float] = {}

    for canonical_name, aliases in PARAM_ALIASES.items():
        for alias in aliases:
            value = _value_after_alias(normalized, alias)
            if value is None:
                value = _value_before_alias(normalized, alias)

            if value is not None:
                values[canonical_name] = value
                break

    return values


def extract_city(text: str) -> str | None:
    """Извлекает город из текста с учётом словоформ."""
    if not text:
        return None

    normalized = normalize_engineering_text(text)

    # Словоформы для городов
    city_forms = {
        "москва": ["москва", "москвы", "москве", "москву", "москвой"],
        "санкт-петербург": ["санкт-петербург", "санкт петербург", "спб", "питер"],
        "новосибирск": ["новосибирск", "нск"],
        "екатеринбург": ["екатеринбург", "екб"],
        "нижний новгород": ["нижний новгород", "нн", "нижний"],
    }

    # Точное совпадение
    for city in sorted(KNOWN_CITIES, key=len, reverse=True):
        if city in normalized:
            return city

    # По словоформам
    for city, forms in city_forms.items():
        for form in forms:
            if f" {form} " in f" {normalized} ":
                return city

    # По токенам (для составных названий)
    tokens = normalized.split()
    for token in tokens:
        for city in KNOWN_CITIES:
            if token in city or city in token:
                return city

    return None


def extract_document_codes(text: str) -> list[str]:
    """Извлекает коды документов (СП, ГОСТ, СНиП)."""
    if not text:
        return []

    pattern = r"\b(?:сп|гост|снип)\s*[-–—]?\s*\d+(?:\.\d+){1,3}(?:-\d{2,4})?\b"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return list(dict.fromkeys(match.strip().upper() for match in matches))


def extract_keywords(text: str) -> list[str]:
    """Извлекает ключевые слова из текста."""
    if not text:
        return []

    words = re.findall(r"[A-Za-zА-Яа-я0-9._/-]{3,}", text)
    result: list[str] = []
    seen: set[str] = set()

    for w in words:
        key = w.lower()
        if key not in seen and len(key) > 2:
            seen.add(key)
            result.append(key)

    return result[:12]


def detect_intent(text: str) -> str:
    """Определяет тип запроса."""
    if not text:
        return "search"

    q = text.lower()

    # Определения
    def_prefixes = [
        "что такое ", "что значит ", "что означает ",
        "дай определение ", "определение ", "определи ",
        "термин ", "понятие ", "расшифруй "
    ]
    if any(q.startswith(p) for p in def_prefixes):
        return "definition"

    # Расчёты
    calc_triggers = [
        "рассчитай", "расчет", "расчёт", "вычисли", "посчитай",
        "гсоп", "теплопотери", "вентиляция", "тепловой поток",
        "сопротивление теплопередаче", "толщина утеплителя",
    ]
    if any(w in q for w in calc_triggers):
        return "calculation"

    # Сравнения
    if any(w in q for w in ["сравни", "сравнение", "разница", "отличие"]):
        return "comparison"

    # Нормативные требования
    if any(w in q for w in ["требования", "норма", "сп ", "гост", "снип"]):
        return "regulatory"

    return "search"


def parse_query(text: str) -> ParsedQuery:
    """Главный метод парсинга запроса."""
    if not text:
        return ParsedQuery(text=text, normalized_text="")

    normalized = normalize_engineering_text(text)
    parameters = extract_variables(text)
    city = extract_city(text)
    document_codes = extract_document_codes(text)
    keywords = extract_keywords(text)
    intent = detect_intent(text)

    # Извлекаем все числа
    raw_numbers: list[float] = []
    for raw in re.findall(NUMBER, normalized):
        try:
            raw_numbers.append(float(raw.replace(",", ".")))
        except ValueError:
            continue

    return ParsedQuery(
        text=text,
        normalized_text=normalized,
        parameters=parameters,
        raw_numbers=raw_numbers,
        city=city,
        document_codes=document_codes,
        keywords=keywords,
        intent=intent,
    )