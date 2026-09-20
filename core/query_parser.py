# core/query_parser.py
"""
Единый парсер инженерных запросов.
Извлекает параметры, город, коды документов.
ЕДИНСТВЕННЫЙ ИСТОЧНИК для всех слоёв.
"""

from __future__ import annotations
import logging
import math
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
NUMBER = r"[-+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:e[-+]?\d+)?"


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
    section_refs: list[str] = field(default_factory=list)


# Алиасы параметров (каноническое имя → варианты написания)
PARAM_ALIASES: dict[str, tuple[str, ...]] = {
    "L": (
        "расход воздуха",
        "расход воздушный",
        "воздухообмен",
        "подача воздуха",
        "l_возд",
        "длина трубы", "длина трубопровода", "длина", "l",
    ),
    "t_v": (
        "tv", "tв", "t_v", "t в", "t_в",
        "температура внутри",
        "температура внутри воздуха",
        "внутренняя температура",
        "температура внутреннего воздуха",
    ),
    "t_n": (
        "tn", "tн", "t_n", "t н", "t_н",
        "температура наружного воздуха",
        "наружная температура",
        "температура снаружи",
    ),
    "t_ot": (
        "tot", "tот", "t_ot", "t от", "t_от",
        "средняя температура отопительного периода",
        "температура отопительного периода",
    ),
    "z_ot": (
        "zot", "zот", "z_ot", "z от", "z_от",
        "продолжительность отопительного периода",
        "длительность отопительного периода",
    ),
    "A": (
        "площадь",
        "площадь ограждения",
        "площадь стены",
        "площадь конструкции",
    ),
    "R": (
        "сопротивление теплопередаче",
        "термическое сопротивление",
        "сопротивление",
        "r_0",
        "r0",
    ),
    "delta_t": (
        "delta_t", "delta t", "dt", "δt", "δ t", "дельта t",
        "разность температур",
        "перепад температур",
    ),
    "delta": (
        "delta", "δ", "дельта",
        "толщина слоя",
        "толщина утеплителя",
        "толщина",
    ),
    "lambda_value": (
        "lambda_value", "lambda", "λ", "лямбда",
        "теплопроводность",
        "коэффициент теплопроводности",
    ),
    "R_tr": (
        "rtr", "r_tr", "r тр",
        "требуемое сопротивление",
        "нормируемое сопротивление",
    ),
    "Q": (
        "мощность",
        "тепловая мощность",
        "тепловая нагрузка",
        "тепловой поток",
        "теплопотери",
        "q_тепл",
    ),
}

# Строгие односимвольные алиасы — только в формате "x=число"
STRICT_SINGLE_CHAR_ALIASES: dict[str, str] = {
    "l": "L",
    "a": "A",
    "r": "R",
    "q": "Q",
}

# Города для климатических данных
KNOWN_CITIES = [
    "москва", "санкт-петербург", "новосибирск", "екатеринбург",
    "казань", "нижний новгород", "челябинск", "омск", "самара",
    "ростов-на-дону", "уфа", "красноярск", "пермь", "воронеж",
    "волгоград", "краснодар", "тюмень", "томск"
]


def normalize_engineering_text(text: str) -> str:
    """Нормализует текст для поиска параметров."""
    if not text:
        return ""

    value = text.lower().strip()

    replacements = {
        "³": "3",
        "²": "2",
        "м³": "м3",
        "°с": "",
        "°c": "",
        "град.с": "",
        "град c": "",
        "×": "x",
        "–": "-",
        "—": "-",
        "−": "-",
        "＝": "=",
        "tв": "t_v",
        "tн": "t_n",
        "tот": "t_ot",
        "zот": "z_ot",
        "rтр": "r_tr",
        "ё": "е",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    # Запятая между цифрами — десятичный знак; после «A=10, R=3»
    # она остаётся разделителем параметров, а не превращает 10 в «10.».
    value = re.sub(r"(?<=\d),(?=\d)|(?<!\w),(?=\d)", ".", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _alias_pattern(alias: str) -> str:
    """Готовит alias для безопасного поиска."""
    escaped = re.escape(alias)
    return rf"(?<![a-zа-я0-9_]){escaped}(?![a-zа-я0-9_])"

def _value_near_alias(text: str, alias: str) -> float | None:
    """Ищет число рядом с алиасом в любом порядке и с любым разделителем."""
    match = _parameter_match(text, alias)
    return float(match.group("number")) if match else None


def _parameter_match(text: str, alias: str) -> re.Match[str] | None:
    """Связывает число и единицу с конкретным обозначением параметра."""
    ap = _alias_pattern(normalize_engineering_text(alias))
    unit = r"(?:м3/часа|м3/час|м3/сек|м3/мин|м3/ч|м3/с|л/мин|л/ч|л/с|м2|см2|мм2|квт|вт|мм|см|м|сут|дней|дня|день|°[cс])(?![a-zа-я])"
    # Минус непосредственно перед числом — знак числа, а не разделитель.
    for pattern in (
        rf"{ap}\s*(?:[=:]|составляет|равн[аоы]?|->)?\s*(?P<number>{NUMBER})(?![\d.])\s*(?P<unit>{unit})?",
        rf"(?P<number>{NUMBER})\s*(?P<unit>{unit})?\s+{ap}",
    ):
        found = re.search(pattern, text, flags=re.IGNORECASE)
        if found is not None:
            return found
    return None


def _convert_unit(value: float, canonical: str, unit: str, alias: str) -> float:
    """Переводит явно указанные единицы в единицы формул."""
    if canonical == "delta" or (canonical == "L" and unit in {"мм", "см", "м"}):
        return value * {"мм": 0.001, "см": 0.01, "м": 1.0}.get(unit, 1.0)
    if canonical == "A":
        return value * {"мм2": 0.000001, "см2": 0.0001}.get(unit, 1.0)
    if canonical == "Q" and unit == "квт":
        return value * 1000.0
    if canonical == "L" and "длина" not in alias:
        return value * {"м3/с": 3600.0, "м3/сек": 3600.0, "м3/мин": 60.0,
                        "л/с": 3.6, "л/мин": 0.06, "л/ч": 0.001}.get(unit, 1.0)
    return value


def _value_strict_single_char(text: str, char: str) -> float | None:
    """Строгий поиск для однобуквенных параметров: только 'x=число'."""
    pattern = rf"(?<![a-zа-я0-9_]){re.escape(char)}\s*=\s*({NUMBER})(?![a-zа-я0-9_])"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        try:
            return float(match.group(1).replace(",", "."))
        except ValueError:
            return None
    return None

def extract_variables(text: str) -> dict[str, float]:
    """Извлекает заданные параметры; длинные алиасы защищают от пересечений."""
    if not text:
        return {}

    normalized = normalize_engineering_text(text)
    values: dict[str, float] = {}

    occupied: list[tuple[int, int]] = []
    # Однобуквенные обозначения принимаются с = или :, чтобы буквы из слов
    # не становились параметрами. Явные обозначения имеют приоритет.
    for char, canonical in STRICT_SINGLE_CHAR_ALIASES.items():
        if not re.search(rf"{_alias_pattern(char)}\s*[=:]", normalized):
            continue
        matched = _parameter_match(normalized, char)
        if matched is not None:
            values[canonical] = _convert_unit(
                float(matched.group("number")), canonical,
                matched.group("unit") or "", char,
            )
            occupied.append(matched.span())

    aliases_by_length = sorted(
        ((canonical, alias) for canonical, aliases in PARAM_ALIASES.items() for alias in aliases),
        key=lambda pair: len(pair[1]), reverse=True,
    )
    for canonical_name, alias in aliases_by_length:
        if canonical_name in values:
            continue
        if len(alias) == 1 and alias in STRICT_SINGLE_CHAR_ALIASES:
            continue
        matched = _parameter_match(normalized, alias)
        if matched is None or any(matched.start() < end and matched.end() > start for start, end in occupied):
            continue
        values[canonical_name] = _convert_unit(
            float(matched.group("number")), canonical_name,
            matched.group("unit") or "", alias,
        )
        occupied.append(matched.span())

    return {name: value for name, value in values.items() if math.isfinite(value)}


def get_parameter_issues(text: str, length_kind: str | None = None) -> list[str]:
    """Находит неоднозначное L и несовместимые явно записанные единицы.

    ``length_kind`` задаёт смысл L в выбранной формуле: ``length`` или
    ``air_flow``. Без единицы сохраняются документированные единицы формул.
    """
    normalized = normalize_engineering_text(text)
    length_units = {"м", "см", "мм"}
    flow_units = {"м3/ч", "м3/час", "м3/часа", "м3/с", "м3/сек", "м3/мин", "л/с", "л/мин", "л/ч"}
    issues: list[str] = []
    has_length = False
    has_flow = False
    occupied: list[tuple[int, int]] = []
    aliases = sorted(
        [(canonical, alias) for canonical, variants in PARAM_ALIASES.items() for alias in variants]
        + [(canonical, alias) for alias, canonical in STRICT_SINGLE_CHAR_ALIASES.items()],
        key=lambda item: len(item[1]), reverse=True,
    )
    for canonical, alias in aliases:
        if len(alias) == 1 and alias in STRICT_SINGLE_CHAR_ALIASES:
            if not re.search(rf"{_alias_pattern(alias)}\s*[=:]", normalized):
                continue
        matched = _parameter_match(normalized, alias)
        if matched is None or any(matched.start() < end and matched.end() > start for start, end in occupied):
            continue
        occupied.append(matched.span())
        unit = matched.group("unit") or ""
        raw_unit = re.match(r"\s*([a-zа-я°]+[23]?(?:/[a-zа-я0-9]+)?)", normalized[matched.end("number"):])
        raw = raw_unit.group(1) if raw_unit else ""
        # Неизвестная составная единица не должна превращаться в отсутствие
        # единицы, например L=100 м3/сут -> L=100 м3/ч.
        if "/" in raw and raw != unit and canonical in {"L", "A", "delta", "Q"}:
            issues.append(f"Не поддерживается единица «{raw}» для {canonical}; укажите единицы формулы.")
            continue
        if canonical == "L":
            is_length = "длина" in alias or unit in length_units
            is_flow = (len(alias) > 1 and "длина" not in alias) or unit in flow_units
            has_length |= is_length
            has_flow |= is_flow
            expected = "length" if "длина" in alias else ("air_flow" if len(alias) > 1 else length_kind)
            if expected == "length" and unit and unit not in length_units:
                issues.append("Для длины L нужны метры, сантиметры или миллиметры; расход воздуха не является длиной.")
            elif expected == "air_flow" and unit and unit not in flow_units:
                issues.append("Для расхода воздуха L нужны единицы объёмного расхода, например м³/ч.")
        elif canonical in {"A", "delta", "Q"}:
            allowed = {"A": {"м2", "см2", "мм2"}, "delta": length_units, "Q": {"вт", "квт"}}[canonical]
            if unit and unit not in allowed:
                issues.append(f"Единица «{unit}» несовместима с параметром {canonical}.")
    if has_length and has_flow:
        issues.insert(0, "В условии одновременно заданы длина и расход воздуха, обозначаемые L. Уточните значение L для выбранного расчёта.")
    elif length_kind == "length" and has_flow:
        issues.insert(0, "Для этого расчёта L — длина трубы, а задан расход воздуха. Укажите длину в метрах.")
    elif length_kind == "air_flow" and has_length:
        issues.insert(0, "Для этого расчёта L — расход воздуха, а задана длина. Укажите расход воздуха в м³/ч.")
    return list(dict.fromkeys(issues))

# core/query_parser.py

CITY_FORMS: dict[str, list[str]] = {
    "москва": ["москва", "москвы", "москве", "москву", "москвой"],
    "санкт-петербург": ["санкт-петербург", "санкт петербург", "спб", "питер", "ленинград"],
    "новосибирск": ["новосибирск", "нск", "новосиб"],
    "екатеринбург": ["екатеринбург", "екб", "екат"],
    "нижний новгород": ["нижний новгород", "нн"],
    "ростов-на-дону": ["ростов-на-дону", "ростов на дону", "ростов"],
    "казань": ["казань", "казани"],
    "самара": ["самара", "самары"],
    "уфа": ["уфа", "уфы"],
    "красноярск": ["красноярск", "крас"],
    "пермь": ["пермь", "перми"],
    "воронеж": ["воронеж", "воронежа"],
    "волгоград": ["волгоград", "волгограда"],
    "краснодар": ["краснодар", "краснодара"],
    "тюмень": ["тюмень", "тюмени"],
    "челябинск": ["челябинск", "челябы"],
    "омск": ["омск", "омска"],
    "томск": ["томск", "томска", "томске"],
}


def extract_city(text: str) -> str | None:
    if not text:
        return None

    normalized = normalize_engineering_text(text)

    # 1. Точное совпадение по формам (приоритет — самые длинные)
    for canonical, forms in sorted(
        CITY_FORMS.items(),
        key=lambda kv: max(len(f) for f in kv[1]),
        reverse=True,
    ):
        for form in forms:
            if re.search(rf"\b{re.escape(form)}\b", normalized):
                return canonical

    # 2. Контекстный шаблон "для/в/по <Слово с большой буквы>"
    match = re.search(
        r"\b(?:для|в|по|город[ае]?|г\.)\s+([А-ЯЁ][а-яё\- ]{2,40})\b",
        text,
    )
    if match:
        candidate = match.group(1).strip().lower()
        # отсекаем заведомо неподходящие слова
        stop = {"помещения", "здания", "расчета", "проекта", "стены", "квартиры"}
        if candidate and candidate not in stop:
            return candidate

    return None
STOPWORDS: frozenset[str] = frozenset({
    "что", "как", "какие", "какой", "какая", "какое", "каков",
    "для", "или", "это", "при", "по", "на", "из", "в", "и", "а",
    "но", "не", "то", "ли", "же", "бы", "мы", "вы", "они", "он",
    "она", "оно", "его", "ее", "их", "наш", "ваш", "свой", "все",
    "весь", "вся", "всё", "где", "когда", "почему", "зачем", "чем",
    "кто", "так", "там", "тут", "здесь", "есть", "было", "быть",
    "будет", "может", "можно", "нужно", "надо", "дай", "дайте",
    "покажи", "скажи", "расскажи", "нужен", "нужна", "нужны",
    "рассчитай", "посчитай", "вычисли", "найди", "найдите",
    "подскажи", "подскажите", "объясни", "объясните",
})


def extract_keywords(text: str) -> list[str]:
    if not text:
        return []

    words = re.findall(r"[A-Za-zА-Яа-я0-9._/-]{3,}", text)
    result: list[str] = []
    seen: set[str] = set()

    for w in words:
        key = w.lower()
        if (
            key not in seen
            and key not in STOPWORDS
            and len(key) > 2
        ):
            seen.add(key)
            result.append(key)

    return result[:12]

def extract_document_codes(text: str) -> list[str]:
    """Извлекает коды документов (СП, ГОСТ, СНиП)."""
    if not text:
        return []

    pattern = r"\b(?:сп|гост(?:\s+[рp])?|снип)\s*[-–—]?\s*\d+(?:\.\d+){0,3}(?:[-–—]\d{2,4})?\b"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return list(dict.fromkeys(match.strip().upper() for match in matches))


def extract_section_refs(text: str) -> list[str]:
    """Номера пунктов/таблиц нужны поиску отдельно от кода СП или ГОСТ."""
    return list(dict.fromkeys(re.findall(
        r"\b(?:пункт|пункта|п\.|раздел|раздела|таблиц[аыуе])\s*(\d+(?:\.\d+)*)",
        text, flags=re.IGNORECASE,
    )))


def detect_intent(text: str) -> str:
    """Определяет тип запроса."""
    if not text:
        return "search"

    q = normalize_engineering_text(text)

    # Определения
    def_prefixes = [
        "что такое ", "что значит ", "что означает ",
        "дай определение ", "дайте определение ", "определение ",
        "термин ", "понятие ", "расшифруй "
    ]
    if any(q.startswith(p) for p in def_prefixes):
        return "definition"

    # «Как рассчитать» — запрос методики; «рассчитай по СП» — действие.
    asks_method = bool(re.search(r"\b(?:как|методика|формула|порядок|пример)\s+(?:рассчит|расчет|вычисл|определ)", q))
    if not asks_method and re.search(r"\b(?:рассчита[йть]|рассчитат|посчита[йть]|вычисл|расчет\b)", q):
        return "calculation"

    # Сравнения
    if any(w in q for w in ["сравни", "сравнение", "разница", "отличие"]):
        return "comparison"

    # Нормативные требования
    if any(w in q for w in ["требования", "норма", "сп ", "гост", "снип"]):
        return "regulatory"

    if not asks_method and any(w in q for w in ("гсоп", "градусо-сутки")):
        return "calculation"

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
    complete_sets = (
        {"L", "t_v", "t_n"}, {"A", "R", "delta_t"},
        {"A", "R", "t_v", "t_n"}, {"t_v", "t_ot", "z_ot"},
        {"delta", "lambda_value"}, {"R_tr", "lambda_value"}, {"Q", "L"},
    )
    if intent in {"search", "regulatory"} and any(required <= parameters.keys() for required in complete_sets):
        intent = "calculation"

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
        section_refs=extract_section_refs(text),
    )


def _run_self_tests() -> None:
    """Регрессии обозначений, единиц и маршрутизации, без сетевых вызовов."""
    assert extract_variables("толщина 100 мм, λ=0,04")["delta"] == 0.1
    mixed = extract_variables("L=100 м³/ч, t_в=20 °C, t_н=-25 °C, t_от=-8,4, z_от=225")
    assert mixed == {"L": 100.0, "t_v": 20.0, "t_n": -25.0, "t_ot": -8.4, "z_ot": 225.0}
    assert extract_variables("толщина 10 см, λ=0,04")["delta"] == 0.1
    assert extract_variables("мощность 10 кВт")["Q"] == 10000.0
    assert extract_variables("расход воздуха 100 л/мин")["L"] == 6.0
    assert get_parameter_issues("длина 5 м, расход воздуха 100 м3/ч", "length")
    assert get_parameter_issues("L=100 м3/сут", "air_flow")
    assert extract_variables("A=10, R=3, dt=50") == {"A": 10.0, "R": 3.0, "delta_t": 50.0}
    assert extract_variables("δ=100 мм λ=4e-2")["lambda_value"] == 0.04
    for notation in ("Δt", "delta_t", "dt"):
        assert extract_variables(f"A=12 R=3 {notation}=50")["delta_t"] == 50
    assert extract_variables("требуемое сопротивление 3,5 λ=0,04") == {"R_tr": 3.5, "lambda_value": 0.04}
    assert parse_query("Рассчитай вентиляцию по СП 60.13330 L=500 tв=20 tн=-30").intent == "calculation"
    assert parse_query("Какие требования к вентиляции по СП 60.13330?").intent == "regulatory"
    assert parse_query("Что такое вентиляция?").intent == "definition"
    assert "какие" not in extract_keywords("Какие требования к вентиляции?")
    assert extract_city("ГСОП для Томска") == "томск"
    assert extract_document_codes("ГОСТ 30494-2011, пункт 5.2") == ["ГОСТ 30494-2011"]
    assert extract_section_refs("пункт 5.2 таблица 3") == ["5.2", "3"]


if __name__ == "__main__":
    _run_self_tests()

# ИСПРАВЛЕНО: единый extract_keywords; алиасы/единицы/Δt и л/мин; мощность в кВт; проверки размерностей и неоднозначного L; intent расчётов по СП; Томск; коды/пункты; тесты.
