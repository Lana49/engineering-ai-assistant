# core/query_parser.py
"""
Единый парсер инженерных запросов.
Извлекает параметры, город, коды документов.
ЕДИНСТВЕННЫЙ ИСТОЧНИК для всех слоёв.
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
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
        "расход воздуха",
        "расход воздушный",
        "воздухообмен",
        "подача воздуха",
        "l_возд",
    ),
    "t_v": (
        "tv", "tв", "t_v", "t в",
        "температура внутри",
        "внутренняя температура",
        "температура внутреннего воздуха",
    ),
    "t_n": (
        "tn", "tн", "t_n", "t н",
        "температура наружного воздуха",
        "наружная температура",
        "температура снаружи",
    ),
    "t_ot": (
        "tot", "tот", "t_ot", "t от",
        "средняя температура отопительного периода",
        "температура отопительного периода",
    ),
"z_ot": (
        "zot", "zот", "z_ot", "z от",
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
        "delta t", "дельта t",
        "разность температур",
        "перепад температур",
    ),
    "delta": (
        "дельта",
        "толщина слоя",
        "толщина утеплителя",
        "толщина",
    ),
"lambda_value": (
        "lambda", "λ", "лямбда",
        "теплопроводность",
        "коэффициент теплопроводности",
    ),
    "R_tr": (
        "rtr", "r_tr", "r тр",
        "требуемое сопротивление",
        "нормируемое сопротивление",
    ),
    "Q": (
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

def _value_near_alias(text: str, alias: str) -> float | None:
    """Ищет число рядом с алиасом в любом порядке и с любым разделителем."""
    ap = _alias_pattern(alias)

    # Приоритет 1: alias = число / alias: число / alias -> число
    patterns = [
        rf"{ap}\s*[=:]\s*({NUMBER})",
        rf"{ap}\s*[-—–]\s*({NUMBER})",
        rf"{ap}\s+({NUMBER})",
        rf"({NUMBER})\s*(?:м3/ч|м³/ч|м2|м²|м|вт|Вт|квт|кВт|°c|°с|сут|мм)?\s*{ap}",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(",", "."))
            except ValueError:
                continue

    return None


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
    if not text:
        return {}

    normalized = normalize_engineering_text(text)
    values: dict[str, float] = {}

    # Сначала строгие однобуквенные (a=, r=, l=, q=)
    for char, canonical in STRICT_SINGLE_CHAR_ALIASES.items():
        if canonical in values:
            continue
        v = _value_strict_single_char(normalized, char)
        if v is not None:
            values[canonical] = v

    # Затем обычные алиасы
    for canonical_name, aliases in PARAM_ALIASES.items():
        if canonical_name in values:
            continue

        for alias in aliases:
            value = _value_near_alias(normalized, alias)
            if value is not None:
                values[canonical_name] = value
                break

    return values

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