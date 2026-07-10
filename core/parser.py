# -*- coding: utf-8 -*-
"""
Парсер строительной документации.
Извлекает:
- Материалы и их свойства
- Конструкции
- Параметры (температура, толщина, плотность и т.д.)
- Стандарты (ГОСТ, СП, СНиП)
- Таблицы
- Формулы
- Разделы документа
- Числовые данные с единицами измерения
"""

import spacy
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


# ========== ЗАГРУЗКА МОДЕЛИ SPACY ==========

def load_spacy_model():
    """Загрузка модели spaCy с автоматической установкой"""
    try:
        nlp = spacy.load('ru_core_news_md')
        print("✅ Модель ru_core_news_md загружена")
        return nlp
    except Exception:
        try:
            nlp = spacy.load('ru_core_news_sm')
            print("✅ Модель ru_core_news_sm загружена")
            return nlp
        except Exception:
            import os
            os.system('python -m spacy download ru_core_news_md')
            nlp = spacy.load('ru_core_news_md')
            print("✅ Модель ru_core_news_md загружена автоматически")
            return nlp


nlp = load_spacy_model()


# ========== РАСШИРЕННЫЙ СЛОВАРЬ МАТЕРИАЛОВ ==========

MATERIALS = {
    # Металлы и сплавы
    'железо', 'железобетон', 'металл', 'сталь', 'алюминий', 'чугун', 'медь',
    'латунь', 'бронза', 'титан', 'свинец', 'цинк', 'сплав', 'нихром', 'инвар',
    'оцинкованная сталь', 'нержавеющая сталь', 'сталь углеродистая',
    'сталь легированная', 'сталь низколегированная', 'сталь высоколегированная',
    'сталь коррозионно-стойкая', 'сталь жаропрочная', 'сталь инструментальная',

    # Бетоны и растворы
    'бетон', 'асфальтобетон', 'цемент', 'цементобетон', 'армопенобетон',
    'пенобетон', 'газобетон', 'керамзитобетон', 'шлакобетон', 'раствор',
    'полистиролбетон', 'газосиликат', 'пеносиликат', 'фибробетон', 'силикатобетон',
    'жаростойкий бетон', 'кислотоупорный бетон', 'гидротехнический бетон',

    # Минераловатные материалы
    'минеральная вата', 'каменная вата', 'стекловолокно', 'стекловата',
    'базальтовое волокно', 'шлаковата', 'войлок', 'маты минераловатные',
    'плиты минераловатные', 'цилиндры минераловатные', 'полуцилиндры минераловатные',
    'сегменты минераловатные', 'прошивные маты', 'рулонированные маты',
    'ламелла-маты', 'иглопробивные маты', 'холсты', 'рулонное стекловолокно',

    # Полимерные материалы
    'пенополиуретан', 'пенополистирол', 'пенополиэтилен', 'пенопласт',
    'пеностекло', 'пенополимерминерал', 'вспененный каучук', 'поропласт', 'полимербетон',
    'пенополиизоцианурат', 'экструзионный пенополистирол', 'вспененный пенополистирол',
    'вспененный полиэтилен', 'полиэтилен высокого давления', 'полиэтилен низкого давления',
    'полипропилен', 'поливинилхлорид', 'ПВХ', 'фторопласт', 'капролон', 'полиамид',

    # Керамические и композитные
    'керамзит', 'перлит', 'перлитоцемент', 'совелит', 'керамика',
    'огнеупорная керамика', 'керамогранит', 'керамблок', 'вермикулит', 'вспученный вермикулит',
    'аглопорит', 'шунгизит', 'туф', 'пемза',

    # Асбестосодержащие
    'асбест', 'асбестоцемент', 'хризотиловое волокно', 'асбестовые шнуры',
    'асбестосодержащие материалы', 'асбестоцементные листы', 'асбестовая бумага',
    'асбестовый картон', 'асбестовая ткань',

    # Рулонные и пленочные материалы
    'стеклоткань', 'стеклохолст', 'фольга', 'алюминиевая фольга',
    'полиэтиленовая пленка', 'рубероид', 'битумный лак', 'битум', 'стеклопластик',
    'стеклорогожа', 'лакостеклоткань', 'стекловолокнистое полотно', 'кэшированный материал',
    'полимерная плёнка', 'армированная плёнка', 'термоусадочная плёнка', 'пароизоляционная плёнка',

    # Штукатурные
    'штукатурка', 'декоративная штукатурка', 'гипсовая штукатурка', 'цементная штукатурка',
    'известковая штукатурка', 'полимерная штукатурка',

    # Древесные
    'дерево', 'древесина', 'фанера', 'ДСП', 'ОСП', 'МДФ', 'брус', 'доска', 'пиломатериалы',
    'бревно', 'клееный брус', 'древесно-стружечная плита', 'древесно-волокнистая плита',
    'ДВП', 'ЛДСП', 'ЛВЛ',

    # Утеплители
    'утеплитель', 'теплоизоляция', 'гидроизоляция', 'пароизоляция', 'звукоизоляция',
    'теплоизоляционный материал', 'гидроизоляционный материал', 'пароизоляционный материал',
    'вспененный полимер', 'эковата', 'пенофол', 'изолон', 'пеноизол',

    # Покрытия
    'краска', 'эмаль', 'лак', 'грунтовка', 'антикоррозийное покрытие',
    'полимерное покрытие', 'порошковая краска', 'мастика', 'герметик',
    'клей', 'силиконовый герметик', 'акриловый герметик', 'полиуретановый герметик',

    # Жидкости и теплоносители
    'вода', 'теплоноситель', 'хладагент', 'фреон', 'антифриз',
    'этиленгликоль', 'пропиленгликоль',

    # Прочие
    'пенопластовые изделия', 'пенополиуретановые изделия', 'полистиролбетонные блоки',
    'газосиликатные блоки', 'пеносиликатные блоки', 'перлитовый песок', 'вермикулитовый песок',
    'шлак', 'зола', 'шлакопортландцемент', 'пуццолановый цемент', 'глинозёмистый цемент',
    'магнезиальный цемент', 'гипс', 'гипсокартон', 'гипсоволокнистый лист', 'ГВЛ',
    'стекломагниевый лист', 'СМЛ', 'целлюлозно-волокнистый утеплитель'
}

MATERIAL_PHRASES = {
    'оцинкованная сталь', 'нержавеющая сталь', 'каменная вата', 'минеральная вата',
    'базальтовое волокно', 'вспененный каучук', 'алюминиевая фольга',
    'полиэтиленовая пленка', 'стекловолокнистое полотно', 'битумный лак',
    'кэшированный материал', 'перлитоцементные плиты', 'совелитовые плиты',
    'пенополиуретан', 'пенополистирол', 'пенополиэтилен', 'пенополимерминерал',
    'антикоррозийное покрытие', 'теплоизоляционный материал', 'гидроизоляционный материал',
    'пароизоляционный материал', 'полимерное покрытие', 'порошковая краска',
    'асбестовые шнуры', 'асбестоцементные листы', 'минераловатные плиты',
    'минераловатные маты', 'минераловатные цилиндры', 'рулонированные маты',
    'прошивные маты', 'иглопробивные маты', 'древесно-стружечная плита',
    'вспененный полиэтилен', 'экструзионный пенополистирол', 'гипсокартон',
    'стекломагниевый лист', 'пенопластовые изделия', 'водный раствор',
    'незамерзающая жидкость', 'охлаждающая жидкость'
}

# ========== СЛОВАРИ ДЛЯ ГОСТОВ И СП ==========

STANDARDS_TYPES = {
    'ГОСТ', 'ГОСТ Р', 'ГОСТ РВ', 'СП', 'СНиП', 'СанПиН', 'ТУ',
    'МСН', 'МСП', 'СТО', 'ЕН', 'ISO', 'ВСН', 'ТСН', 'СН',
    'Технический регламент', 'Методические рекомендации'
}

STANDARD_PATTERNS = [
    r'(ГОСТ Р\s+\d+(?:\.\d+)*-\d{4})',
    r'(ГОСТ\s+\d+(?:\.\d+)*-\d{4})',
    r'(СП\s+\d+(?:\.\d+)*\.\d{4})',
    r'(СНиП\s+\d+(?:\.\d+)*-\d{2}-\d{2})',
    r'(СанПиН\s+\d+(?:\.\d+)*\.\d+\.\d+\.\d+)',
    r'(ТУ\s+\d+(?:\.\d+)*-\d+(?:-\d+)?)',
    r'(МСН\s+\d+(?:\.\d+)*-\d{2})',
    r'(СТО\s+\d+(?:\.\d+)*-\d{4})',
]

# ========== СЛОВАРИ КОНСТРУКЦИЙ И ПАРАМЕТРОВ ==========

CONSTRUCTION_TERMS = {
    'материалы': MATERIALS,
    'конструкции': {
        'здание', 'помещение', 'фундамент', 'стена', 'перекрытие', 'фасад', 'кровля',
        'эстакада', 'галерея', 'тоннель',
        'теплоизоляционная конструкция', 'покровный слой', 'пароизоляционный слой',
        'предохранительный слой', 'выравнивающий слой', 'теплоизоляционный слой',
        'опорные элементы', 'разгружающие устройства', 'крепление', 'температурный шов',
        'фланцевое соединение', 'компенсатор', 'многослойная конструкция',
        'съемная конструкция', 'опорные конструкции', 'крепежные детали', 'изоляция',
        'трубопровод', 'газоход', 'воздуховод', 'канал', 'оборудование',
        'система отопления', 'система вентиляции', 'кондиционер', 'вентилятор',
        'калорифер', 'теплообменник', 'радиатор', 'конвектор', 'насос', 'клапан',
        'воздухораспределитель', 'чиллер', 'фанкойл', 'рекуператор', 'тепловой пункт',
        'метеорологическая станция'
    },
    'параметры': {
        'толщина стенок', 'плотность', 'теплопроводность', 'коэффициент теплопроводности',
        'паропроницаемость', 'температуростойкость', 'уплотнение', 'коэффициент уплотнения',
        'горючесть', 'группа горючести', 'температура применения', 'срок эксплуатации',
        'толщина изоляции', 'потери', 'плотность теплового потока', 'термическое сопротивление',
        'температура воздуха', 'энтальпия', 'солнечная радиация', 'влажность',
        'скорость ветра', 'осадки', 'градусо-сутки', 'расход воздуха', 'расход теплоты',
        'кратность воздухообмена', 'аэродинамическое сопротивление', 'микроклимат',
        'барометрическое давление', 'обеспеченность'
    },
    'нормативы': {
        'ГОСТ', 'СНиП', 'СП', 'ТУ', 'ГОСТ Р', 'СанПиН', 'ВСН', 'ТСН', 'СТО', 'ЕН', 'ISO'
    }
}

# ========== СОЗДАНИЕ ЛЕММ ==========

LEMMA_MATERIALS = {term.lower() for term in CONSTRUCTION_TERMS['материалы']}
LEMMA_STRUCTURES = {term.lower() for term in CONSTRUCTION_TERMS['конструкции']}
LEMMA_PARAMETERS = {term.lower() for term in CONSTRUCTION_TERMS['параметры']}
LEMMA_STANDARDS = CONSTRUCTION_TERMS['normatives']


# ========== ФУНКЦИИ ЧТЕНИЯ ФАЙЛОВ ==========

def read_docx(file_path: str) -> str:
    """Чтение DOCX файла"""
    try:
        from docx import Document
        doc = Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    except ImportError:
        print("⚠️ Для работы с DOCX установите python-docx: pip install python-docx")
        return ""
    except Exception as e:
        print(f"❌ Ошибка при чтении DOCX {file_path}: {e}")
        return ""


def read_pdf(file_path: str) -> str:
    """Чтение PDF файла"""
    try:
        import fitz
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except ImportError:
        print("⚠️ Для работы с PDF установите PyMuPDF: pip install pymupdf")
        return ""
    except Exception as e:
        print(f"❌ Ошибка при чтении PDF {file_path}: {e}")
        return ""


def read_rtf(file_path: str) -> str:
    """Чтение RTF файла"""
    try:
        from striprtf.striprtf import rtf_to_text
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            rtf_content = f.read()
        return rtf_to_text(rtf_content).strip()
    except ImportError:
        print("⚠️ Для работы с RTF установите striprtf: pip install striprtf")
        return ""
    except Exception as e:
        print(f"❌ Ошибка при чтении RTF {file_path}: {e}")
        return ""


def read_file(file_path: str) -> str:
    """Универсальное чтение файла"""
    file_path = str(file_path)
    if file_path.endswith('.docx') or file_path.endswith('.doc'):
        return read_docx(file_path)
    elif file_path.endswith('.pdf'):
        return read_pdf(file_path)
    elif file_path.endswith('.rtf') or file_path.endswith('.RTF'):
        return read_rtf(file_path)
    else:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"❌ Ошибка при чтении файла {file_path}: {e}")
            return ""


# ========== ОСНОВНОЙ КЛАСС ПАРСЕРА ==========

class DocumentParser:
    """Парсер строительной документации"""

    def __init__(self, nlp_model=None):
        self.nlp = nlp_model or nlp
        self.materials_set = MATERIALS
        self.material_phrases = MATERIAL_PHRASES
        self.standards_types = STANDARDS_TYPES
        self.standard_patterns = STANDARD_PATTERNS
        self.construction_terms = CONSTRUCTION_TERMS
        self.lemma_materials = LEMMA_MATERIALS
        self.lemma_structures = LEMMA_STRUCTURES
        self.lemma_parameters = LEMMA_PARAMETERS
        self.lemma_standards = LEMMA_STANDARDS

    def parse(self, file_path: str, max_chars: int = 500000) -> Dict[str, Any]:
        """
        Парсинг документа

        Args:
            file_path: Путь к файлу
            max_chars: Максимальное количество символов для обработки

        Returns:
            Dict с извлечёнными данными
        """
        print(f"📖 Чтение файла: {file_path}")

        text = read_file(file_path)
        if not text:
            return {"error": f"Не удалось прочитать файл {file_path}"}

        print(f"✅ Загружено {len(text)} символов")

        # Ограничиваем текст для обработки
        text_to_process = text[:max_chars]
        doc = self.nlp(text_to_process)

        result = {
            'file': str(file_path),
            'full_text': text[:10000] + "..." if len(text) > 10000 else text,
            'stats': {
                'characters': len(text),
                'words': len([t for t in doc if not t.is_space]),
                'sentences': len(list(doc.sents))
            },
            'materials': set(),
            'structures': set(),
            'parameters': set(),
            'materials_lemmas': set(),
            'structures_lemmas': set(),
            'parameters_lemmas': set(),
            'standards': [],
            'temperatures': [],
            'thicknesses': [],
            'densities': [],
            'speeds': [],
            'flows': [],
            'sections': {},
            'tables': [],
            'formulas': []
        }

        # Поиск разделов
        result['sections'] = self._extract_sections(text)

        # Поиск терминов
        materials, structures, parameters = self._extract_terms(doc, text)
        result['materials'] = list(materials)
        result['structures'] = list(structures)
        result['parameters'] = list(parameters)

        # Поиск стандартов
        result['standards'] = self._extract_standards(text)

        # Поиск таблиц
        result['tables'] = self._extract_tables(text)

        # Поиск формул
        result['formulas'] = self._extract_formulas(text)

        # Числовые значения
        result['temperatures'] = self._extract_numeric_values(text, r'[−-]?\d+(?:[.,]\d+)?\s*°[CС]')
        result['thicknesses'] = self._extract_numeric_values(text, r'\d+(?:[.,]\d+)?\s*мм')
        result['densities'] = self._extract_numeric_values(text, r'\d+(?:[.,]\d+)?\s*кг/м³')
        result['speeds'] = self._extract_numeric_values(text, r'\d+(?:[.,]\d+)?\s*м/с')
        result['flows'] = self._extract_numeric_values(text, r'\d+(?:[.,]\d+)?\s*м³/ч')

        # Преобразуем множества в списки
        result['materials'] = list(result['materials'])
        result['structures'] = list(result['structures'])
        result['parameters'] = list(result['parameters'])
        result['standards'] = list(set(result['standards']))

        print(f"📊 Найдено материалов: {len(result['materials'])}")
        print(f"📊 Найдено стандартов: {len(result['standards'])}")
        print(f"📊 Найдено таблиц: {len(result['tables'])}")
        print(f"📊 Найдено формул: {len(result['formulas'])}")

        return result

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Извлечение разделов документа"""
        sections = {}
        patterns = [
            r'\n(\d+(?:\.\d+)*)\s+([А-ЯЁ][^\n]{5,100})',
            r'\n(Раздел|Глава|Приложение)\s+(\d+[\.\s]+)([А-ЯЁ][^\n]{5,100})',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text[:50000]):
                groups = match.groups()
                if len(groups) >= 2:
                    num = groups[0] if groups[0] else groups[1]
                    title = groups[1] if len(groups) == 2 else groups[2]
                    if num and title:
                        sections[num.strip()] = title.strip()[:80]

        return sections

    def _extract_terms(self, doc, text: str) -> Tuple[Set[str], Set[str], Set[str]]:
        """Извлечение материалов, конструкций и параметров"""
        materials = set()
        structures = set()
        parameters = set()

        # Обработка токенов
        for token in doc:
            if token.is_punct or token.is_space:
                continue

            lemma = token.lemma_.lower()
            text_orig = token.text

            # Поиск материалов
            if lemma in self.lemma_materials or text_orig in self.materials_set:
                materials.add(text_orig)

            # Поиск биграмм
            if token.i + 1 < len(doc):
                bigram = f"{token.text} {doc[token.i + 1].text}".lower()
                if bigram in self.lemma_materials or bigram in self.material_phrases:
                    materials.add(bigram)

            # Поиск триграмм
            if token.i + 2 < len(doc):
                trigram = f"{token.text} {doc[token.i + 1].text} {doc[token.i + 2].text}".lower()
                if trigram in self.material_phrases:
                    materials.add(trigram)

            # Поиск конструкций
            if lemma in self.lemma_structures:
                structures.add(text_orig)

            # Поиск параметров
            if lemma in self.lemma_parameters:
                parameters.add(text_orig)

            # Поиск нормативов
            if token.text.upper() in self.lemma_standards:
                if token.i + 1 < len(doc) and doc[token.i + 1].like_num:
                    pass  # Обработка в _extract_standards

        # Прямой поиск в тексте
        text_lower = text.lower()
        for material in self.materials_set:
            if material in text_lower:
                materials.add(material)

        for phrase in self.material_phrases:
            if phrase in text_lower:
                materials.add(phrase)

        # Поиск по маркерам
        markers = ['материал', 'покрытие', 'изоляция', 'утеплитель', 'плита', 'мат', 'слой', 'жидкость', 'раствор']
        for sent in doc.sents:
            sent_text = sent.text.lower()
            for marker in markers:
                if marker in sent_text:
                    for token in sent:
                        if token.pos_ == 'NOUN' and len(token.text) > 3:
                            lemma = token.lemma_.lower()
                            if lemma in self.lemma_materials:
                                materials.add(token.text)

        return materials, structures, parameters

    def _extract_standards(self, text: str) -> List[str]:
        """Извлечение стандартов (ГОСТ, СП, СНиП)"""
        standards = []

        # По паттернам
        for pattern in self.standard_patterns:
            matches = re.findall(pattern, text)
            standards.extend(matches)

        # По ключевым словам
        for token in re.findall(r'[А-Я]{2,}\s+\d+[\.\d-]*', text):
            if any(token.startswith(st) for st in self.standards_types):
                standards.append(token)

        return list(set(standards))

    def _extract_tables(self, text: str) -> List[Dict]:
        """Извлечение таблиц из текста"""
        tables = []
        lines = text.split('\n')
        in_table = False
        table_lines = []
        table_title = ""
        table_id = ""

        for i, line in enumerate(lines):
            # Поиск заголовка таблицы
            match = re.search(r'(?:Таблица|табл\.?)\s+(\d+)\s*[—\-–]\s*([^\n]+)', line, re.IGNORECASE)
            if match:
                table_id = match.group(1)
                table_title = match.group(2).strip()
                in_table = True
                table_lines = []
                continue

            # Поиск таблицы с разделителями
            if '|' in line and len(line.split('|')) > 2:
                if not in_table:
                    in_table = True
                    table_lines = []
                    if i > 0 and len(lines[i - 1].strip()) < 100:
                        table_title = lines[i - 1].strip()
                table_lines.append(line.strip())
            else:
                if in_table and table_lines:
                    tables.append({
                        'id': table_id or str(len(tables) + 1),
                        'title': table_title or f"Таблица {len(tables) + 1}",
                        'content': '\n'.join(table_lines),
                        'rows': [l for l in table_lines if l.strip()]
                    })
                    table_lines = []
                    in_table = False
                    table_title = ""
                    table_id = ""

        # Добавляем последнюю таблицу
        if in_table and table_lines:
            tables.append({
                'id': table_id or str(len(tables) + 1),
                'title': table_title or f"Таблица {len(tables) + 1}",
                'content': '\n'.join(table_lines),
                'rows': [l for l in table_lines if l.strip()]
            })

        return tables

    def _extract_formulas(self, text: str) -> List[Dict]:
        """Извлечение формул из текста"""
        formulas = []
        seen = set()

        patterns = [
            r'([A-Za-zА-Яа-я][_\w]*)\s*=\s*([^=;\n]+)',
            r'(?:формул[аы]|по формуле)\s*[:;]\s*([^;\n]+)',
            r'\(([^)]*[=+\-*/^][^)]*)\)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) >= 2:
                        var_part, expr_part = match[0], match[1]
                    else:
                        continue
                else:
                    var_part = None
                    expr_part = match

                expr_clean = str(expr_part).strip()
                expr_clean = re.sub(r'\s+', ' ', expr_clean)

                if len(expr_clean) > 3 and expr_clean not in seen:
                    seen.add(expr_clean)
                    variables = re.findall(r'[A-Za-zА-Яа-я][_\w]*', expr_clean)
                    formulas.append({
                        'raw': expr_clean,
                        'variables': list(set(variables)),
                        'has_operator': any(c in expr_clean for c in ['+', '-', '*', '/', '^', '='])
                    })

        return formulas

    def _extract_numeric_values(self, text: str, pattern: str) -> List[str]:
        """Извлечение числовых значений по паттерну"""
        matches = re.findall(pattern, text)
        return list(set(matches))

    def parse_multiple(self, file_paths: List[str]) -> Dict[str, Any]:
        """Парсинг нескольких файлов"""
        combined_result = {
            'files': [],
            'all_materials': set(),
            'all_structures': set(),
            'all_parameters': set(),
            'all_standards': set(),
            'all_tables': [],
            'all_formulas': [],
            'total_stats': {
                'files': len(file_paths),
                'characters': 0,
                'words': 0,
                'sentences': 0
            }
        }

        for file_path in file_paths:
            result = self.parse(file_path)
            if 'error' not in result:
                combined_result['files'].append(result)
                combined_result['all_materials'].update(result.get('materials', []))
                combined_result['all_structures'].update(result.get('structures', []))
                combined_result['all_parameters'].update(result.get('parameters', []))
                combined_result['all_standards'].update(result.get('standards', []))
                combined_result['all_tables'].extend(result.get('tables', []))
                combined_result['all_formulas'].extend(result.get('formulas', []))

                stats = result.get('stats', {})
                combined_result['total_stats']['characters'] += stats.get('characters', 0)
                combined_result['total_stats']['words'] += stats.get('words', 0)
                combined_result['total_stats']['sentences'] += stats.get('sentences', 0)

        # Преобразуем множества в списки
        combined_result['all_materials'] = list(combined_result['all_materials'])
        combined_result['all_structures'] = list(combined_result['all_structures'])
        combined_result['all_parameters'] = list(combined_result['all_parameters'])
        combined_result['all_standards'] = list(combined_result['all_standards'])

        return combined_result
# УТИЛИТНЫЕ ФУНКЦИИ
def save_to_json(result: Dict, output_file: Path):
    """Сохранение результатов парсинга в JSON"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"💾 Результат сохранен в {output_file}")
def load_from_json(input_file: Path) -> Dict:
    """Загрузка результатов парсинга из JSON"""
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)