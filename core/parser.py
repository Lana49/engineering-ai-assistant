import spacy
import re
import json
from docx import Document

# загрузка модели
try:
    nlp = spacy.load('ru_core_news_md')
    print("✅ Модель ru_core_news_md загружена")
except Exception as e:
    try:
        nlp = spacy.load('ru_core_news_sm')
        print("✅ Модель ru_core_news_sm загружена")
    except Exception as e2:
        import os
        os.system('python -m spacy download ru_core_news_md')
        nlp = spacy.load('ru_core_news_md')
        print("✅ Модель ru_core_news_md загружена автоматически")


# ========== РАСШИРЕННЫЙ СЛОВАРЬ МАТЕРИАЛОВ ==========

MATERIALS = {
    # Металлы и сплавы
    'железо', 'железобетон', 'металл', 'сталь', 'алюминий', 'чугун', 'медь',
    'латунь', 'бронза', 'титан', 'свинец', 'цинк', 'сплав', 'нихром', 'инвар',

    # Виды стали
    'оцинкованная сталь', 'нержавеющая сталь', 'сталь углеродистая', 'сталь легированная',
    'сталь низколегированная', 'сталь высоколегированная', 'сталь коррозионно-стойкая',
    'сталь жаропрочная', 'сталь инструментальная',

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

    # Жидкости и теплоносители (ОВК)
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

# ========== РАСШИРЕННЫЕ СЛОВАРИ ДЛЯ ГОСТОВ И СП ==========

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

# ========== СЛОВАРИ КОНСТРУКЦИЙ И ПАРАМЕТРОВ (ДОБАВЛЕНЫ) ==========

Construction_terms = {
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

# ========== СОЗДАНИЕ ЛЕММ (ТЕПЕРЬ ОПРЕДЕЛЕНЫ) ==========

LEMMA_MATERIALS = {term.lower() for term in Construction_terms['материалы']}
LEMMA_STRUCTURES = {term.lower() for term in Construction_terms['конструкции']}
LEMMA_PARAMETERS = {term.lower() for term in Construction_terms['параметры']}
LEMMA_STANDARDS = Construction_terms['нормативы']


# ========== ФУНКЦИИ ЧТЕНИЯ ==========

def read_docx(file_path):
    doc = Document(file_path)
    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

def read_pdf(file_path):
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

def read_rtf(file_path):
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

def read_file(file_path):
    file_path = str(file_path)
    if file_path.endswith('.docx'):
        return read_docx(file_path)
    elif file_path.endswith('.pdf'):
        return read_pdf(file_path)
    elif file_path.endswith('.rtf'):
        return read_rtf(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

# ========== ПАРСИНГ ==========

def parse_construction_document(file_path):
    print(f"📖 Чтение файла: {file_path}")

    text = read_file(file_path)
    print(f"✅ Загружено {len(text)} символов")

    doc = nlp(text[:500000])

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
        'sections': {}
    }

    # Поиск разделов
    section_pattern = r'\n(\d+(?:\.\d+)*)\s+([А-ЯЁ][^\n]{5,100})'
    for match in re.finditer(section_pattern, text[:50000]):
        num = match.group(1)
        title = match.group(2).strip()
        result['sections'][num] = title[:80]

    # Поиск терминов
    for token in doc:
        if token.is_punct or token.is_space:
            continue

        lemma = token.lemma_.lower()
        text_orig = token.text

        if lemma in LEMMA_MATERIALS:
            result['materials'].add(text_orig)
            result['materials_lemmas'].add(lemma)

        if lemma in LEMMA_STRUCTURES:
            result['structures'].add(text_orig)
            result['structures_lemmas'].add(lemma)

        if lemma in LEMMA_PARAMETERS:
            result['parameters'].add(text_orig)
            result['parameters_lemmas'].add(lemma)

        # Поиск биграмм
        if token.i + 1 < len(doc):
            bigram = f"{token.text} {doc[token.i + 1].text}".lower()
            if bigram in LEMMA_MATERIALS or bigram in MATERIAL_PHRASES:
                result['materials'].add(bigram)
                result['materials_lemmas'].add(bigram)

        # Поиск триграмм
        if token.i + 2 < len(doc):
            trigram = f"{token.text} {doc[token.i + 1].text} {doc[token.i + 2].text}".lower()
            if trigram in MATERIAL_PHRASES:
                result['materials'].add(trigram)

        # Поиск нормативов
        if token.text.upper() in LEMMA_STANDARDS:
            if token.i + 1 < len(doc) and doc[token.i + 1].like_num:
                result['standards'].append(f"{token.text} {doc[token.i + 1].text}")
            else:
                result['standards'].append(token.text)

    # Поиск стандартов по паттернам
    for pattern in STANDARD_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            if match not in result['standards']:
                result['standards'].append(match)

    # Поиск по фразам
    for phrase in MATERIAL_PHRASES:
        if phrase in text.lower():
            result['materials'].add(phrase)
            result['materials_lemmas'].add(phrase)

    # Прямой поиск материалов
    for material in MATERIALS:
        if material in text.lower():
            if material not in result['materials_lemmas']:
                result['materials'].add(material)
                result['materials_lemmas'].add(material)

    # Поиск биграмм
    words = text.lower().split()
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i + 1]}"
        if bigram in LEMMA_MATERIALS or bigram in MATERIAL_PHRASES:
            result['materials'].add(bigram)
            result['materials_lemmas'].add(bigram)

    # Поиск по маркерам
    markers = ['материал', 'покрытие', 'изоляция', 'утеплитель', 'плита', 'мат', 'слой', 'жидкость', 'раствор']
    for sent in doc.sents:
        sent_text = sent.text.lower()
        for marker in markers:
            if marker in sent_text:
                for token in sent:
                    if token.pos_ == 'NOUN' and len(token.text) > 3:
                        lemma = token.lemma_.lower()
                        if lemma in LEMMA_MATERIALS and lemma not in result['materials_lemmas']:
                            result['materials'].add(token.text)
                            result['materials_lemmas'].add(lemma)

    # Поиск материалов в контексте
    for sent in doc.sents:
        sent_text = sent.text.lower()
        for material in MATERIALS:
            if material in sent_text and material not in result['materials']:
                result['materials'].add(material)
                result['materials_lemmas'].add(material)

    # Числовые значения
    temp_pattern = r'[−-]?\d+(?:[.,]\d+)?\s*°[CС]'
    result['temperatures'] = list(set(re.findall(temp_pattern, text)))

    thick_pattern = r'\d+(?:[.,]\d+)?\s*мм'
    result['thicknesses'] = list(set(re.findall(thick_pattern, text)))

    density_pattern = r'\d+(?:[.,]\d+)?\s*кг/м³'
    result['densities'] = list(set(re.findall(density_pattern, text)))

    speed_pattern = r'\d+(?:[.,]\d+)?\s*м/с'
    result['speeds'] = list(set(re.findall(speed_pattern, text)))

    flow_pattern = r'\d+(?:[.,]\d+)?\s*м³/ч'
    result['flows'] = list(set(re.findall(flow_pattern, text)))

    # Преобразуем множества в списки
    result['materials'] = list(result['materials'])
    result['structures'] = list(result['structures'])
    result['parameters'] = list(result['parameters'])
    result['standards'] = list(set(result['standards']))

    print(f"📊 Найдено материалов (лемм): {len(result['materials_lemmas'])}")
    print(f"📊 Найдено нормативов: {len(result['standards'])}")
    return result

def save_to_json(result, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"💾 Результат сохранен в {output_file}")