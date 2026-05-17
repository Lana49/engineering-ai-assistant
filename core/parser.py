import spacy
import re
from pathlib import Path
import json
from docx import Document
from collections import defaultdict
import spacy
# загрузка модели
try:
    nlp = spacy.load('ru_core_news_md')
    print("✅ Модель ru_core_news_md загружена")
except:
    try:
        nlp = spacy.load('ru_core_news_sm')
        print("✅ Модель ru_core_news_sm загружена")
    except:
        # Автоматическая загрузка модели для Streamlit Cloud
        import os
        os.system('python -m spacy download ru_core_news_md')
        nlp = spacy.load('ru_core_news_md')
        print("✅ Модель ru_core_news_md загружена автоматически")
# РАСШИРЕННЫЙ СЛОВАРЬ МАТЕРИАЛОВ
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

    # Прочие
    'пенопластовые изделия', 'пенополиуретановые изделия', 'полистиролбетонные блоки',
    'газосиликатные блоки', 'пеносиликатные блоки', 'перлитовый песок', 'вермикулитовый песок',
    'шлак', 'зола', 'шлакопортландцемент', 'пуццолановый цемент', 'глинозёмистый цемент',
    'магнезиальный цемент', 'гипс', 'гипсокартон', 'гипсоволокнистый лист', 'ГВЛ',
    'стекломагниевый лист', 'СМЛ', 'целлюлозно-волокнистый утеплитель'
}
# Дополнительные фразы (биграммы и триграммы)
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
    'стекломагниевый лист', 'пенопластовые изделия'
}

Construction_terms = {
    'материалы': MATERIALS,
    'конструкции': {
        'тоннель', 'теплоизоляционная конструкция', 'покровный слой',
        'пароизоляционный слой', 'предохранительный слой', 'выравнивающий слой',
        'теплоизоляционный слой', 'опорные элементы', 'разгружающие устройства',
        'крепление', 'температурный шов', 'фланцевое соединение', 'компенсатор',
        'трубопровод', 'оборудование', 'газоход', 'воздуховод', 'канал',
        'эстакада', 'галерея', 'фундамент', 'стена', 'перекрытие', 'опорные конструкции',
        'крепежные детали', 'изоляция', 'многослойная конструкция', 'съемная конструкция'
    },
    'параметры': {
        'толщина стенок', 'плотность', 'теплопроводность', 'коэффициент теплопроводности',
        'паропроницаемость', 'температуростойкость', 'уплотнение', 'коэффициент уплотнения',
        'горючесть', 'группа горючести', 'температура применения', 'срок эксплуатации',
        'толщина изоляции', 'потери', 'плотность теплового потока', 'термическое сопротивление'
    },
    'нормативы': {
        'ГОСТ', 'СНиП', 'СП', 'ТУ', 'ГОСТ Р', 'СанПиН', 'ВСН', 'ТСН', 'СТО', 'ЕН', 'ISO'
    }
}

# Создаем множества лемм для быстрого поиска
LEMMA_MATERIALS = {term.lower() for term in Construction_terms['материалы']}
LEMMA_STRUCTURES = {term.lower() for term in Construction_terms['конструкции']}
LEMMA_PARAMETERS = {term.lower() for term in Construction_terms['параметры']}
LEMMA_STANDARDS = Construction_terms['нормативы']


def read_docx(file_path):
    """Чтение DOCX файла"""
    doc = Document(file_path)
    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])


def read_file(file_path):
    """Универсальное чтение файла"""
    if file_path.endswith('.docx'):
        return read_docx(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()


def parse_construction_document(file_path):
    """Парсинг строительной документации"""

    print(f"📖 Чтение файла: {file_path}")

    # Читаем файл
    text = read_file(file_path)
    print(f"✅ Загружено {len(text)} символов")

    # Ограничиваем текст для скорости
    text_for_analysis = text[:500000]

    # Обработка через spaCy
    doc = nlp(text_for_analysis)

    # Результаты
    result = {
        'file': file_path,
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
        'sections': {}
    }

    # 1. Поиск разделов документа
    section_pattern = r'\n(\d+(?:\.\d+)*)\s+([А-ЯЁ][^\n]{5,100})'
    for match in re.finditer(section_pattern, text[:50000]):
        num = match.group(1)
        title = match.group(2).strip()
        result['sections'][num] = title[:80]

    # 2. Поиск строительных терминов с использованием лемматизации
    for token in doc:
        if token.is_punct or token.is_space:
            continue

        lemma = token.lemma_.lower()
        text_orig = token.text

        # Поиск по леммам
        if lemma in LEMMA_MATERIALS:
            result['materials'].add(text_orig)
            result['materials_lemmas'].add(lemma)

        if lemma in LEMMA_STRUCTURES:
            result['structures'].add(text_orig)
            result['structures_lemmas'].add(lemma)

        if lemma in LEMMA_PARAMETERS:
            result['parameters'].add(text_orig)
            result['parameters_lemmas'].add(lemma)

        # Поиск по биграммам (два слова подряд)
        if token.i + 1 < len(doc):
            bigram = f"{token.text} {doc[token.i + 1].text}".lower()
            if bigram in LEMMA_MATERIALS or bigram in MATERIAL_PHRASES:
                result['materials'].add(bigram)
                result['materials_lemmas'].add(bigram)

        # Поиск по триграммам (три слова подряд)
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


    # 1. Поиск по фразам из MATERIAL_PHRASES (только словарные)
    for phrase in MATERIAL_PHRASES:
        if phrase in text.lower():
            result['materials'].add(phrase)
            result['materials_lemmas'].add(phrase)

    # 2. Прямой поиск материалов из MATERIALS (только словарные)
    for material in MATERIALS:
        if material in text.lower():
            if material not in result['materials_lemmas']:
                result['materials'].add(material)
                result['materials_lemmas'].add(material)

    # 3. Поиск биграмм из текста (только если биграмма из словаря)
    words = text.lower().split()
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i + 1]}"
        if bigram in LEMMA_MATERIALS or bigram in MATERIAL_PHRASES:
            result['materials'].add(bigram)
            result['materials_lemmas'].add(bigram)

    # 4. Поиск по маркерам
    markers = ['материал', 'покрытие', 'изоляция', 'утеплитель', 'плита', 'мат', 'слой']
    for sent in doc.sents:
        sent_text = sent.text.lower()
        for marker in markers:
            if marker in sent_text:
                for token in sent:
                    if token.pos_ == 'NOUN' and len(token.text) > 3:
                        lemma = token.lemma_.lower()
                        # ДОБАВЛЯЕМ ТОЛЬКО ЕСЛИ УЖЕ ЕСТЬ В СЛОВАРЕ
                        if lemma in LEMMA_MATERIALS and lemma not in result['materials_lemmas']:
                            result['materials'].add(token.text)
                            result['materials_lemmas'].add(lemma)
                            print(f"  🔍 Добавлен из словаря по маркеру: {lemma}")


    # 5. Поиск материалов в контексте (дополнительно)
    for sent in doc.sents:
        sent_text = sent.text.lower()
        for material in MATERIALS:
            if material in sent_text and material not in result['materials']:
                result['materials'].add(material)
                result['materials_lemmas'].add(material)

    # 6. Числовые значения
    temp_pattern = r'[−-]?\d+(?:[.,]\d+)?\s*°[CС]'
    result['temperatures'] = list(set(re.findall(temp_pattern, text)))

    thick_pattern = r'\d+(?:[.,]\d+)?\s*мм'
    result['thicknesses'] = list(set(re.findall(thick_pattern, text)))

    density_pattern = r'\d+(?:[.,]\d+)?\s*кг/м³'
    result['densities'] = list(set(re.findall(density_pattern, text)))

    # Преобразуем множества в списки
    result['materials'] = list(result['materials'])
    result['structures'] = list(result['structures'])
    result['parameters'] = list(result['parameters'])
    result['standards'] = list(set(result['standards']))

    # Статистика найденного
    print(f"📊 Найдено материалов (оригинальных форм): {len(result['materials'])}")
    print(f"📊 Уникальных материалов (лемм): {len(result['materials_lemmas'])}")
    print(f"📊 Найдено конструкций: {len(result['structures'])}")
    print(f"📊 Найдено параметров: {len(result['parameters'])}")
    print(f"📊 Найдено нормативов: {len(result['standards'])}")

    return result


def save_to_json(result, output_file):
    """Сохраняет результат в JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"💾 Результат сохранен в {output_file}")


# Для тестирования
if __name__ == "__main__":
    file_path = "data/raw/sp_61_13330_2012_27052024.docx"

    if Path(file_path).exists():
        result = parse_construction_document(file_path)
        save_to_json(result, "construction_analysis.json")
        print("\n✅ Анализ завершен!")
    else:
        print(f"❌ Файл {file_path} не найден!")