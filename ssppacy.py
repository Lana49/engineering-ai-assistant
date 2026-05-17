import spacy
import re
from pathlib import Path
from collections import defaultdict
import json
from docx import Document
nlp = spacy.load('ru_core_news_md')
Construction_terms = {
    'материалы': {
      'железо', 'железобетон', 'металл', 'дерево', 'бетон', 'асфальтобетон', 'цемент', 'цементобетон', 'стеклоткань',
        'фольга', 'кэшированный материал', 'пенополиуретан', 'сталь', 'оцинкованная сталь', 'минеральная вата', 'стекловолокно',
        'каменная  вата', 'перлитоцементные плиты', 'совелитовые плиты', 'сталь углеродистая', 'сталь легированная',
        'вспененный каучук', 'пенополиэтилен', 'перлит', 'асбест', 'пенополистирол', 'пеностекло', 'пенополимерминерал',
        'оцинкованная сталь', 'нержавеющая сталь', 'алюминий', 'керамзит', 'штукатурка', 'рубероид', 'полиэтиленовая пленка',
        'стекловолокнистое полотно', 'битумный лак'
    },
    'конструкции': {
        'тоннель', 'теплоизоляционная конструкция', 'покровный слой', 'пароизоляционный слой','предохранительный слой',
        'выравнивающий слой', 'теплоизоляционный слой', 'опорные элементы', 'разгружающие устройства', 'крепление',
        'температурный шов', 'фланцевое соединение', 'компенсатор', 'трубопровод', 'оборудование', 'газоход',
        'воздуховод', 'канал','тоннель', 'эстакада', 'галерея', 'фундамент', 'стена', 'перекрытие', 'опорные конструкции',
        'крепление', 'крепежные детали', 'изоляция'
    },
    'параметры': {
        'толщина стенок', 'плотность', 'теплопроводность', 'коэффициент теплопроводности', 'паропроницаемость', 'температуростойкость',
        'уплотнение', 'коэффициент уплотнения', 'горючесть', 'группа горючести', 'температура применения', 'срок эксплуатации',
        'толщина изоляции', 'потери'
    },
    'нормативы': {
        'ГОСТ', 'СНиП', 'СП', 'ТУ', 'ГОСТ Р', 'СанПиН', 'ВСН', 'ТСН'
    }
}

LEMMA_MATERIALS = {term.lower() for term in Construction_terms['материалы']}
LEMMA_STRUCTURES = {term.lower() for term in Construction_terms['конструкции']}
LEMMA_PARAMETERS = {term.lower() for term in Construction_terms['параметры']}
LEMMA_STANDARDS = Construction_terms['нормативы']

def read_docx(file_path):
    doc = Document(file_path)
    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    return text

def parse_construction_document(file_path):
    """Парсинг строительной документации"""

    print(f"📖 Чтение файла: {file_path}")

    if file_path.endswith('.docx'):
        text = read_docx(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

    print(f"✅ Загружено {len(text)} символов")

    # Обработка через spaCy
    doc = nlp(text[:1000000])  # ограничиваем для скорости

    # Результаты
    result = {
        'file': file_path,
        'full_text': text[:10000] + "..." if len(text) > 10000 else text,  # сохраняем начало текста
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
        'numbers_with_units': [],
        'temperatures': [],
        'thicknesses': [],
        'densities': [],
        'sections': {}
    }

    section_pattern = r'\n(\d+(?:\.\d+)*)\s+([А-ЯЁ][^\n]{5,100})'
    for match in re.finditer(section_pattern, text[:50000]):
        num = match.group(1)
        title = match.group(2).strip()
        result['sections'][num] = title[:80]

    # 2. Поиск строительных терминов с использованием лемматизации
    for token in doc:
        # Пропускаем знаки препинания и пробелы
        if token.is_punct or token.is_space:
            continue

        lemma = token.lemma_.lower()  # начальная форма
        text_orig = token.text

        # Проверяем по леммам (начальным формам)
        if lemma in LEMMA_MATERIALS:
            result['materials'].add(text_orig)
            result['materials_lemmas'].add(lemma)

        if lemma in LEMMA_STRUCTURES:
            result['structures'].add(text_orig)
            result['structures_lemmas'].add(lemma)

        if lemma in LEMMA_PARAMETERS:
            result['parameters'].add(text_orig)
            result['parameters_lemmas'].add(lemma)

        # Для нормативов проверяем оригинальный текст (обычно в верхнем регистре)
        if token.text.upper() in LEMMA_STANDARDS:
            if token.i + 1 < len(doc) and doc[token.i + 1].like_num:
                result['standards'].append(f"{token.text} {doc[token.i + 1].text}")
            else:
                result['standards'].append(token.text)

    # 3. Числовые значения
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

    return result


def print_report(result):
    """Вывод отчета"""
    print("\n" + "=" * 80)
    print(f"📄 ДОКУМЕНТ: {result['file']}")
    print("=" * 80)

    # Статистика
    print(f"\n📊 СТАТИСТИКА:")
    print(f"  • Символов: {result['stats']['characters']:,}")
    print(f"  • Слов: {result['stats']['words']:,}")
    print(f"  • Предложений: {result['stats']['sentences']:,}")

    # Разделы документа
    if result['sections']:
        print(f"\n📑 РАЗДЕЛЫ ДОКУМЕНТА (первые 10):")
        for num, title in list(result['sections'].items())[:10]:
            print(f"  • {num} {title}")

    # Материалы
    if result['materials']:
        print(f"\n🏗️ МАТЕРИАЛЫ (найдено {len(result['materials'])}):")
        for mat in sorted(result['materials'])[:20]:
            print(f"  • {mat}")
        if len(result['materials']) > 20:
            print(f"  ... и еще {len(result['materials']) - 20}")

    if result.get('materials_lemmas'):
        print(f"\n  📖 Леммы (начальные формы):")
        for lem in sorted(result['materials_lemmas'])[:10]:
            print(f"    • {lem}")

    # Конструкции
    if result['structures']:
        print(f"\n🏢 КОНСТРУКЦИИ И ЭЛЕМЕНТЫ (найдено {len(result['structures'])}):")
        for struct in sorted(result['structures'])[:15]:
            print(f"  • {struct}")
        if len(result['structures']) > 15:
            print(f"  ... и еще {len(result['structures']) - 15}")

    # Параметры
    if result['parameters']:
        print(f"\n📊 ПАРАМЕТРЫ И ХАРАКТЕРИСТИКИ (найдено {len(result['parameters'])}):")
        for param in sorted(result['parameters'])[:15]:
            print(f"  • {param}")

    # Нормативы
    if result['standards']:
        print(f"\n📜 НОРМАТИВЫ (найдено {len(result['standards'])}):")
        for std in sorted(result['standards'])[:10]:
            print(f"  • {std}")

    # Температуры
    if result['temperatures']:
        print(f"\n🌡️ ТЕМПЕРАТУРЫ (найдено {len(result['temperatures'])}):")
        for temp in sorted(result['temperatures'])[:10]:
            print(f"  • {temp}")

    # Толщины
    if result['thicknesses']:
        print(f"\n📏 ТОЛЩИНЫ (найдено {len(result['thicknesses'])}):")
        for thick in sorted(result['thicknesses'])[:10]:
            print(f"  • {thick}")

    # Плотности
    if result['densities']:
        print(f"\n⚖️ ПЛОТНОСТИ (найдено {len(result['densities'])}):")
        for dens in sorted(result['densities'])[:10]:
            print(f"  • {dens}")


def save_to_json(result, output_file):
    """Сохраняет результат в JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 Результат сохранен в {output_file}")


# Основная программа
if __name__ == "__main__":
    # Укажите путь к файлу
    file_path = "sp_61_13330_2012_27052024.docx"  # или другой файл

    if Path(file_path).exists():
        result = parse_construction_document(file_path)
        print_report(result)
        save_to_json(result, "construction_analysis.json")

        print("\n✅ Анализ завершен!")
    else:
        print(f"❌ Файл {file_path} не найден!")
        print("\n📝 Создайте файл с текстом строительной документации, например:")
        print("  touch sp_61_13330_2012_27052024.docx")
        print("  # затем скопируйте текст документа в файл")