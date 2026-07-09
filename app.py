import streamlit as st
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))

from core.qa_engine import QASystem
from core.formula_engine import FormulaEngine
from utils.config import RAW_DIR, PROCESSED_DIR

st.set_page_config(page_title="Инженерный чат-бот", page_icon="🏗️", layout="wide")

HISTORY_FILE = PROCESSED_DIR / "chat_history.json"

# ИНИЦИАЛИЗАЦИЯ
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = QASystem(use_llm=False)
    idx_path = PROCESSED_DIR / "qa_index"  # переименовано
    if idx_path.exists():
        st.session_state.qa_system.load_index(idx_path)

if 'formula_engine' not in st.session_state:
    st.session_state.formula_engine = FormulaEngine()

if 'messages' not in st.session_state:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            st.session_state.messages = json.load(f)
    else:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": """🏗️ **Здравствуйте!** Я инженерный помощник по строительной документации.

📖 **База знаний (152 документа):** 
• ГОСТы (строительные стандарты)
• СП (Своды правил) 
• МСН, МСП (межгосударственные нормы)
• Технические регламенты
• Методические рекомендации

**Что я умею:**
• 📖 Отвечать на вопросы по нормативной документации
• 📐 Рассчитывать толщину изоляции и теплопотери
• 🌍 Вычислять ГСОП (градусо-сутки)
• 💨 Определять расход теплоты на вентиляцию

**Задайте свой вопрос или попросите сделать расчет!**"""
            }
        ]

qa_system = st.session_state.qa_system
formula_engine = st.session_state.formula_engine


def auto_load_documents():
    """Автоматически загружает/индексирует документы"""

    idx_path = PROCESSED_DIR / "qa_index"

    # Если индекс уже загружен — показываем статистику
    if idx_path.exists() and qa_system.is_ready:
        st.sidebar.success(f"✅ База знаний готова\n📄 {qa_system.index.ntotal} фрагментов")
        return True

    # Проверяем, есть ли документы
    docs = list(RAW_DIR.glob("*.docx")) + list(RAW_DIR.glob("*.pdf")) + list(RAW_DIR.glob("*.rtf"))

    if not docs:
        st.sidebar.info("📥 Документы будут загружены из Hugging Face...")
        # Пробуем загрузить через snapshot_download
        with st.sidebar:
            with st.spinner("📥 Загрузка документов из Hugging Face..."):
                qa_system.index_documents(RAW_DIR)
                if qa_system.is_ready:
                    idx_path.mkdir(parents=True, exist_ok=True)
                    qa_system.save_index(idx_path)
                    st.sidebar.success(f"✅ Загружено {qa_system.index.ntotal} фрагментов")
                    st.rerun()
                    return True
                else:
                    st.sidebar.warning("📁 Папка `data/raw/` пуста. Добавьте документы вручную")
                    return False

    # Если документы есть — индексируем
    if not qa_system.is_ready:
        with st.sidebar:
            st.info(f"📚 Индексация {len(docs)} документов...")
            progress_bar = st.progress(0)

        success = qa_system.index_documents(RAW_DIR)

        if success:
            idx_path.mkdir(parents=True, exist_ok=True)
            qa_system.save_index(idx_path)
            with st.sidebar:
                progress_bar.progress(1.0)
                st.success(f"✅ Загружено {qa_system.index.ntotal} фрагментов")
            return True
        else:
            st.sidebar.error("❌ Ошибка индексации")
            return False

st.title("🏗️ Инженерный помощник проектировщика")
st.caption("📄 База: СП 61.13330 (Изоляция), СП 131.13330 (Климатология), СП 60.13330 (ОВК)")

with st.sidebar:
    st.header("📚 О системе")
    st.markdown("- ✅ Семантический поиск по тексту\n- ✅ Инженерные расчеты\n- ✅ Извлечение нормативных параметров")
    st.divider()

    if not qa_system.is_ready:
        if st.button("📚 Индексировать документы", key="index_btn"):
            with st.spinner("Индексация..."):
                res = qa_system.index_documents(RAW_DIR)  # переименовано
                if res:
                    idx_path = PROCESSED_DIR / "qa_index"
                    idx_path.mkdir(parents=True, exist_ok=True)
                    qa_system.save_index(idx_path)
                    st.success(f"✅ Проиндексировано {qa_system.index.ntotal} фрагментов")
                    st.rerun()
                else:
                    st.error("❌ Не найдено документов для индексации")

    st.divider()

    st.subheader("📐 Доступные формулы")
    formulas = formula_engine.get_available_formulas()
    for f in formulas:
        with st.expander(f"📖 {f['name']}"):
            st.markdown(f['expression'])
            st.caption(f['description'])
            if f.get('legend'):
                st.markdown("**Обозначения:**")
                st.markdown(f['legend'])

    st.divider()

    if st.button("🗑️ Очистить историю", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
        st.rerun()

    st.divider()

    st.subheader("📊 Статистика базы")
    docs_count = len(list(RAW_DIR.glob("*.docx"))) + len(list(RAW_DIR.glob("*.pdf"))) + len(list(RAW_DIR.glob("*.rtf")))
    chunks_count = qa_system.index.ntotal if qa_system.is_ready else 0
    st.metric("Документов", docs_count)
    st.metric("Фрагментов базы", chunks_count)

    st.divider()

    chat_text = "\n\n".join(
        [f"{'👤' if m['role'] == 'user' else '🤖'}:\n{m['content']}" for m in st.session_state.messages])
    st.download_button(label="📥 Скачать отчет (TXT)", data=chat_text, file_name="engineering_report.txt",
                       mime="text/plain", use_container_width=True)

# ОТОБРАЖЕНИЕ ЧАТА
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ОБРАБОТКА ВОПРОСА
if prompt := st.chat_input("Задайте вопрос по строительной документации...", key="main_chat_input"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Анализирую..."):
            prompt_lower = prompt.lower()

            calc_triggers = ['рассчитай', 'вычисли', 'посчитай', 'толщин', 'температур', 'потери', 'формул', 'вентиляц',
                             'расход']
            def_triggers = ['что такое', 'определение', 'термин', 'понятие', 'что значит', 'что означает', 'расшифруй',
                            'аббревиатура', 'расшифровка', 'что это', 'как понимать', 'объясните', 'поясните']

            is_calc = any(w in prompt_lower for w in calc_triggers)
            is_def = any(w in prompt_lower for w in def_triggers)

            if is_calc:
                result = formula_engine.answer_calculation(prompt)
                response = result['answer']
            elif is_def:
                clean_term = prompt_lower
                for trigger in def_triggers:
                    clean_term = clean_term.replace(trigger, "").strip()
                definition_result = qa_system.find_definition(clean_term)
                if definition_result['found']:
                    response = f"📖 **Определение термина «{clean_term}»:**\n\n{definition_result['definition']}\n\n📚 **Источник:** {definition_result['source']}"
                else:
                    response = f"⚠️ В загруженных документах не найдено определение для термина «{clean_term}»."
            else:
                search_result = qa_system.answer(prompt)
                response = search_result['answer']
                if search_result.get('sources') and not qa_system.use_llm:
                    response += "\n\n📚 **Источники:**\n" + "\n".join(
                        [f"• {src['doc_name']}" for src in search_result['sources'][:2]])

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)

st.divider()
st.caption("💡 Совет: для расчетов указывайте числа и параметры прямо в вопросе")