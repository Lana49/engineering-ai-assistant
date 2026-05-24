
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from core.qa_engine import QASystem
from core.formula_engine import FormulaEngine
from utils.config import RAW_DIR, PROCESSED_DIR

# Настройка страницы
st.set_page_config(
    page_title="Инженерный чат-бот",
    page_icon="🏗️",
    layout="wide"
)

# Инициализация QA системы
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = QASystem(use_llm=False)
    index_path = PROCESSED_DIR / "qa_index"
    if index_path.exists():
        st.session_state.qa_system.load_index(index_path)

# Инициализация Formula Engine
if 'formula_engine' not in st.session_state:
    st.session_state.formula_engine = FormulaEngine()

# Инициализация сообщений чата
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """🏗️ **Здравствуйте!** Я инженерный помощник по строительной документации.

📖 **База знаний:** СП 61.13330.2012 «Тепловая изоляция оборудования и трубопроводов»

**Что я умею:**
• 📖 Отвечать на вопросы по документации
• 📐 Рассчитывать толщину изоляции
• 🌡️ Определять тепловые потери
• 📊 Находить материалы и нормативы

**💡 Примеры вопросов:**
• Какие материалы для тепловой изоляции?
• Как рассчитать толщину изоляции?
• Рассчитай толщину при температуре 150°C
• Какая температура на поверхности изоляции?
• Какие нормативы регулируют изоляцию?

**Задайте свой вопрос!**"""
        }
    ]

qa_system = st.session_state.qa_system
formula_engine = st.session_state.formula_engine

# Заголовок
st.title("🏗️ Инженерный помощник по строительной документации")
st.caption("📄 База знаний: СП 61.13330.2012 «Тепловая изоляция оборудования и трубопроводов»")

# Боковая панель
with st.sidebar:
    st.header("📚 О системе")
    st.markdown("""
    **База знаний:**
    - СП 61.13330.2012
    - Тепловая изоляция
    - 93 714 символов
    - 460 предложений
    
    **Что я умею:**
    - ✅ Отвечать на вопросы
    - ✅ Рассчитывать толщину
    - ✅ Определять теплопотери
    - ✅ Находить нормативы
    """)

    st.divider()

    # Статус системы
    if qa_system.is_ready:
        st.success(f"✅ База знаний готова\n📄 {qa_system.index.ntotal} фрагментов")
    else:
        st.warning("⚠️ База знаний не загружена")
        if st.button("📚 Индексировать документы"):
            with st.spinner("Индексация..."):
                success = qa_system.index_documents(RAW_DIR)
                if success:
                    index_path = PROCESSED_DIR / "qa_index"
                    index_path.mkdir(parents=True, exist_ok=True)
                    qa_system.save_index(index_path)
                    st.success(f"✅ Проиндексировано {qa_system.index.ntotal} фрагментов")
                    st.rerun()
                else:
                    st.error("❌ Не найдено документов для индексации")

    st.divider()

    # Доступные формулы
    st.subheader("📐 Доступные формулы")
    formulas = formula_engine.get_available_formulas()
    for f in formulas:
        with st.expander(f"📖 {f['name'][:25]}"):
            st.write(f"`{f['expression']}`")
            st.caption(f['description'][:50])

    st.divider()

    # Кнопка очистки истории
    if st.button("🗑️ Очистить историю", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

# Отображение чата
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Поле ввода вопроса
if prompt := st.chat_input("Задайте вопрос по строительной документации..."):
    # Добавляем вопрос пользователя
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Генерируем ответ
    with st.chat_message("assistant"):
        with st.spinner("🔍 Анализирую..."):
            # Проверяем, расчетный ли вопрос
            is_calc = any(word in prompt.lower() for word in
                         ['рассчитай', 'вычисли', 'толщин', 'температур', 'потери', 'формул'])

            if is_calc:
                result = formula_engine.answer_calculation(prompt)
                response = result['answer']
            else:
                search_result = qa_system.answer(prompt)
                response = search_result['answer']

                # Добавляем источники, если есть
                if search_result.get('sources'):
                    response += "\n\n---\n📚 **Источники:**\n"
                    for src in search_result['sources'][:2]:
                        response += f"\n• {src['doc_name']} (релевантность: {src['score']:.2f})"

            st.markdown(response)

    # Сохраняем ответ
    st.session_state.messages.append({"role": "assistant", "content": response})

# Подвал
st.divider()
st.caption("💡 Совет: для расчетов указывайте температуру (например, 'при 150°C')")