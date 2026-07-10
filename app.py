import streamlit as st
from pathlib import Path
import sys
import json
from datetime import datetime
import asyncio
import os

print(f"Текущая директория: {os.getcwd()}")
print(f"Права на запись в data/processed: {os.access('data/processed', os.W_OK)}")

sys.path.insert(0, str(Path(__file__).parent))

from core.qa_engine import QASystem
from core.formula_engine import FormulaEngine
from core.agent_loop import AgentLoop
from utils.config import RAW_DIR, PROCESSED_DIR
from core.error_handler import ErrorHandler
from core.prompts import get_quick_definition

st.set_page_config(
    page_title="Инженерный чат-бот",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

HISTORY_FILE = PROCESSED_DIR / "chat_history.json"


# ========= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =========

def run_async_safely(async_func, *args, **kwargs):
    """Безопасный запуск асинхронной функции в Streamlit"""
    loop = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(async_func(*args, **kwargs))
    finally:
        if loop is not None:
            loop.close()
        asyncio.set_event_loop(None)


def call_maybe_async(func, *args, **kwargs):
    """Универсальный вызов sync/async функции"""
    result = func(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return run_async_safely(lambda: result)
    return result


def get_initial_message():
    return [{
        "role": "assistant",
        "content": """🏗️ **Здравствуйте!** Я инженерный помощник по строительной документации.

📖 **База знаний:** ГОСТы, СП, технические регламенты и методические документы по строительству

**Что я умею:**
• 📖 Отвечать на вопросы по нормативной документации
• 📐 Рассчитывать толщину изоляции и теплопотери
• 🌍 Вычислять ГСОП (градусо-сутки)
• 💨 Определять расход теплоты на вентиляцию
• 📊 Находить таблицы и формулы в документах
• 🔍 Искать определения терминов

**Задайте свой вопрос или попросите сделать расчет!**"""
    }]


# ========= ЭКСПОРТ В ИСТОРИЮ =========

def export_history_to_docx():
    try:
        from docx import Document

        doc = Document()
        doc.add_heading("Инженерный чат-бот - История", 0)
        doc.add_paragraph(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        doc.add_paragraph()

        for msg in st.session_state.messages:
            role = "Пользователь" if msg["role"] == "user" else "Ассистент"
            doc.add_heading(role, level=1)
            doc.add_paragraph(msg["content"])
            doc.add_paragraph()

        output_path = PROCESSED_DIR / f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))
        return output_path

    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        return None


# ========= ИНИЦИАЛИЗАЦИЯ =========

def init_session_state():
    """Инициализация состояния сессии"""
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = QASystem(use_llm=False)

        # ПЫТАЕМСЯ ЗАГРУЗИТЬ ИНДЕКС СРАЗУ
        idx_path = PROCESSED_DIR / "qa_index"
        if idx_path.exists():
            loaded = st.session_state.qa_system.load_index(idx_path)
            if loaded:
                print(f"✅ Индекс загружен при старте: {st.session_state.qa_system.index.ntotal} векторов")
            else:
                print("⚠️ Не удалось загрузить индекс")
        else:
            print("📁 Индекс не найден, будет создан при первой необходимости")

    if "formula_engine" not in st.session_state:
        st.session_state.formula_engine = FormulaEngine(st.session_state.qa_system)

    if "agent_loop" not in st.session_state:
        st.session_state.agent_loop = AgentLoop(
            st.session_state.qa_system,
            st.session_state.formula_engine
        )

    if "error_handler" not in st.session_state:
        st.session_state.error_handler = ErrorHandler(log_level="info")

    if "messages" not in st.session_state:
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    st.session_state.messages = json.load(f)
            except (json.JSONDecodeError, IOError):
                st.session_state.messages = get_initial_message()
        else:
            st.session_state.messages = get_initial_message()

    if "current_answer" not in st.session_state:
        st.session_state.current_answer = ""
    if "current_sources" not in st.session_state:
        st.session_state.current_sources = []
    if "current_tables" not in st.session_state:
        st.session_state.current_tables = []
    if "current_formulas" not in st.session_state:
        st.session_state.current_formulas = []

    # ВАЖНО: уникальный id текущего ответа
    if "current_response_id" not in st.session_state:
        st.session_state.current_response_id = 0


# ========= ФУНКЦИИ ЭКСПОРТА =========

def export_to_docx(answer: str, sources: list, tables: list = None, formulas: list = None, filename: str = None):
    """Экспорт в DOCX"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"engineering_report_{timestamp}.docx"

    try:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        doc = Document()

        title = doc.add_heading("Инженерный отчёт", 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER

        doc.add_paragraph(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        doc.add_paragraph()

        doc.add_heading("Ответ", level=1)
        doc.add_paragraph(answer)

        if tables:
            doc.add_heading("Таблицы", level=1)
            for table in tables[:2]:
                if isinstance(table, dict):
                    doc.add_paragraph(table.get("title", "Таблица"))
                    doc.add_paragraph(table.get("content", ""))
                    doc.add_paragraph()

        if formulas:
            doc.add_heading("Формулы", level=1)
            for formula in formulas[:3]:
                if isinstance(formula, dict):
                    raw = formula.get("raw") or formula.get("expression") or formula.get("name", "")
                    doc.add_paragraph(f"`{raw}`")
                    if formula.get("variables"):
                        doc.add_paragraph(f"Переменные: {', '.join(formula['variables'][:5])}")
                else:
                    doc.add_paragraph(str(formula))
                doc.add_paragraph()

        if sources:
            doc.add_heading("Источники", level=1)
            for src in sources:
                if isinstance(src, dict):
                    doc.add_paragraph(src.get("doc_name", "Документ"), style="List Bullet")
                else:
                    doc.add_paragraph(str(src), style="List Bullet")

        doc.add_paragraph()
        doc.add_paragraph("Отчёт сгенерирован автоматически", style="Intense Quote")

        output_path = PROCESSED_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))

        return output_path

    except ImportError:
        st.error("❌ python-docx не установлен. pip install python-docx")
        return None
    except Exception as e:
        st.error(f"❌ Ошибка создания DOCX: {e}")
        return None


def export_to_pdf(answer: str, sources: list, tables: list = None, formulas: list = None, filename: str = None):
    """Экспорт в PDF"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"engineering_report_{timestamp}.pdf"

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_CENTER

        output_path = PROCESSED_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Title"],
            fontSize=24,
            textColor="#1a5276",
            alignment=TA_CENTER,
            spaceAfter=20
        )

        heading_style = ParagraphStyle(
            "HeadingStyle",
            parent=styles["Heading1"],
            fontSize=16,
            textColor="#2e86c1",
            spaceAfter=12,
            spaceBefore=12
        )

        normal_style = ParagraphStyle(
            "NormalStyle",
            parent=styles["Normal"],
            fontSize=11,
            spaceAfter=6
        )

        story = []
        story.append(Paragraph("Инженерный отчёт", title_style))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}", normal_style))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Ответ", heading_style))
        for line in answer.split("\n"):
            if line.strip():
                clean_line = line.replace("**", "").replace("*", "")
                story.append(Paragraph(clean_line, normal_style))

        story.append(Spacer(1, 0.2 * inch))

        if sources:
            story.append(Paragraph("Источники", heading_style))
            for src in sources:
                if isinstance(src, dict):
                    story.append(Paragraph(f"• {src.get('doc_name', 'Документ')}", normal_style))
                else:
                    story.append(Paragraph(f"• {str(src)}", normal_style))

        doc.build(story)
        return output_path

    except ImportError:
        st.warning("⚠️ reportlab не установлен. Использую DOCX...")
        return export_to_docx(answer, sources, tables, formulas, filename.replace(".pdf", ".docx"))
    except Exception as e:
        st.error(f"❌ Ошибка создания PDF: {e}")
        return None


def render_export_buttons(answer, sources, tables, formulas, key_suffix="current", response_id=None):
    """
    Отображение кнопок экспорта.
    Все widget keys теперь уникальны.
    """
    if response_id is None:
        response_id = st.session_state.get("current_response_id", 0)

    widget_id = f"{key_suffix}_{response_id}"

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Экспорт DOCX", key=f"export_docx_{widget_id}"):
            docx_path = export_to_docx(answer, sources, tables, formulas)
            if docx_path and docx_path.exists():
                with open(docx_path, "rb") as f:
                    st.download_button(
                        label="📥 Скачать DOCX",
                        data=f.read(),
                        file_name=docx_path.name,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key=f"download_docx_{widget_id}"
                    )

    with col2:
        if st.button("📄 Экспорт PDF", key=f"export_pdf_{widget_id}"):
            pdf_path = export_to_pdf(answer, sources, tables, formulas)
            if pdf_path and pdf_path.exists():
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="📥 Скачать PDF",
                        data=f.read(),
                        file_name=pdf_path.name,
                        mime="application/pdf",
                        key=f"download_pdf_{widget_id}"
                    )

    with col3:
        if st.button("📋 Копировать", key=f"copy_{widget_id}"):
            st.code(answer, language="text")
            st.success("✅ Текст скопирован!")


# ========= АВТОЗАГРУЗКА ДОКУМЕНТОВ =========

def auto_load_documents():
    """Автоматическая загрузка и индексация документов"""
    qa_system = st.session_state.qa_system
    idx_path = PROCESSED_DIR / "qa_index"

    # === СНАЧАЛА ПРОВЕРЯЕМ ГОТОВНОСТЬ ===
    if qa_system.is_ready and qa_system.index is not None:
        st.sidebar.success(f"✅ База знаний готова\n📄 {qa_system.index.ntotal} фрагментов")
        return True

    # === ПЫТАЕМСЯ ЗАГРУЗИТЬ ===
    if idx_path.exists():
        if qa_system.load_index(idx_path):
            st.sidebar.success(f"✅ Индекс загружен\n📄 {qa_system.index.ntotal} фрагментов")
            return True

    # === ЕСЛИ НЕТ - СОЗДАЁМ ===
    docs = list(RAW_DIR.glob("*.docx")) + list(RAW_DIR.glob("*.pdf")) + list(RAW_DIR.glob("*.rtf"))

    if not docs:
        st.sidebar.info("📥 Документы будут загружены из Hugging Face...")
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

    if not qa_system.is_ready:
        with st.sidebar:
            st.info(f"📚 Индексация {len(docs)} документов...")

        success = qa_system.index_documents(RAW_DIR)

        if success:
            idx_path.mkdir(parents=True, exist_ok=True)
            qa_system.save_index(idx_path)
            with st.sidebar:
                st.success(f"✅ Загружено {qa_system.index.ntotal} фрагментов")
            return True
        else:
            st.sidebar.error("❌ Ошибка индексации")
            return False

    return True


# ========= ОСНОВНОЙ ИНТЕРФЕЙС =========

def main():
    init_session_state()

    qa_system = st.session_state.qa_system
    formula_engine = st.session_state.formula_engine
    agent_loop = st.session_state.agent_loop
    error_handler = st.session_state.error_handler

    st.title("🏗️ Инженерный помощник проектировщика")
    st.caption("📄 База знаний: ГОСТы, СП, технические регламенты и методические документы по строительству")

    # ========= SIDEBAR =========
    with st.sidebar:
        st.header("📚 О системе")
        st.markdown("""
        - ✅ Семантический поиск по тексту
        - ✅ Инженерные расчеты
        - ✅ Извлечение нормативных параметров
        - ✅ Поиск таблиц и формул
        - ✅ Определения терминов
        """)
        st.divider()

        auto_load_documents()
        st.divider()

        # Кнопки управления индексом
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Перезагрузить индекс", use_container_width=True):
                idx_path = PROCESSED_DIR / "qa_index"
                if idx_path.exists():
                    st.session_state.qa_system.load_index(idx_path)
                    st.success(f"✅ Индекс перезагружен: {qa_system.index.ntotal} векторов")
                    st.rerun()
                else:
                    st.warning("⚠️ Индекс не найден")

        with col2:
            if st.button("🗑️ Очистить индекс", use_container_width=True):
                import shutil
                idx_path = PROCESSED_DIR / "qa_index"
                if idx_path.exists():
                    shutil.rmtree(idx_path)
                    st.success("✅ Индекс очищен")
                    st.rerun()

        if not qa_system.is_ready:
            if st.button("📚 Индексировать документы", key="index_btn", use_container_width=True):
                with st.spinner("Индексация..."):
                    res = qa_system.index_documents(RAW_DIR)
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
                st.markdown(f.get("expression", ""))
                st.caption(f.get("description", ""))
                if f.get("legend"):
                    st.markdown("**Обозначения:**")
                    st.markdown(f["legend"])
                st.caption(f"📚 {f.get('source', '')}")

        st.divider()

        st.subheader("📊 Статистика базы")
        docs_count = (
                len(list(RAW_DIR.glob("*.docx")))
                + len(list(RAW_DIR.glob("*.pdf")))
                + len(list(RAW_DIR.glob("*.rtf")))
        )
        chunks_count = qa_system.index.ntotal if qa_system.is_ready else 0
        col1, col2 = st.columns(2)
        col1.metric("Документов", docs_count)
        col2.metric("Фрагментов", chunks_count)

        st.divider()

        st.subheader("💾 Экспорт")
        if st.button("📄 Экспорт истории (DOCX)", use_container_width=True):
            docx_path = export_history_to_docx()
            if docx_path and docx_path.exists():
                with open(docx_path, "rb") as f:
                    st.download_button(
                        label="📥 Скачать DOCX",
                        data=f.read(),
                        file_name=docx_path.name,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True
                    )
            else:
                st.error("❌ Ошибка создания файла")

        if st.button("🗑️ Очистить историю", use_container_width=True):
            st.session_state.messages = get_initial_message()
            if HISTORY_FILE.exists():
                HISTORY_FILE.unlink()
            st.session_state.current_answer = ""
            st.session_state.current_sources = []
            st.session_state.current_tables = []
            st.session_state.current_formulas = []
            st.rerun()

        if error_handler.errors:
            st.divider()
            st.subheader("⚠️ Ошибки")
            with st.expander(f"Показать {len(error_handler.errors)} ошибок"):
                for i, err in enumerate(error_handler.errors[-5:]):
                    st.error(f"{i + 1}. {err.get('type', 'Error')}: {err.get('message', '')[:100]}")

    # ========= ОТОБРАЖЕНИЕ СООБЩЕНИЙ =========
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message == st.session_state.messages[-1]:
                if st.session_state.current_answer:
                    render_export_buttons(
                        st.session_state.current_answer,
                        st.session_state.current_sources,
                        st.session_state.current_tables,
                        st.session_state.current_formulas,
                        key_suffix="last",
                        response_id=st.session_state.current_response_id
                    )

    # ========= ОБРАБОТКА ВОПРОСА =========
    if prompt := st.chat_input("Задайте вопрос по строительной документации...", key="main_chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        response = ""
        sources = []
        tables = []
        formulas = []

        with st.chat_message("assistant"):
            with st.spinner("🔍 Анализирую..."):
                try:
                    prompt_lower = prompt.lower()

                    calc_triggers = [
                        "рассчитай", "вычисли", "посчитай", "толщин", "температур",
                        "потери", "формул", "вентиляц", "расход", "гсоп", "градусо"
                    ]
                    def_triggers = [
                        "что такое", "определение", "термин", "понятие", "что значит",
                        "что означает", "расшифруй", "аббревиатура", "расшифровка",
                        "что это", "как понимать", "объясните", "поясните"
                    ]
                    table_triggers = ["таблиц", "табл", "покажи таблиц", "выведи таблиц"]

                    is_calc = any(w in prompt_lower for w in calc_triggers)
                    is_def = any(w in prompt_lower for w in def_triggers)
                    is_table = any(w in prompt_lower for w in table_triggers)

                    if is_calc:
                        # === РАСЧЁТНЫЙ ЗАПРОС ===
                        result = call_maybe_async(formula_engine.answer_calculation, prompt)
                        response = result.get("answer", "Не удалось выполнить расчёт")
                        sources = result.get("sources", [])
                        tables = result.get("tables", [])
                        formulas = result.get("formulas", [])

                        if not formulas and result.get("formula"):
                            formulas = [result["formula"]]

                    elif is_def:
                        # === ОПРЕДЕЛЕНИЕ ТЕРМИНА ===
                        clean_term = prompt_lower
                        for trigger in def_triggers:
                            clean_term = clean_term.replace(trigger, "").strip(" ?!.,:")

                        quick_def = get_quick_definition(clean_term)
                        if quick_def:
                            response = (
                                f"📖 **Определение термина «{clean_term}»:**\n\n"
                                f"{quick_def['definition']}\n\n"
                                f"📚 **Источник:** {quick_def['source']}"
                            )
                            if quick_def.get("example"):
                                response += f"\n\n**Пример:** {quick_def['example']}"
                        else:
                            definition_result = qa_system.find_definition(clean_term)
                            if definition_result.get("found"):
                                response = (
                                    f"📖 **Определение термина «{clean_term}»:**\n\n"
                                    f"{definition_result.get('definition', '')}\n\n"
                                    f"📚 **Источник:** {definition_result.get('source', 'Нормативная база')}"
                                )
                            else:
                                response = f"⚠️ В загруженных документах не найдено определение для термина «{clean_term}»."

                    elif is_table:
                        # === ПОИСК ТАБЛИЦЫ ===
                        result = qa_system.answer(prompt)
                        response = result.get("answer", "Таблица не найдена")
                        tables = result.get("tables", [])
                        sources = result.get("sources", [])
                        formulas = result.get("formulas", [])

                        if tables:
                            response += "\n\n📊 **Найденные таблицы:**\n"
                            for table in tables[:2]:
                                response += f"\n**{table.get('title', 'Таблица')}**\n"
                                content = table.get("content", "")
                                if len(content) > 500:
                                    content = content[:500] + "..."
                                response += f"```\n{content}\n```\n"

                    else:
                        # === АГЕНТСКИЙ ЦИКЛ ДЛЯ СЛОЖНЫХ ЗАПРОСОВ ===
                        result = call_maybe_async(agent_loop.run, prompt)
                        response = result.get("answer", "Не удалось получить ответ")
                        sources = result.get("sources", [])
                        tables = result.get("tables", [])
                        formulas = result.get("formulas", [])

                        if result.get("needs_clarification"):
                            questions = result.get("questions", [])
                            if questions:
                                response += "\n\n❓ **Уточните:**\n" + "\n".join([f"• {q}" for q in questions])

                    # Показываем цепочку рассуждений
                    with st.sidebar:
                        with st.expander("🔍 Показать цепочку рассуждений"):
                            if is_calc:
                                st.markdown(formula_engine.get_reasoning_chain())
                            else:
                                st.markdown(agent_loop.get_reasoning_chain())

                    # Новый уникальный id ответа
                    st.session_state.current_response_id += 1
                    current_id = st.session_state.current_response_id

                    # Сохраняем для экспорта
                    st.session_state.current_answer = response
                    st.session_state.current_sources = sources
                    st.session_state.current_tables = tables
                    st.session_state.current_formulas = formulas

                    st.markdown(response)

                    render_export_buttons(
                        response,
                        sources,
                        tables,
                        formulas,
                        key_suffix="current",
                        response_id=current_id
                    )

                except Exception as e:
                    error_info = error_handler.handle(e, {"query": prompt})
                    response = error_info["user_message"]
                    st.error(response)

        # Сохраняем историю
        st.session_state.messages.append({"role": "assistant", "content": response})
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)
    # ========= FOOTER =========
    st.divider()
    st.caption("💡 Совет: для расчетов указывайте числа и параметры прямо в вопросе")
    st.caption("📧 По всем вопросам обращайтесь к разработчику")


if __name__ == "__main__":
    main()