# -*- coding: utf-8 -*-
"""
Инженерный чат-бот для работы с документацией.

Интегрирует:
- QASystem для поиска по документам
- FormulaEngine для инженерных расчётов
- AgentLoop для многошаговых рассуждений
- ErrorHandler для обработки ошибок
- Экспорт в DOCX и PDF
- Извлечение таблиц и расчёты по таблицам
"""

from __future__ import annotations

import asyncio
import html
import json
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from core.agent_loop import AgentLoop
from core.error_handler import ErrorHandler
from core.formula_engine import FormulaEngine
from core.qa_engine import QASystem
from core.table_calculator import TableCalculator
from core.table_extractor import patch_qa_system_with_table_extractor
from utils.config import PROCESSED_DIR, RAW_DIR

try:
    patch_qa_system_with_table_extractor()
    print("✅ TableExtractor применён")
except Exception as patch_error:
    print(f"⚠️ Не удалось применить TableExtractor: {patch_error}")

st.set_page_config(
    page_title="Инженерный чат-бот",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

HISTORY_FILE = PROCESSED_DIR / "chat_history.json"


def run_async_safely(async_func, *args, **kwargs):
    """Безопасный запуск асинхронной функции в Streamlit."""
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
    """Универсальный вызов sync/async функции."""
    result = func(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return run_async_safely(lambda: result)
    return result


def get_initial_message() -> list[dict[str, str]]:
    """Начальное приветственное сообщение."""
    return [
        {
            "role": "assistant",
            "content": """🏗️ **Здравствуйте!** Я инженерный помощник по строительной документации.

📖 **База знаний:** ГОСТы, СП, технические регламенты и методические документы по строительству

**Что я умею:**
• 📖 Отвечать на вопросы по нормативной документации
• 📐 Рассчитывать толщину изоляции и теплопотери
• 🌍 Вычислять ГСОП (градусо-сутки отопительного периода)
• 💨 Определять расход теплоты на вентиляцию
• 📊 Находить таблицы и формулы в документах
• 🔍 Искать определения терминов

**Задайте свой вопрос или попросите сделать расчёт!**""",
        }
    ]


def save_history() -> None:
    """Сохраняет историю чата в JSON."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as file_obj:
        json.dump(st.session_state.messages, file_obj, ensure_ascii=False, indent=2)


def export_history_to_docx() -> Optional[Path]:
    """Экспорт истории чата в DOCX."""
    try:
        from docx import Document

        doc = Document()
        doc.add_heading("Инженерный чат-бот — история", 0)
        doc.add_paragraph(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        doc.add_paragraph()

        for message in st.session_state.messages:
            role = "Пользователь" if message["role"] == "user" else "Ассистент"
            doc.add_heading(role, level=1)
            doc.add_paragraph(message["content"])
            doc.add_paragraph()

        output_path = PROCESSED_DIR / f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))
        return output_path

    except ImportError:
        st.error("❌ Для экспорта истории нужен пакет python-docx: pip install python-docx")
        return None
    except OSError as export_error:
        st.error(f"❌ Ошибка записи DOCX: {export_error}")
        return None


def export_to_docx(
    answer: str,
    sources: list,
    tables: Optional[list] = None,
    formulas: Optional[list] = None,
    filename: Optional[str] = None,
) -> Optional[Path]:
    """Экспорт отчёта в DOCX."""
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
                else:
                    doc.add_paragraph(str(table))
                    doc.add_paragraph()

        if formulas:
            doc.add_heading("Формулы", level=1)
            for formula in formulas[:3]:
                if isinstance(formula, dict):
                    raw = formula.get("raw") or formula.get("expression") or formula.get("name", "")
                    doc.add_paragraph(raw)
                    variables = formula.get("variables")
                    if variables:
                        doc.add_paragraph(f"Переменные: {', '.join(variables[:5])}")
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
        doc.add_paragraph("Отчёт сгенерирован автоматически.", style="Intense Quote")

        output_path = PROCESSED_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))
        return output_path

    except ImportError:
        st.error("❌ python-docx не установлен. Выполните: pip install python-docx")
        return None
    except OSError as export_error:
        st.error(f"❌ Ошибка создания DOCX: {export_error}")
        return None


def export_to_pdf(
    answer: str,
    sources: list,
    tables: Optional[list] = None,
    formulas: Optional[list] = None,
    filename: Optional[str] = None,
) -> Optional[Path]:
    """Экспорт отчёта в PDF."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"engineering_report_{timestamp}.pdf"

    try:
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

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
            spaceAfter=20,
        )

        heading_style = ParagraphStyle(
            "HeadingStyle",
            parent=styles["Heading1"],
            fontSize=16,
            textColor="#2e86c1",
            spaceAfter=12,
            spaceBefore=12,
        )

        normal_style = ParagraphStyle(
            "NormalStyle",
            parent=styles["Normal"],
            fontSize=11,
            spaceAfter=6,
        )

        story = [
            Paragraph("Инженерный отчёт", title_style),
            Spacer(1, 0.2 * inch),
            Paragraph(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}", normal_style),
            Spacer(1, 0.2 * inch),
            Paragraph("Ответ", heading_style),
        ]

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
                    story.append(Paragraph(f"• {src}", normal_style))

        doc.build(story)
        return output_path

    except ImportError:
        st.warning("⚠️ reportlab не установлен. Будет создан DOCX вместо PDF.")
        safe_name = filename.replace(".pdf", ".docx")
        return export_to_docx(answer, sources, tables, formulas, safe_name)
    except OSError as export_error:
        st.error(f"❌ Ошибка создания PDF: {export_error}")
        return None


def render_export_buttons(
    answer: str,
    sources: list,
    tables: list,
    formulas: list,
    key_suffix: str = "current",
    response_id: Optional[int] = None,
) -> None:
    """Отображение кнопок экспорта с уникальными ключами."""
    if response_id is None:
        response_id = st.session_state.get("current_response_id", 0)

    if "export_button_counter" not in st.session_state:
        st.session_state.export_button_counter = 0
    st.session_state.export_button_counter += 1

    unique_id = (
        f"{key_suffix}_"
        f"{response_id}_"
        f"{int(time.time() * 1000)}_"
        f"{st.session_state.export_button_counter}_"
        f"{random.randint(1000, 9999)}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Экспорт DOCX", key=f"export_docx_{unique_id}"):
            with st.spinner("Создание DOCX..."):
                docx_path = export_to_docx(answer, sources, tables, formulas)
                if docx_path and docx_path.exists():
                    with open(docx_path, "rb") as file_obj:
                        st.download_button(
                            label="📥 Скачать DOCX",
                            data=file_obj.read(),
                            file_name=docx_path.name,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key=f"download_docx_{unique_id}",
                        )
                else:
                    st.error("❌ Ошибка создания DOCX")

    with col2:
        if st.button("📄 Экспорт PDF", key=f"export_pdf_{unique_id}"):
            with st.spinner("Создание PDF..."):
                pdf_path = export_to_pdf(answer, sources, tables, formulas)
                if pdf_path and pdf_path.exists():
                    with open(pdf_path, "rb") as file_obj:
                        st.download_button(
                            label="📥 Скачать PDF",
                            data=file_obj.read(),
                            file_name=pdf_path.name,
                            mime="application/pdf",
                            key=f"download_pdf_{unique_id}",
                        )
                else:
                    st.error("❌ Ошибка создания PDF")

    with col3:
        if st.button("📋 Копировать", key=f"copy_{unique_id}"):
            escaped_answer = html.escape(answer)
            st.markdown(
                f"""
                <script>
                (function() {{
                    const text = `{escaped_answer}`;
                    navigator.clipboard.writeText(text).then(() => {{
                        console.log("copied");
                    }});
                }})();
                </script>
                """,
                unsafe_allow_html=True,
            )
            st.success("✅ Текст скопирован в буфер обмена!")


def init_session_state() -> None:
    """Инициализация состояния сессии."""
    if "qa_system" not in st.session_state:
        st.session_state.qa_system = QASystem(use_llm=False)

        idx_path = PROCESSED_DIR / "qa_index"
        if idx_path.exists():
            try:
                loaded = st.session_state.qa_system.load_index(idx_path)
                if loaded:
                    print("✅ Индекс загружен при старте")
                else:
                    print("⚠️ Не удалось загрузить индекс")
            except OSError as load_error:
                print(f"⚠️ Ошибка загрузки индекса: {load_error}")
        else:
            print("📁 Индекс пока не найден")

    if "formula_engine" not in st.session_state:
        st.session_state.formula_engine = FormulaEngine(st.session_state.qa_system)

    if "agent_loop" not in st.session_state:
        st.session_state.agent_loop = AgentLoop(
            st.session_state.qa_system,
            st.session_state.formula_engine,
        )

    if "table_calculator" not in st.session_state:
        st.session_state.table_calculator = TableCalculator(st.session_state.qa_system)

    if "messages" not in st.session_state:
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as file_obj:
                    st.session_state.messages = json.load(file_obj)
            except (json.JSONDecodeError, OSError):
                st.session_state.messages = get_initial_message()
        else:
            st.session_state.messages = get_initial_message()

    st.session_state.setdefault("current_answer", "")
    st.session_state.setdefault("current_sources", [])
    st.session_state.setdefault("current_tables", [])
    st.session_state.setdefault("current_formulas", [])
    st.session_state.setdefault("current_response_id", 0)

    if "error_handler" not in st.session_state:
        st.session_state.error_handler = ErrorHandler()


def auto_load_documents() -> bool:
    """Автоматическая загрузка и индексация документов."""
    qa_system = st.session_state.qa_system
    idx_path = PROCESSED_DIR / "qa_index"

    if qa_system.is_ready and qa_system.index is not None:
        st.sidebar.success(f"✅ База знаний готова\n📄 {qa_system.index.ntotal} фрагментов")
        return True

    if idx_path.exists():
        try:
            if qa_system.load_index(idx_path):
                st.sidebar.success(f"✅ Индекс загружен\n📄 {qa_system.index.ntotal} фрагментов")
                return True
        except OSError as load_error:
            st.sidebar.warning(f"⚠️ Ошибка загрузки индекса: {load_error}")

    docs = list(RAW_DIR.glob("*.docx")) + list(RAW_DIR.glob("*.pdf")) + list(RAW_DIR.glob("*.rtf"))

    if not docs:
        st.sidebar.info("📥 Документы будут загружены из Hugging Face...")
        with st.sidebar:
            with st.spinner("📥 Загрузка документов из Hugging Face..."):
                indexed = qa_system.index_documents(RAW_DIR)
                if indexed and qa_system.is_ready:
                    idx_path.mkdir(parents=True, exist_ok=True)
                    qa_system.save_index(idx_path)
                    st.sidebar.success(f"✅ Загружено {qa_system.index.ntotal} фрагментов")
                    st.rerun()
                else:
                    st.sidebar.warning("📁 Папка data/raw пуста. Добавьте документы вручную.")
        return False

    if not qa_system.is_ready:
        with st.sidebar:
            st.info(f"📚 Индексация {len(docs)} документов...")

        indexed = qa_system.index_documents(RAW_DIR)

        if indexed:
            idx_path.mkdir(parents=True, exist_ok=True)
            qa_system.save_index(idx_path)
            with st.sidebar:
                st.success(f"✅ Загружено {qa_system.index.ntotal} фрагментов")
            return True

        st.sidebar.error("❌ Ошибка индексации")
        return False

    return True


def render_sidebar(
    qa_system: QASystem,
    formula_engine: FormulaEngine,
    error_handler: ErrorHandler,
) -> None:
    """Рендер боковой панели."""
    with st.sidebar:
        st.header("📚 О системе")
        st.markdown(
            """
- ✅ Семантический поиск по тексту
- ✅ Инженерные расчёты
- ✅ Извлечение нормативных параметров
- ✅ Поиск таблиц и формул
- ✅ Определения терминов
"""
        )
        st.divider()

        auto_load_documents()
        st.divider()

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
                idx_path = PROCESSED_DIR / "qa_index"
                if idx_path.exists():
                    shutil.rmtree(idx_path)
                    st.success("✅ Индекс очищен")
                    st.rerun()

        if not qa_system.is_ready:
            if st.button("📚 Индексировать документы", key="index_btn", use_container_width=True):
                with st.spinner("Индексация..."):
                    result = qa_system.index_documents(RAW_DIR)
                    if result:
                        idx_path = PROCESSED_DIR / "qa_index"
                        idx_path.mkdir(parents=True, exist_ok=True)
                        qa_system.save_index(idx_path)
                        st.success(f"✅ Проиндексировано {qa_system.index.ntotal} фрагментов")
                        st.rerun()
                    else:
                        st.error("❌ Не найдено документов для индексации")

        st.divider()

        st.subheader("📐 Доступные формулы")
        available_formulas = formula_engine.get_available_formulas()
        for formula in available_formulas:
            with st.expander(f"📖 {formula['name']}"):
                st.markdown(formula.get("expression", ""))
                st.caption(formula.get("description", ""))
                if formula.get("legend"):
                    st.markdown("**Обозначения:**")
                    st.markdown(formula["legend"])
                st.caption(f"📚 {formula.get('source', '')}")

        st.divider()

        st.subheader("📊 Статистика базы")
        docs_count = (
            len(list(RAW_DIR.glob("*.docx")))
            + len(list(RAW_DIR.glob("*.pdf")))
            + len(list(RAW_DIR.glob("*.rtf")))
        )
        chunks_count = qa_system.index.ntotal if qa_system.is_ready else 0

        stat_col1, stat_col2 = st.columns(2)
        stat_col1.metric("Документов", docs_count)
        stat_col2.metric("Фрагментов", chunks_count)

        st.divider()

        st.subheader("💾 Экспорт")
        if st.button("📄 Экспорт истории (DOCX)", use_container_width=True):
            docx_path = export_history_to_docx()
            if docx_path and docx_path.exists():
                with open(docx_path, "rb") as file_obj:
                    st.download_button(
                        label="📥 Скачать DOCX",
                        data=file_obj.read(),
                        file_name=docx_path.name,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
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

        if getattr(error_handler, "errors", None):
            st.divider()
            st.subheader("⚠️ Ошибки")
            with st.expander(f"Показать {len(error_handler.errors)} ошибок"):
                for i, error_item in enumerate(error_handler.errors[-5:], start=1):
                    st.error(f"{i}. {error_item.get('type', 'Error')}: {error_item.get('message', '')[:100]}")


def main() -> None:
    init_session_state()

    qa_system = st.session_state.qa_system
    formula_engine = st.session_state.formula_engine
    agent_loop = st.session_state.agent_loop
    error_handler = st.session_state.error_handler

    st.title("🏗️ Инженерный помощник проектировщика")
    st.caption(
        "📄 База знаний: ГОСТы, СП, технические регламенты и методические документы по строительству"
    )

    render_sidebar(qa_system, formula_engine, error_handler)

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
                        response_id=st.session_state.current_response_id,
                    )

    prompt = st.chat_input("Задайте вопрос по строительной документации...", key="main_chat_input")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        response = "Не удалось сформировать ответ."
        current_sources: list[Any] = []
        current_tables: list[Any] = []
        current_formulas: list[Any] = []

        with st.chat_message("assistant"):
            with st.spinner("🔍 Анализирую запрос..."):
                try:
                    result = call_maybe_async(agent_loop.run, prompt)

                    response = result.get("answer", response)
                    current_sources = result.get("sources", [])
                    current_tables = result.get("tables", [])
                    current_formulas = result.get("formulas", [])

                    st.session_state.current_response_id += 1
                    current_id = st.session_state.current_response_id

                    st.session_state.current_answer = response
                    st.session_state.current_sources = current_sources
                    st.session_state.current_tables = current_tables
                    st.session_state.current_formulas = current_formulas

                    st.markdown(response)

                    if hasattr(agent_loop, "get_reasoning_chain"):
                        try:
                            reasoning_chain = agent_loop.get_reasoning_chain()
                            if reasoning_chain:
                                with st.expander("🔍 Показать цепочку рассуждений"):
                                    st.markdown(reasoning_chain)
                        except AttributeError:
                            pass

                    render_export_buttons(
                        response,
                        current_sources,
                        current_tables,
                        current_formulas,
                        key_suffix="current",
                        response_id=current_id,
                    )

                except (RuntimeError, ValueError, OSError) as run_error:
                    error_info = error_handler.handle(run_error, {"query": prompt})
                    response = error_info.get("user_message", f"Ошибка: {run_error}")
                    st.error(response)
                except Exception as unexpected_error:
                    error_info = error_handler.handle(unexpected_error, {"query": prompt})
                    response = error_info.get("user_message", "Произошла непредвиденная ошибка.")
                    st.error(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        save_history()

    st.divider()
    st.caption("💡 Совет: для расчётов указывайте числа и параметры прямо в вопросе.")
    st.caption("📧 По всем вопросам обращайтесь к разработчику.")


if __name__ == "__main__":
    main()