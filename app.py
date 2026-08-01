# -*- coding: utf-8 -*-
"""
Инженерный чат-бот для работы с документацией.
Объединяет интерфейс чат-бота с технологиями инженерной базы знаний.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

# ИСПРАВЛЕНО: правильный импорт snapshot_download
try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None
    print("⚠️ huggingface-hub не установлен. pip install huggingface-hub")

from core.agent_loop import AgentLoop
from core.config import PROCESSED_DIR, RAW_DIR, HF_DATASET_REPO_ID
from core.error_handler import ErrorHandler
from core.formula_engine import FormulaEngine
from core.parser import parse_directory
from core.prompts import get_quick_definition
from core.qa_engine import QASystem
from core.table_calculator import patch_app_with_table_calculator
# ИСПРАВЛЕНО: правильный импорт из table_extractor
from core.table_extractor import extract_tables, tables_to_dicts, ExtractedTable

st.set_page_config(
    page_title="Инженерный чат-бот",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

HISTORY_FILE = PROCESSED_DIR / "chat_history.json"


# ========= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =========

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
    return [{
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

**Задайте свой вопрос или попросите сделать расчёт!**"""
    }]


def save_history() -> None:
    """Сохраняет историю чата в JSON."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)


# ========= СИНХРОНИЗАЦИЯ ДАТАСЕТА =========

def sync_hf_dataset_to_raw(force: bool = False) -> bool:
    """
    Скачивает документы из Hugging Face Dataset repo в RAW_DIR.
    """
    dataset_repo_id = (HF_DATASET_REPO_ID or "").strip()
    if not dataset_repo_id:
        print("ℹ️ HF_DATASET_REPO_ID не задан, синхронизация dataset пропущена")
        return False

    # ИСПРАВЛЕНО: проверка наличия snapshot_download
    if snapshot_download is None:
        print("❌ huggingface-hub не установлен. Установите: pip install huggingface-hub")
        return False

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing_docs = (
        list(RAW_DIR.glob("*.docx"))
        + list(RAW_DIR.glob("*.pdf"))
        + list(RAW_DIR.glob("*.rtf"))
        + list(RAW_DIR.glob("*.doc"))
    )

    if existing_docs and not force:
        print(f"✅ Документы уже есть в {RAW_DIR}: {len(existing_docs)} шт.")
        return True

    try:
        print(f"📥 Скачиваю dataset {dataset_repo_id} в {RAW_DIR} ...")
        snapshot_download(
            repo_id=dataset_repo_id,
            repo_type="dataset",
            local_dir=str(RAW_DIR),
        )

        downloaded_docs = (
            list(RAW_DIR.glob("*.docx"))
            + list(RAW_DIR.glob("*.pdf"))
            + list(RAW_DIR.glob("*.rtf"))
            + list(RAW_DIR.glob("*.doc"))
        )

        print(f"✅ Dataset синхронизирован. Найдено документов: {len(downloaded_docs)}")
        return bool(downloaded_docs)

    except Exception as dataset_error:
        print(f"⚠️ Ошибка загрузки dataset из Hugging Face: {dataset_error}")
        return False


# ========= ЭКСПОРТ =========

def export_history_to_docx():
    """Экспорт истории чата в DOCX."""
    try:
        from docx import Document

        doc = Document()
        doc.add_heading("Инженерный чат-бот — история", 0)
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

    except ImportError:
        st.error("❌ Для экспорта истории нужен python-docx: pip install python-docx")
        return None
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        return None


def export_to_docx(answer: str, sources: list, tables: list = None, formulas: list = None, filename: str = None):
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
                    if table.get("content"):
                        doc.add_paragraph(table.get("content", ""))
                    elif table.get("rows"):
                        for row in table.get("rows", [])[:10]:
                            doc.add_paragraph(" | ".join(map(str, row)))
                    doc.add_paragraph()

        if formulas:
            doc.add_heading("Формулы", level=1)
            for formula in formulas[:3]:
                if isinstance(formula, dict):
                    raw = formula.get("raw") or formula.get("expression") or formula.get("name", "")
                    doc.add_paragraph(raw)
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
        doc.add_paragraph("Отчёт сгенерирован автоматически.", style="Intense Quote")

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
        return export_to_docx(answer, sources, tables, formulas, filename.replace(".pdf", ".docx"))
    except Exception as e:
        st.error(f"❌ Ошибка создания PDF: {e}")
        return None


def render_export_buttons(answer: str, sources: list, tables: list, formulas: list, key_suffix: str = "current", response_id: int = None):
    """
    Отображение кнопок экспорта с уникальными ключами.
    Исправлена проблема StreamlitDuplicateElementKey.
    """
    if response_id is None:
        response_id = st.session_state.get("current_response_id", 0)

    unique_id = f"{key_suffix}_{response_id}_{int(time.time() * 1000)}"

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Экспорт DOCX", key=f"export_docx_{unique_id}"):
            docx_path = export_to_docx(answer, sources, tables, formulas)
            if docx_path and docx_path.exists():
                with open(docx_path, "rb") as f:
                    st.download_button(
                        label="📥 Скачать DOCX",
                        data=f.read(),
                        file_name=docx_path.name,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key=f"download_docx_{unique_id}",
                    )

    with col2:
        if st.button("📄 Экспорт PDF", key=f"export_pdf_{unique_id}"):
            pdf_path = export_to_pdf(answer, sources, tables, formulas)
            if pdf_path and pdf_path.exists():
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="📥 Скачать PDF",
                        data=f.read(),
                        file_name=pdf_path.name,
                        mime="application/pdf",
                        key=f"download_pdf_{unique_id}",
                    )

    with col3:
        if st.button("📋 Копировать", key=f"copy_{unique_id}"):
            st.code(answer, language="text")
            st.success("✅ Текст скопирован!")


# ========= ИНИЦИАЛИЗАЦИЯ QA СИСТЕМЫ =========

def init_qa_system() -> QASystem:
    """Инициализирует QA-систему с загрузкой или построением индекса."""
    qa = QASystem(
        use_llm=True,
        llm_provider="mixed",
        use_embeddings=True,
    )

    index_path = PROCESSED_DIR / "faiss_index.pkl"

    try:
        if index_path.exists():
            qa.load_index(index_path)
            print(f"📂 Индекс загружен: {index_path}")
        else:
            print("📥 Индекс не найден. Парсим документы...")
            print("🔄 Проверяю наличие документов...")
            sync_hf_dataset_to_raw()
            _build_index(qa, index_path)
    except Exception as exc:
        print(f"⚠️ Ошибка загрузки индекса: {exc}")
        st.warning("⚠️ Индекс повреждён или несовместим. Выполняется пересборка...")
        sync_hf_dataset_to_raw()
        _build_index(qa, index_path)

    return qa


def _build_index(qa: QASystem, index_path) -> None:
    """Строит индекс по документам."""
    if not RAW_DIR.exists() or not list(RAW_DIR.glob("*")):
        st.warning("⚠️ Папка с документами пуста. Загрузите документы в data/raw/")
        return

    parsed_docs = parse_directory(RAW_DIR, recursive=True)
    if parsed_docs:
        qa.build_index(parsed_docs)
        qa.save_index(index_path)
        print(f"✅ Индекс построен и сохранён: {index_path}")
    else:
        st.warning("⚠️ Не удалось распарсить документы")


def init_session_state() -> None:
    """Инициализирует состояние сессии."""
    if "qa_system" not in st.session_state:
        with st.spinner("Загрузка системы..."):
            # Синхронизируем датасет
            with st.status("📥 Загрузка документов...", expanded=True) as status:
                status.write("Скачивание датасета с Hugging Face...")
                sync_hf_dataset_to_raw()
                status.write("✅ Документы загружены")

            st.session_state.qa_system = init_qa_system()
            st.session_state.formula_engine = FormulaEngine(st.session_state.qa_system)
            st.session_state.agent_loop = AgentLoop(
                st.session_state.qa_system,
                st.session_state.formula_engine,
            )
            # ИСПРАВЛЕНО: удален вызов несуществующей функции
            # patch_qa_system_with_table_extractor() - больше не нужна
            patch_app_with_table_calculator()

    if "error_handler" not in st.session_state:
        st.session_state.error_handler = ErrorHandler(log_level="info")

    if "messages" not in st.session_state:
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    st.session_state.messages = json.load(f)
            except (json.JSONDecodeError, OSError):
                st.session_state.messages = get_initial_message()
        else:
            st.session_state.messages = get_initial_message()

    st.session_state.setdefault("current_answer", "")
    st.session_state.setdefault("current_sources", [])
    st.session_state.setdefault("current_tables", [])
    st.session_state.setdefault("current_formulas", [])
    st.session_state.setdefault("current_response_id", 0)


def auto_load_documents() -> bool:
    """Автоматическая загрузка и индексация документов."""
    qa_system = st.session_state.qa_system

    if qa_system.is_ready:
        st.sidebar.success(f"✅ База знаний готова\n📄 {len(qa_system.chunks)} фрагментов")
        return True

    index_path = PROCESSED_DIR / "faiss_index.pkl"

    if index_path.exists():
        try:
            if qa_system.load_index(index_path):
                st.sidebar.success(f"✅ Индекс загружен\n📄 {len(qa_system.chunks)} фрагментов")
                return True
        except Exception as e:
            st.sidebar.warning(f"⚠️ Ошибка загрузки индекса: {e}")

    docs = (
        list(RAW_DIR.glob("*.docx"))
        + list(RAW_DIR.glob("*.pdf"))
        + list(RAW_DIR.glob("*.rtf"))
        + list(RAW_DIR.glob("*.doc"))
    )

    if not docs:
        st.sidebar.info("📁 Папка документов пуста. Проверьте загрузку dataset.")
        return False

    with st.sidebar:
        st.info(f"📚 Индексация {len(docs)} документов...")

    if qa_system.index_documents(RAW_DIR):
        index_path.parent.mkdir(parents=True, exist_ok=True)
        qa_system.save_index(index_path)
        with st.sidebar:
            st.success(f"✅ Загружено {len(qa_system.chunks)} фрагментов")
        return True

    st.sidebar.error("❌ Ошибка индексации")
    return False


# ========= ОСНОВНОЙ ИНТЕРФЕЙС =========

def render_sidebar(qa_system: QASystem, formula_engine: FormulaEngine, error_handler: ErrorHandler) -> None:
    """Рендер боковой панели."""
    with st.sidebar:
        st.header("📚 О системе")
        st.markdown("""
        - ✅ Семантический поиск по тексту
        - ✅ Инженерные расчёты
        - ✅ Извлечение нормативных параметров
        - ✅ Поиск таблиц и формул
        - ✅ Определения терминов
        """)
        st.divider()

        auto_load_documents()
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Перезагрузить индекс", use_container_width=True):
                index_path = PROCESSED_DIR / "faiss_index.pkl"
                if index_path.exists():
                    qa_system.load_index(index_path)
                    st.success(f"✅ Индекс перезагружен: {len(qa_system.chunks)} фрагментов")
                    st.rerun()
                else:
                    st.warning("⚠️ Индекс не найден")

        with col2:
            if st.button("🗑️ Очистить индекс", use_container_width=True):
                index_path = PROCESSED_DIR / "faiss_index.pkl"
                if index_path.exists():
                    index_path.unlink(missing_ok=True)
                    st.success("✅ Индекс очищен")
                    st.rerun()

        if st.button("📥 Синхронизировать dataset", use_container_width=True):
            with st.spinner("Скачивание документов из Hugging Face..."):
                if sync_hf_dataset_to_raw(force=True):
                    st.success("✅ Dataset синхронизирован")
                else:
                    st.error("❌ Не удалось синхронизировать dataset")
                st.rerun()

        if not qa_system.is_ready:
            if st.button("📚 Индексировать документы", key="index_btn", use_container_width=True):
                with st.spinner("Индексация..."):
                    if qa_system.index_documents(RAW_DIR):
                        index_path = PROCESSED_DIR / "faiss_index.pkl"
                        index_path.parent.mkdir(parents=True, exist_ok=True)
                        qa_system.save_index(index_path)
                        st.success(f"✅ Проиндексировано {len(qa_system.chunks)} фрагментов")
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
            + len(list(RAW_DIR.glob("*.doc")))
        )
        chunks_count = len(qa_system.chunks) if qa_system.is_ready else 0

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
                for i, err in enumerate(error_handler.errors[-5:], start=1):
                    st.error(f"{i}. {err.get('type', 'Error')}: {err.get('message', '')[:100]}")


def render_sources(sources) -> None:
    """Рендерит источники."""
    with st.expander("📚 Источники", expanded=False):
        for src in sources[:5]:
            if isinstance(src, dict):
                doc_name = src.get("doc_name") or src.get("docname") or src.get("source") or "Документ"
            else:
                doc_name = str(src)
            st.caption(f"📄 {doc_name}")


def render_tables(tables) -> None:
    """Рендерит таблицы."""
    with st.expander("📊 Таблицы", expanded=False):
        for table in tables[:3]:
            if isinstance(table, dict):
                rows = table.get("rows", [])
                if rows and isinstance(rows, list):
                    st.write(f"**{table.get('title', 'Таблица')}**")
                    st.dataframe(rows, use_container_width=True)
                else:
                    st.write(table)
            else:
                st.write(table)


def render_formulas(formulas) -> None:
    """Рендерит формулы."""
    with st.expander("📐 Формулы", expanded=False):
        for formula in formulas[:3]:
            if isinstance(formula, dict):
                st.code(formula.get("raw", ""))
            else:
                st.code(str(formula))


def render_reasoning(steps) -> None:
    """Рендерит цепочку рассуждений."""
    with st.expander("🧠 Цепочка рассуждений", expanded=False):
        for step in steps:
            st.caption(
                f"Шаг {step['step']}: {step['description']} "
                f"(уверенность: {step['confidence']:.0%})"
            )


def main() -> None:
    """Основная функция приложения."""
    init_session_state()

    qa_system = st.session_state.qa_system
    formula_engine = st.session_state.formula_engine
    agent_loop = st.session_state.agent_loop
    error_handler = st.session_state.error_handler

    st.title("🏗️ Инженерный помощник проектировщика")
    st.caption("📄 База знаний: ГОСТы, СП, технические регламенты и методические документы по строительству")

    render_sidebar(qa_system, formula_engine, error_handler)

    # Отображение сообщений
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

    # Обработка вопроса
    prompt = st.chat_input("Задайте вопрос по строительной документации...", key="main_chat_input")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        response = "Не удалось сформировать ответ."
        sources: list = []
        tables: list = []
        formulas: list = []

        with st.chat_message("assistant"):
            with st.spinner("🔍 Анализирую запрос..."):
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

                    # Быстрое определение
                    quick_def = get_quick_definition(prompt)
                    if quick_def:
                        response = (
                            f"📖 **Быстрое определение:**\n\n"
                            f"{quick_def.get('definition', '')}\n\n"
                            f"📚 **Источник:** {quick_def.get('source', '')}"
                        )
                        if quick_def.get("example"):
                            response += f"\n\n📌 **Пример:** {quick_def['example']}"
                        st.markdown(response)
                        st.stop()

                    if is_calc:
                        # Расчётный запрос
                        result = call_maybe_async(formula_engine.answer_calculation, prompt)
                        response = result.get("answer", "Не удалось выполнить расчёт")
                        sources = result.get("sources", [])
                        tables = result.get("tables", [])
                        formulas = result.get("formulas", [])

                        if not formulas and result.get("formula"):
                            formulas = [result["formula"]]

                    elif is_def:
                        # Определение термина
                        clean_term = prompt_lower
                        for trigger in def_triggers:
                            clean_term = clean_term.replace(trigger, "").strip(" ?!.,:")

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
                        # Поиск таблицы
                        result = qa_system.answer(prompt)
                        response = result.get("answer", "Таблица не найдена")
                        tables = result.get("tables", [])
                        sources = result.get("sources", [])
                        formulas = result.get("formulas", [])

                        if tables:
                            response += "\n\n📊 **Найденные таблицы:**\n"
                            for table in tables[:2]:
                                if isinstance(table, dict):
                                    response += f"\n**{table.get('title', 'Таблица')}**\n"
                                    content = table.get("content", "")
                                    if len(content) > 500:
                                        content = content[:500] + "..."
                                    response += f"```\n{content}\n```\n"

                    else:
                        # Агентский цикл для сложных запросов
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
                        response_id=current_id,
                    )

                except Exception as e:
                    error_info = error_handler.handle(e, {"query": prompt})
                    response = error_info.get("user_message", f"❌ Ошибка: {e}")
                    st.error(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        save_history()

    st.divider()
    st.caption("💡 Совет: для расчётов указывайте числа и параметры прямо в вопросе.")
    st.caption("📧 По всем вопросам обращайтесь к разработчику.")


if __name__ == "__main__":
    main()