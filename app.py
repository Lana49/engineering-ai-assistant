# -*- coding: utf-8 -*-
"""
Инженерный чат-бот для работы с документацией.

Интегрирует:
- QASystem для поиска по документам
- FormulaEngine для инженерных расчётов
- AgentLoop для пошаговых рассуждений
- ErrorHandler для обработки ошибок
- Экспорт в DOCX и PDF
- Извлечение таблиц и расчёты по таблицам
- Mixed-режим с автоматическим выбором Ollama/Gemini
- Автосинхронизацию документов из Hugging Face Dataset
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import streamlit as st
from huggingface_hub import snapshot_download

# Загрузка .env через python-dotenv (установите: pip install python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent))

from core.agent_loop import AgentLoop
from core.error_handler import ErrorHandler
from core.formula_engine import FormulaEngine
from core.prompts import get_quick_definition
from core.qa_engine import QASystem
from core.table_calculator import TableCalculator
from core.table_extractor import patch_qa_system_with_table_extractor
from utils.config import HF_DATASET_REPO_ID, PROCESSED_DIR, RAW_DIR

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
• 📐 Рассчитывать толщину изоляции и тепловые потери
• 🌍 Вычислять ГСОП (градусо-сутки отопительного периода)
• 💨 Определять расход теплоты на вентиляцию
• 📊 Находить таблицы и формулы в документах
• 🔍 Искать определения терминов
• 🤖 Автоматический выбор между Ollama и Gemini

**Задайте свой вопрос или попросите сделать расчёт!**""",
        }
    ]


def save_history() -> None:
    """Сохраняет историю чата в JSON."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as file_obj:
        json.dump(st.session_state.messages, file_obj, ensure_ascii=False, indent=2)


def sync_hf_dataset_to_raw(force: bool = False) -> bool:
    """
    Скачивает документы из Hugging Face Dataset repo в RAW_DIR.

    Переменные окружения:
    - HF_DATASET_REPO_ID: например "Lana49/engineering-docs"
    - HF_TOKEN: токен Hugging Face (необязательно для публичного датасета)
    """
    dataset_repo_id = (HF_DATASET_REPO_ID or "").strip()
    if not dataset_repo_id:
        print("ℹ️ HF_DATASET_REPO_ID не задан, синхронизация dataset пропущена")
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

    hf_token = os.getenv("HF_TOKEN", "").strip() or None

    try:
        print(f"📥 Скачиваю dataset {dataset_repo_id} в {RAW_DIR} ...")
        snapshot_download(
            repo_id=dataset_repo_id,
            repo_type="dataset",
            local_dir=str(RAW_DIR),
            token=hf_token,
            resume_download=True,
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
        if st.button("📋 Копировать в буфер", key=f"copy_{unique_id}"):
            import html
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


def build_qa_system() -> QASystem:
    """
    Создаёт QASystem с поддержкой mixed-режима.

    Переменные окружения:
    - LLM_PROVIDER: "ollama", "gemini", "mixed", "none" (по умолчанию "mixed")
    - OLLAMA_BASE_URL: URL Ollama (по умолчанию "http://localhost:11434")
    - OLLAMA_MODEL: модель Ollama (по умолчанию "llama3.1:8b")
    - GEMINI_API_KEY: ключ для Gemini
    - GEMINI_MODEL: модель Gemini (по умолчанию "gemini-2.0-flash")
    - TOP_K: количество результатов (по умолчанию 5)
    - MIN_SCORE: минимальный score (по умолчанию 0.15)
    - USE_EMBEDDINGS: использовать эмбеддинги (по умолчанию true)
    - SEMANTIC_WEIGHT: вес семантики (по умолчанию 0.7)
    - LEXICAL_WEIGHT: вес лексики (по умолчанию 0.3)
    """
    llm_provider = os.getenv("LLM_PROVIDER", "mixed").strip().lower()
    use_llm = llm_provider not in {"none", ""}

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b").strip() or "llama3.1:8b"

    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip() or "gemini-2.0-flash"

    top_k = int(os.getenv("TOP_K", "5"))
    min_score = float(os.getenv("MIN_SCORE", "0.15"))
    use_embeddings = os.getenv("USE_EMBEDDINGS", "true").strip().lower() in {"1", "true", "yes", "on"}

    semantic_weight = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
    lexical_weight = float(os.getenv("LEXICAL_WEIGHT", "0.3"))

    try:
        qa = QASystem(
            use_llm=use_llm,
            llm_provider=llm_provider,
            top_k=top_k,
            min_score=min_score,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            use_embeddings=use_embeddings,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
        )

        if use_llm:
            if llm_provider == "mixed":
                print("✅ QASystem инициализирован в mixed-режиме")
                print(f"   Ollama: {ollama_model} ({ollama_base_url})")
                print(f"   Gemini: {gemini_model}")
            else:
                print(f"✅ QASystem инициализирован с {llm_provider}")
        else:
            print("ℹ️ QASystem в режиме без LLM")
        return qa
    except Exception as e:
        print(f"⚠️ Ошибка инициализации: {e}")
        print("ℹ️ Запуск в режиме без LLM")
        return QASystem(use_llm=False)


@st.cache_resource(show_spinner=False)
def get_qa_system() -> QASystem:
    """Кэшированное создание QASystem."""
    return build_qa_system()


def init_session_state() -> None:
    """Инициализация состояния сессии."""
    if "qa_system" not in st.session_state:
        st.session_state.qa_system = get_qa_system()

        idx_path = PROCESSED_DIR / "qa_index.pkl"
        if idx_path.exists():
            try:
                loaded = st.session_state.qa_system.load_index(idx_path)
                if loaded:
                    print("✅ Индекс загружен при старте")
                else:
                    print("⚠️ Не удалось загрузить индекс")
            except Exception as load_error:
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
    st.session_state.setdefault("current_provider", "none")

    if "error_handler" not in st.session_state:
        st.session_state.error_handler = ErrorHandler()


def auto_load_documents() -> bool:
    """Автоматическая загрузка и индексация документов."""
    qa_system = st.session_state.qa_system
    idx_path = PROCESSED_DIR / "qa_index.pkl"

    if qa_system.is_ready:
        st.sidebar.success(f"✅ База знаний готова\n📄 {len(qa_system.chunks)} фрагментов")
        return True

    if idx_path.exists():
        try:
            if qa_system.load_index(idx_path):
                st.sidebar.success(f"✅ Индекс загружен\n📄 {len(qa_system.chunks)} фрагментов")
                return True
        except Exception as load_error:
            st.sidebar.warning(f"⚠️ Ошибка загрузки индекса: {load_error}")

    docs = (
        list(RAW_DIR.glob("*.docx"))
        + list(RAW_DIR.glob("*.pdf"))
        + list(RAW_DIR.glob("*.rtf"))
        + list(RAW_DIR.glob("*.doc"))
    )

    if not docs:
        st.sidebar.info("📁 Папка документов пуста. Проверьте загрузку dataset или добавьте документы.")
        return False

    with st.sidebar:
        st.info(f"📚 Индексация {len(docs)} документов...")

    indexed = qa_system.index_documents(RAW_DIR)

    if indexed:
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        qa_system.save_index(idx_path)
        with st.sidebar:
            st.success(f"✅ Загружено {len(qa_system.chunks)} фрагментов")
        return True

    st.sidebar.error("❌ Ошибка индексации")
    return False


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
- 🤖 Mixed-режим (Ollama + Gemini)
"""
        )
        st.divider()

        llm_provider = os.getenv("LLM_PROVIDER", "mixed").strip().lower()

        if llm_provider == "mixed":
            ollama_available = qa_system.is_ollama_available() if hasattr(qa_system, "is_ollama_available") else False
            gemini_available = bool(getattr(qa_system, "gemini_api_key", ""))

            ollama_status = "✅" if ollama_available else "❌"
            gemini_status = "✅" if gemini_available else "❌"
            st.info(f"🤖 Mixed-режим\nOllama: {ollama_status}\nGemini: {gemini_status}")

        elif llm_provider == "ollama":
            ollama_available = qa_system.is_ollama_available() if hasattr(qa_system, "is_ollama_available") else False
            if ollama_available:
                st.success(f"🤖 Ollama: {qa_system.ollama_model}")
            else:
                st.warning("⚠️ Ollama не доступен")

        elif llm_provider == "gemini":
            if getattr(qa_system, "gemini_api_key", ""):
                st.success(f"🤖 Gemini: {qa_system.gemini_model}")
            else:
                st.warning("⚠️ Не задан GEMINI_API_KEY")
        else:
            st.info("🤖 Режим без LLM")

        st.caption(f"📂 RAW_DIR: {RAW_DIR}")
        st.caption(f"📦 Dataset: {HF_DATASET_REPO_ID or 'не задан'}")

        auto_load_documents()
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Перезагрузить индекс", use_container_width=True):
                idx_path = PROCESSED_DIR / "qa_index.pkl"
                if idx_path.exists():
                    st.session_state.qa_system.load_index(idx_path)
                    st.success(f"✅ Индекс перезагружен: {len(qa_system.chunks)} фрагментов")
                    st.rerun()
                else:
                    st.warning("⚠️ Индекс не найден")

        with col2:
            if st.button("🗑️ Очистить индекс", use_container_width=True):
                idx_path = PROCESSED_DIR / "qa_index.pkl"
                if idx_path.exists():
                    idx_path.unlink(missing_ok=True)
                    st.success("✅ Индекс очищен")
                    st.rerun()

        if st.button("📥 Синхронизировать dataset", use_container_width=True):
            with st.spinner("Скачивание документов из Hugging Face..."):
                ok = sync_hf_dataset_to_raw(force=True)
                if ok:
                    st.success("✅ Dataset синхронизирован")
                else:
                    st.error("❌ Не удалось синхронизировать dataset")
                st.rerun()

        if not qa_system.is_ready:
            if st.button("📚 Индексировать документы", key="index_btn", use_container_width=True):
                with st.spinner("Индексация..."):
                    result = qa_system.index_documents(RAW_DIR)
                    if result:
                        idx_path = PROCESSED_DIR / "qa_index.pkl"
                        idx_path.parent.mkdir(parents=True, exist_ok=True)
                        qa_system.save_index(idx_path)
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

    # Синхронизация документов с Hugging Face при старте
    sync_hf_dataset_to_raw()

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
        current_provider = "none"

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
                    calc_from_table_triggers = [
                        "по таблице", "из таблицы", "на основе таблицы",
                        "используя таблицу", "с помощью таблицы"
                    ]

                    is_calc = any(w in prompt_lower for w in calc_triggers)
                    is_def = any(w in prompt_lower for w in def_triggers)
                    is_table = any(w in prompt_lower for w in table_triggers)
                    is_calc_from_table = any(w in prompt_lower for w in calc_from_table_triggers)

                    if is_calc_from_table:
                        # === РАСЧЁТ НА ОСНОВЕ ТАБЛИЦЫ ===
                        calc = TableCalculator(qa_system)

                        cities = [
                            'москва', 'санкт-петербург', 'новосибирск', 'екатеринбург',
                            'казань', 'нижний новгород', 'челябинск', 'омск', 'самара',
                            'ростов-на-дону', 'уфа', 'красноярск', 'пермь', 'воронеж',
                            'волгоград', 'краснодар', 'сочи', 'владивосток', 'иркутск'
                        ]

                        found_city = None
                        for city in cities:
                            if city in prompt_lower:
                                found_city = city
                                break

                        if not found_city:
                            response = (
                                "⚠️ Не удалось определить город в запросе.\n\n"
                                "Поддерживаемые города: Москва, Санкт-Петербург, Новосибирск, "
                                "Екатеринбург, Казань, Нижний Новгород, Челябинск, Омск, Самара, "
                                "Уфа, Красноярск, Пермь, Воронеж, Волгоград, Краснодар, Сочи\n\n"
                                "Примеры:\n"
                                "- «Рассчитай ГСОП для Москвы по таблице»\n"
                                "- «Найди в таблице климат для Новосибирска и посчитай ГСОП»"
                            )
                        else:
                            if "вентиляц" in prompt_lower or "расход теплоты" in prompt_lower:
                                import re
                                flow_match = re.search(r'(\d+[.,]?\d*)\s*м³/ч', prompt_lower)
                                if not flow_match:
                                    flow_match = re.search(r'расход\s*(\d+[.,]?\d*)', prompt_lower)
                                if flow_match:
                                    air_flow = float(flow_match.group(1).replace(',', '.'))
                                    result = calc.calculate_ventilation_from_table(found_city, air_flow)
                                    response = result['answer']
                                    sources = [{'doc_name': result.get('source', 'Таблица')}]
                                else:
                                    response = (
                                        f"⚠️ Для расчёта вентиляции укажите расход воздуха (м³/ч)\n\n"
                                        f"Пример: «Рассчитай вентиляцию для {found_city.title()} с расходом 1000 м³/ч по таблице»"
                                    )
                            elif "теплопотер" in prompt_lower:
                                import re
                                area_match = re.search(r'площадь\s*(\d+[.,]?\d*)', prompt_lower)
                                res_match = re.search(r'сопротивление\s*(\d+[.,]?\d*)', prompt_lower)
                                if not area_match:
                                    area_match = re.search(r'A\s*=\s*(\d+[.,]?\d*)', prompt_lower)
                                if not res_match:
                                    res_match = re.search(r'R\s*=\s*(\d+[.,]?\d*)', prompt_lower)
                                if area_match and res_match:
                                    area = float(area_match.group(1).replace(',', '.'))
                                    resistance = float(res_match.group(1).replace(',', '.'))
                                    result = calc.calculate_heat_loss_from_table(found_city, area, resistance)
                                    response = result['answer']
                                    sources = [{'doc_name': result.get('source', 'Таблица')}]
                                else:
                                    response = (
                                        f"⚠️ Для расчёта теплопотерь укажите:\n"
                                        f"- площадь (м²)\n"
                                        f"- сопротивление теплопередаче (м²·°C/Вт)\n\n"
                                        f"Пример: «Рассчитай теплопотери для {found_city.title()} с площадью 100 м² и сопротивлением 2.5 по таблице»"
                                    )
                            else:
                                result = calc.calculate_gsop_from_table(found_city)
                                response = result['answer']
                                sources = [{'doc_name': result.get('source', 'Таблица')}]
                                if result.get('table'):
                                    tables = [result['table'].to_dict()]

                    elif is_calc:
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
                            if hasattr(qa_system, "find_definition"):
                                definition_result = qa_system.find_definition(clean_term)
                                if definition_result.get("found"):
                                    response = (
                                        f"📖 **Определение термина «{clean_term}»:**\n\n"
                                        f"{definition_result.get('definition', '')}\n\n"
                                        f"📚 **Источник:** {definition_result.get('source', 'Нормативная база')}"
                                    )
                                else:
                                    response = f"⚠️ В загруженных документах не найдено определение для термина «{clean_term}»."
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
                                if isinstance(table, dict):
                                    title = table.get("title", "Таблица")
                                    headers = table.get("headers", [])
                                    rows = table.get("rows", [])
                                    response += f"\n**{title}**\n"
                                    if headers:
                                        response += "| " + " | ".join(headers[:6]) + " |\n"
                                        response += "| " + " | ".join(["---"] * len(headers[:6])) + " |\n"
                                        for row in rows[:5]:
                                            padded = row + [""] * (len(headers[:6]) - len(row))
                                            response += "| " + " | ".join(
                                                str(cell).strip()[:30] for cell in padded[:6]) + " |\n"
                                    else:
                                        for row in rows[:5]:
                                            response += f"- " + " | ".join(row) + "\n"
                                    if len(rows) > 5:
                                        response += f"*... и ещё {len(rows) - 5} строк*\n"

                    else:
                        # === АГЕНТСКИЙ ЦИКЛ ДЛЯ СЛОЖНЫХ ЗАПРОСОВ ===
                        result = call_maybe_async(agent_loop.run, prompt)
                        response = result.get("answer", "Не удалось получить ответ")
                        sources = result.get("sources", [])
                        tables = result.get("tables", [])
                        formulas = result.get("formulas", [])
                        current_provider = result.get("provider", "none")

                        if result.get("needs_clarification"):
                            questions = result.get("questions", [])
                            if questions:
                                response += "\n\n❓ **Уточните:**\n" + "\n".join([f"• {q}" for q in questions])

                    # Показываем цепочку рассуждений
                    with st.sidebar:
                        with st.expander("🔍 Показать цепочку рассуждений"):
                            if is_calc or is_calc_from_table:
                                st.markdown("✅ Расчёт выполнен на основе данных из таблицы")
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
                    st.session_state.current_provider = current_provider

                    st.markdown(response)

                    if current_provider and current_provider != "none":
                        st.caption(f"🤖 Ответ сгенерирован через: **{current_provider.upper()}**")

                    render_export_buttons(
                        response,
                        sources,
                        tables,
                        formulas,
                        key_suffix="current",
                        response_id=current_id
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