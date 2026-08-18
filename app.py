# -*- coding: utf-8 -*-
"""
Инженерный чат-бот для работы с документацией.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from reportlab.lib.colors import HexColor

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None
    print("⚠️ huggingface-hub не установлен")

from core.agent_loop import AgentLoop
from core.config import PROCESSED_DIR, RAW_DIR, HF_DATASET_REPO_ID
from core.error_handler import ErrorHandler
from core.formula_engine import FormulaEngine
from core.parser import parse_directory
from core.prompts import get_quick_definition
from core.qa_engine import QASystem
from core.table_calculator import patch_app_with_table_calculator

st.set_page_config(
    page_title="Инженерный чат-бот",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

HISTORY_FILE = PROCESSED_DIR / "chat_history.json"
INDEX_FILE = PROCESSED_DIR / "faiss_index.pkl"
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".doc", ".rtf"}


def run_async_safely(async_func, *args, **kwargs):
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
    if func is None:
        raise ValueError("Передана пустая функция (None) в call_maybe_async")
    if not callable(func):
        raise TypeError(f"Объект {type(func).__name__} не является вызываемым")
    result = func(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return run_async_safely(lambda: result)
    return result


def get_initial_message() -> list[dict[str, str]]:
    return [{
        "role": "assistant",
        "content": """🏗️ **Здравствуйте!** Я инженерный помощник по строительной документации.

📖 **База знаний:** ГОСТы, СП, технические регламенты и методические документы

**Что я умею:**
• 📖 Отвечать на вопросы по нормативной документации
• 📐 Выполнять инженерные расчёты
• 📊 Находить таблицы и формулы
• 🔍 Искать определения терминов

📥 Для начала работы скачайте документы через кнопку в боковой панели."""
    }]


def save_history() -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)


def get_docs_from_raw() -> list[Path]:
    if not RAW_DIR.exists():
        return []
    return [
        p for p in RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def inspect_raw_directory() -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_files = [p for p in RAW_DIR.rglob("*") if p.is_file()]
    supported = [p for p in all_files if p.suffix.lower() in SUPPORTED_SUFFIXES]

    extensions = {}
    for path in all_files:
        ext = path.suffix.lower() or "[без расширения]"
        extensions[ext] = extensions.get(ext, 0) + 1

    return {
        "all_files": all_files,
        "supported_files": supported,
        "extensions": extensions,
        "total": len(all_files),
        "supported_count": len(supported),
    }


def sync_hf_dataset_to_raw(force: bool = False) -> bool:
    dataset_repo_id = (HF_DATASET_REPO_ID or "").strip()
    if not dataset_repo_id:
        print("❌ HF_DATASET_REPO_ID не задан")
        return False

    if snapshot_download is None:
        print("❌ huggingface-hub не установлен")
        return False

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing_docs = get_docs_from_raw()
    if existing_docs and not force:
        print(f"✅ Документы уже есть: {len(existing_docs)}")
        return True

    try:
        print(f"📥 Скачиваю датасет {dataset_repo_id} в {RAW_DIR} ...")
        snapshot_download(
            repo_id=dataset_repo_id,
            repo_type="dataset",
            local_dir=str(RAW_DIR),
            local_dir_use_symlinks=False,
        )

        all_files = [p for p in RAW_DIR.rglob("*") if p.is_file()]
        downloaded_docs = [p for p in all_files if p.suffix.lower() in SUPPORTED_SUFFIXES]

        print(f"📦 Всего файлов после скачивания: {len(all_files)}")
        print(f"📄 Поддерживаемых документов: {len(downloaded_docs)}")

        if not downloaded_docs:
            print("⚠️ Датасет скачан, но PDF/DOCX/DOC/RTF не найдено.")
            print("⚠️ Проверьте расширения файлов и структуру каталогов.")
            return False

        for path in downloaded_docs[:10]:
            print(f"   ✓ {path.relative_to(RAW_DIR)}")

        return True

    except Exception as e:
        print(f"❌ Ошибка загрузки датасета: {type(e).__name__}: {e}")
        return False


def force_rebuild_index(qa: QASystem) -> bool:
    print("=" * 50)
    print("🔨 ПРИНУДИТЕЛЬНАЯ ПЕРЕСТРОЙКА ИНДЕКСА")
    print("=" * 50)

    if not RAW_DIR.exists():
        print(f"❌ Папка {RAW_DIR} не существует")
        return False

    docs = get_docs_from_raw()
    print(f"📄 Найдено документов в RAW_DIR: {len(docs)}")

    if not docs:
        print("❌ Нет документов для индексации")
        return False

    print("📖 Начинаем парсинг документов...")
    parsed_docs = parse_directory(RAW_DIR, recursive=True)

    if not parsed_docs:
        print("❌ Парсинг не вернул ни одного документа")
        return False

    print(f"📄 Распарсено документов: {len(parsed_docs)}")

    total_chunks = sum(len(doc.get("chunks", [])) for doc in parsed_docs)
    print(f"🧩 Всего чанков: {total_chunks}")

    print("🔨 Строим индекс с эмбеддингами...")
    result = qa.build_index(parsed_docs)

    if not result:
        print("❌ build_index вернул False")
        return False

    print(f"✅ Индекс построен: {len(qa.chunks)} чанков")

    print("💾 Сохраняем индекс...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    save_result = qa.save_index(INDEX_FILE)
    if not save_result:
        print("❌ Ошибка сохранения индекса")
        return False

    if not INDEX_FILE.exists():
        print("❌ Файл индекса не появился")
        return False

    size = INDEX_FILE.stat().st_size
    print(f"✅ Индекс сохранён, размер: {size} байт")

    if size < 1_000_000:
        print(f"⚠️ Подозрительно маленький индекс: {size} байт")
        return False

    print("=" * 50)
    return True


def get_llm_status(qa_system: QASystem) -> dict[str, str]:
    """Возвращает безопасный статус LLM-провайдеров."""
    ollama_alive = False
    check_ollama = getattr(qa_system, "is_ollama_alive", None)

    if callable(check_ollama):
        try:
            ollama_alive = bool(check_ollama())
        except (ConnectionError, OSError, RuntimeError, ValueError):
            ollama_alive = False

    selected_provider = "unknown"
    get_provider = getattr(qa_system, "get_selected_provider", None)

    if callable(get_provider):
        try:
            selected_provider = str(get_provider())
        except (AttributeError, RuntimeError, ValueError):
            selected_provider = "unknown"

    return {
        "use_llm": str(getattr(qa_system, "use_llm", False)),
        "llm_provider": str(getattr(qa_system, "llm_provider", "unknown")),
        "llm_available": str(getattr(qa_system, "llm_available", False)),
        "selected_provider": selected_provider,
        "ollama_alive": str(ollama_alive),
        "ollama_model": str(getattr(qa_system, "ollama_model", ""))
    }

def init_qa_system() -> QASystem:
    use_llm = os.getenv("USE_LLM", "true").lower() == "true"
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
    ollama_model = os.getenv("OLLAMA_MODEL", "phi3:mini").strip()

    auto_sync = os.getenv("AUTO_SYNC_DATASET", "false").lower() == "true"
    auto_rebuild = os.getenv("AUTO_REBUILD_INDEX", "false").lower() == "true"

    print("🔧 Инициализация QASystem:")
    print(f"   use_llm: {use_llm}")
    print(f"   llm_provider: {llm_provider}")
    print(f"   ollama_base_url: {ollama_base_url}")
    print(f"   ollama_model: {ollama_model}")
    print(f"   auto_sync: {auto_sync}")
    print(f"   auto_rebuild: {auto_rebuild}")

    qa = QASystem(
        use_llm=use_llm,
        llm_provider=llm_provider if use_llm else "none",
        use_embeddings=True,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model
    )

    if INDEX_FILE.exists():
        print(f"📂 Индекс найден: {INDEX_FILE}")
        try:
            if qa.load_index(INDEX_FILE):
                print(f"✅ Индекс загружен: {len(qa.chunks)} чанков")
                return qa
        except Exception as e:
            print(f"⚠️ Ошибка загрузки индекса: {e}")

    if auto_sync:
        print("📥 AUTO_SYNC_DATASET=true → синхронизация dataset")
        sync_hf_dataset_to_raw(force=False)

    if auto_rebuild:
        print("🔨 AUTO_REBUILD_INDEX=true → перестройка индекса")
        force_rebuild_index(qa)
    else:
        print("⏭️ Автоперестройка индекса отключена")

    return qa


def init_session_state() -> None:
    if "qa_system" not in st.session_state:
        with st.spinner("Загрузка системы..."):
            st.session_state.qa_system = init_qa_system()
            st.session_state.formula_engine = FormulaEngine(st.session_state.qa_system)
            st.session_state.agent_loop = AgentLoop(
                st.session_state.qa_system,
                st.session_state.formula_engine,
            )
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
    qa_system = st.session_state.qa_system

    if getattr(qa_system, "is_ready", False):
        count = len(getattr(qa_system, "chunks", []))
        st.sidebar.success(f"✅ База знаний готова\n📄 {count} фрагментов")
        return True

    if INDEX_FILE.exists():
        try:
            if qa_system.load_index(INDEX_FILE):
                count = len(getattr(qa_system, "chunks", []))
                st.sidebar.success(f"✅ Индекс загружен\n📄 {count} фрагментов")
                return True
        except Exception as e:
            st.sidebar.warning(f"⚠️ Ошибка загрузки индекса: {e}")

    st.sidebar.info(
        "ℹ️ База знаний пока пуста.\n"
        "1. Скачайте датасет.\n"
        "2. Нажмите «Перестроить индекс»."
    )
    return False


def export_history_to_docx():
    try:
        from docx import Document
        doc = Document()
        doc.add_heading("Инженерный чат-бот — история", 0)
        doc.add_paragraph(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        for msg in st.session_state.messages:
            role = "Пользователь" if msg["role"] == "user" else "Ассистент"
            doc.add_heading(role, level=1)
            doc.add_paragraph(msg["content"])
        output_path = PROCESSED_DIR / f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))
        return output_path
    except ImportError:
        st.error("❌ python-docx не установлен")
        return None
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        return None


def export_to_docx(answer: str, sources: list, tables: list = None, formulas: list = None, filename: str = None):
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
        doc.add_heading("Ответ", level=1)
        doc.add_paragraph(answer)
        if tables:
            doc.add_heading("Таблицы", level=1)
            for table in tables[:2]:
                if isinstance(table, dict):
                    doc.add_paragraph(table.get("title", "Таблица"))
                    if table.get("content"):
                        doc.add_paragraph(table.get("content", ""))
        if formulas:
            doc.add_heading("Формулы", level=1)
            for formula in formulas[:3]:
                if isinstance(formula, dict):
                    doc.add_paragraph(formula.get("raw", ""))
        if sources:
            doc.add_heading("Источники", level=1)
            for src in sources:
                if isinstance(src, dict):
                    doc.add_paragraph(src.get("doc_name", "Документ"), style="List Bullet")
                else:
                    doc.add_paragraph(str(src), style="List Bullet")
        output_path = PROCESSED_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))
        return output_path
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        return None


def export_to_pdf(answer: str, sources: list, tables: list = None, formulas: list = None, filename: str = None):
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
            textColor=HexColor("#1a5276"),
            alignment=TA_CENTER,
            spaceAfter=20,
        )
        heading_style = ParagraphStyle(
            "HeadingStyle",
            parent=styles["Heading1"],
            fontSize=16,
            textColor=HexColor("#2e86c1"),
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
                story.append(Paragraph(line.replace("**", "").replace("*", ""), normal_style))

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
        return export_to_docx(answer, sources, tables, formulas, filename.replace(".pdf", ".docx"))
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        return None


def render_export_buttons(answer: str, sources: list, tables: list, formulas: list, key_suffix: str = "current", response_id: int | None = None):
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


def render_sidebar(qa_system: QASystem, formula_engine: FormulaEngine, error_handler: ErrorHandler) -> None:
    with st.sidebar:
        st.header("📚 О системе")
        st.markdown("""
- Семантический поиск по тексту
- Инженерные расчёты
- Поиск таблиц и формул
- Определения терминов
""")
        st.divider()

        st.subheader("🔍 Диагностика")
        st.write(f"INDEX_FILE: `{INDEX_FILE}`")
        st.write(f"Файл существует: `{INDEX_FILE.exists()}`")
        if INDEX_FILE.exists():
            st.write(f"Размер: `{INDEX_FILE.stat().st_size}` bytes")

        chunks_count = len(getattr(qa_system, "chunks", []))
        st.write(f"Чанков в памяти: `{chunks_count}`")
        st.write(f"is_ready: `{getattr(qa_system, 'is_ready', False)}`")

        diag = inspect_raw_directory()
        st.write(f"Файлов в RAW_DIR: `{diag['total']}`")
        st.write(f"Поддерживаемых документов: `{diag['supported_count']}`")

        if diag["extensions"]:
            with st.expander("📂 Расширения файлов"):
                st.json(diag["extensions"])

        with st.expander("📄 Первые файлы в RAW_DIR"):
            for path in diag["all_files"][:30]:
                try:
                    st.code(str(path.relative_to(RAW_DIR)))
                except Exception:
                    st.code(str(path))

        st.divider()

        st.subheader("🤖 Статус LLM")
        llm_status = get_llm_status(qa_system)
        st.write(f"use_llm: `{llm_status['use_llm']}`")
        st.write(f"llm_provider: `{llm_status['llm_provider']}`")
        st.write(f"llm_available: `{llm_status['llm_available']}`")
        st.write(f"ollama_alive: `{llm_status['ollama_alive']}`")
        st.write(f"ollama_model: `{llm_status['ollama_model']}`")

        st.divider()
        auto_load_documents()
        st.divider()

        st.subheader("📁 Управление данными")

        if st.button("📥 Скачать датасет", use_container_width=True):
            with st.spinner("Скачивание документов из Hugging Face..."):
                ok = sync_hf_dataset_to_raw(force=False)
            if ok:
                st.success("✅ Датасет скачан! Теперь можно строить индекс.")
                st.rerun()
            else:
                st.error("❌ Не удалось скачать датасет. Проверьте логи.")

        if st.button("🔄 Перескачать датасет", use_container_width=True):
            with st.spinner("Перескачиваю датасет..."):
                ok = sync_hf_dataset_to_raw(force=True)
            if ok:
                st.success("✅ Датасет перескачан!")
                st.rerun()
            else:
                st.error("❌ Ошибка перескачивания датасета.")

        st.divider()

        if st.button("🧠 Перестроить индекс", use_container_width=True):
            docs = get_docs_from_raw()
            if not docs:
                st.warning("⚠️ Сначала скачайте датасет: RAW_DIR пуст.")
            else:
                with st.spinner("Перестраиваю индекс..."):
                    ok = force_rebuild_index(qa_system)
                if ok:
                    st.success("✅ Индекс перестроен и сохранён!")
                    st.rerun()
                else:
                    st.error("❌ Ошибка перестройки индекса. Подробности в логах.")

        if st.button("🗑️ Очистить индекс", use_container_width=True):
            if INDEX_FILE.exists():
                INDEX_FILE.unlink(missing_ok=True)
                qa_system.is_ready = False
                st.success("✅ Индекс очищен")
                st.rerun()
            else:
                st.warning("⚠️ Индекс не найден")

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
            st.rerun()

        if getattr(error_handler, "errors", None):
            st.divider()
            st.subheader("⚠️ Ошибки")
            with st.expander(f"Показать {len(error_handler.errors)} ошибок"):
                for i, err in enumerate(error_handler.errors[-5:], start=1):
                    st.error(f"{i}. {err.get('type', 'Error')}: {err.get('message', '')[:100]}")


def main() -> None:
    init_session_state()

    qa_system = st.session_state.qa_system
    formula_engine = st.session_state.formula_engine
    agent_loop = st.session_state.agent_loop
    error_handler = st.session_state.error_handler

    st.title("🏗️ Инженерный помощник проектировщика")
    st.caption("📄 База знаний: ГОСТы, СП, технические регламенты и методические документы по строительству")

    render_sidebar(qa_system, formula_engine, error_handler)

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                has_sources = bool(msg.get("sources"))
                has_tables = bool(msg.get("tables"))
                has_formulas = bool(msg.get("formulas"))
                if has_sources or has_tables or has_formulas:
                    with st.expander("📎 Источники и материалы", expanded=False):
                        if has_sources:
                            st.markdown("**Источники:**")
                            for src in msg.get("sources", []):
                                if isinstance(src, dict):
                                    st.markdown(f"- {src.get('doc_name', 'Документ')}")
                                else:
                                    st.markdown(f"- {src}")
                        if has_tables:
                            st.markdown("**Таблицы:**")
                            for table in msg.get("tables", [])[:2]:
                                if isinstance(table, dict):
                                    st.markdown(f"- {table.get('title', 'Таблица')}")
                        if has_formulas:
                            st.markdown("**Формулы:**")
                            for formula in msg.get("formulas", [])[:5]:
                                if isinstance(formula, dict):
                                    st.markdown(f"- `{formula.get('raw', '')}`")
                                else:
                                    st.markdown(f"- `{formula}`")
                    render_export_buttons(
                        answer=msg["content"],
                        sources=msg.get("sources", []),
                        tables=msg.get("tables", []),
                        formulas=msg.get("formulas", []),
                        key_suffix=f"history_{i}",
                        response_id=i,
                    )

    prompt = st.chat_input("Задайте вопрос по строительной документации...", key="main_chat_input")
    if not prompt:
        return

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
                prompt_clean = prompt.strip()
                prompt_lower = prompt_clean.lower()

                calc_triggers = ["рассчитай", "вычисли", "посчитай", "формул"]
                definition_triggers = ["что такое ", "что значит ", "определение ", "определи "]
                table_triggers = ["таблица", "таблицы", "таблиц"]

                is_calc = any(w in prompt_lower for w in calc_triggers)
                is_definition_query = any(prompt_lower.startswith(t) for t in definition_triggers)
                is_table = any(w in prompt_lower for w in table_triggers)

                if is_definition_query:
                    clean_term = prompt_lower
                    for trigger in definition_triggers:
                        if clean_term.startswith(trigger):
                            clean_term = clean_term[len(trigger):].strip(" ?!.,:;\"'«»()[]")
                            break

                    quick_def = get_quick_definition(clean_term) if clean_term else None

                    if quick_def:
                        response = (
                            f"📖 **Определение:**\n\n"
                            f"{quick_def.get('definition', '')}\n\n"
                            f"📚 **Источник:** {quick_def.get('source', '')}"
                        )
                        if quick_def.get("example"):
                            response += f"\n\n📌 **Пример:** {quick_def['example']}"
                    elif clean_term:
                        definition_result = qa_system.find_definition(clean_term)
                        if definition_result.get("found"):
                            response = (
                                f"📖 **Определение термина «{clean_term}»:**\n\n"
                                f"{definition_result.get('definition', '')}\n\n"
                                f"📚 **Источник:** {definition_result.get('source', 'Нормативная база')}"
                            )
                        else:
                            response = f"⚠️ В документах не найдено определение для термина «{clean_term}»."
                    else:
                        response = "⚠️ Уточните термин для определения."

                elif is_calc:
                    result = call_maybe_async(formula_engine.answer_calculation, prompt_clean)
                    response = result.get("answer", "Не удалось выполнить расчёт")
                    sources = result.get("sources", [])
                    tables = result.get("tables", [])
                    formulas = result.get("formulas", [])
                    if not formulas and result.get("formula"):
                        formulas = [result["formula"]]

                elif is_table:
                    result = qa_system.answer(prompt_clean)
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
                    result = call_maybe_async(agent_loop.run, prompt_clean)
                    response = result.get("answer", "Не удалось получить ответ")
                    sources = result.get("sources", [])
                    tables = result.get("tables", [])
                    formulas = result.get("formulas", [])

                    if result.get("needs_clarification"):
                        questions = result.get("questions", [])
                        if questions:
                            response += "\n\n❓ **Уточните:**\n" + "\n".join([f"• {q}" for q in questions])

            except Exception as e:
                error_info = error_handler.handle(e, {"query": prompt})
                response = error_info.get("user_message", f"❌ Ошибка: {e}")

        st.markdown(response)

        has_sources = bool(sources)
        has_tables = bool(tables)
        has_formulas = bool(formulas)

        if has_sources or has_tables or has_formulas:
            with st.expander("📎 Источники и материалы", expanded=False):
                if has_sources:
                    st.markdown("**Источники:**")
                    for src in sources:
                        if isinstance(src, dict):
                            st.markdown(f"- {src.get('doc_name', 'Документ')}")
                        else:
                            st.markdown(f"- {src}")
                if has_tables:
                    st.markdown("**Таблицы:**")
                    for table in tables[:5]:
                        if isinstance(table, dict):
                            st.markdown(f"- {table.get('title', 'Таблица')}")
                if has_formulas:
                    st.markdown("**Формулы:**")
                    for formula in formulas[:5]:
                        if isinstance(formula, dict):
                            st.markdown(f"- `{formula.get('raw', '')}`")
                        else:
                            st.markdown(f"- `{formula}`")

        current_response_id = st.session_state.get("current_response_id", 0) + 1
        st.session_state.current_response_id = current_response_id

        render_export_buttons(
            answer=response,
            sources=sources,
            tables=tables,
            formulas=formulas,
            key_suffix="current",
            response_id=current_response_id,
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
        "tables": tables,
        "formulas": formulas,
    })
    save_history()


if __name__ == "__main__":
    main()
