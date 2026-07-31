# -*- coding: utf-8 -*-
"""
Главный файл приложения Streamlit.
Инженерная база знаний по строительной документации.
"""

from __future__ import annotations

import asyncio

import streamlit as st
from huggingface_hub import snapshot_download

from core.config import PROCESSED_DIR, RAW_DIR, HF_DATASET_REPO_ID
from core.parser import parse_directory
from core.qa_engine import QASystem
from core.formula_engine import FormulaEngine
from core.agent_loop import AgentLoop
from core.prompts import get_quick_definition
from core.table_extractor import patch_qa_system_with_table_extractor
from core.table_calculator import patch_app_with_table_calculator

st.set_page_config(
    page_title="Инженерная база знаний",
    page_icon="🏗️",
    layout="wide",
)

st.title("🏗️ Инженерная база знаний")
st.caption("Поиск по строительным нормам, ГОСТам и технической документации")


def sync_hf_dataset_to_raw(force: bool = False) -> bool:
    """
    Скачивает документы из Hugging Face Dataset repo в RAW_DIR.
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

    try:
        print(f"📥 Скачиваю dataset {dataset_repo_id} в {RAW_DIR} ...")
        snapshot_download(
            repo_id=dataset_repo_id,
            repo_type="dataset",
            local_dir=str(RAW_DIR),
            # token не нужен для публичного датасета
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


def run_async(coro):
    """Безопасный запуск async-кода внутри Streamlit."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


@st.cache_resource
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

            # АВТОМАТИЧЕСКАЯ ЗАГРУЗКА ДАТАСЕТА
            print("🔄 Проверяю наличие документов...")
            sync_hf_dataset_to_raw()

            _build_index(qa, index_path)
    except Exception as exc:
        print(f"⚠️ Ошибка загрузки индекса: {exc}")
        st.warning("⚠️ Индекс повреждён или несовместим. Выполняется пересборка...")

        # АВТОМАТИЧЕСКАЯ ЗАГРУЗКА ДАТАСЕТА
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
            # Сначала синхронизируем датасет
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
            patch_qa_system_with_table_extractor()
            patch_app_with_table_calculator()


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

    qa = st.session_state.qa_system

    if not getattr(qa, "is_ready", False):
        st.warning("⚠️ Система не готова. Проверьте наличие документов в data/raw/")

        # Кнопка для ручной загрузки
        if st.button("📥 Загрузить документы сейчас"):
            with st.spinner("Загрузка..."):
                sync_hf_dataset_to_raw(force=True)
                st.rerun()
        st.stop()

    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input(
            "Введите ваш вопрос:",
            placeholder="Например: что такое ГСОП? или рассчитай ГСОП для Москвы",
        )

    with col2:
        search_type = st.selectbox(
            "Тип поиска",
            ["hybrid", "semantic", "lexical"],
            index=0,
        )

    if query and st.button("🔍 Найти", type="primary"):
        with st.spinner("Поиск..."):
            quick_def = get_quick_definition(query)
            if quick_def:
                st.markdown("### 📖 Быстрое определение")
                st.markdown(f"**{quick_def.get('definition', '')}**")
                st.caption(f"📚 Источник: {quick_def.get('source', '')}")
                if quick_def.get("example"):
                    st.info(f"📌 Пример: {quick_def['example']}")
                st.stop()

            if search_type != "hybrid" and hasattr(qa, "search"):
                direct = qa.search(query, top_k=5, search_type=search_type)
                if direct:
                    st.markdown("### 📌 Ответ")
                    st.write(direct[0].get("text", "Ответ не найден."))
                    render_sources(direct)
                    st.stop()

            result = run_async(st.session_state.agent_loop.run(query))

            st.markdown("### 📌 Ответ")
            st.write(result.get("answer", "Ответ не найден."))

            if result.get("sources"):
                render_sources(result["sources"])

            if result.get("tables"):
                render_tables(result["tables"])

            if result.get("formulas"):
                render_formulas(result["formulas"])

            if result.get("reasoning_steps"):
                render_reasoning(result["reasoning_steps"])


if __name__ == "__main__":
    main()