import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from utils.config import RAW_DIR, PROCESSED_DIR
from core.parser import parse_construction_document, save_to_json
from core.knowledge_graph import build_knowledge_graph, get_graph_statistics
from ui.components import (
    render_statistics_tab, render_materials_tab,
    render_standards_tab, render_parameters_tab, render_sidebar
)
from ui.pages.graph import render_graph_page

# Настройка страницы
st.set_page_config(
    page_title="Анализ строительной документации",
    page_icon="📄",
    layout="wide"
)

# Заголовок
st.title("📄 Анализ строительной документации")
st.markdown("Загрузите документ для анализа: статистика, материалы, нормативы, параметры, граф знаний")

# Боковая панель
uploaded_file = render_sidebar()
max_nodes = 30
graph_type = "Matplotlib"
# Дополнительная информация в боковой панели
with st.sidebar:
    st.divider()
    st.subheader("📄 Доступные документы")
    docs = list(RAW_DIR.glob("*"))
    if docs:
        for doc in docs:
            st.write(f"• {doc.name}")
    else:
        st.info("Нет документов")

# Основные вкладки (без чата)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Статистика",
    "🏗️ Материалы",
    "📜 Нормативы",
    "🌡️ Параметры",
    "🕸️ Граф знаний"
])

# Загрузка и анализ документа
if uploaded_file is not None:
    file_path = RAW_DIR / uploaded_file.name
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Загружен: {uploaded_file.name}")

    if st.button("🔍 Анализировать", type="primary"):
        with st.spinner("Анализ документа..."):
            try:
                result = parse_construction_document(str(file_path))
                save_to_json(result, PROCESSED_DIR / f"{Path(uploaded_file.name).stem}_analysis.json")

                # Построение графа знаний
                G = build_knowledge_graph(result)
                graph_stats = get_graph_statistics(G)

                with tab1:
                    render_statistics_tab(result)
                with tab2:
                    render_materials_tab(result)
                with tab3:
                    render_standards_tab(result)
                with tab4:
                    render_parameters_tab(result)
                with tab5:
                    render_graph_page(G, graph_stats, max_nodes, graph_type)

                st.success("✅ Анализ завершён")
            except Exception as e:
                st.error(f"Ошибка: {e}")
else:
    for tab in [tab1, tab2, tab3, tab4, tab5]:
        with tab:
            st.info("👈 Загрузите документ для анализа")

st.divider()
st.caption("Инструмент для технического анализа строительной документации 2026")