import streamlit as st
import pandas as pd


def render_statistics_tab(result):
    st.subheader("📊 Статистика документа")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Символов", f"{result['stats']['characters']:,}")
    with col2:
        st.metric("Слов", f"{result['stats']['words']:,}")
    with col3:
        st.metric("Предложений", f"{result['stats']['sentences']:,}")
    if result.get('sections'):
        st.subheader("📑 Разделы документа")
        for num, title in list(result['sections'].items())[:10]:
            st.write(f"**{num}** {title}")


def render_materials_tab(result):
    st.subheader("🏗️ Найденные материалы")
    if result.get('materials_lemmas'):
        lemmas = sorted(result['materials_lemmas'])
        df = pd.DataFrame({
            '№': range(1, len(lemmas) + 1),
            'Материал': lemmas
        })
        st.dataframe(df, width='stretch')
        st.caption(f"📊 **Всего уникальных материалов: {len(lemmas)}**")
    else:
        st.info("Материалы не найдены")


def render_standards_tab(result):
    st.subheader("📜 Нормативные документы")
    if result.get('standards'):
        df = pd.DataFrame({'Норматив': sorted(result['standards'])})
        st.dataframe(df, width='stretch')
        st.caption(f"📊 **Всего нормативов: {len(result['standards'])}**")
    else:
        st.info("Нормативы не найдены")


def render_parameters_tab(result):
    st.subheader("🌡️ Технические параметры")
    if result.get('temperatures'):
        st.write("**Температуры:**")
        st.write(", ".join(sorted(result['temperatures'])[:15]))
    if result.get('thicknesses'):
        st.write("**Толщины:**")
        st.write(", ".join(sorted(result['thicknesses'])[:15]))
    if result.get('densities'):
        st.write("**Плотности:**")
        st.write(", ".join(sorted(result['densities'])[:15]))


def render_sidebar():
    """Боковая панель для загрузки файлов"""
    with st.sidebar:
        st.header("📁 Управление")

        uploaded_file = st.file_uploader(
            "Загрузите документ",
            type=['txt', 'docx', 'pdf'],
            help="Поддерживаются форматы: TXT, DOCX, PDF"
        )

        st.divider()

        st.subheader("📄 Доступные документы")
        from utils.config import RAW_DIR
        docs = list(RAW_DIR.glob("*"))
        if docs:
            for doc in docs:
                st.write(f"• {doc.name}")
        else:
            st.info("Нет документов")

        return uploaded_file
