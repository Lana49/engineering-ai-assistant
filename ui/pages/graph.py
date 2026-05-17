import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from core.knowledge_graph import get_graph_statistics


def render_graph_page(G, graph_stats, max_nodes, graph_type):
    """Отображение страницы графа знаний"""

    st.subheader("🕸️ Граф знаний")

    if graph_stats and graph_stats.get('nodes', 0) > 0:
        # Статистика графа
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Узлов", graph_stats['nodes'])
        with col2:
            st.metric("Связей", graph_stats['edges'])
        with col3:
            st.metric("Плотность", f"{graph_stats['density']:.3f}")

        # Ключевые понятия
        if graph_stats.get('top_nodes'):
            st.subheader("🔥 Ключевые понятия")
            top_df = pd.DataFrame(
                graph_stats['top_nodes'][:10],
                columns=['Понятие', 'Связей', 'Тип']
            )
            st.dataframe(top_df, width='stretch')

        # ВИЗУАЛИЗАЦИЯ ГРАФА
        st.subheader("📊 Визуализация графа")

        if graph_type == "Matplotlib":
            try:
                fig, ax = plt.subplots(figsize=(12, 8))
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

                # Цвета узлов
                color_map = {
                    'материал': '#87CEEB',
                    'конструкция': '#90EE90',
                    'параметр': '#F4A460',
                    'норматив': '#FFD700',
                    'unknown': '#CCCCCC'
                }
                node_colors = [color_map.get(G.nodes[node].get('type', 'unknown'), '#CCCCCC') for node in G.nodes()]
                node_sizes = [G.degree(node) * 150 + 300 for node in G.nodes()]

                nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
                nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray', ax=ax)

                labels = {node: node[:20] + '..' if len(node) > 20 else node for node in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

                plt.title('Граф знаний', fontsize=14)
                plt.axis('off')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Ошибка при построении графа: {e}")

        else:  # Интерактивный Plotly
            st.info("Интерактивный граф требует установки plotly. Используйте Matplotlib или установите: pip install plotly")

    else:
        st.info("⚠️ Недостаточно данных для построения графа знаний")