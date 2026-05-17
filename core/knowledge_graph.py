import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import defaultdict
import re
from utils.config import COLORS, MAX_NODES


def build_knowledge_graph(result):
    """Строит граф знаний на основе извлеченных сущностей (только реальные связи)"""
    G = nx.Graph()

    # Добавляем узлы
    for material in result.get('materials_lemmas', []):
        G.add_node(material, type='материал', color=COLORS['материал'])

    for structure in result.get('structures_lemmas', []):
        G.add_node(structure, type='конструкция', color=COLORS['конструкция'])

    for param in result.get('parameters_lemmas', []):
        G.add_node(param, type='параметр', color=COLORS['параметр'])

    for standard in result.get('standards', []):
        G.add_node(standard, type='норматив', color=COLORS['норматив'])

    # Приводим к спискам
    materials = list(result.get('materials_lemmas', []))
    structures = list(result.get('structures_lemmas', []))
    parameters = list(result.get('parameters_lemmas', []))
    all_lemmas = materials + structures + parameters

    # Реальные связи из текста (по предложениям)
    co_occurrence = defaultdict(int)
    text = result.get('full_text', '').lower()
    sentences = re.split(r'[.!?]\s+', text)

    for sent in sentences:
        sent_lower = sent.lower()
        lemmas_in_sent = [lemma for lemma in all_lemmas if lemma in sent_lower]

        for i, t1 in enumerate(lemmas_in_sent):
            for t2 in lemmas_in_sent[i+1:]:
                co_occurrence[(t1, t2)] += 1

    # Добавляем только реальные связи (без искусственных)
    for (t1, t2), weight in co_occurrence.items():
        if weight >= 1:  # если хоть раз встретились в одном предложении
            G.add_edge(t1, t2, weight=weight)

    # Удаляем изолированные узлы (без связей)
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    print(f"📊 ИТОГО: узлов={G.number_of_nodes()}, связей={G.number_of_edges()}")
    if isolated:
        print(f"🗑️ Удалено изолированных узлов: {len(isolated)}")

    return G


def get_graph_statistics(G):
    """Статистика графа"""
    if len(G.nodes) == 0:
        return {}

    stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'components': nx.number_connected_components(G),
    }

    degrees = dict(G.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    stats['top_nodes'] = [(node, degree, G.nodes[node].get('type', 'unknown'))
                          for node, degree in top_nodes]

    type_counts = defaultdict(int)
    for node, data in G.nodes(data=True):
        type_counts[data.get('type', 'unknown')] += 1
    stats['type_counts'] = dict(type_counts)

    return stats


def visualize_matplotlib(G, max_nodes=MAX_NODES):
    """Визуализация графа с matplotlib (статический)"""
    if len(G.nodes) == 0:
        return None

    # Ограничиваем количество узлов для читаемости
    if len(G.nodes) > max_nodes:
        degrees = dict(G.degree())
        top = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        G = G.subgraph([node for node, _ in top])

    fig, ax = plt.subplots(figsize=(14, 12))
    pos = nx.spring_layout(G, k=2, iterations=80, seed=42)

    # Цвета и размеры
    node_colors = [G.nodes[n].get('color', COLORS['unknown']) for n in G.nodes()]
    node_sizes = [G.degree(n) * 100 + 500 for n in G.nodes()]

    # Рисуем
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color='gray', ax=ax)

    # Подписи
    labels = {n: n[:20] + '..' if len(n) > 20 else n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

    # Легенда
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor=COLORS['материал'], label='Материалы'),
        Patch(facecolor=COLORS['конструкция'], label='Конструкции'),
        Patch(facecolor=COLORS['параметр'], label='Параметры'),
        Patch(facecolor=COLORS['норматив'], label='Нормативы')
    ]
    ax.legend(handles=legend, loc='upper left', fontsize=10)

    plt.title('Граф знаний строительной документации (реальные связи)', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    return fig


def visualize_plotly(G, max_nodes=MAX_NODES):
    """Интерактивная визуализация графа с plotly"""
    if len(G.nodes) == 0:
        return None

    if len(G.nodes) > max_nodes:
        degrees = dict(G.degree())
        top = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        G = G.subgraph([node for node, _ in top])

    pos = nx.spring_layout(G, k=2, iterations=80, seed=42)

    # Рёбра
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Узлы
    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Связей: {G.degree(node)}<br>Тип: {G.nodes[node].get('type', 'unknown')}")
        node_color.append(G.nodes[node].get('color', COLORS['unknown']))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.8, color='#888'), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()),
                             textposition="top center", hovertext=node_text,
                             marker=dict(size=[G.degree(node)*2+20 for node in G.nodes()],
                                         color=node_color, line=dict(width=1, color='black'))))

    fig.update_layout(title='Граф знаний строительной документации', showlegend=False,
                      hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    return fig